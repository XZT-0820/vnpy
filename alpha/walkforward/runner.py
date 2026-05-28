import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Type
import polars as pl
import pandas as pd
import numpy as np
from empyrical import annual_return
from vnpy.trader.constant import Interval
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha import Segment, AlphaDataset
from vnpy.alpha.strategy import BacktestingEngine2
from vnpy.alpha.strategy.strategies.equity_demo_strategy2 import EquityDemoStrategy2
from vnpy.alpha import logger, AlphaStrategy
import lightgbm as lgb
from .window import WalkForwardWindow


class WalkForwardRunner(ABC):
    """
    Walk Forward 滚动回测运行器（纯框架）

    使用方式：
        1. 继承本类，重写 extract_data / train_model / predict_signal
        2. 实例化后调用 run(windows)

    属性说明：
        name                : 实验名称，用于保存路径
        lab                 : AlphaLab 实例
        dataset             : AlphaDataset 实例
        engine              : BacktestingEngine2 实例（run 后可用）
        model_params        : 模型超参数字典
        strategy_params     : 选股、挂单策略超参数字典
        backtest_params     : 回测引擎参数字典
        benchmark_symbol    : 股票池指数代码
        strategy_class      : 选股策略类（默认 EquityDemoStrategy2）
        seed                : 随机种子
        models              : 每轮窗口训练好的模型列表
        signals             : 每轮窗口测试集信号列表（最后 append 拼接后的完整信号）
    """

    def __init__(
        self,
        name: str,
        lab: AlphaLab,
        dataset: AlphaDataset,
        engine_class: Type[BacktestingEngine2],
        strategy_class: Type[AlphaStrategy],
        windows: list[WalkForwardWindow],
        model_params: dict | None = None,
        strategy_params: dict | None = None,
        backtest_params: dict | None = None,
        benchmark_symbol: str = "000300.SSE",
        n_quantiles: int | None = None,  # rank任务需要
        seed: int = 42,
    ) -> None:
        self.name: str = name
        self.lab: AlphaLab = lab
        self.dataset: AlphaDataset = dataset

        # 参数属性化，支持运行时修改
        self.model_params: dict = model_params or {}
        self.strategy_params: dict = strategy_params or {}
        self.backtest_params: dict = backtest_params or {}
        self.benchmark_symbol: str = benchmark_symbol
        self.strategy_class: type = strategy_class or EquityDemoStrategy2
        self.seed: int = seed

        self.engine_class = engine_class
        # 回测引擎（run 时初始化）
        self.engine: BacktestingEngine2
        # 中间结果存储
        self.models: list = []
        self.signals: list[pl.DataFrame] = []

        # 保存路径
        self.model_dir: Path = lab.model_path / "walkforward" / name
        self.signal_dir: Path = lab.signal_path / "walkforward" / name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.signal_dir.mkdir(parents=True, exist_ok=True)

        self.n_quantiles = n_quantiles
        self.windows = windows

    # =====================================================================
    # 抽象方法：必须由子类实现
    # =====================================================================

    @abstractmethod
    def extract_data(
        self,
        window: WalkForwardWindow,
    ) :
        """
        从 dataset 中提取当前窗口的 train/valid/test 数据

        """
        raise NotImplementedError

    @abstractmethod
    def train_model(
        self,
        **params: dict,
    ) -> object:
        """
        训练模型

        Parameters
        ----------
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        group_train: np.ndarray | None = None,
        group_valid: np.ndarray | None = None,


        Returns
        -------
        object
            训练好的模型对象
        """
        raise NotImplementedError

    @abstractmethod
    def predict_signal(
        self,
        model: object,
        **params: dict,
    ) -> pl.DataFrame:
        """
        在测试集上预测，返回信号 DataFrame

        Returns
        -------
        pl.DataFrame
            必须包含列 [datetime, vt_symbol, signal]
        """
        raise NotImplementedError

    # =====================================================================
    # 核心编排方法
    # =====================================================================
    @abstractmethod
    def run_window(self, window: WalkForwardWindow) -> pl.DataFrame:
        """
        执行单个窗口：
        1. extract_data() 切分数据
        2. train_model() 训练
        3. predict_signal() 预测
        4. 模型 append 到 self.models
        5. 返回信号
        """

        raise NotImplementedError

    def run(self, windows: list[WalkForwardWindow], log: bool = True) -> dict:
        """
        执行完整滚动回测：
        1. 遍历所有窗口，run_window() 收集信号
        2. 拼接信号为一个完整 df，append 到 self.signals
        3. 保存拼接信号
        4. 初始化 BacktestingEngine2 并执行一次回测
        5. 返回 calculate_statistics() 结果
        """

        original_handlers = None
        if not log:
            import sys
            original_handlers = logger._core.handlers.copy()
            logger.remove()
            logger.add(sys.stderr, level="WARNING")


        # 1. 逐窗口执行
        for window in windows:
            signal = self.run_window(window)
            self.signals.append(signal)
            logger.info(f"[Window {window.index}] 获取信号完成，shape={signal.shape}")

        # 2. 拼接完整信号
        logger.info("拼接所有窗口测试集信号...")
        all_signal: pl.DataFrame = pl.concat(self.signals).sort(["datetime", "vt_symbol"]).unique()
        self.signals.append(all_signal)
        self.save_all_signal(all_signal)

        # 3. 初始化回测引擎
        test_start = windows[0].test_start
        test_end = windows[-1].test_end

        logger.info(f"初始化回测引擎: {test_start.date()} ~ {test_end.date()}")
        self.engine = self.engine_class(self.lab)

        # 加载股票池
        component_symbols = self.lab.load_component_symbols(
            self.benchmark_symbol,
            test_start,
            test_end,
        )

        # 回测参数（含默认值）
        interval = self.backtest_params.get("interval", Interval.DAILY)
        capital = self.backtest_params.get("capital", 100_000)
        min_commission = self.backtest_params.get("min_commission", 5.0)
        risk_free = self.backtest_params.get("risk_free", 0.0)
        annual_days = self.backtest_params.get("annual_days", 240)
        slippage = self.backtest_params.get("slippage", 0.0006)
        adjust_type = self.backtest_params.get("adjust_type", "none")

        self.engine.set_parameters(
            vt_symbols=component_symbols,
            interval=interval,
            start=test_start,
            end=test_end,
            capital=capital,
            min_commission=min_commission,
            risk_free=risk_free,
            annual_days=annual_days,
            slippage=slippage,
            adjust_type=adjust_type,
        )

        # 4. 添加策略并回测
        logger.info(f'载入选股策略{self.strategy_class}...')
        self.engine.add_strategy(self.strategy_class, self.strategy_params, all_signal)
        logger.info(f'WalkForward {self.name} 回测开始：')
        self.engine.load_data()
        self.engine.run_backtesting()
        self.engine.calculate_result()
        stats = self.engine.calculate_statistics()
        if log:
            self.engine.show_chart()
            # 显示超额收益分析结果
            self.engine.show_performance(benchmark_symbol=self.benchmark_symbol)
        logger.info(f"WalkForward {self.name} 回测完成")

        if not log and original_handlers is not None:
            logger.remove()
            for handler_id, handler_config in original_handlers.items():
                logger.add(
                    handler_config._sink,
                    level=handler_config._level,
                    format=handler_config._format,
                    filter=handler_config._filter,
                )

        return stats




    # =====================================================================
    # 保存 / 加载方法
    # =====================================================================

    def save_window_model(self, index: int, model: object) -> None:
        """保存窗口模型到 MF_lab/model/walkforward/NAME/{index}.pkl"""
        file_path: Path = self.model_dir / f"{index}.pkl"
        with open(file_path, mode="wb") as f:
            pickle.dump(model, f)
        logger.info(f"模型已保存: {file_path}")

    def load_window_model(self, index: int) -> object:
        """加载窗口模型"""
        file_path: Path = self.model_dir / f"{index}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
        with open(file_path, mode="rb") as f:
            return pickle.load(f)

    def save_window_signal(self, index: int, signal_df: pl.DataFrame) -> None:
        """保存窗口信号到 MF_lab/signal/walkforward/NAME/{index}.parquet"""
        file_path: Path = self.signal_dir / f"{index}.parquet"
        signal_df.write_parquet(file_path)
        logger.info(f"信号已保存: {file_path}")

    def save_all_signal(self, signal_df: pl.DataFrame) -> None:
        """保存拼接后的完整信号到 MF_lab/signal/walkforward/NAME/all.parquet"""
        file_path: Path = self.signal_dir / "all.parquet"
        signal_df.write_parquet(file_path)
        logger.info(f"完整信号已保存: {file_path}")

    def load_window_signal(self, index: int) -> pl.DataFrame:
        """加载窗口信号"""
        file_path: Path = self.signal_dir / f"{index}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"信号文件不存在: {file_path}")
        return pl.read_parquet(file_path)

    def load_all_signal(self) -> pl.DataFrame:
        """加载拼接后的完整信号"""
        file_path: Path = self.signal_dir / "all.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"信号文件不存在: {file_path}")
        return pl.read_parquet(file_path)




# --------------------------------LightGBM LambdaRank------------------------------
class LGBMLR_Runner(WalkForwardRunner):
    """
    使用 LightGBM LambdaRank 的滚动回测实现
    """

    def extract_data(self, window):
        """从 dataset 中按时间切分数据"""
        self.dataset.data_periods[Segment.TRAIN] = (window.train_start, window.train_end)
        self.dataset.data_periods[Segment.VALID] = (window.valid_start, window.valid_end)
        self.dataset.data_periods[Segment.TEST] = (window.test_start, window.test_end)

        logger.info('提取 LambdaRank 训练数据...')
        X_train, y_train, meta_train, group_train = self.dataset.extract_lambdarank_data(
            Segment.TRAIN, n_quantiles=self.n_quantiles
        )

        logger.info('提取验证数据...')
        X_valid, y_valid, meta_valid, group_valid = self.dataset.extract_lambdarank_data(
            Segment.VALID, n_quantiles=self.n_quantiles
        )

        logger.info('提取测试数据...')
        X_test, y_test, meta_test, group_test = self.dataset.extract_lambdarank_data(
            Segment.TEST, n_quantiles=self.n_quantiles
        )

        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "X_test": X_test,
            "y_test": y_test,
            "meta_train": meta_train,
            "meta_valid": meta_valid,
            "meta_test": meta_test,
            "group_train": group_train,
            "group_valid": group_valid,
            "group_test": group_test,
        }

        return data

    def train_model(self, **params) -> object:
        """LambdaRank 训练"""
        logger.info('开始训练 LambdaRank 模型...')
        X_train = params["X_train"]
        y_train = params["y_train"]
        X_valid = params["X_valid"]
        y_valid = params["y_valid"]
        group_train = params["group_train"]
        group_valid = params["group_valid"]
        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
        valid_data = lgb.Dataset(
            X_valid, label=y_valid, group=group_valid, reference=train_data
        )

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10],
            'label_gain': [i**2 for i in range(self.n_quantiles)],
            'lambdarank_truncation_level': 100,
            'num_leaves': self.model_params.get('num_leaves', 1024),
            'max_depth': self.model_params.get('max_depth', -1),
            'min_data_in_leaf': self.model_params.get('min_data_in_leaf', 300),
            'learning_rate': self.model_params.get('learning_rate', 0.001),
            'feature_fraction': self.model_params.get('feature_fraction', 0.88),
            'bagging_fraction': self.model_params.get('bagging_fraction', 0.87),
            'bagging_freq': self.model_params.get('bagging_freq', 5),
            'lambda_l1': self.model_params.get('lambda_l1', 30),
            'lambda_l2': self.model_params.get('lambda_l2', 0.0),
            'boosting_type': 'gbdt',
            'device': 'gpu',
            'verbose': -1,
            'seed': self.seed,
            'num_threads': -1,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(100, first_metric_only=True),
                lgb.log_evaluation(period=0),  # 设为 1 可查看训练日志
            ]
        )
        logger.info(f'训练完成！最佳迭代轮数: {model.best_iteration}')
        if model.best_score:
            logger.info(f'VALID BEST NDCG: {model.best_score['valid']}')
            logger.info(f'TRAIN NDCG: {model.best_score['train']}')

        return model

    def predict_signal(self, model, **params) -> pl.DataFrame:
        """在测试集上预测，返回信号"""
        logger.info('---在测试集上预测---')
        X_test = params["X_test"]
        meta_test = params["meta_test"]
        predictions = model.predict(X_test, num_iteration=model.best_iteration)
        logger.info(f'预测完成，预测样本数:{len(predictions)}')

        signal = meta_test.with_columns([
            pl.Series('signal', predictions)
        ])

        logger.info(f'siganl.shape: {signal.shape}')
        logger.info('siganl:')
        logger.info(signal.head(5))
        logger.info(signal.tail(5))

        return signal

    def run_window(self, window: WalkForwardWindow) -> pl.DataFrame:
        logger.info(f"[Window {window.index}] 执行：")
        data = self.extract_data(window)

        model = self.train_model(**data)
        self.models.append(model)

        signal = self.predict_signal(model, **data)

        # 保存中间结果
        self.save_window_model(window.index, model)
        self.save_window_signal(window.index, signal)

        return signal