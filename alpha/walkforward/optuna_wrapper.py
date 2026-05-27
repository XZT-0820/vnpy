from typing import TYPE_CHECKING

from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.dataset import AlphaDataset
from vnpy.alpha import logger

from .window import WalkForwardWindow

if TYPE_CHECKING:
    from .runner import WalkForwardRunner


class WalkForwardOptunaObjective:
    """
    将 WalkForwardRunner.run() 包装为 Optuna 多目标优化

    同时优化：
        - total_return      （最大化）
        - max_ddpercent     （最大化，因该值本身为负，最大化即最小化回撤）

    使用方式：
        study = optuna.create_study(directions=["maximize", "maximize"])
        study.optimize(
            WalkForwardOptunaObjective(runner_class, name_prefix, lab, dataset, windows),
            n_trials=50,
        )
    """

    def __init__(
        self,
        runner_class: type["WalkForwardRunner"],
        name_prefix: str,
        lab: AlphaLab,
        dataset: AlphaDataset,
        windows: list[WalkForwardWindow],
        optimize_targets: list[str] | None = None,
    ) -> None:
        self.runner_class = runner_class
        self.name_prefix = name_prefix
        self.lab = lab
        self.dataset = dataset
        self.windows = windows
        self.optimize_targets = optimize_targets or ["total_return", "max_ddpercent"]

    def __call__(self, trial) -> tuple[float, float]:
        import optuna

        # ---------- 模型超参数空间 ----------
        model_params = {
            "num_leaves": trial.suggest_int("num_leaves", 256, 2048, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.01, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 100.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 100.0),
            "n_quantiles": trial.suggest_int("n_quantiles", 10, 50),
        }

        # ---------- 选股策略超参数空间 ----------
        strategy_params = {
            "top_k": trial.suggest_int("top_k", 10, 100),
            "min_days": trial.suggest_int("min_days", 1, 20),
            "cash_ratio": trial.suggest_float("cash_ratio", 0.8, 1.0),
            "open_rate": 0.0005,
            "close_rate": 0.0015,
            "min_commission": 5,
            "slippage": 0.0006,
            "price_add": 0.05,
        }

        # ---------- 回测参数（固定） ----------
        backtest_params = {
            "interval": "DAILY",
            "capital": 100_000,
            "risk_free": 0.0,
            "annual_days": 240,
            "min_commission": 5.0,
            "slippage": 0.0006,
            "adjust_type": "none",
        }

        # ---------- 创建 runner 并执行 ----------
        name = f"{self.name_prefix}_trial{trial.number}"
        runner = self.runner_class(
            name=name,
            lab=self.lab,
            dataset=self.dataset,
            model_params=model_params,
            strategy_params=strategy_params,
            backtest_params=backtest_params,
            seed=42,  # 固定 seed，保证可复现
        )

        logger.info(f"[Optuna Trial {trial.number}] 开始滚动回测: {name}")
        stats = runner.run(self.windows)

        # ---------- 提取多目标 ----------
        total_return = float(stats.get("total_return", 0.0))
        max_ddpercent = float(stats.get("max_ddpercent", 0.0))

        logger.info(
            f"[Optuna Trial {trial.number}] 完成: "
            f"total_return={total_return:.2f}%, max_ddpercent={max_ddpercent:.2f}%"
        )

        return total_return, max_ddpercent
