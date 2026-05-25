import gc
import json
import shutil
import shelve
import pickle
from pathlib import Path
from datetime import datetime, timedelta, date
from collections import defaultdict
from functools import lru_cache

import polars as pl
from multipart import file_path
from pandas.core.arrays import interval

from vnpy.trader.object import BarData, IndustryData, FactorData, FactorRequest
from vnpy.trader.constant import Interval, FactorType, Exchange
from vnpy.trader.utility import extract_vt_symbol
from win32comext.shell.demos.servers.folder_view import folderViewImplContextMenuIDs
from yfinance import Industry

from .logger import logger
from .dataset import AlphaDataset, to_datetime, Segment
from .model import AlphaModel

#导入因子定义模块
from vnpy.factor_define import (
    FACTOR_REGISTRY,
    PARAMS_REGISTRY,
    FACTOR_NAMES
)

class AlphaLab:
    """Alpha Research Laboratory"""

    def __init__(self, lab_path: str) -> None:
        """Constructor"""
        # Set data paths
        self.lab_path: Path = Path(lab_path)

        self.K_path: Path = self.lab_path.joinpath("K")

        self.daily_path: Path = self.K_path.joinpath("daily")
        self.daily_pre_path: Path = self.K_path.joinpath("daily_pre")
        self.daily_post_path: Path = self.K_path.joinpath("daily_post")

        self.minute_path: Path = self.K_path.joinpath("minute")
        self.minute_pre_path: Path = self.K_path.joinpath("minute_pre")
        self.minute_post_path: Path = self.K_path.joinpath("minute_post")

        self.weekly_path: Path = self.K_path.joinpath("weekly")
        self.weekly_pre_path: Path = self.K_path.joinpath("weekly_pre")
        self.weekly_post_path: Path = self.K_path.joinpath("weekly_post")

        self.monthly_path: Path = self.K_path.joinpath("monthly")
        self.monthly_pre_path: Path = self.K_path.joinpath("monthly_pre")
        self.monthly_post_path: Path = self.K_path.joinpath("monthly_post")

        self.component_path: Path = self.lab_path.joinpath("component")

        self.dataset_path: Path = self.lab_path.joinpath("dataset")
        self.model_path: Path = self.lab_path.joinpath("model")
        self.signal_path: Path = self.lab_path.joinpath("signal")

        self.contract_path: Path = self.lab_path.joinpath("contract.json")

        self.trading_days_path: Path = self.lab_path.joinpath("trading_days.json")  # list[str] 加载后是list[date]

        self.industry_path: Path = self.lab_path.joinpath("industry")

        self.factor_path: Path = self.lab_path.joinpath("factor")
        self.fundamental_facor_path: Path = self.factor_path.joinpath("fundamental")
        self.price_volume_factor_path: Path = self.factor_path.joinpath("price and volume")

        # Create folders
        for path in [
            self.lab_path,
            self.K_path,
            self.daily_path,
            self.daily_pre_path,
            self.daily_post_path,
            self.minute_path,
            self.minute_pre_path,
            self.minute_post_path,
            self.weekly_path,
            self.weekly_pre_path,
            self.weekly_post_path,
            self.monthly_path,
            self.monthly_pre_path,
            self.monthly_post_path,
            self.component_path,
            self.dataset_path,
            self.model_path,
            self.signal_path,
            self.industry_path,
            self.factor_path,
            self.fundamental_facor_path,
            self.price_volume_factor_path
        ]:
            if not path.exists():
                path.mkdir(parents=True)

    def save_bar_data(self, bars: list[BarData]) -> None:
        """Save bar data"""
        if not bars:
            return

        # Get file path
        bar: BarData = bars[0]

        if bar.interval == Interval.DAILY and bar.adjust_type == "none":
            file_path: Path = self.daily_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.DAILY and bar.adjust_type == "pre":
            file_path: Path = self.daily_pre_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.DAILY and bar.adjust_type == "post":
            file_path: Path = self.daily_post_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.MINUTE and bar.adjust_type == "none":
            file_path = self.minute_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.MINUTE and bar.adjust_type == "pre":
            file_path = self.minute_pre_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.MINUTE and bar.adjust_type == "post":
            file_path = self.minute_post_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.WEEKLY and bar.adjust_type == "none":
            file_path = self.weekly_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.WEEKLY and bar.adjust_type == "pre":
            file_path = self.weekly_pre_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.WEEKLY and bar.adjust_type == "post":
            file_path = self.weekly_post_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif interval == Interval.MONTHLY and bar.adjust_type == "none":
            file_path = self.monthly_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif interval == Interval.MONTHLY and bar.adjust_type == "pre":
            file_path = self.monthly_pre_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif interval == Interval.MONTHLY and bar.adjust_type == "post":
            file_path = self.monthly_post_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval:
            logger.error(f"Unsupported interval {bar.interval.value}")
            return

        data: list = []
        for bar in bars:
            bar_data: dict = {
                "datetime": bar.datetime.replace(tzinfo=None),
                "open": bar.open_price,
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
                "turnover": bar.turnover,
                "open_interest": bar.open_interest,
            }
            data.append(bar_data)

        new_df: pl.DataFrame = pl.DataFrame(data)

        # If file exists, read and merge
        if file_path.exists():
            old_df: pl.DataFrame = pl.read_parquet(file_path)

            new_df = pl.concat([old_df, new_df])

            new_df = new_df.unique(subset=["datetime"],keep = "last")

            new_df = new_df.sort("datetime")

        # Save to file
        new_df.write_parquet(file_path)

    def load_bar_data(
        self,
        vt_symbol: str,
        interval: Interval | str,
        start: datetime | str,
        end: datetime | str,
        adjust_type: str
    ) -> list[BarData]:
        """Load bar data"""
        # Convert types
        if isinstance(interval, str):
            interval = Interval(interval)

        start = to_datetime(start)
        end = to_datetime(end)

        # Get folder path
        if interval == Interval.DAILY and adjust_type == "none":
            folder_path: Path = self.daily_path
        elif interval == Interval.DAILY and adjust_type == "pre":
            folder_path: Path = self.daily_pre_path
        elif interval == Interval.DAILY and adjust_type == "post":
            folder_path: Path = self.daily_post_path
        elif interval == Interval.MINUTE and adjust_type == "none":
            folder_path = self.minute_path
        elif interval == Interval.MINUTE and adjust_type == "pre":
            folder_path: Path = self.minute_pre_path
        elif interval == Interval.MINUTE and adjust_type == "post":
            folder_path: Path = self.minute_post_path
        elif interval == Interval.WEEKLY and adjust_type == "none":
            folder_path = self.weekly_path
        elif interval == Interval.WEEKLY and adjust_type == "pre":
            folder_path: Path = self.weekly_pre_path
        elif interval == Interval.WEEKLY and adjust_type == "post":
            folder_path: Path = self.weekly_post_path
        elif interval == Interval.MONTHLY and adjust_type == "none":
            folder_path = self.monthly_path
        elif interval == Interval.MONTHLY and adjust_type == "pre":
            folder_path: Path = self.monthly_pre_path
        elif interval == Interval.MONTHLY and adjust_type == "post":
            folder_path: Path = self.monthly_post_path
        else:
            logger.error(f"Unsupported interval {interval.value}")
            return []

        # Check if file exists
        file_path: Path = folder_path.joinpath(f"{vt_symbol}.parquet")
        if not file_path.exists():
            logger.error(f"File {file_path} does not exist")
            return []

        # Open file
        df: pl.DataFrame = pl.read_parquet(file_path)

        # Filter by date range
        df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))

        # Convert to BarData objects
        bars: list[BarData] = []

        symbol, exchange = extract_vt_symbol(vt_symbol)

        for row in df.iter_rows(named=True):
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=row["datetime"],
                interval=interval,
                open_price=row["open"],
                high_price=row["high"],
                low_price=row["low"],
                close_price=row["close"],
                volume=row["volume"],
                turnover=row["turnover"],
                open_interest=row["open_interest"],
                gateway_name="DB",
                adjust_type=adjust_type
            )
            bars.append(bar)

        return bars

    def load_bar_df(    #-------------------做了市值中性化？
        self,
        vt_symbols: list[str],
        interval: Interval | str,
        start: datetime | str,
        end: datetime | str,
        extended_days: int,
        adjust_type: str
    ) -> pl.DataFrame | None:
        """Load bar data as DataFrame"""
        if not vt_symbols:
            return None

        # Convert types
        if isinstance(interval, str):
            interval = Interval(interval)

        start = to_datetime(start) - timedelta(days=extended_days)
        end = to_datetime(end) + timedelta(days=extended_days // 10)

        # Get folder path
        if interval == Interval.DAILY and adjust_type == "none":
            folder_path: Path = self.daily_path
        elif interval == Interval.DAILY and adjust_type == "pre":
            folder_path: Path = self.daily_pre_path
        elif interval == Interval.DAILY and adjust_type == "post":
            folder_path: Path = self.daily_post_path
        elif interval == Interval.MINUTE and adjust_type == "none":
            folder_path = self.minute_path
        elif interval == Interval.MINUTE and adjust_type == "pre":
            folder_path: Path = self.minute_pre_path
        elif interval == Interval.MINUTE and adjust_type == "post":
            folder_path: Path = self.minute_post_path
        elif interval == Interval.WEEKLY and adjust_type == "none":
            folder_path = self.weekly_path
        elif interval == Interval.WEEKLY and adjust_type == "pre":
            folder_path: Path = self.weekly_pre_path
        elif interval == Interval.WEEKLY and adjust_type == "post":
            folder_path: Path = self.weekly_post_path
        elif interval == Interval.MONTHLY and adjust_type == "none":
            folder_path = self.monthly_path
        elif interval == Interval.MONTHLY and adjust_type == "pre":
            folder_path: Path = self.monthly_pre_path
        elif interval == Interval.MONTHLY and adjust_type == "post":
            folder_path: Path = self.monthly_post_path
        else:
            logger.error(f"Unsupported interval {interval.value}")
            return None

        # Read data for each symbol
        dfs: list = []

        for vt_symbol in vt_symbols:
            # Check if file exists
            file_path: Path = folder_path.joinpath(f"{vt_symbol}.parquet")
            if not file_path.exists():
                logger.error(f"File {file_path} does not exist")
                continue

            df: pl.DataFrame = pl.scan_parquet(file_path)
            df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))
            if interval == Interval.WEEKLY or interval == Interval.MONTHLY: # 周k、月k需要合成 所以没有vwap
                df = df.with_columns(
                    pl.col("open"),
                    pl.col("high"),
                    pl.col("low"),
                    pl.col("close"),
                    pl.col("volume"),
                    pl.col("turnover"),
                    pl.col("open_interest")
                )
            else:
                df = df.with_columns(
                pl.col("open"),
                pl.col("high"),
                pl.col("low"),
                pl.col("close"),
                pl.col("volume"),
                pl.col("turnover"),
                pl.col("open_interest"),
                (pl.col("turnover") / pl.col("volume")).alias("vwap")
            )
            df = df.collect()

            # # Open file
            # df: pl.DataFrame = pl.read_parquet(file_path)
            #
            # # Filter by date range
            # df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))
            #
            # # Specify data types
            # if interval == Interval.WEEKLY or interval == Interval.MONTHLY: # 周k、月k需要合成 所以没有vwap
            #     df = df.with_columns(
            #         pl.col("open"),
            #         pl.col("high"),
            #         pl.col("low"),
            #         pl.col("close"),
            #         pl.col("volume"),
            #         pl.col("turnover"),
            #         pl.col("open_interest")
            #     )
            # else:
            #     df = df.with_columns(
            #     pl.col("open"),
            #     pl.col("high"),
            #     pl.col("low"),
            #     pl.col("close"),
            #     pl.col("volume"),
            #     pl.col("turnover"),
            #     pl.col("open_interest"),
            #     (pl.col("turnover") / pl.col("volume")).alias("vwap")
            # )

            # Check for empty data
            if df.is_empty():
                continue

            # Normalize prices
            # close_0: float = df.select(pl.col("close")).item(0, 0)
            #
            # df = df.with_columns(
            #     (pl.col("open") / close_0).alias("open"),
            #     (pl.col("high") / close_0).alias("high"),
            #     (pl.col("low") / close_0).alias("low"),
            #     (pl.col("close") / close_0).alias("close"),
            # )

            # Convert zeros to NaN for suspended trading days
            numeric_columns: list = df.columns[1:]                              # Extract numeric columns

            mask: pl.Series = df[numeric_columns].sum_horizontal() == 0         # Sum by row, if 0 then suspended

            df = df.with_columns(                                               # Convert suspended day values to NaN
                [pl.when(mask).then(float("nan")).otherwise(pl.col(col)).alias(col) for col in numeric_columns]
            )

            # Add symbol column
            df = df.with_columns(pl.lit(vt_symbol).alias("vt_symbol"))

            # Cache in list
            dfs.append(df)

        # Concatenate results
        result_df: pl.DataFrame = pl.concat(dfs)
        return result_df

    def save_component_data(
        self,
        index_symbol: str,
        index_components: dict[str, list[str]]
    ) -> None:
        """Save index component data"""
        file_path: Path = self.component_path.joinpath(f"{index_symbol}")

        with shelve.open(str(file_path)) as db:
            db.update(index_components)

    @lru_cache      # noqa
    def load_component_data(
        self,
        index_symbol: str,
        start: datetime | str,
        end: datetime | str
    ) -> dict[datetime, list[str]]:
        """Load index component data as DataFrame"""
        file_path: Path = self.component_path.joinpath(f"{index_symbol}")

        start = to_datetime(start)
        end = to_datetime(end)

        with shelve.open(str(file_path)) as db:
            keys: list[str] = list(db.keys())
            keys.sort()

            index_components: dict[datetime, list[str]] = {}
            for key in keys:
                dt: datetime = datetime.strptime(key, "%Y-%m-%d")
                if start <= dt <= end:
                    index_components[dt] = db[key]

            return index_components

    #------------------ 使得每天都有信号使用 ---------------------  对未更新沪深300成分股名单的日期key对应的value进行填充，填充最后记录的成分股
    @lru_cache
    def load_component_data2(
            self,
            index_symbol: str,
            start: datetime | str,
            end: datetime | str
    ) -> dict[datetime, list[str]]:
        """Load index component data, and for dates after the last known record,
        fill with the latest known components."""
        trading_days = self.load_trading_days()
        trading_days = tuple(trading_days)

        file_path: Path = self.component_path.joinpath(f"{index_symbol}")

        start = to_datetime(start)
        end = to_datetime(end)

        # 1. 读取原始数据（只包含有明确记录的日子）
        with shelve.open(str(file_path)) as db:
            keys: list[str] = list(db.keys())
            keys.sort()
            original: dict[datetime, list[str]] = {}
            for key in keys:
                dt: datetime = datetime.strptime(key, "%Y-%m-%d")
                if start <= dt <= end:
                    original[dt] = db[key]

        if not original:
            return {}

        # 2. 找出原始数据中最大的日期
        max_record_date = max(original.keys())

        # 3. 如果 end 超出了最大记录日期，则从 max_record_date+1 到 end 进行填充
        if end > max_record_date:

            # 获取最后一个有效的成分股列表
            last_components = original[max_record_date]

            current = max_record_date + timedelta(days=1)
            while current <= end:
                if current.date() in trading_days:
                    original[current] = last_components
                current += timedelta(days=1)


        return original


    def load_component_symbols(  # 这个只是获取成分股名单，并不需因load_component_data的更新而改变
        self,
        index_symbol: str,
        start: datetime | str,
        end: datetime | str
    ) -> list[str]:
        """Collect index component symbols"""
        index_components: dict[datetime, list[str]] = self.load_component_data(
            index_symbol,
            start,
            end
        )

        component_symbols: set[str] = set()

        for vt_symbols in index_components.values():
            component_symbols.update(vt_symbols)

        return list(component_symbols)


    def load_component_filters(
        self,
        index_symbol: str,
        start: datetime | str,
        end: datetime | str
    ) -> dict[str, list[tuple[datetime, datetime]]]:
        """Collect index component duration filters"""
        index_components: dict[datetime, list[str]] = self.load_component_data(
            index_symbol,
            start,
            end
        )

        # Get all trading dates and sort
        trading_dates: list[datetime] = sorted(index_components.keys())

        # Initialize component duration dictionary
        component_filters: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)

        # Get all component symbols
        all_symbols: set[str] = set()
        for vt_symbols in index_components.values():
            all_symbols.update(vt_symbols)

        # Iterate through each component to identify its duration in the index
        for vt_symbol in all_symbols:
            period_start: datetime | None = None
            period_end: datetime | None = None

            # Iterate through each trading day to identify continuous holding periods
            for trading_date in trading_dates:
                if vt_symbol in index_components[trading_date]:
                    if period_start is None:
                        period_start = trading_date

                    period_end = trading_date
                else:
                    if period_start and period_end:
                        component_filters[vt_symbol].append((period_start, period_end))
                        period_start = None
                        period_end = None

            # Handle the last holding period
            if period_start and period_end:
                component_filters[vt_symbol].append((period_start, period_end))

        return component_filters


    #----------------------修改————————————————---
    def load_component_filters2(
        self,
        index_symbol: str,
        start: datetime | str,
        end: datetime | str
    ) -> dict[str, list[tuple[datetime, datetime]]]:
        """Collect index component duration filters"""

        index_components: dict[datetime, list[str]] = self.load_component_data2(
            index_symbol,
            start,
            end
        )

        # Get all trading dates and sort
        trading_dates: list[datetime] = sorted(index_components.keys())

        # Initialize component duration dictionary
        component_filters: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)

        # Get all component symbols
        all_symbols: set[str] = set()
        for vt_symbols in index_components.values():
            all_symbols.update(vt_symbols)

        # Iterate through each component to identify its duration in the index
        for vt_symbol in all_symbols:
            period_start: datetime | None = None
            period_end: datetime | None = None

            # Iterate through each trading day to identify continuous holding periods
            for trading_date in trading_dates:
                if vt_symbol in index_components[trading_date]:
                    if period_start is None:
                        period_start = trading_date

                    period_end = trading_date
                else:
                    if period_start and period_end:
                        component_filters[vt_symbol].append((period_start, period_end))
                        period_start = None
                        period_end = None

            # Handle the last holding period
            if period_start and period_end:
                component_filters[vt_symbol].append((period_start, period_end))

        return component_filters

    def add_contract_setting(
        self,
        vt_symbol: str,
        long_rate: float,
        short_rate: float,
        size: float,
        pricetick: float
    ) -> None:
        """Add contract information"""
        contracts: dict = {}

        if self.contract_path.exists():
            with open(self.contract_path, encoding="UTF-8") as f:
                contracts = json.load(f)

        contracts[vt_symbol] = {
            "long_rate": long_rate,
            "short_rate": short_rate,
            "size": size,
            "pricetick": pricetick
        }

        with open(self.contract_path, mode="w+", encoding="UTF-8") as f:
            json.dump(
                contracts,
                f,
                indent=4,
                ensure_ascii=False
            )

    def load_contract_setttings(self) -> dict:
        """Load contract settings"""
        contracts: dict = {}

        if self.contract_path.exists():
            with open(self.contract_path, encoding="UTF-8") as f:
                contracts = json.load(f)

        return contracts

    def load_trading_days(self, start: datetime | date = None, end: datetime | date = None) -> list[date]:
        trading_days: list = []

        if self.trading_days_path.exists():
            with open(self.trading_days_path) as f:
                trading_days = json.load(f)
            trading_days = [date.fromisoformat(t) for t in trading_days]

            if start is not None:
                trading_days = [d for d in trading_days if d >= start]
            if end is not None:
                trading_days = [d for d in trading_days if d <= end]

        else:
            logger.error(f"Trading days file not found, path: {self.trading_days_path}")
            return []
        return trading_days

    def save_dataset(self, name: str, dataset: AlphaDataset) -> None:
        """Save dataset as directory with parquet files + json metadata"""
        dataset_dir: Path = self.dataset_path.joinpath(name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save DataFrames to parquet (streaming write, no memory spike)
        if dataset.result_df is not None:
            dataset.result_df.write_parquet(dataset_dir.joinpath("result.parquet"))
            dataset._result_path = dataset_dir.joinpath("result.parquet")
        if dataset.raw_df is not None:
            dataset.raw_df.write_parquet(dataset_dir.joinpath("raw.parquet"))
            dataset._raw_path = dataset_dir.joinpath("raw.parquet")
        if dataset.infer_df is not None:
            dataset.infer_df.write_parquet(dataset_dir.joinpath("infer.parquet"))
            dataset._infer_path = dataset_dir.joinpath("infer.parquet")
        if dataset.learn_df is not None:
            dataset.learn_df.write_parquet(dataset_dir.joinpath("learn.parquet"))
            dataset._learn_path = dataset_dir.joinpath("learn.parquet")

        # Release memory-held DataFrames after persist
        dataset.result_df = None
        dataset.raw_df = None
        dataset.infer_df = None
        dataset.learn_df = None
        gc.collect()

        # Save metadata to json (cross-version compatible)
        meta: dict = {
            "data_periods": {
                k.name if hasattr(k, "name") else str(k): v
                for k, v in dataset.data_periods.items()
            },
            "feature_names": list(dataset.feature_expressions.keys()),
            "label_expression": dataset.label_expression,
            "process_type": dataset.process_type,
            "has_result": dataset._result_path is not None,
            "has_raw": dataset._raw_path is not None,
            "has_infer": dataset._infer_path is not None,
            "has_learn": dataset._learn_path is not None,
        }
        with open(dataset_dir.joinpath("meta.json"), mode="w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    def load_dataset(self, name: str) -> AlphaDataset | None:
        """Load dataset from directory with lazy loading support"""
        dataset_dir: Path = self.dataset_path.joinpath(name)

        # Try new format first
        if dataset_dir.exists() and dataset_dir.is_dir():
            meta_path: Path = dataset_dir.joinpath("meta.json")
            if not meta_path.exists():
                logger.error(f"Dataset {name} directory exists but meta.json missing")
                return None

            with open(meta_path, mode="r", encoding="utf-8") as f:
                meta: dict = json.load(f)

            # Reconstruct data_periods
            periods: dict = {}
            for k, v in meta.get("data_periods", {}).items():
                try:
                    seg = Segment[k]
                except KeyError:
                    seg_map: dict = {
                        "1": Segment.TRAIN, "2": Segment.VALID, "3": Segment.TEST,
                        "TRAIN": Segment.TRAIN, "VALID": Segment.VALID, "TEST": Segment.TEST
                    }
                    seg = seg_map.get(k, Segment.TRAIN)
                v = [datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S") for date_str in v]
                periods[seg] = tuple(v)

            # Create instance with dummy df, then clear it
            dummy_df: pl.DataFrame = pl.DataFrame({"datetime": [], "vt_symbol": []})
            dataset: AlphaDataset = AlphaDataset(
                df=dummy_df,
                train_period=periods.get(Segment.TRAIN, ("", "")),
                valid_period=periods.get(Segment.VALID, ("", "")),
                test_period=periods.get(Segment.TEST, ("", "")),
                process_type=meta.get("process_type", "append"),
            )
            dataset.df = None  # Release dummy immediately

            # Restore metadata
            dataset.data_periods = periods
            dataset.feature_expressions = {
                name: "" for name in meta.get("feature_names", [])
            }
            dataset.label_expression = meta.get("label_expression", "")
            dataset.process_type = meta.get("process_type", "append")
            dataset.infer_processors = []
            dataset.learn_processors = []

            # Set lazy-load paths
            if meta.get("has_result"):
                dataset._result_path = dataset_dir.joinpath("result.parquet")
            if meta.get("has_raw"):
                dataset._raw_path = dataset_dir.joinpath("raw.parquet")
            if meta.get("has_infer"):
                dataset._infer_path = dataset_dir.joinpath("infer.parquet")
            if meta.get("has_learn"):
                dataset._learn_path = dataset_dir.joinpath("learn.parquet")

            return dataset

        # Fallback to legacy .pkl format
        return self._load_dataset_legacy(name)

    def _load_dataset_legacy(self, name: str) -> AlphaDataset | None:
        """Load dataset from legacy .pkl file"""
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Dataset file {name} does not exist")
            return None

        with open(file_path, mode="rb") as f:
            dataset: AlphaDataset = pickle.load(f)
            return dataset

    def remove_dataset(self, name: str) -> bool:
        """Remove dataset"""
        dataset_dir: Path = self.dataset_path.joinpath(name)
        if dataset_dir.exists() and dataset_dir.is_dir():
            shutil.rmtree(dataset_dir)
            return True

        # Fallback to legacy .pkl
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")
        if file_path.exists():
            file_path.unlink()
            return True

        logger.error(f"Dataset {name} does not exist")
        return False

    def list_all_datasets(self) -> list[str]:
        """List all datasets"""
        names: set[str] = set()
        # New format: directories with meta.json
        for item in self.dataset_path.iterdir():
            if item.is_dir() and item.joinpath("meta.json").exists():
                names.add(item.name)
        # Legacy format: .pkl files
        for file in self.dataset_path.glob("*.pkl"):
            names.add(file.stem)
        return sorted(list(names))

    def save_model(self, name: str, model: AlphaModel) -> None:
        """Save model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")

        with open(file_path, mode="wb") as f:
            pickle.dump(model, f)

    def load_model(self, name: str) -> AlphaModel | None:
        """Load model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Model file {name} does not exist")
            return None

        with open(file_path, mode="rb") as f:
            model: AlphaModel = pickle.load(f)
            return model

    def remove_model(self, name: str) -> bool:
        """Remove model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Model file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_models(self) -> list[str]:
        """List all models"""
        return [file.stem for file in self.model_path.glob("*.pkl")]

    def save_signal(self, name: str, signal: pl.DataFrame, OS: int = 0) -> None:
        """Save signal
            0: out of sample TEST
            1: in sample TRAIN
            2: VALID
            3: TRAIN + VALID
            4: ALL
        """
        dir_path: Path = self.signal_path.joinpath(f"{name}")
        dir_path.mkdir(parents=True, exist_ok=True)
        if OS == 0:
            file_path: Path = dir_path.joinpath(f"{name}_OS.parquet")
        elif OS == 1:
            file_path: Path = dir_path.joinpath(f"{name}_IS.parquet")
        elif OS == 2:
            file_path: Path = dir_path.joinpath(f"{name}_VA.parquet")
        elif OS == 3:
            file_path: Path = dir_path.joinpath(f"{name}_TV.parquet")
        elif OS == 4:
            file_path: Path = dir_path.joinpath(f"{name}_ALL.parquet")
        signal.write_parquet(file_path)

    def load_signal(self, name: str, OS: int = 0) -> pl.DataFrame | None:
        """Load signal"""
        dir_path: Path = self.signal_path.joinpath(f"{name}")
        if not dir_path.exists():
            logger.error(f"Signal dir {name} does not exist")
            return None

        if OS == 0:
            file_path: Path = dir_path.joinpath(f"{name}_OS.parquet")
        elif OS == 1:
            file_path: Path = dir_path.joinpath(f"{name}_IS.parquet")
        elif OS == 2:
            file_path: Path = dir_path.joinpath(f"{name}_VA.parquet")
        elif OS == 3:
            file_path: Path = dir_path.joinpath(f"{name}_TV.parquet")
        elif OS == 4:
            file_path: Path = dir_path.joinpath(f"{name}_ALL.parquet")

        if not file_path.exists():
            logger.error(f"Signal file {name} does not exist")
            return None

        return pl.read_parquet(file_path)

    def remove_signal(self, name: str) -> bool:
        """Remove signal"""
        file_path: Path = self.signal_path.joinpath(f"{name}.parquet")
        if not file_path.exists():
            logger.error(f"Signal file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_signals(self) -> list[str]:
        """List all signals"""
        return [file.stem for file in self.model_path.glob("*.parquet")]


    def save_industry_data(self, industrys: list[IndustryData]) -> None:
        """
            保存行业分类mapping: pl.datafram .parquet
            各级行业成分股: list[str] .json
        """
        if not industrys:

            return

        industry = industrys[0]
        market = industry.market
        source = industry.source
        industry_dir = self.industry_path.joinpath(market).joinpath(source)

        mapping_dir = industry_dir.joinpath("mapping")
        component_dir = industry_dir.joinpath("component")
        industry_dir.mkdir(parents=True, exist_ok=True)
        mapping_dir.mkdir(parents=True, exist_ok=True)
        component_dir.mkdir(parents=True, exist_ok=True)

        for ind in industrys:
            # 保存mapping
            date = ind.date
            date_str = date.strftime("%Y-%m-%d")
            mapping_date_dir = mapping_dir.joinpath(date_str)
            mapping_date_dir.mkdir(parents=True, exist_ok=True)
            file_path = mapping_date_dir.joinpath(f"mapping.parquet")
            if not file_path.exists():
                pl.DataFrame(ind.mapping).write_parquet(file_path)

            #保存行业成分股
            component_date_dir = component_dir.joinpath(date_str)
            component_date_dir.mkdir(parents=True, exist_ok=True)

            first_dir = component_date_dir.joinpath(
                f"{ind.first_industry_name}-{ind.first_industry_code}")

            second_dir = first_dir.joinpath(f'{ind.second_industry_name}-{ind.second_industry_code}')

            third_dir = second_dir.joinpath(f'{ind.third_industry_name}-{ind.third_industry_code}')

            third_dir.mkdir(parents=True, exist_ok=True)

            file_path1 = first_dir.joinpath(f"{ind.first_industry_code}.json")
            if not file_path1.exists():
                with open(file_path1, mode="w") as f:
                    json.dump(ind.first_component, f, indent=2)

            file_path2 = second_dir.joinpath(f"{ind.second_industry_code}.json")
            if not file_path2.exists():
                with open(file_path2, mode="w") as f:
                    json.dump(ind.second_component, f, indent=2)

            file_path3 = third_dir.joinpath(f"{ind.third_industry_code}.json")
            if not file_path3.exists():
                with open(file_path3, mode="w") as f:
                    json.dump(ind.third_component, f, indent = 2)



    def cal_factor_daliy(self, req: FactorRequest) -> list[FactorData]:
        """
        请求factor_define, 计算指定时间段的日频因子数据，

        Args:
            is_FD: True
            start: datetime | date
            end: datetime | date
            symbols: list[str]
            factor_name: str 要计算的因子名称
            factor_type: FactorType
        return:
            pl.dataframe  columns: [ "vt_symbol", "datetime", "data" ]


        """
        start = req.start
        end = req.end
        symbols = req.vt_symbols
        factor_name = req.factor_name
        factor_type = req.factor_type
        factor_func = req.factor_func
        market = req.market

        # 获取所有交易日列表
        all_trading_days = self.load_trading_days()
        start_date = start.date() if hasattr(start, 'date') else start

        # 找到开始时间在交易日列表中的索引
        try:
            start_idx = all_trading_days.index(start_date)
        except ValueError:
            # 如果月初不是交易日，找到下一个交易日
            for i, d in enumerate(all_trading_days):
                if d >= start_date:
                    start_idx = i
                    break

        logger.info(f'{factor_name}...')

        # 获取因子配置
        config = PARAMS_REGISTRY.get(factor_name, {})
        constant_params = config.get("constant_params", {})
        df_params = config.get("df_params", {})

        # 标记是否需要start截断
        have_window = False

        # 加载分线数据（如果需要）
        df_minute = None
        if "df_minute" in df_params:
            params = df_params["df_minute"]
            max_window = params.get("max_window", 0)
            if max_window > 0:
                have_window = True
                load_start_idx = max(0, start_idx - max_window - 5)
                load_start_minute = datetime.combine(all_trading_days[load_start_idx], datetime.min.time())
            else:
                load_start_minute = start

            df_minute = self.load_bar_df(
                vt_symbols=symbols,
                interval=Interval.MINUTE,
                start=load_start_minute,
                end=end,
                extended_days=0,
                adjust_type="none"
            )
            if df_minute is None or df_minute.is_empty():
                logger.error(f'无[{load_start_minute.date()}-{end.date()}]分线数据')
                return pl.DataFrame()
            logger.info(f'[{load_start_minute.date()}-{end.date()}] 分线数据加载完成，行数: {df_minute.shape[0]}')

        # 加载日线数据（如果需要）
        df_daily = None
        if "df_daily" in df_params:
            params = df_params["df_daily"]
            max_window = params.get("max_window", 0)
            if max_window > 0:
                have_window = True
                load_start_idx = max(0, start_idx - max_window - 5)
                load_start_daily = datetime.combine(all_trading_days[load_start_idx], datetime.min.time())
            else:
                load_start_daily = start

            df_daily = self.load_bar_df(
                vt_symbols=symbols,
                interval=Interval.DAILY,
                start=load_start_daily,
                end=end,
                extended_days=0,
                adjust_type="none"
            )
            if df_daily is None or df_daily.is_empty():
                logger.error(f'无[{load_start_daily.date()}-{end.date()}]日线数据')
                return pl.DataFrame()
            logger.info(f'[{load_start_daily.date()}-{end.date()}] 日线数据加载完成，行数: {df_daily.shape[0]}')


        # 构建调用参数
        call_kwargs = {}

        # 数据表参数
        if df_minute is not None:
            call_kwargs["df_minute"] = df_minute
        if df_daily is not None:
            call_kwargs["df_daily"] = df_daily

        # constant_params
        call_kwargs.update(constant_params)

        # df_params（移除max_window）
        for params in df_params.values():
            filtered = {k: v for k, v in params.items() if k != "max_window"}
            call_kwargs.update(filtered)

        # start参数
        if have_window:
            call_kwargs["start"] = start_date

        # 调用因子函数
        factor_df = factor_func(**call_kwargs)
        if factor_df.is_empty():
            logger.error('因子数据表为空')
            return []
        else:
            logger.info('计算完成')
            # 转换date为datetime格式
            factor_df = factor_df.with_columns([
                pl.col('date').cast(pl.Datetime).alias('datetime')
            ]).select(['datetime', 'vt_symbol', 'data'])

            data: list[FactorData] = []


            for row in factor_df.iter_rows(named = True):
                vt_symbol, vt_exchange = row['vt_symbol'].split('.')
                Factor: FactorData = FactorData(
                    symbol=vt_symbol,
                    exchange=Exchange(vt_exchange),
                    datetime=row["datetime"],
                    factor_name=factor_name,
                    factor_type=factor_type,
                    data=row["data"],
                    market=market,
                    gateway_name="RQ",
                )

                data.append(Factor)

        return data


    def save_factor(self, factors: list[FactorData], recalcu = True) -> None:
        """
        保存query_factor()或cal_factor_daily()返回的FactorData, 一次只保存一个因子
        """
        if not factors:
            return
        factor0 = factors[0]
        if factor0.factor_type == FactorType.FUNDAMENTAL:
            file_path = self.fundamental_facor_path.joinpath(f"{factor0.factor_name}.parquet")
        else:
            file_path = self.price_volume_factor_path.joinpath(f"{factor0.factor_name}.parquet")
        data: list = []
        for factor in factors:
            factor_data: dict = {
                "vt_symbol": factor.vt_symbol,
                "datetime": factor.datetime,
                "data": factor.data,
            }
            data.append(factor_data)

        factor_df: pl.DataFrame = pl.DataFrame(data)

        if file_path.exists():
            if recalcu:
                old_df = pl.read_parquet(file_path)
                new_df = pl.concat([old_df, factor_df])
                new_df = new_df.unique(subset=["datetime", "vt_symbol"], keep="last")
                new_df.write_parquet(file_path)
            else:
                old_df = pl.read_parquet(file_path)
                new_df = pl.concat([old_df, factor_df])
                new_df = new_df.unique(subset=["datetime", "vt_symbol"], keep="first")
                new_df.write_parquet(file_path)
        else:
            factor_df.write_parquet(file_path)
        logger.info(f'{factor0.factor_name}保存至{file_path}')

    def load_factor(self, req: FactorRequest) -> pl.DataFrame:
        r"""
        从 D:\Aquant project\MF\MF_lab\Factor\FactorType\factorname.parquet 加载因子数据

        req.symbols = None 则加载全部

        return:
        pl.DataFrame  columns: ["vt_symbol", "datetime", "data"]

        """
        start = req.start
        end = req.end
        if req.symbols is not None and req.exchanges is not None:
            symbols = req.vt_symbols
        else:
            symbols = None
        factor_name = req.factor_name
        factor_type = req.factor_type

        if factor_type == FactorType.FUNDAMENTAL:
            file_path = self.fundamental_facor_path.joinpath(f"{factor_name}.parquet")
        else:
            file_path = self.price_volume_factor_path.joinpath(f"{factor_name}.parquet")

        if not file_path.exists():
            logger.error(f"File {file_path} does not exist")
            return pl.DataFrame()

        factor_df = pl.scan_parquet(file_path)

        if start is not None:
            factor_df = factor_df.filter((pl.col("datetime") >= start))
        if end is not None:
            factor_df = factor_df.filter((pl.col("datetime") <= end))
        factor_df = factor_df.filter((pl.lit(True) if symbols is None else pl.col('vt_symbol').is_in(symbols))).sort(['vt_symbol', 'datetime'])
        factor_df = factor_df.collect()

        found_symbols = list(factor_df['vt_symbol'].unique())
        if not found_symbols:
            logger.error(f"File {file_path} exist, but no target symbols found")
            return pl.DataFrame()
        if symbols is not None:
            not_found_symbols = list(set(symbols) - set(found_symbols))
            if len(not_found_symbols) > 0:
                logger.error(f'{len(not_found_symbols)}只股票无此时间段的因子数据:{not_found_symbols}\n')
        logger.info(f'{factor_name}: 成功加载{len(found_symbols)} 只股票')

        return factor_df

    def missing_ratio(
            self,
            req: FactorRequest
    ):
        """
        从因子库中加载因子, 计算指定因子列表中每个因子的 data 列缺失率（null + NaN）

        Args:
            factor_name: 因子名称
            start: 起始时间，None 表示不限制
            end: 结束时间，None 表示不限制
            symbols: 股票代码列表

        Returns:
            dict: {factor_name: missing_ratio}
        """
        start = req.start
        end = req.end
        if req.symbols is not None and req.exchanges is not None:
            symbols = req.vt_symbols
        else:
            symbols = None
        factor_name = req.factor_name
        factor_type = req.factor_type

        factor_df = self.load_factor(req = req)

        # 计算缺失率
        total_rows = factor_df.height
        if total_rows == 0:
            logger.error(f"File {factor_name} exist, but no value")
            return
        # null 计数
        null_cnt = factor_df['data'].null_count()
        # NaN 计数（注意：is_nan() 只对浮点列有效，若 data 非浮点会报错，可先转换或检查）
        # 确保 data 列是浮点类型，否则 is_nan() 可能不适用
        if factor_df['data'].dtype.is_float():
            nan_cnt = factor_df['data'].is_nan().sum()
        else:
            # 对于非浮点列，尝试转换为浮点再统计 NaN，或直接认为 NaN 不存在
            # 简单处理：先 cast 到 float，若失败则设为0
            try:
                nan_cnt = factor_df['data'].cast(pl.Float64).is_nan().sum()
            except:
                nan_cnt = 0
                logger.error(f"{factor_name} \"data\" 列非浮点且无法转换，NaN 计为0")

        null_ratio = null_cnt / total_rows
        nan_ratio = nan_cnt / total_rows
        missing_total = null_cnt + nan_cnt
        logger.info(
            f"{factor_name}: 总行数 {total_rows:,}, null数 {null_cnt:,} ({null_ratio:.4%}), NaN数 {nan_cnt:,} ({nan_ratio:.4%})")

        return
    # def save_factor_by_month(self, year, month, factor_name, factor_df, recalcu = True):
    #     """
    #     year: int
    #     month: int
    #     factor_name: str
    #     factor_df: pl.DataFrame
    #     接收factor_df 按月分区保存因子 例: ".../MF_lab/factor/adjusted_range/2018-01.parquet"
    #
    #     """
    #
    #     factor_dir = self.factor_path / factor_name
    #     factor_dir.mkdir(parents=True, exist_ok=True)
    #     file_path = factor_dir / f'{year}-{month:02d}.parquet'
    #
    #     if file_path.exists():
    #         if recalcu:
    #             old_df = pl.read_parquet(file_path)
    #             new_df = pl.concat([old_df, factor_df])
    #             new_df = new_df.unique(subset=["datetime", "vt_symbol"], keep="last")
    #             new_df.write_parquet(file_path)
    #         else:
    #             old_df = pl.read_parquet(file_path)
    #             new_df = pl.concat([old_df, factor_df])
    #             new_df = new_df.unique(subset=["datetime", "vt_symbol"], keep="first")
    #             new_df.write_parquet(file_path)
    #     else:
    #         factor_df.write_parquet(file_path)
    #
    # def load_factor_from_month(self, start, end, factor_name, symbols):
    #     """
    #     从月分区的因子库中加载单个因子的指定时间跨度和股票池的数据
    #
    #     Args:
    #         factor_name: str
    #         start: datetime | None
    #         end: datetime | None
    #         symbols: list[str] | None
    #     None则加载全部
    #     Returns:
    #         pl.dataframe ["vt_symbol", "datetime", "data"]
    #     """
    #
    #     print("开始加载...")
    #
    #     factor_dir = self.factor_path / factor_name
    #     if not factor_dir.exists():
    #         print(f'{factor_name} 不存在')
    #         return pl.DataFrame()
    #
    #     # 筛选在[start, end]范围内的月份文件
    #     files = []
    #     for f in factor_dir.glob('*.parquet'):
    #         # 文件名格式: YYYY-MM.parquet
    #         try:
    #             file_year = int(f.stem.split('-')[0])
    #             file_month = int(f.stem.split('-')[1])
    #             file_date = datetime(file_year, file_month, 1)
    #             if start is not None and end is not None:
    #                 # 检查该月份是否在时间范围内
    #                 if file_date >= datetime(start.year, start.month, 1) and file_date <= datetime(end.year, end.month, 1):
    #                     files.append(f)
    #             elif start is not None and end is None:
    #                 end = datetime.max
    #                 if file_date >= datetime(start.year, start.month, 1):
    #                     files.append(f)
    #             elif start is None and end is not None:
    #                 start = datetime.min
    #                 if file_date <= datetime(end.year, end.month, 1):
    #                     files.append(f)
    #             elif start is None and end is None:
    #                 start = datetime.min
    #                 end = datetime.max
    #                 files.append(f)
    #         except (ValueError, IndexError):
    #             continue
    #
    #
    #     if not files:
    #         print(f'{factor_name} 存在, 但没有指定日期文件')
    #         return pl.DataFrame()
    #
    #     # 加载所有月份文件并合并
    #     dfs = []
    #     for f in files:
    #         try:
    #             df = pl.read_parquet(f)
    #             dfs.append(df)
    #         except Exception as e:
    #             print(f'读取 {f} 失败: {e}')
    #
    #     if not dfs:
    #         print(f'{factor_name} 存在, 指定日期文件内容为空')
    #         return pl.DataFrame()
    #
    #     # 合并并过滤时间范围和股票池
    #     merged = pl.concat(dfs)
    #     factor_df = merged.filter(
    #         (pl.col('datetime') >= start) &
    #         (pl.col('datetime') <= end) &
    #         (pl.lit(True) if symbols is None else pl.col('vt_symbol').is_in(symbols))
    #     ).sort(['vt_symbol', 'datetime'])
    #
    #
    #     found_symbols = list(factor_df['vt_symbol'].unique())
    #     if not found_symbols:
    #         print(f'{factor_name} 存在, 但指定日期文件不含该股票池')
    #         return pl.DataFrame()
    #     if symbols is not None:
    #         not_found_symbols = list(set(symbols) - set(found_symbols))
    #         if len(not_found_symbols) > 0:
    #             print(f'{len(not_found_symbols)}只股票无此时间段的因子数据：{not_found_symbols}\n')
    #
    #     print(f'{factor_name}: 加载完成，共 {len(found_symbols)} 只股票，{len(files)} 个月数据')
    #
    #     return factor_df




