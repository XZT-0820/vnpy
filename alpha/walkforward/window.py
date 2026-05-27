from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class WalkForwardWindow:
    """单个滚动窗口"""
    index: int
    train_start: datetime
    train_end: datetime
    valid_start: datetime
    valid_end: datetime
    test_start: datetime
    test_end: datetime


class WindowGenerator:
    """
    根据总时间范围和窗口长度生成 WalkForwardWindow 列表

    支持测试集被 end 截断（最后一个窗口的测试集可能不完整）

    支持 float 年份（如 0.5 年）
    """

    @staticmethod
    def _years_to_delta(years: float) -> timedelta:
        """将 float 年份转换为精确/近似的 timedelta

        使用 dateutil.relativedelta 保证整年精度，
        小数部分按天近似（0.5 年 ≈ 182/183 天）
        """
        from dateutil.relativedelta import relativedelta

        full_years = int(years)
        fraction = years - full_years

        delta = relativedelta(years=full_years)
        if fraction:
            # 0.5 年按 365/2 ≈ 182 或 183 天
            delta += timedelta(days=int(fraction * 365.25))

        return delta

    @classmethod
    def generate(
        cls,
        start: datetime,
        end: datetime,
        train_years: float,
        valid_years: float,
        test_years: float,
        step_years: float,
    ) -> list[WalkForwardWindow]:
        """
        生成 WalkForwardWindow 列表

        Parameters
        ----------
        start : datetime
            总数据起始时间
        end : datetime
            总数据结束时间
        train_years : float
            训练集长度（年），支持小数如 0.5
        valid_years : float
            验证集长度（年）
        test_years : float
            测试集长度（年）
        step_years : float
            滚动步长（年）

        Returns
        -------
        list[WalkForwardWindow]
        """
        train_delta = cls._years_to_delta(train_years)
        valid_delta = cls._years_to_delta(valid_years)
        test_delta = cls._years_to_delta(test_years)
        step_delta = cls._years_to_delta(step_years)

        windows: list[WalkForwardWindow] = []
        t0 = start
        index = 0

        while True:
            train_start = t0
            train_end = t0 + train_delta
            valid_start = train_end
            valid_end = train_end + valid_delta
            test_start = valid_end
            test_end = valid_end + test_delta

            # 截断：test_end 不超过 end
            if test_start >= end:
                break

            actual_test_end = min(test_end, end)

            windows.append(WalkForwardWindow(
                index=index,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=actual_test_end,
            ))

            t0 = t0 + step_delta
            index += 1

        return windows
