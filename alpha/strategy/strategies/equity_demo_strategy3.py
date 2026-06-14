import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.utility import floor_to

from vnpy.alpha import AlphaStrategy3


class EquityDemoStrategy3(AlphaStrategy3):
    """Equity Long-Only Demo Strategy"""

    top_k: int = 50                 # Maximum number of stocks to hold
    min_days: int = 3               # Minimum holding period in days
    cash_ratio: float = 0.95        # Cash utilization ratio
    min_volume: int = 100           # Minimum trading unit
    open_rate: float = 0.0005       # Opening commission rate
    close_rate: float = 0.0015      # Closing commission rate
    min_commission: int = 5         # Minimum commission value
    price_add: float = 0.05         # Order price adjustment ratio
    slippage = 0.0001
    def on_init(self) -> None:
        """Strategy initialization callback"""
        self.write_log("Strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """Trade execution callback"""
        pass

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """Daily bar callback \u2014 signal-weighted TopK strategy with open-price execution.

        Logic:
        1. Get yesterday's close signal, sort descending, select TopK.
        2. total_assets = cash + holding_value; allocated = total_assets * cash_ratio.
        3. Weighted by signal: target_value_i = allocated * (signal_i / sum_signals).
        4. target_volume_i = floor_to(target_value_i / open, min_volume).
        5. Stocks held but not in TopK: target = 0 (sell all).
        6. Execute via open-price orders.
        """
        # 1. Get yesterday's close signal
        last_signal: pl.DataFrame = self.strategy_engine.get_previous_signal()
        if last_signal.is_empty():
            return

        last_signal = last_signal.sort("signal", descending=True)

        # 2. Select TopK target stocks
        buy_symbols: list[str] = list(last_signal["vt_symbol"][:self.top_k])
        self.buy_symbols = buy_symbols

        # 3. Calculate total portfolio value and allocated cash
        total_assets: float = self.get_portfolio_value()
        allocated_cash: float = total_assets * self.cash_ratio

        # 4. Calculate signal-weighted target positions
        buy_df: pl.DataFrame = last_signal.filter(
            pl.col("vt_symbol").is_in(buy_symbols)
        )
        signal_sum: float = buy_df["signal"].sum()

        if signal_sum <= 0:
            self.write_log("Signal sum <= 0, skipping trading")
            return

        for row in buy_df.iter_rows(named=True):
            vt_symbol: str = row["vt_symbol"]
            signal_val: float = row["signal"]

            bar: BarData | None = bars.get(vt_symbol)

            # Skip suspended stocks      not bar可能是今天指数调仓，某只股票有信号但是已经被去除了
            if not bar or not bar.open_price or bar.volume == 0:
                continue

            weight: float = signal_val / signal_sum
            target_value: float = allocated_cash * weight
            target_volume: float = floor_to(target_value / bar.open_price, self.min_volume)

            self.set_target(vt_symbol, target_volume)

        # 5. For currently held stocks NOT in TopK: set target to 0
        for vt_symbol, pos in self.pos_data.items():
            if pos > 0 and vt_symbol not in buy_symbols:
                self.set_target(vt_symbol, 0)

        # 6. Execute trading at open price
        self.execute_trading_open(bars, self.price_add)





