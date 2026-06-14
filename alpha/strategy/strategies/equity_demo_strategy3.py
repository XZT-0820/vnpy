import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
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
        self.holding_days: dict[str, int] = {}
        self.cost_basis: dict[str, float] = {}      # 当前持仓加权平均成本
        self.sell_cost: dict[str, float] = {}        # 卖出时刻成本快照 (vt_tradeid → cost)
        self.write_log("Strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """Trade execution callback"""
        if trade.direction == Direction.SHORT:
            # Snapshot cost basis at time of sale
            self.sell_cost[trade.vt_tradeid] = self.cost_basis.get(trade.vt_symbol, trade.price)

            # Reset holding days
            self.holding_days.pop(trade.vt_symbol, None)

            # Clear cost basis if fully exited
            if self.pos_data.get(trade.vt_symbol, 0) == 0:
                self.cost_basis.pop(trade.vt_symbol, None)

        elif trade.direction == Direction.LONG:
            pos_after: float = self.pos_data.get(trade.vt_symbol, 0)
            pos_before: float = pos_after - trade.volume

            old_cost: float = self.cost_basis.get(trade.vt_symbol, 0)
            if pos_before > 0 and old_cost > 0:
                self.cost_basis[trade.vt_symbol] = (
                    pos_before * old_cost + trade.volume * trade.price
                ) / pos_after
            else:
                self.cost_basis[trade.vt_symbol] = trade.price

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """Daily bar callback \u2014 signal-weighted TopK strategy with holding_days constraint.

        Logic:
        1. Get yesterday's close signal, sort descending, select TopK.
        2. Update holding_days for all current positions.
        3. Locked stocks (held, not in TopK, days < min_days) are kept.
        4. Available slots = top_k - locked_count; select at most that many from TopK.
        5. Signal-weighted allocation among selected stocks.
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

        # Build signal lookup for buy_symbols
        buy_df: pl.DataFrame = last_signal.filter(
            pl.col("vt_symbol").is_in(buy_symbols)
        )
        signal_dict: dict[str, float] = {
            row["vt_symbol"]: row["signal"]
            for row in buy_df.iter_rows(named=True)
        }

        # 3. Update holding_days for all current positions
        for vt_symbol, pos in self.pos_data.items():
            if pos > 0:
                self.holding_days[vt_symbol] = self.holding_days.get(vt_symbol, 0) + 1

        # 4. Identify locked stocks (held, not in TopK, holding_days < min_days)
        locked: set[str] = set()
        for vt_symbol, pos in self.pos_data.items():
            if pos > 0 and vt_symbol not in buy_symbols:
                if self.holding_days.get(vt_symbol, 0) < self.min_days:
                    locked.add(vt_symbol)

        # 5. Sell non-locked stocks not in TopK
        for vt_symbol, pos in self.pos_data.items():
            if pos > 0 and vt_symbol not in buy_symbols and vt_symbol not in locked:
                self.set_target(vt_symbol, 0)

        # 6. Determine available slots and select buy stocks
        available_slots: int = max(0, self.top_k - len(locked))

        held_in_buy: list[str] = [s for s in buy_symbols if self.pos_data.get(s, 0) > 0]
        new_in_buy: list[str] = [s for s in buy_symbols if self.pos_data.get(s, 0) == 0]

        if len(held_in_buy) <= available_slots:
            # Enough slots: keep all held TopK stocks + fill with new by signal order
            selected: list[str] = list(held_in_buy) + new_in_buy[:available_slots - len(held_in_buy)]
        else:
            # Too many held TopK stocks: keep highest-signal ones, sell the rest
            held_sorted = sorted(held_in_buy, key=lambda s: signal_dict.get(s, 0), reverse=True)
            selected = held_sorted[:available_slots]
            for s in held_sorted[available_slots:]:
                self.set_target(s, 0)

        # 7. Filter to positive-signal stocks for capital allocation
        positive_selected: list[str] = [s for s in selected if signal_dict.get(s, 0) > 0]

        if not positive_selected:
            self.write_log("No positive-signal stocks, skipping buys")
            if locked:
                self.write_log(f"{len(locked)} stocks locked by holding_days")
            return

        total_assets: float = self.get_portfolio_value()
        allocated_cash: float = total_assets * self.cash_ratio

        positive_signals: list[float] = [signal_dict[s] for s in positive_selected]
        signal_sum: float = sum(positive_signals)

        for vt_symbol in positive_selected:
            bar: BarData | None = bars.get(vt_symbol)
            # Skip suspended stocks
            if not bar or not bar.open_price or bar.volume == 0:
                continue

            weight: float = signal_dict[vt_symbol] / signal_sum
            target_value: float = allocated_cash * weight
            target_volume: float = floor_to(target_value / bar.open_price, self.min_volume)

            self.set_target(vt_symbol, target_volume)

        # 8. Execute trading at open price
        self.execute_trading_open(bars, self.price_add)





