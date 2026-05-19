from collections import defaultdict

import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
from vnpy.trader.utility import round_to, floor_to

from vnpy.alpha import AlphaStrategy


class EquityDemoStrategy(AlphaStrategy):
    """Equity Long-Only Demo Strategy"""

    top_k: int = 50                 # Maximum number of stocks to hold
    n_drop: int = 5                 # Number of stocks to sell each time
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
        # Dictionary to track stock holding days
        self.holding_days: defaultdict = defaultdict(int)

        self.write_log("Strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """Trade execution callback"""
        # Remove holding days record when selling
        if trade.direction == Direction.SHORT:
            self.holding_days.pop(trade.vt_symbol, None)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K-line slice callback"""
        # Get the latest signals and sort them
        last_signal: pl.DataFrame = self.get_signal()
        last_signal = last_signal.sort("signal", descending=True)

        # Get position symbols and update holding days
        pos_symbols: list[str] = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos]

        for vt_symbol in pos_symbols:
            self.holding_days[vt_symbol] += 1

        # Generate sell list
        active_symbols: set[str] = set(last_signal["vt_symbol"][:self.top_k])                         # Extract symbols with highest signals
        active_symbols.update(pos_symbols)                                                            # Merge with currently held symbols
        active_df: pl.DataFrame = last_signal.filter(pl.col("vt_symbol").is_in(active_symbols))       # Filter signals for these symbols
        component_symbols: set[str] = set(last_signal["vt_symbol"])                 # Extract current index component symbols
        sell_symbols: set[str] = set(pos_symbols).difference(component_symbols)     # Sell positions not in components

        for vt_symbol in active_df["vt_symbol"][-self.n_drop:]:                     # Iterate through lowest signal portion
            if vt_symbol in pos_symbols:                                            # If the contract is in current positions
                sell_symbols.add(vt_symbol)                                         # Add it to sell list

        # Generate buy list
        buyable_df: pl.DataFrame = last_signal.filter(~pl.col("vt_symbol").is_in(pos_symbols))  # Filter contracts available for purchase
        buy_quantity: int = max(len(sell_symbols) + self.top_k - len(pos_symbols), 0)             # Calculate number of contracts to buy
        buy_thres: int = 15               #最低可接受的信号排名
        buy_symbols: list = list(buyable_df[:buy_thres]["vt_symbol"])                        # Select buy contract code list



        # Sell rebalancing
        cash: float = self.get_cash_available()                     # Get available cash after yesterday's settlement

        for vt_symbol in sell_symbols:
            if self.holding_days[vt_symbol] < self.min_days:        # Check if holding period exceeds threshold
                continue

            bar: BarData | None = bars.get(vt_symbol)               # Get current price of the contract
            if not bar:
                continue
            sell_price: float = bar.close_price*(1 - self.slippage)

            sell_volume: float = self.get_pos(vt_symbol)            # Get current holding volume

            self.set_target(vt_symbol, target=0)                    # Set target volume to 0

            turnover: float = sell_price * sell_volume                                  # Calculate selling turnover
            cost: float = max(turnover * self.close_rate, self.min_commission)          # Calculate selling cost
            cash += turnover - cost                                                     # Update available cash

        # 初始买入资金预算
        available_cash: float = cash * self.cash_ratio

        # Buy rebalancing
        if buy_symbols:
            buyed_count: int = 0   #已买股票数统计
            for vt_symbol in buy_symbols:
                if buyed_count == buy_quantity:
                    break

                buy_price: float = bars[vt_symbol].close_price * (1 + self.slippage)
                if not buy_price:
                    continue

                # 剩余目标股票数量（包括当前这只）
                remaining: int = buy_quantity - buyed_count
                # 当前股票最多能分配的金额：剩余资金 / 剩余股票数
                max_value: float = available_cash / remaining

                # 按 min_volume 向下取整计算实际买入量
                buy_volume: float = floor_to(max_value / buy_price, self.min_volume)

                if buy_volume == 0:
                    continue

                buyed_count += 1
                turnover: float = buy_price * buy_volume
                cost: float = max(turnover * self.open_rate, self.min_commission)
                available_cash -= turnover + cost
                cash -= turnover - cost

                self.set_target(vt_symbol, buy_volume)

        # Execute trading
        self.execute_trading(bars, price_add=self.price_add)
