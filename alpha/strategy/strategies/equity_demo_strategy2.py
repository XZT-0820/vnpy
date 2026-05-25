from collections import defaultdict

import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
from vnpy.trader.utility import round_to, floor_to

from vnpy.alpha import AlphaStrategy


class EquityDemoStrategy2(AlphaStrategy):
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
        # Dictionary to track stock holding days
        self.holding_days: defaultdict = defaultdict(int)

        self.write_log("Strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        """Trade execution callback"""
        # Remove holding days record when selling
        if trade.direction == Direction.SHORT:
            self.holding_days.pop(trade.vt_symbol, None)

    def on_bars(self, bars: dict[str, BarData]) -> None: #这里的bars是回测当天的股票池！！！！
        """K-line slice callback"""
        # Get the latest signals and sort them
        last_signal: pl.DataFrame = self.get_signal()
        last_signal = last_signal.sort("signal", descending=True)
        for vt_symbol in bars.keys(): #！！！！！
            self.set_target(vt_symbol, self.get_pos(vt_symbol))
        # Get position symbols and update holding days
        pos_symbols: list[str] = [vt_symbol for vt_symbol, pos in self.pos_data.items() if pos]

        for vt_symbol in pos_symbols:
            self.holding_days[vt_symbol] += 1

        # Generate sell list
        buy_symbols = list(last_signal["vt_symbol"][:self.top_k])                  # Extract symbols with highest signals
        # print(f'{len(buy_symbols)} buy symbols')
        # 保存 buy_symbols 顺序，供 cross_order 使用
        self._buy_symbols = buy_symbols

        buy_df: pl.DataFrame = last_signal.filter(pl.col("vt_symbol").is_in(buy_symbols))       # Filter signals for these symbols

        sell_symbols: set[str] = set(pos_symbols).difference(buy_symbols)
        # print(f'{len(sell_symbols)} sell symbols')

        actual_sell: list[str] = []
        # Sell rebalancing
        cash: float = self.get_cash_available()                     # Get available cash after yesterday's settlement
        # print(f'{cash} cash available')
        if cash < 0:
            return
        for vt_symbol in sell_symbols:
            if self.holding_days[vt_symbol] < self.min_days:        # Check if holding period exceeds threshold
                continue

            bar: BarData | None = bars.get(vt_symbol)               # Get current price of the contract
            if not bar:
                continue
            sell_price: float = bar.close_price*(1 - self.slippage)

            sell_volume: float = self.get_pos(vt_symbol)            # Get current holding volume

            self.set_target(vt_symbol, target=0)                    # Set target volume to 0

            actual_sell.append(vt_symbol)

            turnover: float = sell_price * sell_volume                                  # Calculate selling turnover
            cost: float = max(turnover * self.close_rate, self.min_commission)          # Calculate selling cost
            cash += turnover - cost                                                     # Update available cash

        buy_quantity = max(self.top_k - (len(pos_symbols) - len(actual_sell)), 0)
        # print(f'{buy_quantity} buy symbols')
        # 初始买入资金预算
        available_cash: float = cash * self.cash_ratio
        # print(f'{available_cash} cash available')
        # Buy rebalancing
        if buy_symbols:
            buyed_count = 0
            for vt_symbol in buy_symbols:
                if buyed_count == buy_quantity or available_cash <= 0:
                    break
                buy_price: float = bars[vt_symbol].close_price * (1 + self.slippage)
                # print(vt_symbol, f'{buy_price} buy_price')
                if not buy_price:
                    continue

                # 剩余目标股票数量（包括当前这只）
                remaining: int = buy_quantity - buyed_count
                # print(f'{remaining}remaining')
                # 当前股票最多能分配的金额：剩余资金 / 剩余股票数
                max_value: float = available_cash / remaining
                # print(f'{max_value}max_value')
                # 按 min_volume 向下取整计算实际买入量
                buy_volume: float = floor_to(max_value / buy_price, self.min_volume)
                # print(f'{buy_volume}buy_volume')
                if buy_volume == 0:
                    continue
                buyed_count += 1
                turnover: float = buy_price * buy_volume
                cost: float = max(turnover * self.open_rate, self.min_commission)

                available_cash -= turnover + cost
                cash -= turnover + cost

                self.set_target(vt_symbol, buy_volume + self.get_pos(vt_symbol))

        # Execute trading
        self.execute_trading(bars, price_add=self.price_add)
