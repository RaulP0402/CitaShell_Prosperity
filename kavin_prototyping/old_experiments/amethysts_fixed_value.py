class Trader:
    def run(self, state: TradingState):
        return run_traders({'AMETHYSTS': FixedValueTrader(20, 10000)}, state)
