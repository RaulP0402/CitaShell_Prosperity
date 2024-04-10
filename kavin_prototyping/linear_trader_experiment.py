class Trader:
    def run(self, state: TradingState):
        return run_traders({'AMETHYSTS': FixedValueTrader(20, 10000),
                            'STARFRUIT': DiminishingReturnsTrader(
                                20,
                                15, 20, 20, 20, 5, 10, 5,
                                1
                                )},
                           state)

