class FixedValueTrader(AbstractIntervalTrader):
    def get_price(self):
        # Note that I have access to self.state here as well
        return self.value

    def get_orders(self, price):
        if self.position == 0:
            self.buy(0, self.limit)
        else:
            self.export(self.position)

    def __init__(self, limit, value):
        super().__init__(limit)
        self.value = value
        self.data = "None"

