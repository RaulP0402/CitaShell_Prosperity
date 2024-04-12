class FixedValueTrader(AbstractIntervalTrader):
    def get_interval(self):
        # Note that I have access to self.state here as well

        return (self.value, self.value)

    def __init__(self, limit, value):
        super().__init__(limit)
        self.value = value
        self.data = "None"

