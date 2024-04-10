class FixedValueTrader(AbstractIntervalTrader):
    def get_interval(self):
        # Note that I have access to self.state here as well

        return (self.value, self.value)

    def next_state(self):
        return "None"

    def __init__(self, limit, value):
        super().__init__(limit)
        self.value = value

