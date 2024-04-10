class DiminishingReturnsTrader(LeastSquaresRegression):
    def __init__(self, limit, buff_size, n_MA, n_mean_BB, n_sigma_BB, n_RSI, n1_MACD, n2_MACD, returns_slope: int):
        super().__init__(limit, buff_size, n_MA, n_mean_BB, n_sigma_BB, n_RSI, n1_MACD, n2_MACD)

        self.returns_slope = returns_slope

    def position_buy(self, position: int, price: float):
        return price - position * self.returns_slope

    def position_sell(self, position: int, price: float):
        return price + position * self.returns_slope

