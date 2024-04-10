class LeastSquaresRegression(AbstractIntervalTrader):

    def __init__(self, limit, buff_size, n_MA, n_mean_BB, n_sigma_BB, n_RSI, n1_MACD, n2_MACD):
        super().__init__(limit)
        self.buff_size = buff_size
        self.Parameters = {"n_MA":       n_MA,  # time-span for MA
              "n_mean_BB":  n_mean_BB,  # time-span BB
              "n_sigma_BB": n_sigma_BB,  # time-span for sigma of BB
              "n_RSI":      n_RSI,  # time-span for RSI
              "n1_MACD":    n1_MACD,  # time-span for the first (longer) MACD EMA
              "n2_MACD":    n2_MACD,  # time-span for the second (shorter) MACD EMA
              }
        self.row_dims = max(self.Parameters.values())

    def get_price(self):

        d = {
            "Y": [],
            "A": []
        }
        data = self.data if self.data else d
        # print(f'data price history {data["Y"]} and data regression matrix {data["A"]}')
        order_depth : OrderDepth = self.state.order_depth

        osell = OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        market_trades: List[Trade] = self.state.market_trades

        price_at_timestamp = (next(iter(osell))  + next(iter(obuy))) / 2
        data["Y"].append(price_at_timestamp)

        if len(data["Y"]) > self.row_dims: data["Y"].pop(0)

        price_history = data["Y"]

        # Calculate Bollinger Bands (lower band, middle band, upper band)
        middle_BB = np.mean(price_history[-self.Parameters["n_mean_BB"]:])
        upper_BB = middle_BB + 2*np.var(price_history[-self.Parameters["n_sigma_BB"]:])
        lower_BB = middle_BB - 2*np.var(price_history[-self.Parameters["n_sigma_BB"]:])

        # RSI (relative strength index)
        RSI_increments = np.diff(price_history[-self.Parameters["n_RSI"]:])
        sum_up = np.sum([max(val, 0) for val in RSI_increments])
        sum_down = np.sum([-min(val, 0) for val in RSI_increments])

        avg_up = np.mean(sum_up)
        avg_down = np.mean(sum_down)
        RSI = avg_up / (avg_up + avg_down) if avg_down + avg_down != 0 else 0

        # MACD (Moving average convergence/divergence)
        alpha_1 = 2 / (self.Parameters["n1_MACD"] + 1) # Time span for longer MACD EMA
        alpha_2 = 2 / (self.Parameters["n2_MACD"] + 1) # Time span for shorted MACD EMA
        EMA_1 = EMA(price_history[-self.Parameters["n1_MACD"]:], alpha_1)
        EMA_2 = EMA(price_history[-self.Parameters["n2_MACD"]:], alpha_2 )
        MACD = EMA_2 - EMA_1

        # Build A (regression matrix)
        indicators = [lower_BB, middle_BB, upper_BB, RSI, MACD]
        regresssion_matrix = data["A"]
        regresssion_matrix.append(indicators)

        if len(regresssion_matrix) > self.row_dims:
            data["A"].pop(0)

        regresssion_matrix = np.array(regresssion_matrix)
        price_history = np.array([price_history])
        price_history = np.reshape(price_history, (len(price_history[0]), 1))

        model = OrdinaryLeastSquares()
        model.fit(regresssion_matrix, price_history)
        fair_price = model.predict(regresssion_matrix[-1])

        if not self.data:
            self.data = data

        return fair_price

    def next_state(self) -> Any:
        if len(self.data) > self.buff_size:
            self.data.pop(0)

        return self.data
