import numpy as np

class LinearRegressionTrader(AbstractIntervalTrader):
    def get_interval(self):
        sell_arr = list(self.state.order_depth.sell_orders.keys())
        buy_arr = list(self.state.order_depth.buy_orders.keys())

        if sell_arr and buy_arr:
            sell_price = min(sell_arr)
            buy_price = max(buy_arr)

            new_dat = (sell_price + buy_price)/2
        elif sell_arr:
            new_dat = min(sell_arr)
        elif buy_arr:
            new_dat = max(buy_arr)
        else:
            return (-np.inf, np.inf)

        if len(self.data) > 2:
            self.data.append(new_dat)

            coef = np.polyfit(range(len(self.data) - 1),[self.data[i] - self.data[i - 1] for i in range(1, len(self.data))],1)
            poly1d_fn = np.poly1d(coef)

            return (poly1d_fn(len(self.data)) + new_dat, poly1d_fn(len(self.data)) + new_dat)
        else:
            self.data.append(new_dat)
            return (new_dat, new_dat)

    def next_state(self):
        if len(self.data) > self.buff_size:
            self.data.pop(0)

        return self.data

    def __init__(self, limit, buff_size):
        super().__init__(limit)
        self.buff_size = buff_size
        self.data = []

