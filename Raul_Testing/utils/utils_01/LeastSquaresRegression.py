from typing import List, Any, Dict, Tuple
import json
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from Driver_v5 import PartTradingState, AbstractIntervalTrader

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LEAST SQUARES REGRESSION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class OrdinaryLeastSquares:

    def __init__(self):
        self.coef = []

    def _reshape_x(self, X):
        return X.reshape(-1, 1)

    def _concatenate_ones(self, X):
        X_with_intercept = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        return X_with_intercept

    def fit(self, X, y):
        X = self._concatenate_ones(X)
        self.coef = np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    def predict(self, entry):
        b0 = self.coef[0]
        other_betas = self.coef[1:]
        prediction = b0

        for xi, bi in zip(entry, other_betas):
            prediction += (bi*xi)

        return prediction

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

    def EMA(self, x, alpha):
        if len(x) == 0:
            return 1
        if len(x) == 1:
            return x[0]
        return alpha*x[-1] + (1-alpha)*self.EMA(x[:-1], alpha)

    def get_price(self):
        
        d = {
            "Y": [5049.5, 5052.5, 5052.5, 5052.5, 5052.5, 5053.0, 5053.5, 5052.5, 5053.5, 5053.0, 5053.0, 5052.0, 5051.5, 5052.0, 5051.5, 5052.5, 5051.0, 5053.5, 5049.5, 5051.0],
            "A": [[5043.945000000001, 5053.85, 5063.755, 0.3, -1.2269647185339636], [5043.929999999999, 5053.7, 5063.47, 0.6428571428571429, -0.4816208737065608], [5044.005, 5053.55, 5063.095, 0.5833333333333334, -0.008095140340628859], [5043.905, 5053.45, 5062.995, 0.5454545454545454, 0.3656174897441815], [5044.35125, 5053.275, 5062.19875, 0, 0.02978131315740029], [5044.745, 5053.15, 5061.554999999999, 0, 0.5849725895532174], [5045.45125, 5053.025, 5060.598749999999, 0, 0.7393071959195368], [5046.195000000001, 5052.85, 5059.505, 0.5, 0.4031170056641713], [5046.775, 5052.75, 5058.725, 0.6666666666666666, 0.36922946154754754], [5047.0112500000005, 5052.675, 5058.33875, 0.5, 0.6993406418814629], [5048.75125, 5052.475, 5056.1987500000005, 0.4, 0.22164561907266034], [5050.225, 5052.25, 5054.275, 0.4, -0.12056155296977522], [5050.245, 5052.15, 5054.054999999999, 0.0, -0.13623947487576515], [5050.27, 5052.1, 5053.93, 0.25, -0.23436428415698174], [5050.27, 5052.1, 5053.93, 0.2, -0.36278384965316945], [5050.344999999999, 5052.15, 5053.955, 0.6, -0.413421182140155], [5050.23125, 5052.125, 5054.01875, 0.42857142857142855, -0.4221171133558528], [5050.501249999999, 5052.275, 5054.04875, 0.6363636363636364, -0.12682499373113387], [5049.71125, 5052.175, 5054.63875, 0.3888888888888889, -0.5270107447149712], [5049.53125, 5052.125, 5054.71875, 0.42105263157894735, -0.34533202345301106]]
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
        EMA_1 = self.EMA(price_history[-self.Parameters["n1_MACD"]:], alpha_1)
        EMA_2 = self.EMA(price_history[-self.Parameters["n2_MACD"]:], alpha_2 )
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

        return fair_price[0]

    def next_state(self) -> Any:
        if len(self.data) > self.buff_size:
            self.data.pop(0)

        return self.data

    def calculate_orders(self, state: PartTradingState, pred_price: int):
        print(f'The Y {self.data["Y"]} and the X = {self.data["A"]}')
        osell = OrderedDict(sorted(state.order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(state.order_depth.buy_orders.items(), reverse=True))

        _, best_sell_pr = self.values_extract(osell)
        _, best_buy_pr = self.values_extract(obuy, 1)

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, pred_price) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, pred_price)

        if self.position < self.limit:
            self.buy_lenient(undercut_buy)
            self.buy_strict(best_sell_pr - 1)

        # if 0 <= self.position <= 15:
        #     self.buy_lenient(undercut_buy)
        
        # if 15 < self.position <= self.limit:
        #     self.buy_strict(best_buy_pr - 1)
        
        for bid, vol in obuy.items():
            if ((bid > pred_price) or (self.position > 0 and bid >= pred_price + 1)):
                self.sell_lenient(bid, vol=vol)
        
        if self.position > -self.limit:
            self.sell_lenient(undercut_sell)
            self.sell_strict(best_sell_pr + 1)


