from typing import List, Any, Dict, Tuple
import json
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from Driver_v3 import PartTradingState
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from Driver_v3 import AbstractIntervalTrader

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LEAST SQUARES REGRESSION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def EMA(x, alpha):
    if len(x) == 0:
        return 1
    if len(x) == 1:
        return x[0]
    return alpha*x[-1] + (1-alpha)*EMA(x[:-1], alpha)

class OrdinaryLeastSquares():

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

    def get_interval(self):
        
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

        return (fair_price, fair_price)

    def next_state(self) -> Any:
        if len(self.data) > self.buff_size:
            self.data.pop(0)

        return self.data

    def calculate_orders(self, state: PartTradingState, interval: Tuple[int]):
        osell = OrderedDict(sorted(state.order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(state.order_depth.buy_orders.items(), reverse=True))

        _, best_sell_pr = self.values_extract(osell)
        _, best_buy_pr = self.values_extract(obuy, 1)

        for ask, vol in osell.items():
            if ((ask < interval[0]) or (self.position < 0 and ask <= interval[0] + 1)):
                self.buy_lenient(ask, vol=-vol)
        
        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, interval[0]) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, interval[1])

        if self.position < self.limit:
            self.buy_lenient(undercut_buy)
            self.buy_strict(best_sell_pr - 1)

        # if 0 <= self.position <= 15:
        #     self.buy_lenient(undercut_buy)
        
        # if 15 < self.position <= self.limit:
        #     self.buy_strict(best_buy_pr - 1)
        
        for bid, vol in obuy.items():
            if ((bid > interval[1]) or (self.position > 0 and bid >= interval[1] + 1)):
                self.sell_lenient(bid, vol=vol)
        
        if self.position > -self.limit:
            self.sell_lenient(undercut_sell)
            self.sell_strict(best_sell_pr + 1)

        # if -15 <= self.position <= 0:
        #     self.sell_lenient(undercut_sell)
        
        # if self.limit < self.position < -15:
        #     self.sell_strict(best_sell_pr + 1)