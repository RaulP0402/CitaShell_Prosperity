from typing import List, Any, Dict, Tuple
import json
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from abstract_trader import AbstractIntervalTrader, PartTradingState

"""
Linear Regression
"""
class LinearModel:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.beta = None 
        self.intercept_ = None 
        self.coef_ = None 

    def least_squares(self, x, y):

        if np.linalg.det(x.T @ x) != 0:     # if determinant is not 0
            return np.linalg.inv((x.T @ x)) @ (x.T @ y)     #computes coefficients of LR model
        return np.linalg.pinv((x.T @ x)) @ (x.T @ y) # matrix not invertible uses pusedo inverse to give coefficients for LR model

    def _prepare_features(self, x):
        if self.fit_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x

    def fit(self, x, y):
        x = self._prepare_features(x)
        self.beta = self.least_squares(x, y)
        if self.fit_intercept:
            self.intercept_ = self.beta[0]
            self.coef_ = self.beta[1:]
        else:
            self.coef_ = self.beta

    def predict(self, x):
        x = self._prepare_features(x)
        return x @ self.beta

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)


"""
ARIMA Model 
"""
class ARIMA(LinearModel):
    def __init__(self, q, d, p):
        """
        An ARIMA model.
        :param q: (int) Order of the autoregressiv part
        :param p: (int) Order of the moving average part
        :param d: (int) Number of times the data needs to be differenced to be stationary
        """
        super().__init__(True) 
        self.p = p
        self.d = d
        self.q = q
        self.ar = None
        self.resid = None # will hold residual errors for ma

    def undo_difference(self, x, d=1):
        if d == 1:
            return np.cumsum(x)
        else:
            x = np.cumsum(x)
            return self.undo_difference(x, d - 1)
        
        """
    Differencing 
    """
    def difference(self, x, d=1):
        if d == 0:
            return x
        else:
            # calculates different between ith and ith+1
            x = np.r_[x[0], np.diff(x)]
            return self.difference(x, d - 1)
        
    """
    Auto Regressor
    """
    n = 500
    eps = np.random.normal(size=n) #generating random distribution

    def lag_view(self, x, order):

        y = x.copy()
        x = np.array([y[-(i + order):][:order] for i in range(y.shape[0])])
        x = np.stack(x)[::-1][order - 1: -1]
        y = y[order:]  
        return x, y
    
    def prepare_features(self, x):
        if self.d > 0: # determines if differencing is needed, if so makes data stationary
            x = self.difference(x, self.d)
        ar_features = None
        ma_features = None
        if self.q > 0: # order of MA 
            if self.ar is None:
                self.ar = ARIMA(0, 0, self.p) 
                self.ar.fit_predict(x)
            eps = self.ar.resid  
            eps[0] = 0 
            ma_features, _ = self.lag_view(np.r_[np.zeros(self.q), eps], self.q)
        if self.p > 0:
            ar_features = self.lag_view(np.r_[np.zeros(self.p), x], self.p)[0]
        if ar_features is not None and ma_features is not None:
            n = min(len(ar_features), len(ma_features))
            ar_features = ar_features[:n]
            ma_features = ma_features[:n]
            features = np.hstack((ar_features, ma_features))
        elif ma_features is not None:
            n = len(ma_features)
            features = ma_features[:n]
        else:
            n = len(ar_features)
            features = ar_features[:n]
        return features, x[:n] 

    def fit(self, x):
        features, x = self.prepare_features(x)
        super().fit(features, x) 
        return features

    def fit_predict(self, x):
        features = self.fit(x)
        return self.predict(x, prepared=(features))

    def predict(self, x, **kwargs):
        features = kwargs.get('prepared', None)
        if features is None:
            features, x = self.prepare_features(x)

        y = super().predict(features)
        self.resid = x - y # calc dif between actual and predcited

        return self.return_output(y)

    def return_output(self, x):
        if self.d > 0: # differenced > 0
            x = self.undo_difference(x, self.d) # undo difference b/c d > 0 
        return x

    def forecast(self, x, n):
        features, x = self.prepare_features(x)
        y = super().predict(features)

        # appends n zeros to end of y predictions
        # essentially making space for future predictions
        y = np.r_[y, np.zeros(n)]
        
        for i in range(n):
            feat = np.r_[y[-(self.p + n) + i: -n + i], np.zeros(self.q)]
            y[x.shape[0] + i] = super().predict(feat[None, :])
        return self.return_output(y)


class ARIMAModel(AbstractIntervalTrader):

    def __init__(self, limit):
        super().__init__(limit)
        # self.starfruit_cache = []
        self.starfruit_dim = 4

    def calculate_orders(self, state: PartTradingState, pred_price: int):
        open_sells = OrderedDict(sorted(self.state.order_depth.sell_orders.items()))
        open_buys = OrderedDict(sorted(self.state.order_depth.buy_orders.items(), reverse=True))

        _, best_sell_pr = self.values_extract(open_sells)
        _, best_buy_pr = self.values_extract(open_buys, 1)

        for ask, vol in open_sells.items():
            if ((ask <= pred_price) or (self.position < 0 and ask == pred_price + 1)) and self.position < self.limit:
                self.buy(ask, -vol)

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, pred_price)
        sell_pr = max(undercut_sell, pred_price)

        if self.position < self.limit:
            num = self.limit - self.position
            self.buy(bid_pr, num)

        for bid, vol in open_buys.items():
            if ((bid >= pred_price) or (self.position > 0 and bid + 1 == pred_price)) and self.position > -self.limit:
                self.sell(bid, vol)

        if self.position > -self.limit:
            num = -self.limit - self.position
            self.sell(sell_pr, num)


    def get_price(self) -> int:
        d = {
            "starfruit_cache": []
        }
        self.data = self.data if self.data else d

        if len(self.data['starfruit_cache']) == self.starfruit_dim:
            self.data['starfruit_cache'].pop(0)
        
        _, best_sell = self.values_extract(
            OrderedDict(sorted(self.state.order_depth.sell_orders.items()))
        )
        _, best_buy = self.values_extract(
            OrderedDict(sorted(self.state.order_depth.buy_orders.items(), reverse=True)), 1
        )
        self.data['starfruit_cache'].append((best_sell + best_buy) / 2)

        if len(self.data["starfruit_cache"]) < self.starfruit_dim:
            return 0

        model = ARIMA(4,0,4)
        pred = model.fit_predict(self.data['starfruit_cache'])
        forecasted_price = model.forecast(pred, 1)[-1]

        # if not self.data:
        #     self.data = data

        return int(round(forecasted_price))