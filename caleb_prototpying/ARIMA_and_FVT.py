from typing import Dict
from collections import defaultdict, OrderedDict
import json
from typing import Any
import numpy as np
from typing import List
import collections
import copy
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId



empty_dict = {'AMETHYSTS': 0, 'STARFRUIT': 0}
INF = int(1e9)

def least_squares(x, y):

    if np.linalg.det(x.T @ x) != 0:     # if determinant is not 0
        return np.linalg.inv((x.T @ x)) @ (x.T @ y)     #computes coefficients of LR model
    return np.linalg.pinv((x.T @ x)) @ (x.T @ y) # matrix not invertible uses pusedo inverse to give coefficients for LR model



"""
Auto Regressor
"""
n = 500
eps = np.random.normal(size=n) #generating random distribution
def lag_view(x, order):

    y = x.copy()
    x = np.array([y[-(i + order):][:order] for i in range(y.shape[0])])
    x = np.stack(x)[::-1][order - 1: -1]
    y = y[order:]  
    return x, y

"""
Differencing 
"""
def difference(x, d=1):
    if d == 0:
        return x
    else:
        # calculates different between ith and ith+1
        x = np.r_[x[0], np.diff(x)]
        return difference(x, d - 1)


def undo_difference(x, d=1):
    if d == 1:
        return np.cumsum(x)
    else:
        x = np.cumsum(x)
        return undo_difference(x, d - 1)


"""
Linear Regression
"""
class LinearModel:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.beta = None 
        self.intercept_ = None 
        self.coef_ = None 

    def _prepare_features(self, x):
        if self.fit_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x

    def fit(self, x, y):
        x = self._prepare_features(x)
        self.beta = least_squares(x, y)
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

    def prepare_features(self, x):
        if self.d > 0: # determines if differencing is needed, if so makes data stationary
            x = difference(x, self.d)
        ar_features = None
        ma_features = None
        if self.q > 0: # order of MA 
            if self.ar is None:
                self.ar = ARIMA(0, 0, self.p) 
                self.ar.fit_predict(x)
            eps = self.ar.resid  
            eps[0] = 0 
            ma_features, _ = lag_view(np.r_[np.zeros(self.q), eps], self.q)
        if self.p > 0:
            ar_features = lag_view(np.r_[np.zeros(self.p), x], self.p)[0]
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
            x = undo_difference(x, self.d) # undo difference b/c d > 0 
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

class Trader:
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    volume_traded = copy.deepcopy(empty_dict)
    cpnl = defaultdict(lambda: 0)
    starfruit_cache = []
    starfruit_dim = 4
    steps = 0

    def calc_next_price_starfruit(self):

        
        model = ARIMA(5, 0, 4)
        pred = model.fit_predict(self.starfruit_cache)
        forecasted_price = model.forecast(pred, 1)[-1]

        return int(round(forecasted_price))

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if buy == 0:
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask

        return tot_vol, best_val

    def compute_orders_amethysts(self, product, order_depth, fixed_value):
        orders = []
        osell = OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        _, best_sell_pr = self.values_extract(osell)
        _, best_buy_pr = self.values_extract(obuy, 1)

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        # Buying logic
        for ask, vol in osell.items():
            if ask < fixed_value or (self.position[product] < 0 and ask <= fixed_value):
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - self.position[product])
                if order_for > 0:
                    orders.append(Order(product, ask, order_for))
                    self.position[product] += order_for

        if 0 <= self.position[product] <= 15:
            order_for = min(40, self.POSITION_LIMIT['AMETHYSTS'] - self.position[product])
            if order_for > 0:
                orders.append(Order(product, undercut_buy, order_for))
                self.position[product] += order_for

        if 15 < self.position[product] <= self.POSITION_LIMIT['AMETHYSTS']:
            order_for = min(5, self.POSITION_LIMIT['AMETHYSTS'] - self.position[product])
            if order_for > 0:
                orders.append(Order(product, best_buy_pr - 1, order_for))
                self.position[product] += order_for

        # Selling logic
        for bid, vol in obuy.items():
            if bid > fixed_value or (self.position[product] > 0 and bid >= fixed_value):
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS'] - self.position[product])
                if order_for < 0:
                    orders.append(Order(product, bid, order_for))
                    self.position[product] += order_for

        if -15 <= self.position[product] <= 0:
            order_for = max(-40, -self.POSITION_LIMIT['AMETHYSTS'] - self.position[product])
            if order_for < 0:
                orders.append(Order(product, undercut_sell, order_for))
                self.position[product] += order_for

        if -self.POSITION_LIMIT['AMETHYSTS'] < self.position[product] < -15:
            order_for = max(-5, -self.POSITION_LIMIT['AMETHYSTS'] - self.position[product])
            if order_for < 0:
                orders.append(Order(product, best_sell_pr + 1, order_for))
                self.position[product] += order_for

        return orders


    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)
        cpos= self.position[product]
        
        # go through each sell order and create buy order
        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product] < 0) and (ask == acc_bid + 1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert (order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid)  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)
        #create more buy orders if still room 
        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        #reset before sell
        cpos = self.position[product]
        # go through each buy order and create sell order
        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product] > 0) and (bid + 1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert (order_for <= 0)
                orders.append(Order(product, bid, order_for))
        #create addiontional sell order if room
        if cpos > -LIMIT:
            num = -LIMIT - cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def compute_orders(self, product, order_depth, acc_bid, acc_ask):
        if product == "AMETHYSTS":
            return self.compute_orders_amethysts(product, order_depth, acc_bid)
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS': [], 'STARFRUIT': []}

        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            self.position[key] = val


        timestamp = state.timestamp

        if len(self.starfruit_cache) == self.starfruit_dim:
            self.starfruit_cache.pop(0)

        _, bs_starfruit = self.values_extract(
            collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, bb_starfruit = self.values_extract(
            collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)

        self.starfruit_cache.append((bs_starfruit + bb_starfruit) / 2)

        INF = 1e9

        starfruit_lb = -INF
        starfruit_ub = INF

        if len(self.starfruit_cache) == self.starfruit_dim:
            starfruit_lb = self.calc_next_price_starfruit() - 1
            starfruit_ub = self.calc_next_price_starfruit() + 1

        amethysts_lb = 1000
        amethysts_ub = 1000

        acc_bid = {'AMETHYSTS': amethysts_lb, 'STARFRUIT': starfruit_lb}  
        acc_ask = {'AMETHYSTS': amethysts_ub, 'STARFRUIT': starfruit_ub}  

        for product in state.market_trades.keys():
            for trade in state.market_trades[product]:
                if trade.buyer == trade.seller:
                    continue
                self.person_position[trade.buyer][product] = 1.5
                self.person_position[trade.seller][product] = -1.5
                self.person_actvalof_position[trade.buyer][product] += trade.quantity
                self.person_actvalof_position[trade.seller][product] += -trade.quantity

        for product in ['AMETHYSTS', 'STARFRUIT']:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product])
            result[product] += orders
        
        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                if trade.timestamp != state.timestamp - 100:
                    continue
                self.volume_traded[product] += abs(trade.quantity)
                if trade.buyer == "SUBMISSION":
                    self.cpnl[product] -= trade.quantity * trade.price
                else:
                    self.cpnl[product] += trade.quantity * trade.price

        totpnl = 0

        for product in state.order_depths.keys():
            settled_pnl = 0
            best_sell = min(state.order_depths[product].sell_orders.keys())
            best_buy = max(state.order_depths[product].buy_orders.keys())

            if self.position[product] < 0:
                settled_pnl += self.position[product] * best_buy
            else:
                settled_pnl += self.position[product] * best_sell
            totpnl += settled_pnl + self.cpnl[product]
        


        # String value holding Trader state data required.
        traderData = "SAMPLE"
        conversions = 1

        #use for submissions on prosperity
        return result, conversions, traderData
    
        #use for backtester
        #return result, conversions
    