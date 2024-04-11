from typing import Dict
from collections import defaultdict
import json
from typing import Any
import numpy as np
from typing import List
import collections
import copy
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId

class Logger:
    # Set this to true, if u want to create
    # local logs
    local: bool
    # this is used as a buffer for logs
    # instead of stdout
    local_logs: dict[int, str] = {}

    def __init__(self, local=False) -> None:
        self.logs = ""
        self.local = local

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        output = json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True)
        if self.local:
            self.local_logs[state.timestamp] = output
        print(output)

        self.logs = ""

    def compress_state(self, state: TradingState) -> dict[str, Any]:
        listings = []
        for listing in state.listings.values():
            listings.append([listing["symbol"], listing["product"], listing["denomination"]])

        order_depths = {}
        for symbol, order_depth in state.order_depths.items():
            order_depths[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return {
            "t": state.timestamp,
            "l": listings,
            "od": order_depths,
            "ot": self.compress_trades(state.own_trades),
            "mt": self.compress_trades(state.market_trades),
            "p": state.position,
            "o": state.observations,
        }

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.buyer,
                    trade.seller,
                    trade.price,
                    trade.quantity,
                    trade.timestamp,
                ])

        return compressed

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed
# This is provisionary, if no other algorithm works.
# Better to loose nothing, then dreaming of a gain.

empty_dict = {'AMETHYSTS': 0, 'STARFRUIT': 0}

INF = int(1e9)

def def_value():
    return copy.deepcopy(empty_dict)

def least_squares(x, y):
    """
    Compute the coefficients of a linear regression model using the least squares method.

    Parameters:
        x (array-like): The independent variable(s) of the linear regression model.
        y (array-like): The dependent variable of the linear regression model.

    Returns:
        array-like: The coefficients of the linear regression model.

    Notes:
        - If the determinant of x.T @ x is not equal to 0, the function computes the coefficients using the inverse of x.T @ x.
        - If the determinant of x.T @ x is equal to 0, the function computes the coefficients using the pseudo inverse of x.T @ x.

    """
    if np.linalg.det(x.T @ x) != 0:     # if determinant is not 0
        return np.linalg.inv((x.T @ x)) @ (x.T @ y)     #computes coefficients of LR model
    return np.linalg.pinv((x.T @ x)) @ (x.T @ y) # matrix not invertible uses pusedo inverse to give coefficients for LR model



"""
Moving Average 
"""
n = 500
eps = np.random.normal(size=n) #generating random distribution


def lag_view(x, order):
    """
    For every value X_i create a row that lags k values: [X_i-1, X_i-2, ... X_i-k]

    Parameters:
    - x: (array-like) The input array.
    - order: (int) The number of lagged values to create.

    Returns:
    - x: (array-like) The lagged array with shape (n-k, k), where n is the length of the input array and k is the order.
    - y: (array-like) The adjusted labels with shape (n-k,), where n is the length of the input array and k is the order.

    Note:
    - The lagged array x is created by shifting a window of size order by one step for each value in the input array x.
    - The adjusted labels y are obtained by removing the first order elements from the input array x.

    Example:
    >>> x = [1, 2, 3, 4, 5]
    >>> order = 2
    >>> lag_view(x, order)
    (array([[1, 2],
            [2, 3],
            [3, 4]]), array([3, 4, 5]))
    """
    y = x.copy()  # copy of x array
    # Create features by shifting the window of `order` size by one step.
    # This results in a 2D array [[t1, t2, t3], [t2, t3, t4], ... [t_k-2, t_k-1, t_k]]
    # look at each y & gathers 'order' num of prev nums -> creating lagged array of size order
    x = np.array([y[-(i + order):][:order] for i in range(y.shape[0])])

    # Reverse the array as we started at the end and remove duplicates.
    # Note that we truncate the features [order -1:] and the labels [order]
    # This is the shifting of the features with one time step compared to the labels
    # ensuring lagged items correspond w target values
    x = np.stack(x)[::-1][order - 1: -1]
    y = y[order:]  # removes first order element from y and ensures y is matched w lagged items

    #x -> lagged items
    #y -> adjusted labels
    return x, y

"""
Differencing 
"""


def difference(x, d=1):
    """
    Calculate the difference of a time series.

    Parameters:
        x (array-like): The input time series.
        d (int, optional): The number of times the time series needs to be differenced. Default is 1.

    Returns:
        array-like: The differenced time series.

    Notes:
        The difference of a time series is calculated by taking the difference between consecutive elements.
        If d is 0, the original time series is returned unchanged.

    Examples:
        >>> x = [1, 2, 4, 7, 11]
        >>> difference(x)
        [1, 2, 3, 4]
        >>> difference(x, d=2)
        [1, 1, 1]

    """
    if d == 0:
        return x
    else:
        # calculates different between ith and ith+1
        x = np.r_[x[0], np.diff(x)]
        return difference(x, d - 1)


def undo_difference(x, d=1):
    """
    Undo Difference

    Reverses the differencing operation performed on a time series.

    Parameters:
        x (array-like): The differenced time series.
        d (int, optional): The number of times the time series was differenced. Default is 1.

    Returns:
        array-like: The original time series.

    Notes:
        The undo_difference function reverses the differencing operation performed by the difference function.
        It calculates the cumulative sum of the differenced time series.
        If d is 1, the cumulative sum is returned.
        If d is greater than 1, the function recursively applies the cumulative sum d times.

    Examples:
        >>> x = [1, 2, 3, 4]
        >>> undo_difference(x)
        [1, 3, 6, 10]
        >>> undo_difference(x, d=2)
        [1, 4, 10, 20]
    """
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
        self.beta = None # stores coefficients
        self.intercept_ = None # stores intercepts
        self.coef_ = None # stores coeffecients of independent vars

    def _prepare_features(self, x):
        #takes independent variables x
        # adds column of ones to each feature
        if self.fit_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x

    def fit(self, x, y):
        """
        Takes independent variables x and adds a column of ones to each feature.

        Parameters:
            x (array-like): The independent variables.

        Returns:
            array-like: The modified independent variables with a column of ones added to each feature.
        """
        x = self._prepare_features(x)
        self.beta = least_squares(x, y)
        if self.fit_intercept:
            self.intercept_ = self.beta[0]
            self.coef_ = self.beta[1:]
        else:
            self.coef_ = self.beta

    def predict(self, x):
        """
        Predict the dependent variable using the trained linear regression model.

        Parameters:
            x (array-like): The independent variables.

        Returns:
            array-like: The predicted values of the dependent variable.

        Notes:
            - The independent variables are prepared by adding a column of ones to each feature.
            - The prediction is made by multiplying the prepared independent variables with the coefficients of the linear regression model.
        """
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
        :param q: (int) Order of the Moving Average part
        :param p: (int) Order of the autoregressive part
        :param d: (int) Number of times the data needs to be differenced to be stationary
        """
        super().__init__(True) # calls constructor of linearmodel
        self.p = p
        self.d = d
        self.q = q
        self.ar = None
        self.resid = None # will hold residual errors

    def prepare_features(self, x):
        """
        Prepare the features for the ARIMA model.

        Parameters:
        - x: (array-like) The input time series.

        Returns:
        - features: (array-like) The prepared features for the ARIMA model.
        - x: (array-like) The truncated time series.

        Notes:
        - If the value of d is greater than 0, the input time series is differenced d times.
        - The AR features are determined by lagging the time series x by p time steps.
        - The MA features are determined by lagging the residuals of the AR process by q time steps.
        - If both AR and MA features are present, the minimum length of the two feature arrays is determined and the features are truncated accordingly.
        - If only MA features are present, the length of the MA features array is determined and the features are truncated accordingly.
        - If only AR features are present, the length of the AR features array is determined and the features are truncated accordingly.

        Example:
        >>> x = [1, 2, 3, 4, 5]
        >>> arima = ARIMA(1, 1, 1)
        >>> arima.prepare_features(x)
        (array([[0, 1],
                [1, 2],
                [2, 3]]), [3, 4, 5])
        """
        if self.d > 0: # determines if differencing is needed, if so makes data stationary
            x = difference(x, self.d)

        ar_features = None
        ma_features = None

        # Determine the features and the epsilon terms for the MA process
        if self.q > 0: # order of MA 
            if self.ar is None:
                self.ar = ARIMA(0, 0, self.p) # initialize ARIMA & only focus on autoregressie to fit model to data
                self.ar.fit_predict(x)
            eps = self.ar.resid # storing residuals froms fitted ar model 
            eps[0] = 0 

            # prepend with zeros as there are no residuals_t-k in the first X_t
            # prepare moving avg features by creating lagged versions of residuals
            ma_features, _ = lag_view(np.r_[np.zeros(self.q), eps], self.q)

        # Determine the features for the AR process
        if self.p > 0:
            # prepend with zeros as there are no X_t-k in the first X_t
            ar_features = lag_view(np.r_[np.zeros(self.p), x], self.p)[0]
        
        #combining and truncating ma and ar features
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

        return features, x[:n] #returns prepared features & x truncated to match len(features)

    def fit(self, x):
        features, x = self.prepare_features(x)
        super().fit(features, x) # calls fit from LR class to fit using prepared features
        return features

    def fit_predict(self, x):
        """
        Fit and transform the input time series

        Parameters:
        - x: (array-like) The input time series.

        Returns:
        - y: (array-like) The predicted values of the dependent variable.

        Notes:
        - The method first fits the ARIMA model to the input time series using the 'fit' method.
        - Then, it predicts the values of the dependent variable using the 'predict' method.
        - The 'fit' method prepares the features for the ARIMA model and fits the linear regression model to the prepared features.
        - The 'predict' method prepares the features for the ARIMA model and predicts the values of the dependent variable using the fitted linear regression model.
        """

        features = self.fit(x)
        return self.predict(x, prepared=(features))

    def predict(self, x, **kwargs):
        """
        Predict the values of the dependent variable using the trained ARIMA model.
        
        Provides features if not provided
        
        Parameters:
            x (array-like): The input time series.

        Keyword Arguments:
            prepared (tuple, optional): A tuple containing the prepared features, residuals, and truncated time series.

        Returns:
            array-like: The predicted values of the dependent variable.

        Notes:
            - If the 'prepared' keyword argument is provided, the method uses the prepared features to make predictions.
            - If the 'prepared' keyword argument is not provided, the method prepares the features using the 'prepare_features' method and then makes predictions.
            - The residuals are calculated as the difference between the actual values and the predicted values.
            - The method returns the predicted values of the dependent variable.

        :kwargs:
            prepared: (tpl) containing the features, eps and x
        """
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
        """
        Forecast the time series.

        Parameters:
            x (array-like): Current time steps.
            n (int): Number of time steps in the future.

        Returns:
            array-like: The predicted values of the dependent variable.

        Notes:
            - The method takes the current time steps and the number of future time steps as input.
            - It prepares the features for the ARIMA model using the 'prepare_features' method.
            - It predicts the values of the dependent variable using the 'predict' method.
            - It appends n zeros to the end of the predicted values, creating space for future predictions.
            - It iteratively predicts the future values by using the previous predictions as input features.
            - The method returns the predicted values of the dependent variable.
        """
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

    logger = Logger(local=True)

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    volume_traded = copy.deepcopy(empty_dict)

    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)

    cpnl = defaultdict(lambda: 0)
    starfruit_cache = []
    starfruit_dim = 4
    steps = 0
    #a
    halflife_diff = 5
    alpha_diff = 1 - np.exp(-np.log(2) / halflife_diff)

    halflife_price = 5
    alpha_price = 1 - np.exp(-np.log(2) / halflife_price)

    halflife_price_dip = 20
    alpha_price_dip = 1 - np.exp(-np.log(2) / halflife_price_dip)

    begin_diff_dip = -INF
    begin_diff_bag = -INF
    begin_bag_price = -INF
    begin_dip_price = -INF

    std = 25
    basket_std = 117

# list of 4 numbers: starfruit_cache
# return int(round(nxt_price))

    def calc_next_price_starfruit(self):
        # starfruit cache stores price from 1 day ago, current day resp
        # by price, here we mean mid price
        #
        # coef = [-0.01869561, 0.0455032, 0.16316049, 0.8090892]
        # intercept = 4.481696494462085

        #               q   d  p
        # most success 5, 0, 4 thus far
        # highest q val u can go up to is 4
        
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

    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product] < 0) and (ask == acc_bid))) and cpos < \
                    self.POSITION_LIMIT[
                        'AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                cpos += order_for
                assert (order_for >= 0)
                orders.append(Order(product, int(round(ask)), order_for))

        mprice_actual = (best_sell_pr + best_buy_pr) / 2
        mprice_ours = (acc_bid + acc_ask) / 2

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid - 2)  # we will shi ft this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask + 2)

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < 0):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid - 1), num))
            cpos += num

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 15):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid - 1), num))
            cpos += num

        if cpos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, int(round(bid_pr)), num))
            cpos += num

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product] > 0) and (bid == acc_ask))) and cpos > - \
            self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert (order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, max(undercut_sell - 1, acc_ask + 1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < -15):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, max(undercut_sell + 1, acc_ask + 1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

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

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product] > 0) and (bid + 1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert (order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT - cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def compute_orders(self, product, order_depth, acc_bid, acc_ask):

        if product == "AMETHYSTS":
            return self.compute_orders_amethysts(product, order_depth, acc_bid, acc_ask)
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
        print()
        for key, val in self.position.items():
            print(f'{key} position: {val}')

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

        amethysts_lb = 10000
        amethysts_ub = 10000

        # CHANGE FROM HERE

        acc_bid = {'AMETHYSTS': amethysts_lb, 'STARFRUIT': starfruit_lb}  # we want to buy at slightly below
        acc_ask = {'AMETHYSTS': amethysts_ub, 'STARFRUIT': starfruit_ub}  # we want to sell at slightly above

        self.steps += 1

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
        #
        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                if trade.timestamp != state.timestamp - 100:
                    continue
                # print(f'We are trading {product}, {trade.buyer}, {trade.seller}, {trade.quantity}, {trade.price}')
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
            print(
                f"For product {product}, {settled_pnl + self.cpnl[product]}, {(settled_pnl + self.cpnl[product]) / (self.volume_traded[product] + 1e-20)}")

        for person in self.person_position.keys():
            for val in self.person_position[person].keys():

                if person == 'Olivia':
                    self.person_position[person][val] *= 0.995
                if person == 'Pablo':
                    self.person_position[person][val] *= 0.8
                if person == 'Camilla':
                    self.person_position[person][val] *= 0

        print(f"Timestamp {timestamp}, Total PNL ended up being {totpnl}")
        # print(f'Will trade {result}')
        print("End transmission")

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"
        # Sample conversion request. Check more details below.
        conversions = 1

        self.logger.flush(state, result)
        #use for submissions on prosperity
        #return result, conversions, traderData
    
        #use for backtester
        return result, conversions
    