
import heapq
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import string
import json
from dataclasses import dataclass
from typing import List, Any, Dict, Tuple
import json
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId

@dataclass
class PartTradingState:
    """
    A version of Trading State, but
    for only one product at a time
    """
    # product name
    product_name: str

    # user defined state
    data: Any

    # You will always want all the listings, so no partitions
    listings: Dict[str, Listing]

    # Order depths for this particular product
    order_depth : OrderDepth

    # own trades for this particular proudct
    own_trades: List[Trade]

    # market trades for this particular product
    market_trades: List[Trade]

    # current position
    position: int

    # You will always want all the observations
    observations: Observation

    # full trading state, in case you want to
    # do some multivar stuff
    full_state: TradingState

    @staticmethod
    def partition_trading_state(state: TradingState, json_data):
        # Partitions Trading State into a dictionary of PartTradingState
        return {
                sym:PartTradingState(
                        product_name = sym,
                        data = json_data[sym] if json_data and (sym in json_data) else None, # in the future, this will also be partitioned
                        listings = state.listings,
                        order_depth = state.order_depths[sym],
                        own_trades = state.own_trades[sym] if sym in state.own_trades else [],
                        market_trades = state.market_trades[sym] if sym in state.market_trades else [],
                        position = state.position[sym] if sym in state.position else 0,
                        observations = state.observations,
                        full_state = state
                    )
            for sym, _ in state.listings.items()
        }

# A class meant to abstract away alot of the
# basic operations needed to design simple algos
class AbstractIntervalTrader:
    """
    An Abstract Base Class for all traders which
    trade using an interval approch.

    The method the user implements if "get_interval",
    which returns a pair (low, high)

    At each timestep, "get_interval" is called to obtain
    (low, high). Then
    - Every sell order of the product priced less than 'low'
      is bought
    - Every buy order of the product priced more than 'high'
      is bought

    Note, this class protects the user from accidentally trying
    to pass the position limit.

    After this, "next_state" is called to define the state for the
    next iteration (must be json-serializable)
    """
    orders: List[Order]
    position: int
    state: PartTradingState
    data: Any
    limit: int
    conversions: int

    def buy(self, price, vol, prod = ""):
        # Helper function to let you buy without worrying about
        # position limits

        if not prod:
            prod = self.state.product_name

        vol = min(vol, self.limit - self.position - self.buys)
        if vol > 0:
            #print(f"Buy {self.state.product_name} - {vol}")
            self.orders.append(Order(prod, price, vol))
            self.buys += vol

    def sell(self, price, vol, prod = ""):
        # Helper function to let you sell without worrying about
        # position limits

        if not prod:
            prod = self.state.product_name

        vol =  max(-vol, -self.position - self.limit - self.sells)
        if vol < 0:
            #(f"Sell {self.state.product_name} - {-vol}")
            self.orders.append(Order(prod, price, vol))
            self.sells += vol

    def setup(self, state: PartTradingState):
        self.state = state
        self.position = state.position
        self.data = state.data
        self.orders = []
        self.buys = 0
        self.sells = 0
        self.conversions = 0

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

    def run(self, state: PartTradingState):
        self.setup(state)

        pred_price = self.get_price()

        # print(f'Cnversions before {self.conversions}')
        if pred_price > 0:
            self.calculate_orders(state, pred_price - 1, pred_price + 1)

        # print(f'Conversions after {self.conversions}')
        return self.orders[:], self.conversions, self.data

    def get_price(self) -> int:
        ## Define some function using self.state to get the price you wanna trade at
        raise NotImplementedError


    def __init__(self, limit: int):
        self.limit = abs(limit)


def run_traders(traders: Dict[str, AbstractIntervalTrader], state: TradingState):
    # Helper function to run a diferent AbstractIntervalTrader
    # on each product of your choice
    try:
        j_data = json.loads(state.traderData)
    except:
        j_data = None

    part = PartTradingState.partition_trading_state(state, j_data)
    results = {}
    next_data = {}
    total_converstions = 0

    for (sym, part_state) in part.items():
        if sym in traders:
            results[sym],converstions, next_data[sym] = traders[sym].run(part_state) if sym in traders else ([], '')
            total_converstions += converstions

    logger.flush(state, results, 0, "")
    return results, total_converstions , json.dumps(next_data)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# JASPERS LOGGER
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def run(self, state: TradingState):

        return run_traders({#'AMETHYSTS': FixedValueTrader(20, 10000),
                            #'STARFRUIT': ARIMAModel(20),
                            'ORCHIDS': OrchidModel(100)
                            }, state)

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

    def calculate_orders(self, state: PartTradingState, acc_bid: int, acc_ask: int):
        open_sells = OrderedDict(sorted(self.state.order_depth.sell_orders.items()))
        open_buys = OrderedDict(sorted(self.state.order_depth.buy_orders.items(), reverse=True))

        _, best_sell_pr = self.values_extract(open_sells)
        _, best_buy_pr = self.values_extract(open_buys, 1)

        cpos = self.position

        for ask, vol in open_sells.items():
            if ((ask <= acc_bid) or (self.position < 0 and ask == acc_bid + 1)) and cpos < self.limit:
                order_for = min(-vol, self.limit - cpos)
                self.orders.append(Order(self.state.product_name, ask, order_for))
                cpos += order_for

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid)
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < self.limit:
            num = self.limit - cpos
            self.orders.append(Order(self.state.product_name, bid_pr, num))
            cpos += num

        
        cpos = self.position
        for bid, vol in open_buys.items():
            if ((bid >= acc_ask) or (self.position > 0 and bid + 1 == acc_ask)) and cpos > -self.limit:
                order_for = max(-vol, -self.limit - cpos)
                self.orders.append(Order(self.state.product_name, bid, order_for))
                cpos += order_for

        if cpos > -self.limit:
            num = -self.limit - cpos
            self.orders.append(Order(self.state.product_name, sell_pr, num))
            cpos += num


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



        return int(round(forecasted_price))
from typing import List, Any, Dict, Tuple
import json
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FIXED VALUE TRADER
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FixedValueTrader(AbstractIntervalTrader):

    def __init__(self, limit, value):
        super().__init__(limit)
        self.value = value

    def get_price(self):
        # Note that I have access to self.state here as well
        return self.value

    def next_state(self):
        return "None"

    
    def calculate_orders(self, state: PartTradingState, acc_bid: int, acc_ask: int):
        open_sells = OrderedDict(sorted(self.state.order_depth.sell_orders.items()))
        open_buys = OrderedDict(sorted(self.state.order_depth.buy_orders.items(), reverse=True))
        acc_ask, acc_bid = 10000, 10000

        _, best_sell_pr = self.values_extract(open_sells)
        _, best_buy_pr = self.values_extract(open_buys, 1)
        cpos = self.position

        for ask, vol in open_sells.items():
            if ((ask < acc_bid) or (self.position < 0 and ask == acc_bid)) and cpos < self.limit:
                # self.buy(ask, -vol)
                order_for = min(-vol, self.limit - cpos)
                self.orders.append(Order(self.state.product_name, int(round(ask)), order_for))
                cpos += order_for

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid - 2)
        sell_pr = max(undercut_sell, acc_ask + 2)

        if (cpos < self.limit) and (self.position < 0):
            num = min(40, self.limit - cpos)
            self.orders.append(Order(
                self.state.product_name, min(undercut_buy + 1, acc_bid - 1), num
                ))
            cpos += num
        
        if (cpos < self.limit) and (self.position > 15):
            num = min(40, self.limit - cpos)
            self.orders.append(Order(
                self.state.product_name, min(undercut_buy -1, acc_bid -1), num
            ))
            cpos += num

        if cpos < self.limit:
            num = min(40, self.limit - cpos)
            self.orders.append(Order(
                self.state.product_name, int(round(bid_pr)), num
            ))
            cpos += num
        
        cpos = self.position

        for bid, vol in open_buys.items():
            if ((bid > acc_ask) or (self.position > 0 and bid == acc_ask)) and cpos > -self.limit:
                order_for = max(-vol, -self.limit - cpos)
                self.orders.append(Order(
                    self.state.product_name, bid, order_for
                ))
                cpos += order_for

        if (cpos > -self.limit) and (self.position > 0):
            num = max(-40, -self.limit - cpos)
            self.orders.append(Order(
                self.state.product_name, max(undercut_sell - 1, acc_ask + 1), num
            ))
            cpos += num

        if (cpos > -self.limit) and (self.position < -15):
            num = max(-40, -self.limit - cpos)
            self.orders.append(Order(
                self.state.product_name, max(undercut_sell + 1, acc_ask + 1), num
            ))
            cpos += num

        if cpos > -self.limit:
            num = max(-40, -self.limit - cpos)
            self.orders.append(Order(
                self.state.product_name, sell_pr, num
            ))
            cpos += num
        


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

class OrchidModel(AbstractIntervalTrader):
    
    def __init__(self, limit: int):
        super().__init__(limit)
        self.humidity_cache = 32
        self.sunlight_cache = 16
        self.dims = 20

    def get_price(self) -> int:
        d = {
            "humidity": [],
            "sunlight": [],
            "A": [],
            "y": []
        }
        self.data = self.data if self.data else d
        converstionObservation = self.state.observations.conversionObservations['ORCHIDS']

        _, best_sell = self.values_extract(
            OrderedDict(sorted(self.state.order_depth.sell_orders.items()))
        )
        _, best_buy = self.values_extract(
            OrderedDict(sorted(self.state.order_depth.buy_orders.items(), reverse=True)), 1
        )

        humidity = converstionObservation.humidity
        sunlight = converstionObservation.sunlight

        self.data['humidity'].append(humidity)
        self.data['sunlight'].append(sunlight / 2500)
        self.data['y'].append((best_buy + best_sell) / 2)
        if len(self.data['humidity']) > self.humidity_cache:
            self.data['humidity'].pop(0)
        if len(self.data['sunlight']) > self.sunlight_cache:
            self.data['sunlight'].pop(0)
        if len(self.data['y']) > self.dims:
            self.data['y'].pop(0)

        sunlight_MA = np.convolve(self.data['sunlight'][-self.sunlight_cache:], np.ones(self.sunlight_cache) / self.sunlight_cache, mode='valid')[-1]
        humidity_MA = np.convolve(self.data['humidity'][-self.humidity_cache:], np.ones(self.humidity_cache) / self.humidity_cache, mode='valid')[-1]

        indicators = [sunlight_MA, humidity_MA]
        regresssion_matrix = self.data['A']
        regresssion_matrix.append(indicators)

        if len(regresssion_matrix) > self.dims:
            self.data['A'].pop(0)

        regresssion_matrix = np.array(regresssion_matrix[:10])
        price_history = np.array([self.data['y'][:10]])
        price_history = np.reshape(price_history, (len(price_history[0]), 1))

        model = OrdinaryLeastSquares()
        model.fit(regresssion_matrix, price_history)
        fair_price = model.predict(regresssion_matrix[-1])

        logger.print(f'fair price predicted {fair_price}')
        return fair_price[0]

    def calculate_orders(self, state: PartTradingState, acc_bid: int, acc_ask: int):        
        open_sells = OrderedDict(sorted(self.state.order_depth.sell_orders.items()))
        open_buys = OrderedDict(sorted(self.state.order_depth.buy_orders.items(), reverse=True))

        _, best_sell_pr = self.values_extract(open_sells)
        _, best_buy_pr = self.values_extract(open_buys, 1)

        cpos = self.position

        for ask, vol in open_sells.items():
            if ((ask <= acc_bid) or (self.position < 0 and ask == acc_bid + 1)) and cpos < self.limit:
                order_for = min(-vol, self.limit - cpos)
                self.orders.append(Order(self.state.product_name, ask, order_for))
                cpos += order_for

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid)
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < self.limit:
            num = self.limit - cpos
            self.orders.append(Order(self.state.product_name, bid_pr, num))
            cpos += num

        
        cpos = self.position
        for bid, vol in open_buys.items():
            if ((bid >= acc_ask) or (self.position > 0 and bid + 1 == acc_ask)) and cpos > -self.limit:
                order_for = max(-vol, -self.limit - cpos)
                self.orders.append(Order(self.state.product_name, bid, order_for))
                cpos += order_for

        if cpos > -self.limit:
            num = -self.limit - cpos
            self.orders.append(Order(self.state.product_name, sell_pr, num))
            cpos += num