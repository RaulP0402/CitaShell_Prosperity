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

#  class meant to abstract away alot of the
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

    def buy(self, price, vol):
        # Helper function to let you buy without worrying about
        # position limits
        vol = min(vol, self.limit - self.position)
        if vol > 0:
            print(f"Buy {self.state.product_name} - {vol}")
            self.orders.append(Order(self.state.product_name, price, vol))

    def sell(self, price, vol):
        # Helper function to let you sell without worrying about
        # position limits
        vol =  max(-vol, -self.position - self.limit)
        if vol < 0:
            print(f"Sell {self.state.product_name} - {-vol}")
            self.orders.append(Order(self.state.product_name, price, vol))

    def setup(self, state: PartTradingState):
        self.state = state
        self.position = state.position
        self.data = state.data
        self.orders = []

    def run(self, state: PartTradingState):
        self.setup(state)

        interval = self.get_interval()

        for (price, vol) in state.order_depth.sell_orders.items():
            if price < interval[0]:
                self.buy(price, -vol) # volumes are negative in sell orders

        for (price, vol) in state.order_depth.buy_orders.items():
            if price > interval[1]:
                self.sell(price, vol) # volumes are negative in sell orders

        return self.orders[:], self.next_state()

    def get_interval(self) -> Tuple[int, int]:
        ## Define some function using self.state to get the price you wanna trade at
        # The first elem of the tuple is the maximum price you buy at
        # The second elem of the tuple is the minimum price you sell at
        raise NotImplementedError

    def next_state(self) -> Any:
        ## Define what your next state will be, must be json serializable
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

    for (sym, part_state) in part.items():
        if sym in traders:
            results[sym], next_data[sym] = traders[sym].run(part_state) if sym in traders else ([], '')

    return results, 1, json.dumps(next_data)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LEAST SQUARES REGRESSION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Parameters = {"n_MA":       20,  # time-span for MA
              "n_mean_BB":  20,  # time-span BB
              "n_sigma_BB": 20,  # time-span for sigma of BB
              "n_RSI":      5,  # time-span for RSI
              "n1_MACD":    10,  # time-span for the first (longer) MACD EMA
              "n2_MACD":    5,  # time-span for the second (shorter) MACD EMA
              }

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

    def __init__(self, limit, buff_size):
        super().__init__(limit)
        self.buff_size = buff_size
        self.row_dims = max(Parameters.values())

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
        middle_BB = np.mean(price_history[-Parameters["n_mean_BB"]:])
        upper_BB = middle_BB + 2*np.var(price_history[-Parameters["n_sigma_BB"]:])
        lower_BB = middle_BB - 2*np.var(price_history[-Parameters["n_sigma_BB"]:])

        # RSI (relative strength index)
        RSI_increments = np.diff(price_history[-Parameters["n_RSI"]:])
        sum_up = np.sum([max(val, 0) for val in RSI_increments])
        sum_down = np.sum([-min(val, 0) for val in RSI_increments])

        avg_up = np.mean(sum_up)
        avg_down = np.mean(sum_down)
        RSI = avg_up / (avg_up + avg_down) if avg_down + avg_down != 0 else 0

        # MACD (Moving average convergence/divergence)
        alpha_1 = 2 / (Parameters["n1_MACD"] + 1) # Time span for longer MACD EMA
        alpha_2 = 2 / (Parameters["n2_MACD"] + 1) # Time span for shorted MACD EMA
        EMA_1 = EMA(price_history[-Parameters["n1_MACD"]:], alpha_1)
        EMA_2 = EMA(price_history[-Parameters["n2_MACD"]:], alpha_2 )
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


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FIXED VALUE TRADER
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FixedValueTrader(AbstractIntervalTrader):
    def get_interval(self):
        # Note that I have access to self.state here as well
        return (self.value, self.value)

    def next_state(self):
        return "None"

    def __init__(self, limit, value):
        super().__init__(limit)
        self.value = value


class Trader:
    def run(self, state: TradingState):
        return run_traders({'AMETHYSTS': FixedValueTrader(20, 10000),
                            'STARFRUIT': LeastSquaresRegression(20, 15)},
                           state)