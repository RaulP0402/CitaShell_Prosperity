from typing import List, Any, Dict, Tuple
import json
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PART TRADING STATE
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ABSTRACT TRADER 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

    def buy_strict(self, price):
        vol = min(5, self.limit - self.position)
        if vol > 0:
            print(f"Buy {self.state.product_name} - {vol}")
            self.orders.append(Order(self.state.product_name, price, vol))

    def sell_strict(self, price):
        vol = max(-5, -self.position - self.limit)
        if vol < 0:
            print(f"Sell {self.state.product_name} - {-vol}")
            self.orders.append(Order(self.state.product_name, price, vol))

    def buy_lenient(self, price, vol=30):
        # Helper function to let you buy without worrying about
        # position limits
        vol = min(vol, self.limit - self.position - 5)
        if vol > 0:
            print(f"Buy {self.state.product_name} - {vol}")
            self.orders.append(Order(self.state.product_name, price, vol))

    def sell_lenient(self, price, vol=30):
        # Helper function to let you sell without worrying about
        # position limits
        vol =  max(-vol, -self.position - self.limit + 5)
        if vol < 0:
            print(f"Sell {self.state.product_name} - {-vol}")
            self.orders.append(Order(self.state.product_name, price, vol))

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

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val

    def setup(self, state: PartTradingState):
        self.state = state
        self.position = state.position
        self.data = state.data
        self.orders = []

    def run(self, state: PartTradingState):
        self.setup(state)

        pred_price = self.get_price()

        self.calculate_orders(state, pred_price)

        return self.orders[:], self.next_state()

    def calculate_orders(self, state: PartTradingState, pred_price: int):
        raise NotImplementedError

    def get_price(self) -> int:
        ## Define some function using self.state to get the price you wanna trade at
        raise NotImplementedError

    def position_buy(self, position: int, price: float) -> float:
        # If I think the stock price is "price" and
        # I am currently at position "position", how
        # much am I willing to pay to go from position to
        # position + 1
        return price

    def position_sell(self, position: int, price: float) -> float:
        # If I think the stock price is "price" and
        # I am currently at position "position", how
        # much will I need to go from position to position - 1
        return price


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

    logger.flush(state, results, 0, "")
    return results, 1, json.dumps(next_data)

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

        return run_traders({'AMETHYSTS': FixedValueTrader(20, 10000),
                            'STARFRUIT': LeastSquaresRegression(20, 10, 20, 20, 20, 5, 10, 5)
                            }, state)
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

    
    def calculate_orders(self, state: PartTradingState, pred_price: int):
        osell = OrderedDict(sorted(state.order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(state.order_depth.buy_orders.items(), reverse=True))

        _, best_sell_pr = self.values_extract(osell)
        _, best_buy_pr = self.values_extract(obuy, 1)

        # Look buy to cheap or sell low on a current short
        for ask, vol in osell.items():
            if ((ask < pred_price) or (self.position < 0 and ask <= pred_price)):
                self.buy_lenient(ask, vol=-vol)
        
        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        if 0 <= self.position <= 15:
            self.buy_lenient(undercut_buy)

        if 15 < self.position <= self.limit:
            self.buy_strict(best_buy_pr - 1)

        for bid, vol in obuy.items():
            if ((bid > pred_price) or (self.position > 0 and bid >= pred_price)):
                self.sell_lenient(bid, vol=vol)

        if -15 <= self.position <= 0:
            self.sell_lenient(undercut_sell)
        
        if self.limit < self.position < -15:
            self.sell_strict(best_sell_pr + 1)
from typing import List, Any, Dict, Tuple
import json
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LEAST SQUARES REGRESSION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


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
            self.buy_strict(best_sell_pr - 3)
            
        for bid, vol in obuy.items():
            if ((bid > pred_price) or (self.position > 0 and bid >= pred_price + 1)):
                self.sell_lenient(bid, vol=vol)
        
        if self.position > -self.limit:
            self.sell_lenient(undercut_sell)
            self.sell_strict(best_sell_pr + 3)


