from datamodel import OrderDepth, UserId, TradingState, Order, Listing, Trade, Observation
from typing import List, Any, Dict, Tuple
import string
import json
from dataclasses import dataclass

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

        if self.data and len(self.data) > 2:
            self.data.append(new_dat)

            coef = np.polyfit(range(len(self.data) - 1),[-(self.data[i] - self.data[i - 1]) for i in range(1, len(self.data))],1)
            poly1d_fn = np.poly1d(coef)

            return (poly1d_fn(len(self.data)) + new_dat, poly1d_fn(len(self.data)) + new_dat)
        else:
            if self.data:
                self.data.append(new_dat)
            else:
                self.data = [new_dat]

            return (new_dat, new_dat)

    def next_state(self):
        if len(self.data) > self.buff_size:
            self.data.pop(0)

        return self.data

    def __init__(self, limit, buff_size):
        super().__init__(limit)
        self.buff_size = buff_size

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
        return run_traders({#'AMETHYSTS': FixedValueTrader(20, 10000),
                            'STARFRUIT': LinearRegressionTrader(20, 10)},
                           state)

