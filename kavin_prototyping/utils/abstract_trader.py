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
    def partition_trading_state(state: TradingState, json_data) -> PartTradingState:
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

    def buy(self, price, vol, prod = ""):
        # Helper function to let you buy without worrying about
        # position limits

        if not prod:
            prod = self.state.product_name

        vol = min(vol, self.limit - self.position - self.buys)
        if vol > 0:
            print(f"Buy {self.state.product_name} - {vol}")
            self.orders.append(Order(prod, price, vol))
            self.buys += vol

    def sell(self, price, vol, prod = ""):
        # Helper function to let you sell without worrying about
        # position limits

        if not prod:
            prod = self.state.product_name

        vol =  max(-vol, -self.position - self.limit - self.sells)
        if vol < 0:
            print(f"Sell {self.state.product_name} - {-vol}")
            self.orders.append(Order(prod, price, vol))
            self.sells += vol


    def setup(self, state: PartTradingState):
        self.state = state
        self.position = state.position
        if state.data:
            self.data = state.data
        self.orders = []

    def run(self, state: PartTradingState):
        self.setup(state)

        pred_price = self.get_price()

        # Place buy orders
        for gain in range(self.limit - self.position):
            self.buy(
                    self.position_buy(self.position + gain, pred_price),
                    1)

        # place sell orders
        for loss in range(self.position - self.limit):
            self.sell(
                    self.position_sell(self.position - gain, pred_price),
                    1)

        return self.orders[:], self.data

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

