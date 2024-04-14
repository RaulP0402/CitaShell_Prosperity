from datamodel import OrderDepth,UserId, TradingState, Order, Listing, Trade, Observation, ConversionObservation
from typing import List, Any, Dict, Tuple
import string
import json
from dataclasses import dataclass

@dataclass
class GeneralOrder:
    price: int
    quantity: int
    tariff: float | None = None

    @staticmethod
    def from_observation(
            obs: ConversionObservation,
            position: int):
        if position < 0:
            price = obs.ask + obs.transportFees
            quantity = -position
            tarrif = obs.importTariff
        elif position > 0:
            price = observation.bid - observation.transportFees
            quantity = position
            tarrif = obs.exportTariff

        return GeneralOrder(
                True,
                price,
                quantity
                )

    def is_international(self):
        if self.tariff:
            return True
        else:
            return False

    def item(self):
        return (self.price, self.quantity, self.tariff)



class GeneralOrderDepth:
    # A list sorted by price of
    # buy orders
    buy_orders: List[GeneralOrder]

    # A list sorted by price of
    # sell orders
    sell_orders: List[GeneralOrder]

    def __init__(self, state: TradingState, product: str):
        sell_orders = []
        buy_orders = []

        for (ask, vol) in state.order_depths[product].sell_orders:
            sell_orders.append(
                    GeneralOrder(ask, vol, None)
                    )

        for (bid, vol) in state.order_depths[product].buy_orders:
            buy_orders.append(
                    GeneralOrder(bid, vol, None)
                    )

        if product in state.observations.conversionObservations:
            obs = state.observations.conversionObservation[product]
            position = state.position[product]

            if position > 0:
                # If you have a positive position, this
                # means that there are people who want to
                # buy from you in the international market.
                # so you can export, so you have a buy order
                buy_orders.append(GeneralOrder.from_observation(obs, position))
            elif position < 0:
                # reverse logic for a short position
                sell_orders.append(GeneralOrder.from_observation(obs, position))

        self.buy_orders = sorted(buy_orders, key = lambda x : x.price)
        self.sell_orders = sorted(sell_orders, key = lambda x : -x.price)

    def buy_items(self):
        return [(item.price, item.quantity, item.tariff) for item in self.buy_orders]

    def sell_items(self):
        return [(item.price, item.quantity, item.tariff) for item in self.sell_orders]


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

    # General Order Depths for this product (including import/export)
    general_order_depth: GeneralOrderDepth

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
                        general_order_depth = GeneralOrderDepth(state, sym),
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

    def import_ext(self, quantity) -> int:
        # returns how much you successfully
        # imported
        if self.position >= 0:
            return 0

        old_imports = self.net_import

        self.net_import = min(quantity + self.net_import, -self.position)

        return self.net_import - old_imports

    def export_ext(self, quantity) -> int:
        if self.position <= 0:
            return 0

        old_imports = self.net_import

        self.net_import = max(self.net_import - quantity, -self.position)

        return -(self.net_import - old_imports)

    def setup(self, state: PartTradingState):
        self.state = state
        self.position = state.position
        self.sells = 0
        self.buys = 0
        self.net_import = 0
        if state.data:
            self.data = state.data
        self.orders = []

    def cpos(self) -> int:
        return self.position + self.buys - self.sells

    def run(self, state: PartTradingState):
        self.setup(state)

        pred_price = self.get_price()

        self.get_orders(pred_price)

        return self.orders[:], self.new_imports, self.data

    def get_price(self) -> int:
        ## Define some function using self.state to get the price you wanna trade at
        raise NotImplementedError

    def get_orders(self, price) -> None:
        # Use self.buy, self.sell, self.import
        # and self.export to do as the names
        # suggest
        pass

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
    conversions = 0
    next_data = {}

    for (sym, part_state) in part.items():
        if sym in traders:
            results[sym], val, next_data[sym] = traders[sym].run(part_state) if sym in traders else ([], 0, '')
            conversions += val

    return results, conversions, json.dumps(next_data)

