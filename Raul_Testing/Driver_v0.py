from collections import defaultdict, OrderedDict
from typing import List, Any, Dict
import numpy as np
import copy
import string
import json
import re

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId


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
    
class OrdinaryLeastSquares():
    def __init__(self):
        self.coef = []

    def _reshape_x(self, X):
        return X.reshape(-1, 1)

    def _concatenate_ones(self, X):
        ones = np.ones(shape=X.shape[0]).reshape(-1,1)
        return np.concatenate((ones,X), 1)

    def _normalize(self, matrix):
        normalized_matrix = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0))
        return normalized_matrix

    def fit(self, X, y):
        if len(X) == 1: 
            X = self._reshape_x(X)

        weights = np.arange(1, len(X) + 1)
        W = np.diag(weights)

        X = self._concatenate_ones(X)
        self.coef = np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
        # self.coef = np.linalg.pinv(X.transpose().dot(W).dot(X)).dot(X.transpose()).dot(W).dot(y)

    def predict(self, entry):
        b0 = self.coef[0]
        other_betas = self.coef[1:]
        prediction = b0
        # coef = [0.192259,0.250758,0.224051,0.332031]

        for xi, bi in zip(entry, other_betas):
            prediction += (bi*xi)

        return prediction

empty_dict = {'AMETHYSTS': 0, 'STARFRUIT': 0}
INF = int(1e9)

def def_value():
    return copy.deepcopy(empty_dict)

class Trader:

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}

    # STARFRUIT DETAILS
    startfruit_window = 4
    startfruit_features = 3 # Number of X variables in Y = b0 + b1X + b2X2 ...
    startfruit_X_cache = []
    startfruit_y_cache = []

    def market_values_extract(self, trades: List[Trade]):
        if trades == None or len(trades) == 0:
            return 0
        n = len(trades)

        ssum = 0
        for trade in trades:
            ssum += trade.price
        
        return ssum / n
    
    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: List[Order] = []

        osell = OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        _, best_sell_pr, bs_vol = self.values_extract(osell)
        _, best_buy_pr, bb_vol = self.values_extract(obuy, 1)

        curr_pos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product] < 0) and (ask == acc_bid+1))) and curr_pos < LIMIT:
                order_for = min(-vol, LIMIT - curr_pos)
                curr_pos += order_for
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid)# we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)


        if curr_pos < LIMIT:
            num = LIMIT - curr_pos
            orders.append(Order(product, bid_pr, num))
            curr_pos += num

        curr_pos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product] > 0) and (bid+1 == acc_ask))) and curr_pos > -LIMIT:
                order_for = max(-vol, -LIMIT-curr_pos)
                curr_pos += order_for
                orders.append(Order(product, bid, order_for))
                
        if curr_pos > -LIMIT:
            num = -LIMIT-curr_pos
            orders.append(Order(product, sell_pr, num))
            curr_pos += num
        
        return orders

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
        
        return tot_vol, best_val, mxvol

    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask):
        orders: List[Order] = []

        osell = OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr,_ = self.values_extract(osell)
        buy_vol, best_buy_pr,_ = self.values_extract(obuy, 1)

        mx_with_buy = -1

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product]<0) and (ask <= acc_bid))) and cpos < self.POSITION_LIMIT['AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))


        mprice_actual = (best_sell_pr + best_buy_pr)/2
        mprice_ours = (acc_bid+acc_ask)/2

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid-1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask+1)

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < 0):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            cpos += num

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 15):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
            cpos += num

        if cpos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid >= acc_ask))) and cpos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < -15):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        
        results = defaultdict(list); conversions = 1; traderData = "Sample"

        # Get current posiions 
        for key, val in state.position.items():
            self.position[key] = val

        for product in ["AMETHYSTS"]:
            amethysts_lb = 10000
            amethysts_ub = 10000

            acc_bid = {"AMETHYSTS": amethysts_lb}
            acc_ask = {"AMETHYSTS": amethysts_ub}
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders_amethysts(product, order_depth, acc_bid[product], acc_ask[product])
            results[product] += orders

        for product in ["STARFRUIT"]:
            
            if len(self.startfruit_y_cache) == self.startfruit_window:
                for _ in range(self.startfruit_features):
                    self.startfruit_X_cache.pop(0)
                self.startfruit_y_cache.pop(0)
            
            tot_sell_vol, bs_product, bs_vol= self.values_extract(OrderedDict(sorted(state.order_depths[product].sell_orders.items())))
            tot_buy_vol, bb_product, bb_vol = self.values_extract(OrderedDict(sorted(state.order_depths[product].buy_orders.items(), reverse=True)), 1)
            mean_market_price = self.market_values_extract(state.market_trades.get("STARFRUIT", None))

            # y
            self.startfruit_y_cache.append((bs_product + bb_product) / 2)
            # X
            self.startfruit_X_cache.append(mean_market_price)
            self.startfruit_X_cache.append(tot_buy_vol)
            self.startfruit_X_cache.append(tot_sell_vol)

            product_lb = INF
            product_ub = -INF

            if len(self.startfruit_y_cache) == self.startfruit_window:
                model = OrdinaryLeastSquares()
                X, y = np.array(self.startfruit_X_cache[:-self.startfruit_features]), np.array(self.startfruit_y_cache[:-1])
                print(f"Length of X: {len(X)} ")
                X = np.reshape(X, (self.startfruit_window - 1, self.startfruit_features)) # Turn it into aa matrix [[x1, x2, xn] * n]

                print(f"and Y: {len(y)}")

                model.fit(X, y)
                product_lb = model.predict(X[-1])
                product_ub = model.predict(X[-1])

            acc_bid = product_lb
            acc_ask = product_ub

            # print("Predicted Price: ", (acc_bid + acc_ask) / 2)
            # print("Actual Price: ", mean_market_price)

            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
            results[product] += orders

        logger.flush(state, results, conversions, traderData)
        return results, conversions, traderData