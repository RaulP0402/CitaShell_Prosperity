import copy
from typing import Any, List
import string
import json
import numpy as np
import re
from collections import defaultdict

from collections import OrderedDict

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

class OrdinaryLeastSquares():
    def __init__(self):
        self.coef = []

    def _reshape_x(self, X):
        return X.reshape(-1, 1)

    def _concatenate_ones(self, X):

        X_with_intercept = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # ones = np.ones(shape=X.shape[0]).reshape(-1,1)
        # return np.concatenate((ones,X), 1)
        return X_with_intercept

    def _normalize(self, matrix):
        normalized_matrix = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0))
        return normalized_matrix

    def fit(self, X, y):
        # if len(X) == 1: 
        #     X = self._reshape_x(X)
        X = self._concatenate_ones(X)
        self.coef = np.linalg.pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    def predict(self, entry):
        b0 = self.coef[0]
        other_betas = self.coef[1:]
        prediction = b0
        # coef = [0.192259,0.250758,0.224051,0.332031]
        # print(f'coefficients : {self.coef}')

        for xi, bi in zip(entry, other_betas):
            prediction += (bi*xi)

        return prediction

logger = Logger()
Limits = {"AMETHYSTS" : 20, "STARFRUIT" : 20}
empty_dict = {'AMETHYSTS': 0, 'STARFRUIT': 0}
INF = int(1e9)
Indicators = ("mid_price", "market_sentiment", "lower_BB", "middle_BB", "upper_BB", "RSI", "MACD", "stat_regression")
sep = ','

# Parameters of modell and indicators
Parameters = {"n_MA":       50,  # time-span for MA
              "n_mean_BB":  50,  # time-span BB
              "n_sigma_BB": 50,  # time-span for sigma of BB
              "n_RSI":      14,  # time-span for RSI
              "n1_MACD":    26,  # time-span for the first (longer) MACD EMA
              "n2_MACD":    12,  # time-span for the second (shorter) MACD EMA
              "days_ADX":   5,   # how many time steps to consider a "day" in ADX
              "n_ADX":      5,   # time-span for smoothing in ADX
              "n_Model":    2,    # time-span for smoothing the regression model
            #   "Intercept":  -29.53989, # regression parameters
            #   "coeff_MS":   -0.03916,
            #   "coeff_BB":   0.83486,
            #   "coeff_RSI":  2.72222,
            #   "coeff_MACD": 0.13197,
            #   "coeff_MP":   0.17064
              }

n = max(Parameters.values())

def EMA(x, alpha):
    if len(x) == 0:
        return 1
    if len(x) == 1:
        return x[0]
    return alpha*x[-1] + (1-alpha)*EMA(x[:-1], alpha)


class Trader:

    def __init__(self):
        self.price_history = {"AMETHYSTS": [10000], "STARFRUIT": []}
        self.regression_matrix = {"AMETHYSTS": [], "STARFRUIT": []}
        self.positions = copy.deepcopy(empty_dict)
        self.limits = copy.deepcopy(Limits)
        self.starfruit_sentiment = 0

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

    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask):
        orders: List[Order] = []

        osell = OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        cpos = self.positions[product]

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.positions[product] < 0) and (ask <= acc_bid))) and cpos < self.limits['AMETHYSTS']:
                order_for = max(-vol, self.limits['AMETHYSTS'] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))
        
        cpos = self.positions[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.positions[product]>0) and (bid >= acc_ask))) and cpos > -self.limits['AMETHYSTS']:
                order_for = max(-vol, -self.limits['AMETHYSTS']-cpos)
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        return orders

    def compute_orders_starfruit(self, product, order_depth, fair_price, sentiment):
        orders: List[Order] = []

        osell = OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        cpos = self.positions[product]
        for ask, vol in osell.items():
            if (ask < fair_price) and cpos < self.limits['STARFRUIT']:
            # if ((ask < fair_price) or ((self.positions[product] < 0) and ask <= fair_price)) and cpos < self.limits['STARFRUIT']:
                order_for = min(-vol, self.limits['STARFRUIT'] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = next(iter(osell))
        undercut_sell = next(iter(obuy))

        undercut_buy = undercut_buy + 1
        undercut_sell = undercut_sell - 1

        bid_pr = min(undercut_buy, fair_price )# we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, fair_price)

        # if cpos < self.limits['STARFRUIT']:
        #     num = self.limits['STARFRUIT'] - cpos
        #     orders.append(Order(product, bid_pr, num))
        #     cpos += num

        cpos = self.positions[product]

        cpos = self.positions[product]
        for bid, vol in obuy.items():
            if (bid > fair_price) and cpos > -self.limits['STARFRUIT']:
            # if ((bid > fair_price) or ((self.positions[product]>0) and (bid == fair_price))) and cpos > -self.limits['STARFRUIT']:
                order_for = max(-vol, -self.limits['STARFRUIT']-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))
        
        # if cpos > -self.limits['STARFRUIT']:
        #     num = -self.limits['STARFRUIT'] - cpos
        #     orders.append(Order(product, sell_pr, num))
        #     cpos += num
            
        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = "@"
        # row_dims = 30
        row_dims = 20
        result = defaultdict(list)

        for product, position in state.position.items():
            self.positions[product] = position

        for product in sorted(state.listings):
            
            order_depth: OrderDepth = state.order_depths[product]

            osell = OrderedDict(sorted(order_depth.sell_orders.items()))
            obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
            # print(f'open sells {osell} and open buys {obuy}')
            # print(f'best sell price {next(iter(osell))} and best buy price {next(iter(obuy))}')
            # Previusly made trades (by bots)
            trades: List[Trade] = state.market_trades.get(product, [])

            price_at_timestamp = (next(iter(osell))  + next(iter(obuy))) / 2
            self.price_history[product].append(price_at_timestamp)

            if(len( self.price_history[product]) > row_dims): self.price_history[product].pop(0)

            if product == "AMETHYSTS": 
                fair_value_regression = 10000
                orders = self.compute_orders_amethysts(product, order_depth, fair_value_regression, fair_value_regression)
                result[product] = orders
            
            if product == "STARFRUIT":
                
                price_history = self.price_history[product]

                mid_price = price_history[-1]
                # Market sentiment -> shows if the market is bullish (> 1) or bearish (< 1)
                # market_sentiment = len(order_depth.buy_orders) / (len(order_depth.buy_orders)+len(order_depth.sell_orders))
                sentiment = (sum(order_depth.buy_orders.values()) + sum(order_depth.sell_orders.values()))
                if sentiment > 0:
                    self.starfruit_sentiment = 1
                elif sentiment < 0:
                    self.starfruit_sentiment = -1
                else:
                    self.starfruit_sentiment = 0

                # Bollinger Bands (lower band, middle band aka MA, upper band)
                middle_BB = np.mean(price_history[-Parameters["n_mean_BB"]:])   
                upper_BB = middle_BB + 2*np.var(price_history[-Parameters["n_sigma_BB"]:])
                lower_BB = middle_BB - 2*np.var(price_history[-Parameters["n_sigma_BB"]:])

                # RSI (relative strength index)            
                RSI_increments  = np.diff(price_history[-Parameters["n_RSI"]:])
                sum_up = np.sum([max(val,0) for val in RSI_increments])
                sum_down = np.sum([-min(val,0) for val in RSI_increments])

                avg_up = np.mean(sum_up)
                avg_down = np.mean(sum_down)
                RSI = avg_up / (avg_up + avg_down) if avg_up + avg_down != 0 else 0

                # MACD (moving average convergence/divergence)
                alpha_1 = 2/(Parameters["n1_MACD"]+1)
                alpha_2 = 2/(Parameters["n2_MACD"]+1)
                EMA_1 =  EMA(price_history[-Parameters["n1_MACD"]:], alpha_1)
                EMA_2 =  EMA(price_history[-Parameters["n2_MACD"]:], alpha_2)
                MACD = EMA_2 - EMA_1

                total_sell_volume, _ = self.values_extract(osell)
                total_buy_volume, _= self.values_extract(obuy,1)

                indicators = [lower_BB, middle_BB, upper_BB, RSI, MACD, total_buy_volume + total_sell_volume]
                regression_matrix = self.regression_matrix[product]
                regression_matrix.append(indicators)

                if len(regression_matrix) > row_dims:
                    self.regression_matrix[product].pop(0)
                
                regression_matrix = np.array(regression_matrix)
                # regression_matrix = np.reshape(regression_matrix, (len(indicators), len(self.price_history[product])))
                Y_value = np.array([self.price_history[product]])
                Y_value = np.reshape(Y_value, (len(Y_value[0]), 1))

                model = OrdinaryLeastSquares()
                model.fit(regression_matrix, Y_value)
                fair_price = model.predict(regression_matrix[-1])
                # print(f'fair price predicted {fair_price[0]}')

                orders: List[Order] = []
                orders = self.compute_orders_starfruit(product, order_depth, fair_price[0], self.starfruit_sentiment)
            
            result[product] = orders               

        logger.flush(state, result, 0, trader_data)
        return result, 0, trader_data

