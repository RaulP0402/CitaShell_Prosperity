import copy
from typing import Any, List
import string
import json
import numpy as np
import re
from collections import defaultdict

from collections import OrderedDict

from Raul_Testing.old_drivers.datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId


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

Limits = {"AMETHYSTS" : 20, "STARFRUIT" : 20}

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
              "Intercept":  -29.53989, # regression parameters
              "coeff_MS":   -0.03916,
              "coeff_BB":   0.83486,
              "coeff_RSI":  2.72222,
              "coeff_MACD": 0.13197,
              "coeff_MP":   0.17064
              }

n = max(Parameters.values())

def EMA(x, alpha):
    if len(x) == 1:
        return x[0]
    return alpha*x[-1] + (1-alpha)*EMA(x[:-1], alpha)


class Trader:

    def __init__(self):
        self.price_history = {key: [] for key in Limits}
        self.regression_history = {key: [] for key in Limits}


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = "@"

        result = {}

        for product in sorted(state.listings):
            
            Limit = Limits[product]

            # Why is this here? 
            if product not in state.market_trades.keys():
                continue

            order_depth: OrderDepth = state.order_depths[product]
            trades: List[Trade] = state.market_trades[product]
            orders: List[Order] = []

            # log value of last n timestamps
            volume = 0
            price = 0

            if(len(trades) > 0):
                price_at_timestamp = sum([tr.price for tr in trades]) / len(trades)
                self.price_history[product].append(price_at_timestamp)
            else:
                self.price_history[product].append(self.price_history[product][-1])

            if(len( self.price_history[product]) > n): self.price_history[product].pop(0)

            if product == "AMETHYSTS": fair_value_regression = 10000


            ### indicators  

            if product == "STARFRUIT":

                price_history = self.price_history[product]

                # Current market value
                mid_price = price_history[-1]

                # Market sentiment -> shows if the market is bullish (> 1) or bearish (< 1)
                market_sentiment = len(order_depth.buy_orders)/(len(order_depth.buy_orders)+len(order_depth.sell_orders))

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
            
                self.regression_history[product].append(Parameters["Intercept"] + Parameters["coeff_MS"]*market_sentiment + Parameters["coeff_BB"]*middle_BB + Parameters["coeff_RSI"]*RSI + Parameters["coeff_MACD"]*MACD + Parameters["coeff_MP"]*mid_price)
                # Older regressions using more variables (also possibly good)...
                # .append(-27.341845 -0.17*market_sentiment -0.004363*lower_BB + 0.800606*middle_BB + 0.066560*RSI + 0.204905*MACD + 0.209055*mid_price)
                # .append(-26.76876 -0.17*market_sentiment -0.02954*lower_BB + 0.84520*middle_BB + 0.28838*RSI + 0.26612*MACD + 0.18943*mid_price)
                # .append(-60.31162 -0.04426*market_sentiment + 0.09908*lower_BB + 0.91190*middle_BB + 13.02079*RSI + 0.20754*MACD)
                # .append(-83.38018 + 1.73501*market_sentiment + 0.12058*lower_BB + 0.89438*middle_BB + 18.41226*RSI -0.05367*MACD)
                fair_value_regression = np.mean(self.regression_history[product])
                print(f'fair value regression is : {fair_value_regression}')
                if(len(self.regression_history[product]) > Parameters["n_Model"]): self.regression_history[product].pop(0)        

                # add indicators to trader data
           
                trader_data += f'{round(mid_price,4)}{sep}'
                trader_data += f'{round(market_sentiment,4)}{sep}'
                trader_data += f'{round(lower_BB,4)}{sep}'
                trader_data += f'{round(middle_BB,4)}{sep}'
                trader_data += f'{round(upper_BB,4)}{sep}'
                trader_data += f'{round(RSI,4)}{sep}'
                trader_data += f'{round(MACD,6)}{sep}'
                trader_data += f'{round(fair_value_regression,4)}@'     


            # place orders
            bid = round(fair_value_regression) - 2 
            ask = round(fair_value_regression) + 2 

            bid_volume_percentage = 1
            ask_volume_percentage = 1

            if(product == "STARFRUIT"):
                bid_volume_percentage = 0.5 + 0.2*np.tanh(fair_value_regression - mid_price)
                ask_volume_percentage = 0.5 + 0.2*np.tanh(mid_price - fair_value_regression)
            
            bid_volume = int(bid_volume_percentage*(Limit - state.position.get(product, 0)))
            ask_volume = int(ask_volume_percentage*(-Limit - state.position.get(product, 0)))

            if (state.position.get(product, 0) / Limit > 0.8):
                bid -= 1
                ask -= 1
            if (state.position.get(product, 0) / Limit < -0.8):
                bid += 1
                ask += 1
            

            orders.append(Order(product, bid, bid_volume))
            orders.append(Order(product, ask, ask_volume))
            
            result[product] = orders               

        logger.flush(state, result, 0, trader_data)
        return result,0,  trader_data
