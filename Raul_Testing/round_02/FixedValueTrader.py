from typing import List, Any, Dict, Tuple
import json
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from abstract_trader import AbstractIntervalTrader, PartTradingState

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
        

        