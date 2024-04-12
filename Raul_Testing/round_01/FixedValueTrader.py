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

    
    def calculate_orders(self, state: PartTradingState, pred_price: int):
        SKEWED_RANGE = 3

        if self.position + SKEWED_RANGE >= self.limit:
            volume_to_leave = self.position - self.limit + SKEWED_RANGE + 1

            self.buy(pred_price-3, 40)
            self.sell(pred_price, volume_to_leave)
            self.sell(pred_price+2, self.position + 20 - volume_to_leave)

        elif self.position - SKEWED_RANGE <= -self.limit:
            volume_to_leave = -self.position - self.limit + SKEWED_RANGE + 1

            self.sell(pred_price+3, 40)
            self.buy(pred_price, volume_to_leave)
            self.buy(pred_price-2, self.position + 20 - volume_to_leave)
        
        else:
            self.buy(pred_price-2, 40)
            self.sell(pred_price+2, 40)

        # open_sells = OrderedDict(sorted(self.state.order_depth.sell_orders.items()))
        # open_buys = OrderedDict(sorted(self.state.order_depth.buy_orders.items(), reverse=True))

        # _, best_sell_pr = self.values_extract(open_sells)
        # _, best_buy_pr = self.values_extract(open_buys, 1)

        # for ask, vol in open_sells.items():
        #     if ((ask < pred_price) or (self.position < 0 and ask == pred_price)) and self.position < self.limit:
        #         self.buy(ask, -vol)

        # undercut_buy = best_buy_pr + 1
        # undercut_sell = best_sell_pr - 1

        # bid_pr = min(undercut_buy, pred_price - 3)
        # sell_pr = max(undercut_sell, pred_price + 3)

        # if (self.position < self.limit) and (self.position < 0):
        #     num = min(40, self.limit - self.position)
        #     self.buy(min(undercut_buy + 1, pred_price - 1), num)
        
        # if (self.position < self.limit) and (self.position > 15):
        #     num = min(40, self.limit - self.position)
        #     self.buy(min(undercut_buy - 1, pred_price - 1), num)

        # if self.position < self.limit:
        #     num = min(40, self.limit - self.position)
        #     self.buy(int(round(bid_pr)), num)

        # for bid, vol in open_buys.items():
        #     if ((bid > pred_price) or (self.position > 0 and bid == pred_price)) and self.position > -self.limit:
        #         self.sell(bid, vol)

        # if (self.position > -self.limit) and (self.position > 0):
        #     num = max(-40, -self.limit - self.position)
        #     self.sell(max(undercut_sell - 1, pred_price + 1), num)

        # if (self.position > -self.limit) and (self.position < -15):
        #     num = max(-40, -self.limit - self.position)
        #     self.sell(max(undercut_sell + 1, pred_price + 1), num)

        # if self.position > -self.limit:
        #     num = max(-40, -self.limit - self.position)
        #     self.sell(sell_pr, num)
        

        