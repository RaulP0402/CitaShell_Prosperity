from typing import List, Any, Dict, Tuple
import json
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from Driver_v3 import PartTradingState
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from Driver_v3 import AbstractIntervalTrader

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FIXED VALUE TRADER
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FixedValueTrader(AbstractIntervalTrader):

    def __init__(self, limit, value):
        super().__init__(limit)
        self.value = value

    def get_interval(self):
        # Note that I have access to self.state here as well
        return (self.value, self.value)

    def next_state(self):
        return "None"
    
    def calculate_orders(self, state: PartTradingState, interval: Tuple[int]):
        osell = OrderedDict(sorted(state.order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(state.order_depth.buy_orders.items(), reverse=True))

        _, best_sell_pr = self.values_extract(osell)
        _, best_buy_pr = self.values_extract(obuy, 1)

        # Look buy to cheap or sell low on a current short
        for ask, vol in osell.items():
            if ((ask < interval[0]) or (self.position < 0 and ask <= interval[0])):
                self.buy_lenient(ask, vol=-vol)
        
        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        if 0 <= self.position <= 15:
            self.buy_lenient(undercut_buy)

        if 15 < self.position <= self.limit:
            self.buy_strict(best_buy_pr - 1)

        for bid, vol in obuy.items():
            if ((bid > interval[1]) or (self.position > 0 and bid >= interval[1])):
                self.sell_lenient(bid, vol=vol)

        if -15 <= self.position <= 0:
            self.sell_lenient(undercut_sell)
        
        if self.limit < self.position < -15:
            self.sell_strict(best_sell_pr + 1)
