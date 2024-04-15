from typing import List, Any, Dict, Tuple
import json
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from abstract_trader import AbstractIntervalTrader, PartTradingState


class OrchidModel(AbstractIntervalTrader):
    
    def __init__(self, limit: int):
        super().__init__(limit)
        self.dims = 10

    def get_price(self) -> int:
        d = {
            "sunlight": [],
            "humidity": []
        }
        self.data = self.data if self.data else d

        if len(self.data['sunlight']) == self.dims:
            self.data['sunlight'].pop(0)
            self.data['humidity'].pop(0)
        
        self.data['sunlight'].append(
            self.observations.sunlight
        )
        self.data['humidity'].append(
            self.observations.humidity
        )

        print(f'data is {self.data}')

        return 0
    
    def calculate_orders(self, state: PartTradingState, acc_bid: int, acc_ask: int):
        print("CHECK CHECK")
        
