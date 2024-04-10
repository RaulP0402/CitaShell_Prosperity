from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

				# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            result[product] = []

		    # String value holding Trader state data required.
				# It will be delivered as TradingState.traderData on next execution.
        traderData = f''

				# Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
