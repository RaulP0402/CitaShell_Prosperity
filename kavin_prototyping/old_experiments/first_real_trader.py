from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:

    def run(self, state: TradingState):
        print(state)
		# Orders to be placed on exchange matching engine
        result = {}

        order_depth: OrderDepth = state.order_depths['AMETHYSTS']
        orders: List[Order] = []

        posn = state.position['AMETHYSTS'] if 'AMETHYSTS' in state.position else 0

        buy_total = 0

        for (price, vol) in order_depth.sell_orders.items():
            # all volumes will be negative since these are sell
            # orders
            if price < 10000:
                orders.append(Order('AMETHYSTS', price, min(-vol, 20 - posn)))
                posn += min(-vol, 20 - posn)
                print(f"BUY - {min(-vol, 20 - posn)}")

        for (price, vol) in order_depth.buy_orders.items():
            # We need to place a negative value in the order so we can sell
            if price > 10000:
                orders.append(Order('AMETHYSTS', price, max(-vol, -posn - 20)))
                posn += max(-vol, -posn - 20)
                print(f"SELL - {max(-vol, -posn - 20)}")


        result['AMETHYSTS'] = orders

		    # String value holding Trader state data required.
			# It will be delivered as TradingState.traderData on next execution.
        traderData = f'{0}'

				# Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
