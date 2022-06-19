#!/usr/bin/env python3

#
# Example that shows all active positions in the 0.3% USDC/ETH
# pool using data from the Uniswap v3 subgraph.
#

from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import numpy as np

import sys
import pandas as pd

#### polygon data set
#POOL_ID = '0x45dda9cb7c25131df268515131f647d726f50608'
#URL = 'https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-polygon'

### arbitrum data set
#POOL_ID = '0xc31e54c7a869b9fcbecc14363cf510d1c41fa443'
#URL = 'https://api.thegraph.com/subgraphs/name/ianlapham/arbitrum-dev'

### ethereum data set
POOL_ID = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'
URL = 'https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-subgraph'

# default pool id is the 0.3% USDC/ETH pool
#POOL_ID = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
#URL = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'

# if passed in command line, use an alternative pool ID
if len(sys.argv) > 1:
    POOL_ID = sys.argv[1]

TICK_BASE = 1.0001

pool_query = """query get_pools($pool_id: ID!) {
  pools(where: {id: $pool_id}) {
    tick
    sqrtPrice
    liquidity
    feeTier
    token0 {
      symbol
      decimals
    }
    token1 {
      symbol
      decimals
    }
  }
}"""

# return open positions only (with liquidity > 0)
position_query = """query get_positions($num_skip: Int, $pool_id: ID!) {
  positions(skip: $num_skip, where: {pool: $pool_id, liquidity_gt: 0}) {
    id
    tickLower { tickIdx }
    tickUpper { tickIdx }
    liquidity
  }
}"""


def from_tick_to_inv_adj_price(tick, decimal0, decimal1):
    price = tick_to_price(tick)
    inv_price = inv_adjust_price(price, decimal0, decimal1)
    return inv_price

def tick_to_price(tick):
    return TICK_BASE ** tick

def inv_adjust_price(price, decimal0, decimal1):
    adjusted_current_price_ = price / (10 ** (decimal1 - decimal0))
    return 1./adjusted_current_price_

def from_inv_adj_price_to_tick(adj_inv_price,decimal0, decimal1):
    adj_price = 1./adj_inv_price
    price = adj_price * (10 ** (decimal1 - decimal0))
    tick = int(np.log(price)/np.log(TICK_BASE))
    return tick


client = Client(
    transport=RequestsHTTPTransport(
        url=URL,
        verify=True,
        retries=5,
    ))

# get pool info
try:
    variables = {"pool_id": POOL_ID}
    response = client.execute(gql(pool_query), variable_values=variables)

    if len(response['pools']) == 0:
        print("pool not found")
        exit(-1)

    pool = response['pools'][0]
    pool_liquidity = int(pool["liquidity"])
    current_tick = int(pool["tick"])

    token0 = pool["token0"]["symbol"]
    token1 = pool["token1"]["symbol"]
    decimals0 = int(pool["token0"]["decimals"])
    decimals1 = int(pool["token1"]["decimals"])
except Exception as ex:
    print("got exception while querying pool data:", ex)
    exit(-1)

# Compute and print the current price
current_price = tick_to_price(current_tick)
current_sqrt_price = tick_to_price(current_tick / 2)
adjusted_current_price = current_price / (10 ** (decimals1 - decimals0))
print("Current price={:.6f} {} for {} at tick {}".format(adjusted_current_price, token1, token0, current_tick))
print("Inversed durrent price={:.6f} {} for {} at tick {}".format(1./adjusted_current_price, token1, token0, current_tick))

##### to get rid of
lower_tick = current_tick - 100
upper_tick = current_tick + 100
test_lower_adj_inv_price = from_tick_to_inv_adj_price(lower_tick, decimal0=decimals0, decimal1=decimals1)
test_upper_price = from_tick_to_inv_adj_price(upper_tick, decimal0=decimals0, decimal1=decimals1)
test_back_tick = from_inv_adj_price_to_tick(test_lower_adj_inv_price, decimal0=decimals0, decimal1=decimals1)
assert lower_tick == test_back_tick

lower_tick_from_lower_bound = from_inv_adj_price_to_tick(2500., decimal0=decimals0, decimal1=decimals1)
upper_tick_from_lower_bound = from_inv_adj_price_to_tick(3500., decimal0=decimals0, decimal1=decimals1)

def get_fork_spreading_factor(lower_bound, upper_bound, decimal0=decimals0, decimal1=decimals1):
    upper_tick_from_upper_bound = from_inv_adj_price_to_tick(lower_bound, decimal0=decimals0, decimal1=decimals1)
    lower_tick_from_lower_bound = from_inv_adj_price_to_tick(upper_bound, decimal0=decimals0, decimal1=decimals1)
    sa = tick_to_price(lower_tick_from_lower_bound / 2)
    sb = tick_to_price(upper_tick_from_upper_bound / 2)
    return (sb-sa)


spreading_factor = get_fork_spreading_factor(2500., 3500., decimal0=decimals0, decimal1=decimals1)


print('done')
#### end of to get rid of

# get position info
positions = []
num_skip = 0
try:
    while True:
        print("Querying positions, num_skip={}".format(num_skip))
        variables = {"num_skip": num_skip, "pool_id": POOL_ID}
        response = client.execute(gql(position_query), variable_values=variables)

        if len(response["positions"]) == 0:
            break
        num_skip += len(response["positions"])
        for item in response["positions"]:
            tick_lower = int(item["tickLower"]["tickIdx"])
            tick_upper = int(item["tickUpper"]["tickIdx"])
            liquidity = int(item["liquidity"])
            id = int(item["id"])
            positions.append((tick_lower, tick_upper, liquidity, id))
except Exception as ex:
    print("got exception while querying position data:", ex)
    exit(-1)



# Sum up all the active liquidity and total amounts in the pool
active_positions_liquidity = 0
total_amount0 = 0
total_amount1 = 0

# Print all active positions
for tick_lower, tick_upper, liquidity, id in sorted(positions):

    sa = tick_to_price(tick_lower / 2)
    sb = tick_to_price(tick_upper / 2)

    if tick_upper <= current_tick:
        # Only token1 locked
        amount1 = liquidity * (sb - sa)
        total_amount1 += amount1

    elif tick_lower < current_tick < tick_upper:
        # Both tokens present
        amount0 = liquidity * (sb - current_sqrt_price) / (current_sqrt_price * sb)
        amount1 = liquidity * (current_sqrt_price - sa)
        adjusted_amount0 = amount0 / (10 ** decimals0)
        adjusted_amount1 = amount1 / (10 ** decimals1)

        total_amount0 += amount0
        total_amount1 += amount1
        active_positions_liquidity += liquidity

        print("  position {: 7d} in range [{},{}]: {:.2f} {} and {:.2f} {} at the current price".format(
              id, tick_lower, tick_upper,
              adjusted_amount0, token0, adjusted_amount1, token1))
    else:
        # Only token0 locked
        amount0 = liquidity * (sb - sa) / (sa * sb)
        total_amount0 += amount0


print("In total (including inactive positions): {:.2f} {} and {:.2f} {}".format(
      total_amount0 / 10 ** decimals0, token0, total_amount1 / 10 ** decimals1, token1))
print("Total liquidity from active positions: {}, from pool: {} (should be equal)".format(
      active_positions_liquidity, pool_liquidity))


