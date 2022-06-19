import json
import requests
import urllib3
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import pandas as pd

token_ticker = 'USDC'
reward_token_ticker = 'WMATIC'

RAY = 10.0**27
WAD = 10.0**18
SECONDS_PER_YEAR = 31556926.0

#################################################
def get_token_price_in_eth(token_ticker):

    if token_ticker!='USDT':
      while True:
        r = requests.get(
          f'https://api.binance.com/api/v3/ticker/price?symbol={token_ticker}USDT')
        if r.status_code == 200:
          break
      data = r.json()
      token_in_usdt=  float(data['price'])
    else:
      token_in_usdt = 1.0

    while True:
      r = requests.get(
        'https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT')
      if r.status_code == 200:
        break
    data = r.json()
    eth_in_usdt=  float(data['price'])

    return token_in_usdt/eth_in_usdt

#################################################

MATIC_URL_API = 'https://api.thegraph.com/subgraphs/name/aave/aave-v2-matic'
MAINNET_URL_API = 'https://api.thegraph.com/subgraphs/name/aave/protocol-v2'

URL_API = MAINNET_URL_API

query = gql('''
query {
  reserves (where: {
    usageAsCollateralEnabled: true
  }) {
    id
    name
    price {
      id
      priceInEth
    }
    liquidityRate
    variableBorrowRate
    stableBorrowRate
    aEmissionPerSecond
    vEmissionPerSecond
    decimals
    totalATokenSupply
    totalCurrentVariableDebt
    symbol
  }
}
''')

sample_transport=RequestsHTTPTransport(
    url=URL_API,
    verify=True,
    retries=3,
)
client = Client(
    transport=sample_transport
)

response = client.execute(query)

reserves_id_dictionnary = {}
for token in response['reserves']:
    reserves_id_dictionnary[token['name']] = token['id']

    if token_ticker in token['symbol']:
        variableBorrowRate = float(token['variableBorrowRate'])
        liquidityRate = float(token['liquidityRate'])
        TOKEN_DECIMALS = 10**float(token['decimals'])
        aEmissionPerSecond = float(token['aEmissionPerSecond'])
        vEmissionPerSecond = float(token['vEmissionPerSecond'])
        totalCurrentVariableDebt = float(token['totalCurrentVariableDebt'])
        totalATokenSupply  = float(token['totalATokenSupply'])
    if reward_token_ticker in token['symbol']:
        REWARD_DECIMALS = 10**float(token['decimals'])


REWARD_PRICE_ETH = get_token_price_in_eth('MATIC')
TOKEN_PRICE_ETH = get_token_price_in_eth(token_ticker.replace('W', ''))

# deposit and borrow calculation

percentDepositAPY = 100.0 * liquidityRate/RAY
percentVariableBorrowAPY = 100.0 * variableBorrowRate/RAY
percentStableBorrowAPY = 100.0 * variableBorrowRate/RAY

print(f"{token_ticker} deposit APY: {percentDepositAPY:.2f}%")
print(f"{token_ticker} borrow APY: {percentVariableBorrowAPY:.2f}%")

percentDepositAPR = 100 * (aEmissionPerSecond*SECONDS_PER_YEAR * REWARD_PRICE_ETH * TOKEN_DECIMALS) / (totalATokenSupply * TOKEN_PRICE_ETH * REWARD_DECIMALS)
percentBorrowAPR = 100 * (vEmissionPerSecond*SECONDS_PER_YEAR * REWARD_PRICE_ETH * TOKEN_DECIMALS) / (totalCurrentVariableDebt * TOKEN_PRICE_ETH * REWARD_DECIMALS)

print(f"{token_ticker} WMATIC reward deposit APR: {percentDepositAPR:.2f}%")
print(f"{token_ticker} WMATIC reward borrow APR: {percentBorrowAPR:.2f}%")

print('requesting historical reserve data')
historical_data = {}
#names = ['SushiToken (PoS)', 'Wrapped Matic', 'CRV (PoS)', '(PoS) Wrapped BTC', 'USD Coin (PoS)', 'Aavegotchi GHST Token (PoS)', 'ChainLink Token', 'Wrapped Ether', 'DefiPulse Index (PoS)', '(PoS) Dai Stablecoin', 'Balancer (PoS)', 'Aave (PoS)']
names = ['(PoS) Wrapped BTC', 'USD Coin (PoS)',  'Wrapped Ether',  '(PoS) Dai Stablecoin']
for name, id in reserves_id_dictionnary.items():
    if name in names:
        reserve_query = """query get_reserves($reserve_id: ID!) {
            reserve (id: $reserve_id) { 
                id
                paramsHistory(skip:100, first: 100) {
                  id
                  variableBorrowRate
                  utilizationRate
                  liquidityRate
                  timestamp
                }
            }
        }"""

        reserve_id = id
        variables = {"reserve_id": reserve_id}
        print('executing query')
        response = client.execute(gql(reserve_query), variable_values=variables)
        print('parsing data')
        historical_df = pd.DataFrame(response['reserve']['paramsHistory'])
        historical_df['date'] = pd.to_datetime(historical_df['timestamp'], unit = 's')
        historical_df['index'] = pd.to_datetime(historical_df['timestamp'], unit = 's')
        historical_df = historical_df.set_index('index')
        historical_df = historical_df.sort_index()
        historical_df['liquidityRatePerc'] = 100.0*historical_df['liquidityRate'].astype(float)/RAY
        historical_df['borrowRatePerc'] = 100.0*historical_df['variableBorrowRate'].astype(float)/RAY
        historical_data[name] = historical_df
        from cryptotoolbox.realtime import realtime_plotting_utility
        fig = realtime_plotting_utility.plot_multiple_time_series(data_df=historical_df[['borrowRatePerc']], logy=False, split=False,
                                                                  put_on_same_scale=False, title=f'{name} borrow rate')
        fig.show()

        #percentDepositAPY = 100.0 * liquidityRate/RAY
        #percentVariableBorrowAPY = 100.0 * variableBorrowRate/RAY
        #percentStableBorrowAPY = 100.0 * variableBorrowRate/RAY

print('done')
