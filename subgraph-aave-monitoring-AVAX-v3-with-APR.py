import json
import requests
import urllib3
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import pandas as pd
import numpy as np

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
local_root_directory = '/Users/stefanduprey/Documents/My_Data/My_AaveHistoricalData/'
symbols = ['DAI.e','WBTC.e','WETH.e','USDC','USDt']
recompute_all = False
results = {}
if recompute_all:
    AVAX_URL_API = 'https://api.thegraph.com/subgraphs/name/aave/protocol-v3-avalanche'
    URL_API = AVAX_URL_API

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
    incentives_deposit_APR = {}
    incentives_borrow_APR = {}

    for token in response['reserves']:
        name = token['name']
        symbol = token['symbol']
        reserves_id_dictionnary[symbol] = token['id']

        variableBorrowRate = float(token['variableBorrowRate'])
        liquidityRate = float(token['liquidityRate'])
        TOKEN_DECIMALS = 10**float(token['decimals'])
    #    aEmissionPerSecond = float(token['aEmissionPerSecond'])
    #    vEmissionPerSecond = float(token['vEmissionPerSecond'])
        totalCurrentVariableDebt = float(token['totalCurrentVariableDebt'])
        totalATokenSupply  = float(token['totalATokenSupply'])
        REWARD_DECIMALS = 10**float(token['decimals'])


        #REWARD_PRICE_ETH = get_token_price_in_eth('AVAX')
        #TOKEN_PRICE_ETH = get_token_price_in_eth(symbol.replace('.e', '').replace('W', ''))

        # deposit and borrow calculation

        percentDepositAPY = 100.0 * liquidityRate/RAY
        percentVariableBorrowAPY = 100.0 * variableBorrowRate/RAY
        percentStableBorrowAPY = 100.0 * variableBorrowRate/RAY

        print(f"{symbol} deposit APY: {percentDepositAPY:.2f}%")
        print(f"{symbol} borrow APY: {percentVariableBorrowAPY:.2f}%")

        # percentDepositAPR = 100 * (aEmissionPerSecond*SECONDS_PER_YEAR * REWARD_PRICE_ETH * TOKEN_DECIMALS) / (totalATokenSupply * TOKEN_PRICE_ETH * REWARD_DECIMALS)
        # #percentBorrowAPR = 100 * (vEmissionPerSecond*SECONDS_PER_YEAR * REWARD_PRICE_ETH * TOKEN_DECIMALS) / (totalCurrentVariableDebt * TOKEN_PRICE_ETH * REWARD_DECIMALS)
        # if totalCurrentVariableDebt>0.:
        #     percentBorrowAPR = 100 * (vEmissionPerSecond*SECONDS_PER_YEAR * REWARD_PRICE_ETH * TOKEN_DECIMALS) / (totalCurrentVariableDebt * TOKEN_PRICE_ETH * REWARD_DECIMALS)
        # else:
        #     percentBorrowAPR = np.nan
        # incentives_deposit_APR[symbol] = percentDepositAPR
        # incentives_borrow_APR[symbol] = percentBorrowAPR

        #print(f"{symbol} AVAX reward deposit APR: {percentDepositAPR:.2f}%")
        #print(f"{symbol} AVAX reward borrow APR: {percentBorrowAPR:.2f}%")

    print('requesting historical reserve data')
    historical_data = {}
    #names = ['SushiToken (PoS)', 'Wrapped Matic', 'CRV (PoS)', '(PoS) Wrapped BTC', 'USD Coin (PoS)', 'Aavegotchi GHST Token (PoS)', 'ChainLink Token', 'Wrapped Ether', 'DefiPulse Index (PoS)', '(PoS) Dai Stablecoin', 'Balancer (PoS)', 'Aave (PoS)']
    matic_symbols = ['(PoS) Wrapped BTC', 'USD Coin (PoS)',  'Wrapped Ether',  '(PoS) Dai Stablecoin']
    ethereum_symbols = ['TUSD', 'AmmUniWBTCUSDC', 'YFI', 'BAT', 'MANA', 'DPI', 'AmmBptWBTCWETH', 'UNI', 'AmmWBTC', 'WBTC', 'AmmUniYFIWETH', 'AmmUniCRVWETH', 'REN', 'AmmUniSNXWETH', 'AmmGUniDAIUSDC', 'LINK', 'AmmBptBALWETH', 'AmmDAI', 'DAI', 'AAVE', 'XSUSHI', 'AmmUniRENWETH', 'FEI', 'MKR', 'AmmUSDC', 'USDC', 'AmmUniLINKWETH', 'AmmUniDAIWETH', 'AmmUniDAIUSDC', 'STETH', 'AmmUniUSDCWETH', 'AmmUniBATWETH', 'BAL', 'AmmUniWBTCWETH', 'SNX', 'AmmWETH', 'WETH', 'ENS', 'AmmUniMKRWETH', 'AmmGUniUSDCUSDT', 'AmmUniUNIWETH', 'CRV', 'KNC', 'AmmUniAAVEWETH', 'ZRX', 'ENJ']

    #symbols = ['DAI', 'WBTC', 'TUSD', 'WETH', 'USDT', 'USDC']


    all_in_df = None

    for symbol, id in reserves_id_dictionnary.items():
        if symbol in symbols:
            print(f'fetching {symbol}')
            try:
                reserve_query = """query get_reserves($reserve_id: ID!) {
                    reserve (id: $reserve_id) { 
                        id
                        paramsHistory(skip:5000, first: 5000) {
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
                historical_df['symbol'] = symbol
                historical_data[symbol] = historical_df
                from cryptotoolbox.realtime import realtime_plotting_utility
                fig = realtime_plotting_utility.plot_multiple_time_series(data_df=historical_df[['liquidityRatePerc','borrowRatePerc']], logy=False, split=False,
                                                                          put_on_same_scale=False, title=f'{symbol} supply_borrow rate')
                fig.show()

                historical_df.to_pickle(local_root_directory + f'{symbol}_rates.pkl')
                historical_df.to_csv(local_root_directory + f'{symbol}_rates.csv')

                temp_df = historical_df[['liquidityRatePerc', 'borrowRatePerc']].copy()
                temp_df.columns = [f'{symbol}_supply', f'{symbol}_borrow']
                results[symbol] = temp_df.copy()

                if all_in_df is None:
                    all_in_df = temp_df.copy()
                    results[symbol] = temp_df.copy()
                else :
                    all_in_df = pd.concat([all_in_df,temp_df.copy()])



            except Exception as e:
                print(f'no data {e}')
            #percentDepositAPY = 100.0 * liquidityRate/RAY
            #percentVariableBorrowAPY = 100.0 * variableBorrowRate/RAY
            #percentStableBorrowAPY = 100.0 * variableBorrowRate/RAY
else :
    for symbol in ['DAI.e','USDC','USDt']:

        for LTV in [0.3,0.4,0.5]:

            historical_df1 = pd.read_csv(local_root_directory + f'{symbol}_rates.csv')
            historical_df2 = pd.read_csv(local_root_directory + 'WBTC.e_rates.csv')
            historical_df1.index = pd.to_datetime(historical_df1.index)
            historical_df2.index = pd.to_datetime(historical_df2.index)
            historical_df1 = historical_df1.set_index('index')
            historical_df2 = historical_df2.set_index('index')

            #historical_df1 = pd.read_pickle(local_root_directory + f'{symbol}_rates.pkl')
            #historical_df2 = pd.read_pickle(local_root_directory + 'WBTC.e_rates.pkl')

            supply_incentive_APR = np.nan
            borrow_incentive_APR = np.nan
            if symbol == 'DAI.e':
                supply_incentive_APR = 1.14
                borrow_incentive_APR = 0.77
            elif symbol == 'WBTC.e':
                supply_incentive_APR = 0.48
                borrow_incentive_APR = 0.
            elif symbol == 'WETH.e':
                supply_incentive_APR = 2.36
                borrow_incentive_APR = 0.
            elif symbol == 'USDt':
                supply_incentive_APR = 2.25
                borrow_incentive_APR = 2.97
            elif symbol == 'USDC':
                supply_incentive_APR = 0.92
                borrow_incentive_APR = 0.64

            borrow_df = historical_df1[['borrowRatePerc']].copy()
            supply_df = historical_df2[['liquidityRatePerc']].copy()
            data_df = pd.concat([supply_df, borrow_df])

            data_df = data_df.sort_index()
            data_df = data_df.ffill()

            # historical_df = pd.read_csv(local_root_directory + f'{symbol}_rates.csv')
            results[symbol] = data_df
            from cryptotoolbox.realtime import realtime_plotting_utility


            # fig = realtime_plotting_utility.plot_multiple_time_series(
            #     data_df=historical_df[['liquidityRatePerc', 'borrowRatePerc']], logy=False, split=False,
            #     put_on_same_scale=False, title=f'{symbol} supply_borrow rate')
            # fig.show()

            def compute_total_apy(total_dollar_amount=np.nan, collateral_factor=0.5, nb_loops=0, supply=np.nan,
                                  borrow=np.nan,
                                  base_rate=0, slope1=0.2, slope2=0, kink=0.5, reserve_factor=0.25):
                total_supplied_amount = total_dollar_amount * (1 - pow(collateral_factor, nb_loops + 1)) / (
                            1 - collateral_factor)
                if nb_loops == 0:
                    total_borrowed_amount = 0.
                else:
                    total_borrowed_amount = total_dollar_amount * collateral_factor * (
                                1 - pow(collateral_factor, nb_loops + 1)) / (
                                                    1 - collateral_factor)
                utilization_rate = (borrow + total_borrowed_amount) / (supply + total_supplied_amount)
                if utilization_rate >= 0.5:
                    raise Exception('to implement : slope regime change')
                borrow_rate = base_rate + slope1 * utilization_rate
                supply_rate = utilization_rate * borrow_rate * (1 - reserve_factor)
                print(f'supply rate {supply_rate}')
                print(f'borrow rate {borrow_rate}')
                earned = supply_rate * total_supplied_amount - borrow_rate * total_borrowed_amount
                return earned


            def compute_rates_to_beat_no_dilution_simplistic(row=np.nan, LTV=0.5, supply_incentive_APR=np.nan,
                                                             borrow_incentive_APR=np.nan, stable_farming_yield = np.nan):
                supply_rate = row['liquidityRatePerc']
                borrow_rate = row['borrowRatePerc']
                total_rate = supply_rate + supply_incentive_APR - LTV * (borrow_rate + borrow_incentive_APR) + LTV*stable_farming_yield
                return total_rate


            cmp_rate3 = lambda x: compute_rates_to_beat_no_dilution_simplistic(row=x, LTV=LTV,
                                                                              supply_incentive_APR=supply_incentive_APR,
                                                                              borrow_incentive_APR=borrow_incentive_APR,
                                                                              stable_farming_yield=3.)
            cmp_rate6 = lambda x: compute_rates_to_beat_no_dilution_simplistic(row=x, LTV=LTV,
                                                                              supply_incentive_APR=supply_incentive_APR,
                                                                              borrow_incentive_APR=borrow_incentive_APR,
                                                                              stable_farming_yield=6.)

            data_df[f'totalRate{LTV}_stable3%'] = data_df.apply(cmp_rate3, axis=1)
            data_df[f'totalRate{LTV}_stable6%'] = data_df.apply(cmp_rate6, axis=1)

            fig = realtime_plotting_utility.plot_multiple_time_series(
                data_df=data_df[[f'totalRate{LTV}_stable3%',f'totalRate{LTV}_stable6%']], logy=False, split=False,
                put_on_same_scale=False, title=f'supply WBTC.e borrow{symbol} {LTV} supply_minus_borrow rate')
            fig.show()
            print('done')

print('computing the minimum APY over time')
