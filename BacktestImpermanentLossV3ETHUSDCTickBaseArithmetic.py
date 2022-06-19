# -*- coding: utf-8 -*-


import numpy as np
import warnings

warnings.filterwarnings('ignore')
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['figure.figsize'] = [12, 7]

plt.rc('font', size=14)
import plotly.graph_objects as go

import pandas as pd
from datetime import datetime
from cryptotoolbox.connector import crypto_connector
from cryptotoolbox.realtime import realtime_plotting_utility
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import numpy as np


#
# def getBBands(df, period=10, stdNbr=2):
#     try:
#         upper, middle, lower = talib.BBANDS(
#             df['Close'].values,
#             timeperiod=period,
#             # number of non-biased standard deviations from the mean
#             nbdevup=stdNbr,
#             nbdevdn=stdNbr,
#             # Moving average type: simple moving average here
#             matype=0)
#     except Exception as ex:
#         return None
#     data = dict(upper=upper, middle=middle, lower=lower)
#     df = pd.DataFrame(data, index=df.index, columns=['upper', 'middle', 'lower']).dropna()
#     return df

def annualize_percent(first_value=np.nan, last_value=np.nan, nb_years=np.nan):
    return (last_value / first_value) ** (1. / nb_years) - 1.


def compute_impermanent_loss_v2(p_0=np.nan, p_out=np.nan):
    tau = p_out / p_0
    return 2 * np.sqrt(tau) / (1. + tau) - 1, tau


def compute_impermanent_loss_v3(p_0=np.nan, p_a=np.nan, p_b=np.nan, p_out=np.nan):
    ILv2, tau = compute_impermanent_loss_v2(p_0=p_0, p_out=p_out)
    square_one = np.sqrt(p_a / p_0)
    square_two = np.sqrt(p_0 / p_b)
    num_one = square_one + tau * square_two
    factor = 1. / (1. - num_one / (1. + tau))
    ILv3 = ILv2 * factor
    return ILv3, ILv2


def getBBands(df, period=10, stdNbr=2):
    df['middle'] = get_sma(df['Close'], period)
    df['std'] = df['Close'].rolling(period).std()
    df['upper'] = df['middle'] + df['std'] * stdNbr
    df['lower'] = df['middle'] - df['std'] * stdNbr
    return df


def get_sma(prices, rate):
    return prices.rolling(rate).mean()


def get_bollinger_bands(prices, rate=20):
    sma = get_sma(prices, rate)
    std = prices.rolling(rate).std()
    bollinger_up = sma + std * 2  # Calculate top band
    bollinger_down = sma - std * 2  # Calculate bottom band
    return bollinger_up, bollinger_down


def getBBands(df, period=10, stdNbr=2):
    df['middle'] = get_sma(df['Close'], period)
    df['std'] = df['Close'].rolling(period).std()
    df['upper'] = df['middle'] + df['std'] * stdNbr
    df['lower'] = df['middle'] - df['std'] * stdNbr
    return df


def get_sma(prices, rate):
    return prices.rolling(rate).mean()


def get_bollinger_bands(prices, rate=20):
    sma = get_sma(prices, rate)
    std = prices.rolling(rate).std()
    bollinger_up = sma + std * 2  # Calculate top band
    bollinger_down = sma - std * 2  # Calculate bottom band
    return bollinger_up, bollinger_down

def from_tick_to_adj_price(tick, decimal0, decimal1):
    price = tick_to_price(tick)
    inv_price = comp_adjust_price(price, decimal0, decimal1)
    return inv_price

def from_tick_to_inv_adj_price(tick, decimal0, decimal1):
    price = tick_to_price(tick)
    inv_price = comp_inv_adjust_price(price, decimal0, decimal1)
    return inv_price

def tick_to_price(tick):
    return TICK_BASE ** tick

def comp_adjust_price(price, decimal0, decimal1):
    adjusted_current_price_ = price / (10 ** (decimal1 - decimal0))
    return adjusted_current_price_

def comp_inv_adjust_price(price, decimal0, decimal1):
    adjusted_current_price_ = price / (10 ** (decimal1 - decimal0))
    return 1./adjusted_current_price_

def from_inv_adj_price_to_tick(adj_inv_price,decimal0, decimal1):
    adj_price = 1./adj_inv_price
    price = adj_price * (10 ** (decimal1 - decimal0))
    tick = int(np.log(price)/np.log(TICK_BASE))
    return tick

def from_adj_price_to_tick(adj_price,decimal0, decimal1):
    price = adj_price * (10 ** (decimal1 - decimal0))
    tick = int(np.log(price)/np.log(TICK_BASE))
    return tick

### ethereum data set
POOL_ID = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'
URL = 'https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-subgraph'


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

current_adj_price = from_tick_to_adj_price(current_tick, decimal0=decimals0, decimal1=decimals1)
current_tick_bis = from_adj_price_to_tick(current_adj_price, decimal0=decimals0, decimal1=decimals1)
assert current_tick_bis == current_tick

get_tick = lambda x : from_adj_price_to_tick(x, decimal0=decimals0, decimal1=decimals1)
get_price = lambda x : from_tick_to_adj_price(x, decimal0=decimals0, decimal1=decimals1)

cp = get_price(current_tick)
ct = get_tick(cp)

lower_tick = get_tick(0.0004)
upper_tick = get_tick(0.0008)
lower_price = get_price(lower_tick)
upper_price = get_price(upper_tick)

base_threshold = abs(upper_tick - lower_tick)
print(f'base threshold {base_threshold}')


frequence = 'daily'
number_per_year = 252

freqly_pkl_file_name_suffix = '_' + frequence + '_returns.pkl'
localrootdirectory = '/Users/stefanduprey/Documents/My_Data/My_ILLossBacktest'

#for me_pair in [('ETH', 'USDT'),('DOT','BNB'),('ETH','USDT'),('LINK','ETH')]:
for me_pair in [('USDT', 'ETH')]:
    ssj = me_pair[0]
    ssj_against = me_pair[1]
    #ssj = 'LINK'
    #ssj_against = 'ETH'

    #daily_crypto_starting_day = '2017-03-01'
    daily_crypto_starting_day = '2021-05-16'

    daily_crypto_starting_date = datetime.strptime(daily_crypto_starting_day, '%Y-%m-%d')
    starting_date = daily_crypto_starting_date
    running_date = datetime.strptime('2022-05-16', '%Y-%m-%d')

    refetch_all = True
    data_df = crypto_connector.fetch_crypto_daily_data(ssj=ssj, ssj_against=ssj_against,
                                                       local_root_directory='None',
                                                       daily_return_pkl_filename_suffix=freqly_pkl_file_name_suffix,
                                                       refetch_all=refetch_all,
                                                       daily_crypto_starting_day=daily_crypto_starting_date,
                                                       daily_crypto_ending_day=running_date)
    data_df = data_df[['open', 'high', 'low', 'close']]

    data_df.columns = ['Open', 'High', 'Low', 'Close']
    data_df['Date'] = data_df.index
    print('done')

    data_df['CloseTick'] = data_df['Close'].apply(get_tick)
    data_df['OpenTick'] = data_df['Open'].apply(get_tick)
    data_df['HighTick'] = data_df['High'].apply(get_tick)
    data_df['LowTick'] = data_df['Low'].apply(get_tick)

    plotTick = False
    if plotTick:
        fig = realtime_plotting_utility.plot_multiple_time_series(
            data_df=data_df[['CloseTick', 'OpenTick', 'HighTick', 'LowTick']].copy(), logy=False, split=False,
            put_on_same_scale=False,
            title=f'v3 bounds over time{ssj}{ssj_against}')
        fig.show()

    # bb_periods = range(5,35,5)
    # stdNbrs = [i / 10 for i in range(1, 10, 2)] + [i for i in range(1, 10)]
    # bound_proximities = [0,0.05,0.1]
    # plotHtml = False

    base_thresholds = [2400]
    plotHtml = True

    plotResistance = False

    # stdNbrs = [3]
    # slope_windows = [5]
    # slope_thresholds = [190]
    # plotHtml = True
    # plotResistance = False

    # relaxed parameters
    # # #### you name it (bon maintenant mais pas bon avant)
    # bb_periods = [5]
    # stdNbrs = [3]
    # slope_windows = [5]
    # slope_thresholds = [590]
    # plotHtml = True
    # plotResistance = False

    max_sum_in_between = -np.inf
    max_il = +np.inf
    max_slope_window = np.nan
    max_slope_threshold = np.nan
    max_bb_period = np.nan
    max_stdNbr = np.nan
    objectives = []

    for base_threshold in base_thresholds:
        df = data_df.copy()
        print(f'params_{base_threshold}')
        stage_nb = 0
        upper_bound_tick = df['CloseTick'].iloc[0] + base_threshold
        lower_bound_tick = df['CloseTick'].iloc[0] - base_threshold
        upper_bound_price = get_price(upper_bound_tick)
        lower_bound_price = get_price(lower_bound_tick)

        initial_price = df['Close'].iloc[0]
        stages = []
        counter = 0
        for i, row in df.iterrows():
            if counter == 0:
                counter = counter + 1
                continue
            ILv3, ILv2 = np.nan, np.nan
            if row['High'] > upper_bound_price or row['Low'] < lower_bound_price:
                up = row['High'] > upper_bound_price
                p_out = np.nan
                if up:
                    p_out = upper_bound_price
                else:
                    p_out = lower_bound_price
                ILv3, ILv2 = compute_impermanent_loss_v3(p_0=initial_price, p_a=lower_bound_price, p_b=upper_bound_price,
                                                         p_out=p_out)

                if up:
                    initial_price = row['High']
                else:
                    initial_price = row['Low']
                # here compute the previous impermanent loss

                stage_nb = stage_nb + 1
                upper_bound_tick = row['HighTick'] + base_threshold
                lower_bound_tick = row['LowTick'] - base_threshold
                upper_bound_price = get_price(upper_bound_tick)
                lower_bound_price = get_price(lower_bound_tick)

            stages.append({
                'date': row['Date'],
                'incurred_v3': ILv3,
                'incurred_v2': ILv2,
                'initial_price': initial_price,
                'upper_bound_price': upper_bound_price,
                'lower_bound_price': lower_bound_price,
                'upper_bound_tick': upper_bound_tick,
                'lower_bound_tick': lower_bound_tick,
                'stage_nb': stage_nb,
            })

        stages_df = pd.DataFrame(stages)
        stages_df = stages_df.fillna(0.)
        stages_df['loss_v3'] = 100. * np.cumprod(1 + stages_df['incurred_v3'].values)
        stages_df['loss_v2'] = 100. * np.cumprod(1 + stages_df['incurred_v2'].values)

        stages_df['date'] = pd.to_datetime(stages_df['date'])
        stages_df.set_index('date', inplace=True)
        all_df = pd.merge(stages_df.copy(), df.copy(), left_index=True, right_index=True)
        if plotHtml:
            fig = realtime_plotting_utility.plot_multiple_time_series(
                data_df=all_df[['lower_bound_price', 'upper_bound_price', 'Close']].copy(), logy=False, split=False,
                put_on_same_scale=False,
                title=f'v3 bounds over time{ssj}{ssj_against}')
            fig.show()

            fig1 = realtime_plotting_utility.plot_multiple_time_series(data_df=all_df[['loss_v2', 'loss_v3']].copy(),
                                                                       logy=False, split=False,
                                                                       put_on_same_scale=False,
                                                                       title=f'v2/v3 loss over time {ssj}{ssj_against}')
            fig1.show()

        #computing the annulized loss
        delta = stages_df.index[len(stages_df) - 1] - stages_df.index[0]
        nb_years = delta.days/365

        print(f'nb years {nb_years}')
        first_v3_value = stages_df['loss_v3'].iloc[0]
        first_v2_value = stages_df['loss_v2'].iloc[0]

        last_v3_value = stages_df['loss_v3'].iloc[len(stages_df) - 1]
        last_v2_value = stages_df['loss_v2'].iloc[len(stages_df) - 1]


        v2_loss_apy = annualize_percent(first_value = first_v2_value, last_value= last_v2_value, nb_years = nb_years)
        v3_loss_apy = annualize_percent(first_value = first_v3_value, last_value= last_v3_value, nb_years = nb_years)

        sum_rebalance = all_df['stage_nb'].max()

        # objective 1 : minimiser le nombre de rebalancements
        obj_rebalance = sum_rebalance
        # objective 2 : minimiser l'impermanent loss
        all_df['spread'] = all_df['upper_bound_price'] - all_df['lower_bound_price']
        mean_spread = all_df['spread'].mean()
        objectives.append({
            'base_threshold': base_threshold,
            'mean_spread': mean_spread,
            'nb_rebalancing_days': obj_rebalance,
            'v2_loss_apy': v2_loss_apy,
            'v3_loss_apy': v3_loss_apy
        })


    print('done')
    objectives_df = pd.DataFrame(objectives)
    print(objectives_df['v3_loss_apy'].max())
    print(objectives_df)
print('done')
