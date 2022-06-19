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


frequence = 'daily'
number_per_year = 252

freqly_pkl_file_name_suffix = '_' + frequence + '_returns.pkl'
localrootdirectory = '/Users/stefanduprey/Documents/My_Data/My_ILLossBacktest'

#for me_pair in [('ETH', 'USDT'),('DOT','BNB'),('ETH','USDT'),('LINK','ETH')]:
for me_pair in [('BTC', 'ETH')]:
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

    # bb_periods = range(5,35,5)
    # stdNbrs = [i / 10 for i in range(1, 10, 2)] + [i for i in range(1, 10)]
    # bound_proximities = [0,0.05,0.1]
    # plotHtml = False

    bb_periods = [15]
    stdNbrs = [3.]
    bound_proximities = [0]
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

    for bb_period in bb_periods:
        for stdNbr in stdNbrs:
            for bound_proximity in bound_proximities:
                df = data_df.copy()
                print(f'params_{bb_period}_{stdNbr}')
                df = getBBands(df.copy(), period=bb_period, stdNbr=stdNbr)
                df = df.dropna()
                stage_nb = 0
                upper_bound = df['Close'].iloc[0] + stdNbr * df['std'].iloc[0]
                lower_bound = df['Close'].iloc[0] - stdNbr * df['std'].iloc[0]
                real_upper_bound = upper_bound - bound_proximity*(upper_bound-df['Close'].iloc[0] )
                real_lower_bound = lower_bound + bound_proximity*(df['Close'].iloc[0]-lower_bound )

                initial_price = df['Close'].iloc[0]
                stages = []
                counter = 0
                for i, row in df.iterrows():
                    if counter == 0:
                        counter = counter + 1
                        continue
                    ILv3, ILv2 = np.nan, np.nan
                    if row['High'] > real_upper_bound or row['Low'] < real_lower_bound:
                        up = row['High'] > real_upper_bound
                        p_out = np.nan
                        if up:
                            p_out = real_upper_bound
                        else:
                            p_out = real_lower_bound
                        ILv3, ILv2 = compute_impermanent_loss_v3(p_0=initial_price, p_a=lower_bound, p_b=upper_bound,
                                                                 p_out=p_out)

                        if up:
                            initial_price = row['High']
                        else:
                            initial_price = row['Low']
                        # here compute the previous impermanent loss

                        stage_nb = stage_nb + 1
                        upper_bound = row['High'] + stdNbr * row['std']
                        lower_bound = row['Low'] - stdNbr * row['std']

                        real_upper_bound = upper_bound - bound_proximity * (upper_bound - row['High'])
                        real_lower_bound = lower_bound + bound_proximity * (row['Low'] - lower_bound)

                    stages.append({
                        'date': row['Date'],
                        'incurred_v3': ILv3,
                        'incurred_v2': ILv2,
                        'initial_price': initial_price,
                        'real_upper_bound': real_upper_bound,
                        'real_lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'lower_bound': lower_bound,
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
                        data_df=all_df[['lower_bound', 'upper_bound', 'Close']].copy(), logy=False, split=False,
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
                all_df['spread'] = all_df['upper'] - all_df['lower']
                mean_spread = all_df['spread'].mean()
                objectives.append({
                    'bb_period': bb_period,
                    'stdNbr': stdNbr,
                    'bound_proximity': bound_proximity,
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
