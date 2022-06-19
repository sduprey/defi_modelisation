import requests
import json
import time
import pandas as pd
import datetime
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import numpy as np

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

def get_pool_state(POOL_ID,URL):
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
    return decimals0, decimals1, current_tick

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


def send_slack_message(message):
    payload = '{"text":"%s"}'% message
    response = requests.post('https://hooks.slack.com/services/********', data=payload)
    print(response.text)


def get_last_price(ssj = 'BTC', ssj_against = 'USDT'):
    r = requests.get(f'https://min-api.cryptocompare.com/data/price?fsym={ssj}&tsyms={ssj_against}')
    return json.loads(r.text)[ssj_against]


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

def request_day_data_paquet(url, me_ts, ssj):
    r = requests.get(url.format(ssj, me_ts))
    dataframe = None
    try:
        dataframe = pd.DataFrame(json.loads(r.text)['Data'])
    except Exception as e:
        print('no data')
    return dataframe

def fetch_crypto_daily_data(ssj=None, daily_crypto_starting_day='2012-01-01', daily_crypto_ending_day=None,ssj_against='USDT'):
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    hour = datetime.datetime.utcnow().hour
    ts = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc).timestamp() + hour * 3600
    ts1 = ts - 2001 * 3600 * 24
    ts2 = ts1 - 2001 * 3600 * 24
    ts3 = ts2 - 2001 * 3600 * 24
    ts4 = ts3 - 2001 * 3600 * 24
    ts5 = ts4 - 2001 * 3600 * 24
    ts6 = ts5 - 2001 * 3600 * 24
    ts7 = ts6 - 2001 * 3600 * 24
    ts8 = ts7 - 2001 * 3600 * 24
    print('Loading data')
    day_url_request = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym='+ssj_against+'&toTs={}&limit=2000'
    dataframe = None
    for me_timestamp in [ts8,ts7,ts6,ts5,ts4,ts3,ts2,ts1,ts]:
        print('waiting')
        time.sleep(1)
        df = request_day_data_paquet(day_url_request, me_timestamp, ssj)
        if df is not None:
            if dataframe is not None:
                dataframe = dataframe.append(df, ignore_index=True)
            else:
                dataframe = df.copy()
    dataframe['time'] = pd.to_datetime(dataframe['time'], unit='s')
    dataframe = dataframe.sort_values(by=['time'])
    dataframe = dataframe.rename(columns={"time": "date", 'volumeto':'volume'}, errors="raise")
    dataframe = dataframe.set_index(dataframe['date'])
    dataframe = dataframe.drop(columns=['date'])
    print('size fetched')
    print(dataframe.shape)
    dataframe = dataframe[dataframe.index >= daily_crypto_starting_day]
    dataframe = dataframe[dataframe.index <= daily_crypto_ending_day]
    print('size filtered after '+str(daily_crypto_starting_day))
    print(dataframe.shape)
    dataframe = dataframe[['open', 'high', 'low', 'close']]

    dataframe.columns = ['Open', 'High', 'Low', 'Close']
    dataframe['Date'] = dataframe.index
    return dataframe


def output_alert_bollinger_band(ssj,ssj_against, daily_crypto_starting_day, stdNbr, proximity_bound, nb_period):
    daily_crypto_starting_date = datetime.datetime.strptime(daily_crypto_starting_day, '%Y-%m-%d')
    starting_date = daily_crypto_starting_date
    running_date = datetime.datetime.now()

    objectives = []
    data_df = fetch_crypto_daily_data(ssj=ssj, ssj_against=ssj_against,
                                      daily_crypto_starting_day=daily_crypto_starting_date,
                                      daily_crypto_ending_day=running_date)

    df = getBBands(data_df.copy(), period=nb_period, stdNbr=stdNbr)
    df = df.dropna()
    stage_nb = 0
    upper_bound = df['Close'].iloc[0] + stdNbr * df['std'].iloc[0]
    lower_bound = df['Close'].iloc[0] - stdNbr * df['std'].iloc[0]
    real_upper_bound = upper_bound - proximity_bound * (upper_bound - df['Close'].iloc[0])
    real_lower_bound = lower_bound + proximity_bound * (df['Close'].iloc[0] - lower_bound)

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

            real_upper_bound = upper_bound - proximity_bound * (upper_bound - row['High'])
            real_lower_bound = lower_bound + proximity_bound * (row['Low'] - lower_bound)

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
    last_upper_bound = stages_df['upper_bound'].iloc[-1]
    last_lower_bound = stages_df['lower_bound'].iloc[-1]

    previous_upper_bound = stages_df['upper_bound'].iloc[-2]
    previous_lower_bound = stages_df['lower_bound'].iloc[-2]

    print(f'current upper bound {last_upper_bound}')
    print(f'current lower bound {last_lower_bound}')

    last_price = get_last_price(ssj, ssj_against)

    if abs(last_upper_bound -previous_upper_bound )> 1e-3:
        print('ALERT ALERT bounds broken')
        send_slack_message(
            f'ALERT ALERT : bounds broken last price: {int(last_price)}, last upperbound: {float(last_upper_bound)},  last lowerbound: {float(last_lower_bound)} , previous upperbound: {float(previous_upper_bound)},  previous lowerbound: {float(previous_lower_bound)} ')

    # send_slack_message(f'everything fine: last price: {int(last_price)}, last lowerbound: {int(last_lower_bound)}, last upperbound: {int(last_upper_bound)}')
    now = datetime.datetime.now()
    print(f'LOG INFO (no rebal) : {ssj}/{ssj_against} last price {now} {last_price}')
    print(f'LOG INFO (no rebal) : last price: {float(last_price)}, last upperbound: {float(last_upper_bound)},  last lowerbound: {float(last_lower_bound)} ')
    print(f'LOG INFO (no rebal) : last price: {float(last_price)}, previous upperbound: {float(previous_upper_bound)},  previous lowerbound: {float(previous_lower_bound)} ')


def output_alert_tick_spread(ssj,ssj_against, daily_crypto_starting_day,base_threshold,rescaling_factor):
    daily_crypto_starting_date = datetime.datetime.strptime(daily_crypto_starting_day, '%Y-%m-%d')
    running_date = datetime.datetime.now()
    data_df = fetch_crypto_daily_data(ssj=ssj, ssj_against=ssj_against,
                                      daily_crypto_starting_day=daily_crypto_starting_date,
                                      daily_crypto_ending_day=running_date)
    data_df['Date'] = data_df.index
    ##### ETHBTC
    if ssj == 'BTC' and ssj_against == 'ETH':
        ### ethereum mainnet (wbtc/eth 0.05%)
        POOL_ID = '0x4585fe77225b41b697c938b018e2ac67ac5a20c0'
        URL = 'https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-subgraph'
        decimals0, decimals1, current_tick = get_pool_state(POOL_ID,URL)
        current_adj_price = from_tick_to_adj_price(current_tick, decimal0=decimals0, decimal1=decimals1)
        current_tick_bis = from_adj_price_to_tick(current_adj_price, decimal0=decimals0, decimal1=decimals1)
        #assert current_tick_bis == current_tick
        get_tick = lambda x: from_adj_price_to_tick(x, decimal0=decimals0, decimal1=decimals1)
        get_price = lambda x: from_tick_to_adj_price(x, decimal0=decimals0, decimal1=decimals1)

    elif ssj == 'USDT' and ssj_against == 'ETH':
        #### ETH USDC
        ### ethereum data set
        POOL_ID = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'
        URL = 'https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-subgraph'
        decimals0, decimals1, current_tick = get_pool_state(POOL_ID,URL)

        current_adj_price = from_tick_to_inv_adj_price(current_tick, decimal0=decimals0, decimal1=decimals1)
        current_tick_bis = from_inv_adj_price_to_tick(current_adj_price, decimal0=decimals0, decimal1=decimals1)
        #assert current_tick_bis == current_tick

        get_tick = lambda x: from_adj_price_to_tick(x, decimal0=decimals0, decimal1=decimals1)
        get_price = lambda x: from_tick_to_adj_price(x, decimal0=decimals0, decimal1=decimals1)

    else :
        raise Exception('pair not handled')

    data_df['CloseTick'] = data_df['Close'].apply(get_tick)
    data_df['OpenTick'] = data_df['Open'].apply(get_tick)
    data_df['HighTick'] = data_df['High'].apply(get_tick)
    data_df['LowTick'] = data_df['Low'].apply(get_tick)

    df = data_df.copy()
    print(f'params_{base_threshold}')
    stage_nb = 0
    upper_bound_tick = df['CloseTick'].iloc[0] + base_threshold
    lower_bound_tick = df['CloseTick'].iloc[0] - rescaling_factor*base_threshold
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
            ILv3, ILv2 = compute_impermanent_loss_v3(p_0=initial_price, p_a=lower_bound_price,
                                                     p_b=upper_bound_price,
                                                     p_out=p_out)

            if up:
                initial_price = row['High']
            else:
                initial_price = row['Low']
            # here compute the previous impermanent loss

            stage_nb = stage_nb + 1
            upper_bound_tick = row['HighTick'] + base_threshold
            lower_bound_tick = row['LowTick'] - rescaling_factor*base_threshold
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

    plotHtml = True
    if plotHtml:
        all_df = pd.merge(stages_df.copy(), df.copy(), left_index=True, right_index=True)
        from cryptotoolbox.realtime import realtime_plotting_utility
        fig = realtime_plotting_utility.plot_multiple_time_series(
            data_df=all_df[['lower_bound_price', 'upper_bound_price', 'Close']].copy(), logy=False, split=False,
            put_on_same_scale=False,
            title=f'v3 bounds over time{ssj}{ssj_against}')
        fig.show()

    last_upper_bound = stages_df['upper_bound_price'].iloc[-1]
    last_lower_bound = stages_df['lower_bound_price'].iloc[-1]

    previous_upper_bound = stages_df['upper_bound_price'].iloc[-2]
    previous_lower_bound = stages_df['lower_bound_price'].iloc[-2]

    print(f'current upper bound {last_upper_bound}')
    print(f'current lower bound {last_lower_bound}')

    last_price = get_last_price(ssj, ssj_against)

    if abs(last_upper_bound - previous_upper_bound) > 1e-3:
        print('ALERT ALERT bounds broken')
        send_slack_message(
            f'ALERT ALERT : bounds broken last price: {int(last_price)}, last upperbound: {float(last_upper_bound)},  last lowerbound: {float(last_lower_bound)} , previous upperbound: {float(previous_upper_bound)},  previous lowerbound: {float(previous_lower_bound)} ')

    # send_slack_message(f'everything fine: last price: {int(last_price)}, last lowerbound: {int(last_lower_bound)}, last upperbound: {int(last_upper_bound)}')
    now = datetime.datetime.now()
    inv_last_price = 1./last_price
    inv_last_upper_bound = 1./last_upper_bound
    inv_last_lower_bound = 1./last_lower_bound
    print(f'LOG INFO (no rebal) : {ssj}/{ssj_against} last price {now} {last_price}')
    print(
        f'LOG INFO (no rebal) : last price: {float(last_price)}, last upperbound: {float(last_upper_bound)},  last lowerbound: {float(last_lower_bound)} ')
    print(
        f'LOG INFO (no rebal) : last price: {float(last_price)}, previous upperbound: {float(previous_upper_bound)},  previous lowerbound: {float(previous_lower_bound)} ')
    inv_last_price = 1./last_price
    inv_last_upper_bound = 1./last_upper_bound
    inv_last_lower_bound = 1./last_lower_bound
    print(f'LOG INFO (no rebal) : {ssj}/{ssj_against} last inverted price {now} {inv_last_price}')
    print(
        f'LOG INFO (no rebal) : last inverted price: {float(inv_last_price)}, last inverted upperbound: {float(inv_last_lower_bound)},  last inverted lowerbound: {float(inv_last_upper_bound)} ')
    print(
        f'LOG INFO (no rebal) : last inverted price: {float(inv_last_price)}, previous inverted upperbound: {float(inv_last_lower_bound)},  previous inverted lowerbound: {float(inv_last_upper_bound)} ')


##### second strat
ssj = 'USDT'
ssj_against = 'ETH'
#daily_crypto_starting_day = '2022-01-01'
daily_crypto_starting_day = '2022-05-10'
stdNbr = 2.5
proximity_bound = 0.
nb_period=15
#output_alert_bollinger_band(ssj,ssj_against, daily_crypto_starting_day, stdNbr, proximity_bound, nb_period)
base_threshold = 5000
rescaling_factor = 0.5
output_alert_tick_spread(ssj,ssj_against,daily_crypto_starting_day,base_threshold,rescaling_factor)

###### first strat
ssj = 'BTC'
ssj_against = 'ETH'
#daily_crypto_starting_day = '2021-05-16'
daily_crypto_starting_day = '2022-05-10'
stdNbr = 3.
proximity_bound = 0.
nb_period=15
#output_alert_bollinger_band(ssj,ssj_against, daily_crypto_starting_day, stdNbr, proximity_bound, nb_period)
base_threshold = 2600
rescaling_factor = 0.5
output_alert_tick_spread(ssj,ssj_against, daily_crypto_starting_day,base_threshold,rescaling_factor)

