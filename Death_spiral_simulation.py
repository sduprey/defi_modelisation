import pandas as pd
import numpy as np
import numpy as np


class curve_pool():

    def __init__(self, A, coins_list):

        self.N_COINS = len(coins_list)
        self.coins_list = coins_list
        self.A = A
        self.A_PRECISION = A * 100
        self._fee = 0.0004 * self.N_COINS / (4 * (self.N_COINS - 1))
        self._admin_fee = 0.5
        self.rates = [1, 1]
        self.FEE_DENOMINATOR = 1
        self.invariant = 1

        self.lps = 0
        deposit = [0, 0]
        # use the add liqudity function to get correct number of LPs minted based on coins_list
        mint_amount, D2, D0 = self.add_liquidity(deposit)

    def add_liquidity(self, deposits):

        lp0 = self.lps
        A = self.A
        N_COINS = self.N_COINS
        _fee = self._fee
        x = self.coins_list[0]
        y = self.coins_list[1]
        rates = self.rates
        old_balances = [x, y]
        # get_D: D0
        D0 = stable_swap_D(x * rates[0], y * rates[1], A)
        new_balances = old_balances.copy()

        for i in range(N_COINS):
            # balances store amounts of c-tokens
            new_balances[i] = old_balances[i] + deposits[i]

        # get D after adding liquidity: D1
        D1 = stable_swap_D(new_balances[0] * rates[0], new_balances[1] * rates[1], A)
        balances = [0, 0]
        fees = [0, 0]

        if lp0 > 0:
            for i in range(self.N_COINS):
                ideal_balance = D1 * old_balances[i] / D0
                difference = 0
                # print('ideal_balance=', ideal_balance)
                if ideal_balance > new_balances[i]:
                    difference = ideal_balance - new_balances[i]
                else:
                    difference = new_balances[i] - ideal_balance
                # print('to_trade:',difference)
                fees[i] = _fee * difference / self.FEE_DENOMINATOR
                balances[i] = new_balances[i] - (fees[i] * self._admin_fee / self.FEE_DENOMINATOR)
                new_balances[i] -= fees[i]

            D2 = stable_swap_D(new_balances[0] * rates[0], new_balances[1] * rates[1], A)

            mint_amount = lp0 * (D2 - D0) / D0
            # print('fees paid from ideal balance:',fees)
        else:
            D2 = D1
            balances = new_balances
            mint_amount = D1

        self.coins_list = balances.copy()
        # Calculate, how much pool tokens to mint

        self.lps += mint_amount
        self.invariant = D2
        return mint_amount, D2, D0

    def get_virtual_price(self):

        return self.invariant / self.lps

    def compute_deposit_slippage(self, deposits, verbose=False):

        lp0 = self.lps
        A = self.A
        rates = self.rates
        N_COINS = self.N_COINS
        _fee = 0
        x = self.coins_list[0]
        y = self.coins_list[1]

        old_balances = [x, y]
        # get_D: D0
        D0 = stable_swap_D(x * rates[0], y * rates[1], A)
        new_balances = old_balances.copy()

        for i in range(N_COINS):
            # balances store amounts of c-tokens
            new_balances[i] = old_balances[i] + deposits[i]

        # get D after adding liquidity: D1
        D1 = stable_swap_D(new_balances[0] * rates[0], new_balances[1] * rates[1], A)
        balances = [0, 0]
        fees = [0, 0]
        # print(f'D1. {D1}')
        if lp0 > 0:

            for i in range(self.N_COINS):
                ideal_balance = D1 * old_balances[i] / D0
                difference = 0
                # print('ideal_balance=', ideal_balance)
                if ideal_balance > new_balances[i]:
                    difference = ideal_balance - new_balances[i]
                else:
                    difference = new_balances[i] - ideal_balance
                # print('to_trade:',difference)
                fees[i] = _fee * difference / self.FEE_DENOMINATOR
                balances[i] = new_balances[i] - fees[i] * self._admin_fee / self.FEE_DENOMINATOR
                new_balances[i] -= fees[i]

            D2 = stable_swap_D(new_balances[0] * rates[0], new_balances[1] * rates[1], A)
            mint_amount = lp0 * (D2 - D0) / D0
        else:
            D2 = D1
            balances = new_balances
            mint_amount = D1

        virtual_price = D2 / (self.lps + mint_amount)
        slippage = -(1 - mint_amount * virtual_price / (sum(deposits))) * 100

        if verbose:
            print('fees not included:')
            print(f'-- deposit slippage of: {slippage}%')
            print(f'-- value in pool: {mint_amount * virtual_price}')

        return slippage

    def withdrawal_slippage(self, _token_amount, i, verbose):

        # param _token_amount Amount of LP tokens to burn in the withdrawal
        # param i Index value of the coin to withdraw
        dy = 0
        dy_fee = 0

        dy, dy_fee, D1 = self._calc_withdraw_one_coin(_token_amount, i, verbose=verbose)

        virtual_price = self.invariant / (self.lps)
        slippage = -(1 - (dy + dy_fee) / (_token_amount * virtual_price)) * 100
        if verbose:
            print('value in pool:', _token_amount * virtual_price)
            print('-- withdrawal slippage:', slippage)
            print('-- value withdrawal:', dy)
        return dy

    def remove_liquidity_one_coin(self, _token_amount, i, verbose=True):
        # param _token_amount Amount of LP tokens to burn in the withdrawal
        # param i Index value of the coin to withdraw
        dy = 0
        dy_fee = 0

        dy, dy_fee, D1 = self._calc_withdraw_one_coin(_token_amount, i, verbose=verbose)

        self.invariant = D1
        self.coins_list[i] -= (dy)  # + dy_fee * self._admin_fee / self.FEE_DENOMINATOR)
        # register the burn of lps
        self.lps = self.lps - _token_amount
        if verbose:
            print('fees paid:', dy_fee)

        return dy

    def _calc_withdraw_one_coin(self, _token_amount, i, verbose=False):
        # First, need to calculate
        # * Get current D
        # * Solve Eqn against y_i for D - _token_amount

        A = self.A
        rates = self.rates
        old_balances = np.multiply(np.array(self.coins_list.copy()), np.array(rates))
        D0 = stable_swap_D(old_balances[0], old_balances[1], A)

        total_supply = self.lps

        D1 = D0 - _token_amount * D0 / total_supply
        new_y = self.get_y_D_2(A, i, old_balances.copy(), D1)  # self.get_y_D(A, i, old_balances, D1)

        xp_reduced = old_balances

        for j in range(self.N_COINS):
            dx_expected = 0
            xp_j = old_balances[j]
            if j == i:
                dx_expected = xp_j * D1 / D0 - new_y
            else:
                dx_expected = xp_j - xp_j * D1 / D0
            xp_reduced[j] -= self._fee * dx_expected / self.FEE_DENOMINATOR

        dy = xp_reduced[i] - self.get_y_D_2(A, i, xp_reduced, D1)  # self.get_y_D(A, i, xp_reduced, D1)
        # dy = (dy - 1) * PRECISION / rate  # Withdraw less to account for rounding errors
        dy_0 = (old_balances[i] - new_y)  # w/o fees

        if verbose:
            print('expected new amount of ETH in Pool:', new_y)
            # print(f'w fees:{dy}, wo fees: {dy_0}')

        return dy, dy_0 - dy, D1

    def get_y_D(self, A_, i, xp, D):
        """
        Calculate x[i] if one reduces D from being calculated for xp to D

        Done by solving quadratic equation iteratively.
        x_1**2 + x_1 * (sum' - (A*n**n - 1) * D / (A * n**n)) = D ** (n + 1) / (n ** (2 * n) * prod' * A)
        x_1**2 + b*x_1 = c

        x_1 = (x_1**2 + c) / (2*x_1 + b)
        """
        # x in the input is converted to the same price/precision

        assert i >= 0  # dev: i below zero
        assert i < self.N_COINS  # dev: i above N_COINS

        Ann = A_ * self.N_COINS
        c = D
        S_ = 0
        _x = 0
        y_prev = 0

        for _i in range(self.N_COINS):
            if _i != i:
                _x = xp[_i]
            else:
                continue
            S_ += _x
            c = c * D / (_x * self.N_COINS)
        c = c * D * self.A_PRECISION / (Ann * self.N_COINS)
        b = S_ + D * self.A_PRECISION / Ann
        y = D

        for _i in range(2555):
            y_prev = y
            y = (y * y + c) / (2 * y + b - D)
            # Equality with the precision of 1
            if y > y_prev:
                if y - y_prev <= 1:
                    return y
            else:
                if y_prev - y <= 1:
                    return y
        raise

    def get_y_D_2(self, A_, i, xp, D):
        if i == 0:
            i = 1
        elif i == 1:
            i = 0
        x = xp[i]
        a = A_ * x
        b = A_ * x * x + 1 / 4 * x * D - x * A_ * D
        c = -(D ** 3) * 1 / 16

        disc = np.sqrt(b * b - 4 * a * c)

        if disc >= 0:
            x1 = (-b + disc) / (2 * a)
            x2 = (-b - disc) / (2 * a)
            # print("The roots of the equation are:", x1, x2)
        else:
            print("The equation has no solutions")

        return x1


def numerical_D(x, y, A):
    # x and y should be token quantity * rate
    S = 0
    A_PRECISION = 0.5
    coins_list = [x, y]
    N_COINS = len(coins_list)

    for _x in coins_list:
        S += _x
    if S == 0:
        D = 0

    Dprev = 0
    D = S
    Ann = A * N_COINS
    for _i in range(5000):
        D_P = D
        for _x in coins_list:
            D_P = D_P * D / (_x * N_COINS)  # +1 is to prevent /0
        Dprev = D
        D = (Ann * S / A_PRECISION + D_P * N_COINS) * D / ((Ann - A_PRECISION) * D / A_PRECISION + (N_COINS + 1) * D_P)
        # Equality with the precision of 1
        if D > Dprev:
            if D - Dprev <= 0.001:
                break
        else:
            if Dprev - D <= 0.001:
                break

    return D


p = lambda x_, y_, A_: 4 * x_ * y_ * (4 * A_ - 1)
q = lambda x_, y_, A_: -16 * A_ * x_ * y_ * (x_ + y_)


def stable_swap_D(x, y, A):
    D = np.power(-q(x, y, A) / 2 + np.power((q(x, y, A) ** 2) / 4 + (p(x, y, A) ** 3) / 27, 0.5), 1 / 3) - \
        np.power((1 / 27 * p(x, y, A) ** 3) / (
        (np.power((1 / 4 * q(x, y, A) ** 2 + 1 / 27 * p(x, y, A) ** 3), 1 / 2) - 1 / 2 * q(x, y, A))), 1 / 3)
    # np.power(-q(x,y,A)/2-np.power((q(x,y,A)**2)/4+(p(x,y,A)**3)/27,0.5),1/3)

    return D


def get_health_ratio_from_price(price, steth_div_eth_ratio = 3.2/2.2, liquidation_threshold = 0.75):
  health_ratio = price * liquidation_threshold * steth_div_eth_ratio
  return health_ratio

def get_price_from_health_ratio(health_ratio, steth_div_eth_ratio = 3.2/2.2, liquidation_threshold = 0.75):
  price = health_ratio / (liquidation_threshold * steth_div_eth_ratio)
  return price
def get_ratio_from_health_ratio_and_depeg(health_ratio, price, liquidation_threshold = 0.75):
    return health_ratio / (price * liquidation_threshold)




def exchange_ether_coin(curvePool, amount_traded):
    lps,_,_ = curvePool.add_liquidity([amount_traded, 0])
    coin2 = curvePool.remove_liquidity_one_coin(lps,1)
    return coin2
def exchange_stether_coin(curvePool, amount_traded):
    lps,_,_ = curvePool.add_liquidity([0, amount_traded])
    coin2 = curvePool.remove_liquidity_one_coin(lps,0)
    return coin2
# curveStEth = curve_pool(25, [271355.80713136998866, 636549.63617423780386]) # ETH / STEth
# curveStEth.lps = 871852.759674796802182533
#
# coingotsteth = exchange_ether_coin(curveStEth, 100)
# coingoteth = exchange_stether_coin(curveStEth, 100)


lido_analytics = [
    {'stETH': 244097, 'health_factor': 1.42},
    {'stETH': 42380,  'health_factor': 1.21},
    {'stETH': 329471, 'health_factor': 1.14},
    {'stETH': 57197, 'health_factor': 1.07},
    {'stETH': 24316, 'health_factor': 1.03},
    {'stETH': 7245, 'health_factor': 1.000001},
]

init_liquidation_df = pd.DataFrame(lido_analytics)
init_liquidation_df = init_liquidation_df.sort_values(by= ['health_factor'], ascending= True)
init_liquidation_df = init_liquidation_df.reset_index()
clean_depeg_price = 1
liquidation_threshold = 0.75
init_liquidation_df['rationstethneth'] = init_liquidation_df['health_factor']/(clean_depeg_price*liquidation_threshold)
init_liquidation_df['cumStETH'] = init_liquidation_df['stETH'].cumsum()

depeg_prices = [i/100. for i in range(99,-1,-1)]
liquidation_dic = {}
for dp_price in depeg_prices:
#  print(f'price depeg {dp_price}')
  key = f'{dp_price}_health_factor'
  init_liquidation_df[key] = init_liquidation_df['rationstethneth'] * (dp_price * liquidation_threshold)
  key_bis = f'{dp_price}_health_factor_liquidated'
  init_liquidation_df[key_bis] = init_liquidation_df[key] <= 1.
  key_df = init_liquidation_df[init_liquidation_df[key_bis] ].copy()
  liquidated_stether = key_df['cumStETH'].sum()
  liquidation_dic[dp_price] = [liquidated_stether]

liquidation_df = pd.DataFrame.from_records(liquidation_dic).T
init_depeg_prices = [0.80,0.85,0.90,0.95,0.99]
init_depeg_prices = [0.98]

global_results = {}
for init_depeg_price in init_depeg_prices:
    curveStEth = curve_pool(25, [271355.80713136998866, 636549.63617423780386])  # ETH / STEth
    curveStEth.lps = 871852.759674796802182533
    depeg_price = init_depeg_price
    nb_iteration = 0
    ### death spiral
    results = []
    result = {}
    result['depeg'] = depeg_price
    result['coin0'] = curveStEth.coins_list[0]
    result['coin1'] = curveStEth.coins_list[1]
    #result['lps'] = curveStEth.lps
    results.append(result)
    while True:
        result = {}
        print(f'depeg price')
        liquidation_dic = liquidation_df.to_dict()[0]
        value_to_liquididate = liquidation_dic[depeg_price]
        if value_to_liquididate <= 0 :
            print('cascade ended')
            break

        print(f'value to liquidate {value_to_liquididate}')
        coingot = exchange_stether_coin(curveStEth, value_to_liquididate)

        depeg_price = round(coingot / value_to_liquididate,2)

        result['depeg'] = depeg_price
        result['coin0'] = curveStEth.coins_list[0]
        result['coin1'] = curveStEth.coins_list[1]
        #result['lps']   = curveStEth.lps

        liquidation_df = liquidation_df.copy() - value_to_liquididate
        nb_iteration = nb_iteration + 1
        results.append(result)

    global_results[init_depeg_price] = pd.DataFrame(results)


    print('end liquidation round')
    print(f'depeg_price {depeg_price}')
    print(f'nb_iteration {nb_iteration}')

print('done')


