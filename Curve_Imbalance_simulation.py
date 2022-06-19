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
A = 50
nb_eth = 286849.02838659
nb_steth = 582013.77832417
pool_proportion = nb_eth/nb_steth
print(f'pool proportion {pool_proportion}')


prop_proportion = 8148/(16532+8148)
prop_amount = 8148. + 16532.
print(f'prop proportion {prop_proportion}')

nb_provided_eth = 33
nb_provided_steth = 67
provided_proportion = 33/(67+33)
print(f'provided proportion {provided_proportion}')

###### premier scenario

curveStEth = curve_pool(A, [nb_eth, nb_steth])  # ETH / STEth
curveStEth.lps =   834615.315990784522406036
prov_lps, _, _ = curveStEth.add_liquidity([33, 67])
eth_got_coins   = curveStEth.remove_liquidity_one_coin(prov_lps, 0)
print(f'eth got coins {eth_got_coins}')

###### second scenario
curveStEth = curve_pool(A, [nb_eth, nb_steth])  # ETH / STEth
curveStEth.lps =   834615.315990784522406036
prov_lps, _, _ = curveStEth.add_liquidity([33, 67])
steth_got_coins   = curveStEth.remove_liquidity_one_coin(prov_lps, 1)
print(f'steth got coins {steth_got_coins}')

#steth_got_coins = curveStEth.remove_liquidity_one_coin(prov_lps, 1)
#print(f'steth got coins {steth_got_coins}')

print(f'lps got {prov_lps}')



curveStEth = curve_pool(A, [nb_eth, nb_steth])  # ETH / STEth
curveStEth.lps =   834615.315990784522406036
prov_lps, _, _ = curveStEth.add_liquidity([33, 67])
arb_quantity = nb_steth-nb_eth
arb_lps, _, _ = curveStEth.add_liquidity([arb_quantity, 0])

print(f'new curve lps {curveStEth.lps}')
print(f'new curve nb eths {curveStEth.coins_list[0]}')
print(f'new curve nb steths {curveStEth.coins_list[1]}')

eth_got_coins   = curveStEth.remove_liquidity_one_coin(prov_lps, 0)
print(f'eth got coins {eth_got_coins}')

print(f'basis point IL {(1 - eth_got_coins/100)*100*100}')
print(f'incurred loss {25190 * (1 - eth_got_coins/100)}')


##############################
###############################
###### REEQUILIBRATING SCENARIO
##############################
##############################

curveStEth = curve_pool(A, [nb_eth, nb_steth])  # ETH / STEth
curve_proportion = nb_eth / (nb_eth + nb_steth)
print(f'curve pool proportion {curve_proportion} {1-curve_proportion}')
curveStEth.lps =   834615.315990784522406036

prov_lps, _, _ = curveStEth.add_liquidity([33, 67])
arb_quantity = nb_steth-nb_eth
arb_lps, _, _ = curveStEth.add_liquidity([arb_quantity, 0])

print(f'new curve lps {curveStEth.lps}')
new_curve_proportion = curveStEth.coins_list[0] / (curveStEth.coins_list[0] + curveStEth.coins_list[1])
print(f'new curve proportion {new_curve_proportion} {1-new_curve_proportion}')
print(f'new curve nb eths {curveStEth.coins_list[0]}')
print(f'new curve nb steths {curveStEth.coins_list[1]}')

steth_got_coins = curveStEth.remove_liquidity_one_coin(prov_lps, 1)
print(f'steth got coins {steth_got_coins}')
diff_perc = (steth_got_coins-100)
diff_perc_in_bps = diff_perc * 100
print(f'diff perc in bps {diff_perc_in_bps}')

print('done')

##############################
###############################
###### DESEQUILIBRATING SCENARIO
##############################
##############################
curveStEth = curve_pool(A, [nb_eth, nb_steth])  # ETH / STEth
curve_proportion = nb_eth / (nb_eth + nb_steth)
print(f'curve pool proportion {curve_proportion} {1-curve_proportion}')
curveStEth.lps =   834615.315990784522406036

prov_lps, _, _ = curveStEth.add_liquidity([33, 67])

des_quantity = (nb_steth+nb_eth)*15./100.
des_lps, _, _ = curveStEth.add_liquidity([0, des_quantity])
print(f'new curve lps {curveStEth.lps}')
new_curve_proportion = curveStEth.coins_list[0] / (curveStEth.coins_list[0] + curveStEth.coins_list[1])
print(f'new curve proportion {new_curve_proportion} {1-new_curve_proportion}')
print(f'new curve nb eths {curveStEth.coins_list[0]}')
print(f'new curve nb steths {curveStEth.coins_list[1]}')

steth_got_coins = curveStEth.remove_liquidity_one_coin(prov_lps, 1)
print(f'steth got coins {steth_got_coins}')
diff_perc = (steth_got_coins-100)
diff_perc_in_bps = diff_perc * 100
print(f'diff perc in bps {diff_perc_in_bps}')


##################@
### state one
nb_eth = 336921.
nb_steth = 673613.
tot_one = nb_eth + nb_steth

### state one
nb_eth = 273376.
nb_steth = 738603.
prop = nb_eth/(nb_eth+nb_steth)
print(f'proportion {prop}')
tot_two = nb_eth + nb_steth

diff = tot_two - tot_one
diff_perc = (diff)/tot_one
diff_in_bp = diff_perc*10000
print(f'diff in bps two {diff_in_bp}')

print('done')

gain_one =prop_amount * 14/10000
gain_two =prop_amount * 40/10000

loss_one =prop_amount * 8/10000

print('done')