import numpy as np
# Price $174.78203832
# Market Liquidity 11,138.86317938 AAVE
#
# # of Suppliers 521
#
# # of Borrowers 144
#
# Reserves 58.16009152 AAVE
#
# Reserve Factor 25%
#
# Collateral Factor 55%
#
# Total Supply $2,848,228.19   =====> 2848228 = 2.8 millions
#
# Total Borrow $911,520.75 ======> 911520 = 0.91 millions
#
# Exchange Rate 1 AAVE = 49.382379 vAAVE
#
#
# utilization ratio : 0,325
# borrow_rate = 0.065 == 6.5%
# supply_rate = 0.0158
#
# ####### https://app.venus.io/vote/proposal/39
# 4. Support AAVE
# Jump Rate model: Base0bps_Slope2000bps_Jump30000bps_Kink50
# XVS Distribution is 25 XVS /day
# Risk factors for Risk evaluation as below:
# Reserve Factor: 25%
# Collateral Factor: 50%
#
# https://blog.venus.io/venus-interest-rate-model-v1-347f551289c4
#
# kink : 50 %
# base : 0 bps
# slope : 2000 bps = 20 % = 0.2
# jump : 30000 bps =
#
# as long as u < 0.5
# borrow_rate = 0 + 0.2*U
# supply_rate = borrow_rate*u*(1-reserve_factor)

def compute_total_apy(amount = np.nan, collateral_factor =0.5, nb_loops = 0,  supply=np.nan, borrow=np.nan, base_rate = 0, slope1=0.2, slope2 = 0, kink = 0.5, reserve_factor = 0.25):
    total_supplied_amount = amount *(1-pow(collateral_factor,nb_loops+1))/(1-collateral_factor)
    if nb_loops == 0:
        total_borrowed_amount = 0.
    else:
        total_borrowed_amount = amount * collateral_factor * (1-pow(collateral_factor,nb_loops+1))/(1-collateral_factor)
    utilization_rate = (borrow+total_borrowed_amount)/(supply+total_supplied_amount)
    if utilization_rate >= 0.5:
        raise Exception('to implement')
    borrow_rate = base_rate + slope1 * utilization_rate
    supply_rate = utilization_rate * borrow_rate * (1-reserve_factor)
    print(f'supply rate {supply_rate}')
    print(f'borrow rate {borrow_rate}')
    earned = supply_rate*total_supplied_amount - borrow_rate*total_borrowed_amount
    return earned


earned=compute_total_apy(amount = 2700000, supply= 2848228, borrow= 911520,collateral_factor=0.5, nb_loops = 1)

print('done')

