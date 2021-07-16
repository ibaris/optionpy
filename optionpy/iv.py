# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 11.07.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import numpy as np

from optionpy.auxiliary import *
from optionpy.price import *

__all__ = ["compute_iv"]


@maybe_jit()
def compute_iv(kind, s0, sigma, k, r, t, q, price):
    """
    Use a binary search to find the implied volatility.

    Parameters
    ----------
    kind : int {-1, 1}
        Type 1 or 'c' for call option and -1 or 'p' for put option.
    s0 : int, float, array_like
        Equity Price
    k : int, float, array_like
        Strike Price
    r : int, float, array_like
        Risk Free Interest Rate (RFR) per annum.
    sigma : int, float, array_like
        Annual volatility of the underlying.
    q : int, float, array_like
        Dividend rate. Default is 0.
    t : int, optional, array_like
        Annualized time till maturity in days.
    price : int, float, array_like
        Actual option price.

    Returns
    -------
    float or array_like
    """
    upper_range = 5
    lower_range = 0
    MOE = 0.0001  # Minimum margin of error
    max_iters = 100
    iter = 0

    s0, sigma, k, r, t, q, price, upper_range, lower_range = align_all((s0, sigma, k, r, t, q, price, upper_range, lower_range))

    iv_array = np.zeros_like(s0)

    while iter < max_iters:  # Don't iterate too much
        fair_value = compute_price_bsm(kind, s0, sigma, k, r, t, q)  # BS Model Pricing

        cost = np.abs((fair_value - price) / price) < MOE

        if np.all(cost == True):
            return sigma

        delta_index = np.where(cost == True)[0]

        iv_array[delta_index] = sigma[delta_index]

        bigger = fair_value > price
        smaller = fair_value < price

        b_index = np.where(bigger == True)[0]
        s_index = np.where(smaller == True)[0]

        tmp = sigma[b_index]
        sigma[b_index] = (sigma[b_index] + lower_range[b_index]) / 2
        upper_range[b_index] = tmp

        tmp = sigma[s_index]
        sigma[s_index] = (sigma[s_index] + upper_range[s_index]) / 2
        lower_range[s_index] = tmp

        iter += 1

    return iv_array
