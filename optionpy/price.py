# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 11.07.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import numpy as np
from scipy.special import ndtr

from optionpy.auxiliary import *
from optionpy.auxiliary.utility import maybe_jit

__all__ = ["compute_price_bsm", "compute_price_ms"]


@maybe_jit()
def compute_price_bsm(kind, s0, sigma, k, r, t, q=0.0):
    """
    Calculate the option price with the Black-Scholes-Merton Model approache.

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
    q : int, float, optional, array_like
        Dividend rate. Default is 0.
    t : int, optional, array_like
        Annualized time till maturity in days.

    Returns
    -------
    float, array_like
    """
    d1 = compute_d1(s0, sigma, k, r, t, q)
    d2 = compute_d2(s0, sigma, k, r, t, q)

    nd1 = ndtr(kind * d1)
    nd2 = ndtr(kind * d2)

    price = (kind * s0 * np.exp(-q * t) * nd1 -
             kind * k * np.exp(-r * t) * nd2)

    return price


@maybe_jit()
def compute_price_ms(kind, s0, sigma, k, r, t, q=0.0, iteration=100000):
    """
    Calculate the option price with the Monte Carlo approache.

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
    q : int, float, optional, array_like
        Dividend rate. Default is 0.
    t : int, optional, array_like
        Annualized time till maturity in days.
    iteration : int
        Amount of iteration. Default is 100000.

    Returns
    -------
    float, array_like
    """
    zt = np.random.normal(0, 1, iteration)
    st = s0 * np.exp((r - q - .5 * sigma ** 2) * t + sigma * t ** .5 * zt)
    st = np.maximum(kind * (st - k), 0)
    return np.average(st) * np.exp(-r * t)
