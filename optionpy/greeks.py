# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 02.07.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from optionpy.auxiliary import *
from optionpy.auxiliary.utility import maybe_jit

__all__ = ["compute_delta", "compute_vega", "compute_rho", "compute_theta", "compute_gamma",
           "compute_epsilon"]


@maybe_jit()
def compute_delta(kind, s0, sigma, k, r, t, q=0.0):
    """Calculate the Delta of the option.
    Delta is the rate of change of the option price with respect to the price of the underlying. Deltas can be
    positive or negative. Deltas can also be thought of as the probability that the option will expire ITM. Having
    a delta neutral portfolio can be a great way to mitigate directional risk from market moves.

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
    float or array_like
    """
    nd1 = compute_nd1(kind, s0, sigma, k, r, t, q)

    delta = kind * np.exp(-q * t) * nd1
    return delta


@maybe_jit()
def compute_vega(s0, sigma, k, r, t, q=0.0):
    """Calculate the Vega of the option.

    Vega is the greek metric that allows us to see our exposure to changes in implied volatility. Vega values
    represent the change in an option’s price given a 1% move in implied volatility, all else equal.

    Parameters
    ----------
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
    float or array_like
    """
    d1 = compute_d1(s0, sigma, k, r, t, q)
    v = s0 * np.exp(-q * t) * norm._pdf(d1) * np.sqrt(t) * 0.01

    return v


@maybe_jit()
def compute_theta(kind, s0, sigma, k, r, t, q=0.0):
    """Calculate the Theta of the option.

    Theta measures the rate of change in an options price relative to time. This is also referred to as time decay.
    Theta values are negative in long option positions and positive in short option positions. Initially,
    out of the money options have a faster rate of theta decay than at the money options, but as expiration
    nears, the rate of theta decay for OTM options slows and the ATM options begin to experience theta decay at
    a faster rate. This is a function of theta being a much smaller component of an OTM option's price, the closer
    the option is to expiring.

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
    float or array_like
    """

    d1 = compute_d1(s0, sigma, k, r, t, q)
    d2 = compute_d2(s0, sigma, k, r, t, q)

    nd1 = ndtr(kind * d1)
    nd2 = ndtr(kind * d2)

    phi_d1 = norm._pdf(d1)

    term_1 = -np.exp(-q * t) * (s0 * phi_d1 * sigma) / (2 * np.sqrt(t))
    term_2 = -kind * r * k * np.exp(-r * t) * nd2
    term_3 = -kind * q * s0 * np.exp(-q * t) * nd1

    theta = (term_1 + term_2 + term_3) / 365

    return theta


@maybe_jit()
def compute_rho(kind, s0, sigma, k, r, t, q=0.0):
    """Compute the Rho of the option.

    Rho is the rate at which the price of a derivative changes relative to a change in the risk-free rate of
    interest. Rho measures the sensitivity of an option or options portfolio to a change in interest rate.
    Rho may also refer to the aggregated risk exposure to interest rate changes that exist for a book of
    several options positions.

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
    float or array_like
    """

    d2 = compute_d2(s0, sigma, k, r, t, q)
    nd2 = ndtr(kind * d2)

    rho = kind * k * t * np.exp(-r * t) * nd2 * 0.01

    return rho


@maybe_jit()
def compute_epsilon(kind, s0, sigma, k, r, t, q=0.0):
    """Compute the Epsilon of the option.

    Epsilon is the percentage change in option value per percentage change in the underlying dividend yield, a
    measure of the dividend risk. The dividend yield impact is in practice determined using a 10% increase in
    those yields. Obviously, this sensitivity can only be applied to derivative instruments of equity products.

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
    float or array_like
    """

    d1 = compute_d1(s0, sigma, k, r, t, q)
    nd1 = ndtr(kind * d1)

    epsilon = -kind * s0 * t * np.exp(-q * t) * nd1

    return epsilon


@maybe_jit()
def compute_gamma(s0, sigma, k, r, t, q=0.0):
    """Calculate the Gamma of the option.

    Gamma is the rate of change in an option's delta per 1-point move in the underlying asset's price. Gamma is an
    important measure of the convexity of a derivative's value, in relation to the underlying.

    Parameters
    ----------
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
    float or array_like
    """
    d1 = compute_d1(s0, sigma, k, r, t, q)
    phi_d1 = norm._pdf(d1)

    gamma = np.exp(-q * t) * phi_d1 / (s0 * sigma * np.sqrt(t))

    return gamma
