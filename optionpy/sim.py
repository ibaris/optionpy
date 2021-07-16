# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 12.07.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import datetime as dt

import numpy as np
import pandas as pd

from optionpy.auxiliary import *

__all__ = ["simulate_option_returns"]


@maybe_jit()
def compute_terminal_price(s0, sigma, r, t, q=0.0):
    """
    Compute the terminal price of the underlying with the BSM model.

    Parameters
    ----------
    s0 : int, float, array_like
        Equity Price
    r : int, float, array_like
        Risk Free Interest Rate (RFR) per annum.
    sigma : int, float, array_like
        Annual volatility of the underlying.
    q : int, float, array_like
        Dividend rate. Default is 0.
    t : int, optional, array_like
        Annualized time till maturity in days.

    Returns
    -------
    float or array_like
    """
    zt = np.random.normal(0, 1)

    st = s0 * np.exp((r - q - 0.5 * sigma ** 2) * t + np.sqrt(sigma) * zt)

    return st


@maybe_jit()
def generate_terminal_values(s0, sigma, r, t, q=0.0, iteration=1000, n=365):
    """
    Generate terminal over a range of days. The terminal values are the results from a monte carlo simulation.

    Parameters
    ----------
    s0 : int, float, array_like
        Equity Price
    r : int, float, array_like
        Risk Free Interest Rate (RFR) per annum.
    sigma : int, float, array_like
        Annual volatility of the underlying.
    q : int, float, array_like
        Dividend rate. Default is 0.
    t : int, optional, array_like
        Annualized time till maturity in days.
    iteration : int
        Amount of iteration of each day. Default is 10000.
    n : int
        Amount of days to simulate. Default is 365.

    Returns
    -------
    pd.DataFrame
    """

    values = list()
    for i in range(n):
        sim_st = np.zeros(iteration)
        sim_st[0] = s0

        for j in range(1, iteration):
            sim_st[j] = compute_terminal_price(sim_st[j - 1], sigma, r, t, q)

        values[i] = sim_st.mean()

    base = dt.datetime.today()
    date_list = [base + dt.timedelta(days=x) for x in range(n)]
    index = [dt.date(item.year, item.month, item.day) for item in date_list]

    df = pd.DataFrame(values, columns=["TV Option"], index=index)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

    return df


@maybe_jit()
def compute_payoff(kind, st, k):
    """
    Compute the payoff of the simulated options.

    Parameters
    ----------
    kind : int {-1, 1}
        Type 1 or 'c' for call option and -1 or 'p' for put option.
    st : int, float, array_like
        Terminal values.
    k : int, float, array_like
        Strike Price

    Returns
    -------
    float, array_like
    """
    po = np.maximum(st - k, 0) if kind == 1 else np.maximum(0, k - st)

    return po


@maybe_jit()
def discount(values, r, t):
    """
    Discount terminal values.

    Parameters
    ----------
    values : int, float, array_like
        Terminal values to discount.
    r : int, float, array_like
        Risk Free Interest Rate (RFR) per annum.
    t : int, optional, array_like
        Annualized time till maturity in days.

    Returns
    -------
    array_like
    """
    return values * np.exp(-r * t)


def simulate_option_returns(kind, s0, sigma, k, r, t, q=0.00, iteration=10000, n=365):
    """
    Simulate option returns vor `n` days.

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
    iteration : int
        Amount of iteration of each day. Default is 10000.
    n : int
        Amount of days to simulate. Default is 365.

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    Read [here](https://financetrainingcourse.com/education/2012/01/calculating-value-at-risk-for-options-and-futures) how it
    works.
    """
    st = generate_terminal_values(s0, sigma, r, t, q, iteration, n)
    exp_payoff = discount(compute_payoff(kind, st, k), r, t)
    payoff = np.log(exp_payoff)

    payoff = payoff.replace([np.inf, -np.inf], np.nan).dropna()

    payoff["Returns"] = payoff.pct_change()
    payoff = payoff.replace([np.inf, -np.inf], np.nan).dropna()

    return payoff["Returns"].to_frame()
