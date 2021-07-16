# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 11.07.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import numpy as np

from optionpy.auxiliary.utility import maybe_jit

__all__ = ["expected_value", "variance", "std", "expected_movement"]


@maybe_jit()
def expected_value(s0, t, mu):
    """
    Return the expected value of the underlying.

    Parameters
    ----------
    s0 : int, float, array_like
        Equity Price
    t : int, optional, array_like
        Annualized time till maturity in days.
    mu : float, int
        Expected yearly return.

    Returns
    -------
    float or array_like
    """
    return s0 * np.exp(mu * t)


@maybe_jit()
def variance(s0, sigma, t, mu):
    """
    Return the variance of the underlying.

    Parameters
    ----------
    s0 : int, float, array_like
        Equity Price
    sigma : int, float, array_like
        Annual volatility of the underlying.
    t : int, optional, array_like
        Annualized time till maturity in days.
    mu : float, int
        Expected yearly return.

    Returns
    -------
    float or array_like
    """
    var = s0 ** 2 * np.exp(2 * mu * t) * (np.exp(sigma ** 2 * t) - 1)
    return var


def std(s0, sigma, t, mu):
    """
    Return the standard deviation of the underlying.

    Parameters
    ----------
    s0 : int, float, array_like
        Equity Price
    sigma : int, float, array_like
        Annual volatility of the underlying.
    t : int, optional, array_like
        Annualized time till maturity in days.
    mu : float, int
        Expected yearly return.

    Returns
    -------
    float or array_like
    """
    return np.sqrt(variance(s0, sigma, t, mu))


def expected_movement(s0, sigma, t, mu):
    """
    Return the expected movement range of the underlying.

    Parameters
    ----------
    s0 : int, float, array_like
        Equity Price
    sigma : int, float, array_like
        Annual volatility of the underlying.
    t : int, optional, array_like
        Annualized time till maturity in days.
    mu : float, int
        Expected yearly return.

    Returns
    -------
    float or array_like
    """
    tmp_std = np.sqrt(variance(s0, sigma, t, mu))
    ev = expected_value(s0, t, mu)
    return [ev - tmp_std, ev, ev + tmp_std]
