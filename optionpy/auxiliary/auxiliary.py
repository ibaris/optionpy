# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 02.07.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import numpy as np
from scipy.special import ndtr

from optionpy.auxiliary.utility import maybe_jit

EPS = 1e-6

__all__ = ["compute_d1", "compute_d2", "compute_nd1", "compute_nd2", "align_all", "asarrays"]


def align_all(data, constant_values='default', dtype=np.double):
    """
    Align the lengths of arrays.

    Parameters
    ----------
    data : tuple
        A tuple with (mixed) array_like, int, float.
    constant_values : int, float or 'default'
        The value at which the smaller values are expand. If 'default' (default) the last value will be choosed.
    dtype : np.dtype
        Data type of output.
    Returns
    -------
    aligned data : tuple
        Aligned tuple with array_like.
    """
    data = asarrays(data)
    max_len = max_length(data)

    if constant_values == 'default':
        return np.asarray(
            [np.pad(item, (0, max_len - len(item)), 'constant', constant_values=item[-1]) for item in data], dtype=dtype)
    else:
        return np.asarray(
            [np.pad(item, (0, max_len - len(item)), 'constant', constant_values=constant_values) for item in data],
            dtype=dtype)


def max_length(data):
    """
    Find the maximum length of the longest object in a tuple.
    Parameters
    ----------
    data : tuple
        A tuple with (mixed) array_like, int, float.
    Returns
    -------
    len : int
    """
    return max([len(item) for item in data])


def asarrays(data, dtype=None):
    """
    A wrapper of numpys asarrays for multiple data in a tuple.
    Parameters
    ----------
    data : tuple
        A tuple with (mixed) array_like, int, float.
    dtype : np.dtype
        Data type of output.
    Returns
    -------
    arrays : tuple
        A tuple with array_like.
    """
    if dtype is None:
        return [np.atleast_1d(item).flatten() for item in data]
    else:
        return [np.atleast_1d(item).flatten().astype(dtype) for item in data]


def same_len(data):
    """
    Determine if the items in a tuple has the same length.
    Parameters
    ----------
    data : tuple
        A tuple with (mixed) array_like, int, float.
    Returns
    -------
    bool
    """
    return all(len(item) == len(data[0]) for item in data)


@maybe_jit()
def compute_d1(s0, sigma, k, r, t, q=0):
    """
    Calculate the '$d_1$ parameter.'

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
    d1_num = (np.log(s0 / k) + (r - q + .5 * sigma ** 2) * t)
    d1_denominator = (sigma * np.sqrt(t)) + EPS
    return d1_num / d1_denominator


@maybe_jit()
def compute_d2(s0, sigma, k, r, t, q=0):
    """
    Calculate the '$d_1$ parameter.'

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
    return compute_d1(s0, sigma, k, r, t, q) - ((sigma * np.sqrt(t)) + EPS)


@maybe_jit()
def compute_nd1(kind, s0, sigma, k, r, t, q=0):
    """
    Delta is the probability of the option being ITM under the stock measure

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

    nd1 = ndtr(kind * compute_d1(s0, sigma, k, r, t, q))
    return nd1


@maybe_jit()
def compute_nd2(kind, s0, sigma, k, r, t, q=0):
    """
    The probability of the event that the underlying price is over the strike price ($S_t≥K$) in the
    risk-neutral world.

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
    nd2 = ndtr(kind * compute_d2(s0, sigma, k, r, t, q))
    return nd2
