# -*- coding: utf-8 -*-
"""
Option
======
*Created on 11.07.2021 by bari_is*

*For COPYING and LICENSE details, please refer to the LICENSE file*

"""

import numpy as np
import pandas as pd

from optionpy.auxiliary import *
from optionpy.greeks import *
from optionpy.iv import *
from optionpy.price import *
from optionpy.sim import *

__all__ = ["Option"]

# ----------------------------------------------------------------------------------------------
# Environmental Settings
# ----------------------------------------------------------------------------------------------
# Pandas DataFrame display settings.
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 15)


# ----------------------------------------------------------------------------------------------
# Options Class
# ----------------------------------------------------------------------------------------------
class Option(object):
    """
    A class to define one or multiple options.

    Attributes
    ----------
        kind : int {-1, 1}
            1 for call option and -1 for put option.
        s0 : int, float, array_like
            Initial equity Price
        k : int, float, array_like
            Strike Price
        sigma : int, float, array_like
            Annual volatility of the underlying equity.
        q : int, float, array_like
            Dividend rate.
        t : int, array_like
            Time till maturity in days.
        iv : int, float, array_like
            Define the implied volatility.
        premium : int, float, array_like
            The actual premium of the option.
            volatility.
        start, end : str
            Start and end date of format "YYY-MM-DD".
        data : pd.DataFrame
            A dataframe with all available information:
                * Kind : 1 for call option and -1 for put option.
                * Start, End : Start and end date of format "YYY-MM-DD".
                * Maturity : Time till maturity in days.
                * S0 : Initial equity Price
                * Strike : Strike Price
                * RFR : Risk Free Rate
                * Volatility : Annual volatility of the underlying equity.
                * Dividend : Dividend rate.
                * Fair Value : Fair value calculated with the BSM model and the volatility of the underlying (`sigma`)
                * Premium : Premium calculated with the BSM model and the implied volatility of the underlying (`iv`)
                * ITM : A bool that indicates if the option in In The Money.
                * IV : Implied volatility
                * Delta, Vega, Theta, Rho, Epsilon, Gamma : The greeks.
                * Nd1, Nd2 : The probability of the event that the underlying price is over the strike price ($S_t≥K$) in the
                  risk-neutral world.
    """

    def __init__(self, kind, s0, k, r, sigma, q=0.0, t=None, iv=None, premium=0.0, start="", end="",
                 **kwargs):
        """
        Initialize one or multiple options.

        Parameters
        ----------
        kind : int {-1, 1}
            Typ 1 for call option and -1 for put option.
        s0 : int, float, array_like
            Initial equity Price
        k : int, float, array_like
            Strike Price
        sigma : int, float, array_like
            Annual volatility of the underlying equity.
        q : int, float, array_like
            Dividend rate. Default is 0.
        t : int, array_like
            Time till maturity in days. Default is None.
        iv : int, float, array_like
            Define the implied volatility. Default is the same as the parameter `sigma`.
        premium : int, float, array_like
            The actual premium of the option. Default is 0.0. This parameter is mandatory if you want to estimate the implied
            volatility.
        start, end : str
            Start and end date of format "YYY-MM-DD". Default is None.

        Notes
        -----
        It is mandatory to define the parameter `end` or `t`.
        """
        data = kwargs.pop("data", None)

        if data is None:
            if end == "" and t is None:
                raise AssertionError

            iv = sigma if iv in [None, 0] else iv
            t = 0 if t is None else t

            self.kind, self.s0, self.k, self.r, self.sigma, self.q, t, self.__iv, self.__premium = align_all((kind, s0, k, r,
                                                                                                              sigma, q, t, iv,
                                                                                                              premium))
            _, start, end = align_all((t, start, end), dtype=str)

            start[start == ""] = np.datetime64('today', 'D')
            self.start = start.astype("datetime64[D]")

            if self.start[end == ""].shape[0] != 0:
                end[end == ""] = self.start[end == ""] + t.astype('timedelta64[D]')
            self.end = end.astype("datetime64[D]")

            self.t = (self.end - self.start).astype(int) / 365

            puts = np.where(self.kind == -1)[0]
            calls = np.where(self.kind == 1)[0]

            itm = np.zeros_like(self.s0, dtype=bool)

            itm[puts] = self.s0[puts] <= self.k[puts]
            itm[calls] = self.s0[calls] >= self.k[calls]

            self.__data = pd.DataFrame(
                columns=["Kind", "Start", "End", "Maturity", "S0", "Strike", "RFR", "Volatility", "Dividend",
                         "Fair Value", "Premium", "ITM", "IV", "Delta", "Vega", "Theta", "Rho",
                         "Epsilon", "Gamma", "Nd1", "Nd2"])

            self.__premium = premium
            self.__iv = iv

            self.__data["Kind"] = self.kind
            self.__data["Start"] = self.start
            self.__data["End"] = self.end
            self.__data["Maturity"] = self.t
            self.__data["S0"] = self.s0
            self.__data["Strike"] = self.k
            self.__data["RFR"] = self.r
            self.__data["Volatility"] = self.sigma
            self.__data["IV"] = iv
            self.__data["Dividend"] = self.q
            self.__data["Premium"] = premium

            self.__data["ITM"] = itm

            self.compute_greeks()
            self.compute_fair_value()

            if not np.all(self.premium == 0.0) and np.all(self.iv == self.sigma):
                self.compute_iv()

            _, _ = self.nd1, self.nd2
        else:
            self.__data = data

            self.kind = data["Kind"]
            self.start = data["Start"].values
            self.end = data["End"].values
            self.t = data["Maturity"].values
            self.__premium = data["Premium"].values
            self.__iv = data["IV"].values

    # ----------------------------------------------------------------------------------------------
    # Magic Methods
    # ----------------------------------------------------------------------------------------------
    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self.data)):
            yield self.data.iloc[0]

    def __getitem__(self, item):
        return self.data[item]

    # ----------------------------------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------------------------------
    @property
    def data(self):
        """
        A dataframe with all available information:
            * Kind : 1 for call option and -1 for put option.
            * Start, End : Start and end date of format "YYY-MM-DD".
            * Maturity : Time till maturity in days.
            * S0 : Initial equity Price
            * Strike : Strike Price
            * RFR : Risk Free Rate
            * Volatility : Annual volatility of the underlying equity.
            * Dividend : Dividend rate.
            * Fair Value : Fair value calculated with the BSM model and the volatility of the underlying (`sigma`)
            * Premium :  : Premium calculated with the BSM model and the implied volatility of the underlying (`iv`)
            * IV : Implied volatility
            * Delta, Vega, Theta, Rho, Epsilon, Gamma : The greeks.
            * Nd1, Nd2 : The probability of the event that the underlying price is over the strike price ($S_t≥K$) in the
              risk-neutral world.

        Returns
        -------
        pd.DataFrame
        """
        return self.__data

    @property
    def iloc(self):
        """
        Purely integer-location based indexing for selection by position. This is a wrapper function of `pd.DataFrame.iloc`

        Returns
        -------
        pd.DataFrame
        """
        return self.data.iloc

    @property
    def loc(self):
        """
        Purely label-location based indexer for selection by label. This is a wrapper function of `pd.DataFrame.loc`

        Returns
        -------
        pd.DataFrame
        """
        return self.data.loc

    @property
    def nd1(self):
        """
        Delta is the probability of the option being ITM under the stock measure

        Returns
        -------
        float, array_like
        """
        nd1 = compute_nd1(self.kind, self.s0, self.iv, self.k, self.r, self.t, self.q)
        self.data["Nd1"] = nd1
        return nd1

    @property
    def nd2(self):
        """
        The probability of the event that the underlying price is over the strike price ($S_t≥K$) in the
        risk-neutral world.

        Returns
        -------
        float, array_like
        """
        nd2 = compute_nd2(self.kind, self.s0, self.iv, self.k, self.r, self.t, self.q)
        self.data["Nd2"] = nd2
        return nd2

    @property
    def iv(self):
        """
        Implied volatility.

        Returns
        -------
        float, array_like
        """
        return self.data["IV"].values

    @iv.setter
    def iv(self, item):
        """
        Implied volatility.

        Parameters
        ----------
        item : int, float, array_like
            Define the implied volatility. Default is the same as the parameter `sigma`.

        Returns
        -------
        None

        Notes
        -----
        This function update the greeks and the premium calculation.
        """
        _, item = align_all((self.__iv, item))
        self.data["IV"] = item
        self.__update()

    @property
    def premium(self):
        """
        Premium

        Returns
        -------
        float, array_like
        """
        return self.data["Premium"].values

    @premium.setter
    def premium(self, value):
        """
        Premium.

        Parameters
        ----------
        value : int, float, array_like
            The actual premium of the option. Default is 0.0. This parameter is mandatory if you want to estimate the implied
            volatility.

        Returns
        -------
        None

        Notes
        -----
        This function update the greeks and the premium calculation.
        """
        _, value = align_all((self.iv, value))
        self.data["Premium"] = value
        self.__update()

    # ----------------------------------------------------------------------------------------------
    # Public Methods
    # ----------------------------------------------------------------------------------------------
    def select_where(self, column, start, stop=None):
        """
        A query function that make should make the `DataFrame` queries easier.

        Parameters
        ----------
        column : str
            The name of the column where the query is pointed to. Possible columns are:
               * 'Kind', 'Start', 'End', 'Maturity', 'S0', 'Strike', 'RFR', 'Volatility', 'Dividend', 'Fair Value', 'Premium',
               'IV', 'Delta', 'Vega', 'Theta', 'Rho', 'Epsilon', 'Gamma', 'Nd1', 'Nd2'.
        start : str
            The start value and its operation (see Notes).
        stop : str or None
            The end value and its operation (see Notes).

        Returns
        -------
        Option

        Notes
        -----
        For example, of you want to select all the Delta that is between 0.30 and 0.50 you use this function like:
            >>> self.select_where("Delta", ">=0.30", "<=0.50")

        """
        if column in self.data.columns:
            if stop is None:
                run = "self.data[self.data[column] {0}]".format(start)

            else:
                run = "self.data[(self.data[column] {0}) & " \
                      "(self.data[column] {1})]".format(start, stop)

            df = eval(run)

            return Option(kind=df["Kind"].values,
                          s0=df["S0"].values,
                          r=df["RFR"].values,
                          sigma=df["Volatility"].values,
                          k=df["Strike"].values,
                          data=df)

    def compute_premium(self, method="BSM", iteration=100000, update=True):
        """
        Calculate the current value of the option. The current value will use the implied volatility of the underlying.

        Parameters
        ----------
        method : str {"BSM", "MC"}
            Calculation Method:
                * BSM : Black-Scholes-Merton Model
                * MC : Monte Carlo
        iteration : int, optional
            Amount of iteration. Only relevant for MC. Default is 100000
        update : bool
            If True (default) the dataframe (self.data) will be updated.

        Returns
        -------
        float, array_like
        """
        premium = self.__compute_premium(iv=self.iv, method=method, iteration=iteration)

        if update:
            self.data["Premium"] = premium

        return premium

    def compute_fair_value(self, method="BSM", iteration=100000, update=True):
        """
        Calculate the fair value of the option. The fair value will use the actual volatility of the underlying.

        Parameters
        ----------
        method : str {"BSM", "MC"}
            Calculation Method:
                * BSM : Black-Scholes-Merton Model
                * MC : Monte Carlo
        iteration : int, optional
            Amount of iteration. Only relevant for MC. Default is 100000
        update : bool
            If True (default) the dataframe (self.data) will be updated.

        Returns
        -------
        float, array_like
        """
        premium = self.__compute_premium(iv=self.sigma, method=method, iteration=iteration)

        if update:
            self.data["Fair Value"] = premium

        return premium

    def compute_iv(self, update=True):
        """
        Compute the implied volatility of the underlying with a binary search.

        Parameters
        ----------
        update : bool
            If True (default) the dataframe (self.data) will be updated.

        Returns
        -------
        float, array_like
        """
        if np.all(self.data["Premium"].values == 0.0):
            raise ValueError("To compute the implied volatility, parameter `premium` must be set.")

        iv = compute_iv(self.kind, self.s0, self.sigma, self.k, self.r, self.t, self.q, self.data["Premium"].values)

        if update:
            self.data["IV"] = iv

        return iv

    def compute_greeks(self, update=True):
        """
        Delta is the rate of change of the option price with respect to the price of the underlying. Deltas can be
        positive or negative. Deltas can also be thought of as the probability that the option will expire ITM. Having
        a delta neutral portfolio can be a great way to mitigate directional risk from market moves.

        Parameters
        ----------
        update : bool
            If True (default) the dataframe (self.data) will be updated.

        Returns
        -------
        float, array_like
        """
        iv = self.data["IV"].values

        delta = compute_delta(self.kind, self.s0, iv, self.k, self.r, self.t, self.q)
        vega = compute_vega(self.s0, iv, self.k, self.r, self.t, self.q)
        theta = compute_theta(self.kind, self.s0, iv, self.k, self.r, self.t, self.q)
        rho = compute_rho(self.kind, self.s0, iv, self.k, self.r, self.t, self.q)
        epsilon = compute_epsilon(self.kind, self.s0, iv, self.k, self.r, self.t, self.q)
        gamma = compute_gamma(self.s0, iv, self.k, self.r, self.t, self.q)

        if update:
            self.data["Delta"] = delta
            self.data["Vega"] = vega
            self.data["Theta"] = theta
            self.data["Rho"] = rho
            self.data["Epsilon"] = epsilon
            self.data["Gamma"] = gamma

        result = dict()
        result["Delta"] = delta
        result["Vega"] = vega
        result["Theta"] = theta
        result["Rho"] = rho
        result["Epsilon"] = epsilon
        result["Gamma"] = gamma

        return result

    # ----------------------------------------------------------------------------------------------
    # Private Method
    # ----------------------------------------------------------------------------------------------
    def __update(self):
        self.compute_iv()
        self.compute_greeks()

    def __compute_premium(self, iv, method="BSM", iteration=100000):
        if method == "BSM":
            premium = compute_price_bsm(self.kind, self.s0, iv, self.k, self.r, self.t, self.q)
        else:
            premium = compute_price_ms(self.kind, self.s0, iv, self.k, self.r, self.t, self.q, iteration)

        return premium
