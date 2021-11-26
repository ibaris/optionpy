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

    Examples
    --------
    >>> # Import OptionPy and other packages.
    >>> import optionpy as opy
    >>> import numpy as np

    >>> # Generate random values.
    >>> kind = np.random.choice([-1, 1], 1000)
    >>> s0 = np.random.uniform(low=50, high=150, size=(1000,))
    >>> k = np.random.uniform(low=50, high=150, size=(1000,)).round(2)
    >>> r = 0.01
    >>> sigma = np.random.uniform(low=0.05, high=0.5, size=(1000,)).round(2)
    >>> t = np.random.randint(1, 28, 1000)

    >>> # Initialize the option class.
    >>> option = opy.Option(kind=kind, s0=s0, k=k, r=r, sigma=sigma, t=t)
    >>> option
         Kind      Start        End  Maturity          S0  Strike   RFR  \
    0     1.0 2021-07-29 2021-08-08  0.027397   58.004210   51.21  0.01
    1    -1.0 2021-07-29 2021-08-14  0.043836   98.451153  103.22  0.01
    2    -1.0 2021-07-29 2021-08-07  0.024658   80.159073   59.73  0.01
    3    -1.0 2021-07-29 2021-08-24  0.071233   56.408618   95.96  0.01
    4    -1.0 2021-07-29 2021-08-23  0.068493   53.316540  125.57  0.01
    ..    ...        ...        ...       ...         ...     ...   ...
    995   1.0 2021-07-29 2021-08-02  0.010959   74.442832  135.33  0.01
    996  -1.0 2021-07-29 2021-08-07  0.024658   93.799969   55.20  0.01
    997   1.0 2021-07-29 2021-08-17  0.052055   93.193626  149.94  0.01
    998   1.0 2021-07-29 2021-08-19  0.057534  109.998472  108.17  0.01
    999  -1.0 2021-07-29 2021-07-30  0.002740  139.401736   74.22  0.01
         Volatility  Dividend     Fair Value  Premium    ITM    IV          Delta  \
    0          0.10       0.0   6.808238e+00      0.0   True  0.10   1.000000e+00
    1          0.39       0.0   6.170877e+00      0.0   True  0.39  -7.030166e-01
    2          0.37       0.0   1.492924e-07      0.0  False  0.37  -1.729793e-07
    3          0.37       0.0   3.948305e+01      0.0   True  0.37  -9.999999e-01
    4          0.24       0.0   7.216748e+01      0.0   True  0.24  -1.000000e+00
    ..          ...       ...            ...      ...    ...   ...            ...
    995        0.18       0.0  3.234953e-222      0.0  False  0.18  7.329169e-221
    996        0.46       0.0   7.106681e-14      0.0  False  0.46  -7.936171e-14
    997        0.21       0.0   1.008802e-23      0.0  False  0.21   2.289562e-23
    998        0.35       0.0   4.674266e+00      0.0   True  0.35   5.981234e-01
    999        0.29       0.0   0.000000e+00      0.0  False  0.29  -0.000000e+00
                  Vega          Theta            Rho        Epsilon  \
    0     1.592872e-14  -1.402629e-03   1.402629e-02  -1.589156e+00
    1     7.133997e-02  -8.488029e-02  -3.304490e-02   3.033986e+00
    2     1.149605e-07  -2.359237e-07  -3.455791e-09   3.418979e-07
    3     4.210952e-08   2.627139e-03  -6.830639e-02   4.018148e+00
    4     4.060452e-42   3.437918e-03  -8.594796e-02   3.651818e+00
    ..             ...            ...            ...            ...
    995  1.812494e-220 -4.079606e-220  5.975678e-223 -5.979223e-221
    996   8.779333e-14  -2.241548e-13  -1.853061e-15   1.835538e-13
    997   4.863145e-23  -2.693345e-23   1.105456e-24  -1.110707e-22
    998   1.020590e-01  -8.672366e-02   3.516401e-02  -3.785331e+00
    999   0.000000e+00   0.000000e+00  -0.000000e+00   0.000000e+00
                 Gamma            Nd1            Nd2
    0     1.728043e-13   1.000000e+00   1.000000e+00
    1     4.305262e-02   7.030166e-01   7.306406e-01
    2     1.961064e-07   1.729793e-07   2.346996e-07
    3     5.021196e-08   9.999999e-01   1.000000e+00
    4     8.689446e-42   1.000000e+00   1.000000e+00
    ..             ...            ...            ...
    995  1.658026e-219  7.329169e-221  4.029708e-221
    996   8.797284e-14   7.936171e-14   1.361784e-13
    997   5.122297e-23   2.289562e-23   1.417063e-23
    998   4.188742e-02   5.981234e-01   5.653469e-01
    999   0.000000e+00   0.000000e+00   0.000000e+00
    [1000 rows x 21 columns]

    >>> # Select where the Delta is greater than 0.3 and smaller than 0.35
    >>> option.select_where("Delta", ">=0.30", "<=0.35")
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
            _, _ = self.end1, self.end2

        else:
            self.__data = data

            self.kind = data["Kind"].values
            self.s0 = data["S0"].values
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
    def end1(self):
        """
        The effective Delta, which is the probability of the option being at the Strike + Premium under the stock measure

        Returns
        -------
        float, array_like
        """
        nd_list = list()

        try:
            tmp_put = self.select_where("Kind", "==-1")

            nd1_put = compute_nd1(tmp_put.data["Kind"].values,
                                  tmp_put.data["S0"].values,
                                  tmp_put.data["IV"].values,
                                  tmp_put["Strike"] - tmp_put.data["Premium"].values,
                                  tmp_put["RFR"],
                                  tmp_put["Maturity"].values,
                                  tmp_put["Dividend"].values)

            nd1_put = nd1_put.values.tolist()
            nd_list.extend(nd1_put)

        except ValueError:
            pass

        try:
            tmp_call = self.select_where("Kind", "==1")

            nd1_call = compute_nd1(tmp_call.data["Kind"].values,
                                   tmp_call.data["S0"].values,
                                   tmp_call.data["IV"].values,
                                   tmp_call["Strike"] + tmp_call.data["Premium"].values,
                                   tmp_call["RFR"],
                                   tmp_call["Maturity"].values,
                                   tmp_call["Dividend"].values)

            nd1_call = nd1_call.values.tolist()
            nd_list.extend(nd1_call)

        except ValueError:
            pass

        end1 = np.atleast_1d(nd_list)

        self.data["ENd1"] = end1

        return end1

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
    def end2(self):
        """
        The effective probability of the event that the underlying price is over the strike price ($S_t≥K + Premium$) in the
        risk-neutral world.

        Returns
        -------
        float, array_like
        """
        nd_list = list()

        try:
            tmp_put = self.select_where("Kind", "==-1")

            nd2_put = compute_nd2(kind=tmp_put.data["Kind"].values,
                                  s0=tmp_put.data["S0"].values,
                                  sigma=tmp_put.data["IV"].values,
                                  k=tmp_put["Strike"].values - tmp_put.data["Premium"].values,
                                  r=tmp_put["RFR"],
                                  t=tmp_put["Maturity"].values,
                                  q=tmp_put["Dividend"].values)

            nd2_put = nd2_put.values.tolist()
            nd_list.extend(nd2_put)

        except ValueError:
            pass

        try:
            tmp_call = self.select_where("Kind", "==1")

            nd2_call = compute_nd2(kind=tmp_call.data["Kind"].values,
                                   s0=tmp_call.data["S0"].values,
                                   sigma=tmp_call.data["IV"].values,
                                   k=tmp_call["Strike"] + tmp_call.data["Premium"].values,
                                   r=tmp_call["RFR"],
                                   t=tmp_call["Maturity"].values,
                                   q=tmp_call["Dividend"].values)

            nd2_call = nd2_call.values.tolist()
            nd_list.extend(nd2_call)

        except ValueError:
            pass

        end2 = np.atleast_1d(nd_list)

        self.data["ENd2"] = end2

        return end2

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
