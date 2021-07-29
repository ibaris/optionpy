<div align="center">
  <p>
    <a href='https://www.freepik.com/'>
      <img src="./resources/logo/optionpy_logo.jpg" width="700" height="400">
    </a>
  </p>

<h4 align="center">OptionPy</h4>

<p align="center">
  <a href="https://media4.giphy.com/media/jVNR0r95F9TAbMxaXl/giphy.gif?cid=ecf05e47alhot5wrab95hjqs7327mxtqhioqtw16q7j2h9o9&rid=giphy.gif&ct=g">
    <img src="https://forthebadge.com/images/badges/fo-shizzle.svg"
         alt="Gitter">
  </a>
  <a href="https://media1.giphy.com/media/3o752jdW2dmll8zlvy/giphy.gif?cid=ecf05e47z9qu4hhk2et05v0qm8ajt9ag1vjzz81tupbqk2j6&rid=giphy.gif&ct=g">
    <img src="https://forthebadge.com/images/badges/powered-by-overtime.svg">
  </a>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#examples">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#dependencies">Dependencies</a> •
</p>
</div>

# Introduction

The `optionpy` package makes pricing of option contracts and calculating the *Greeks* fast.

# Key Features

* Calculate the quantities, like:
    * Fair Value : Fair value calculated with the BSM model and the volatility of the underlying (`sigma`)
    * ITM : A bool that indicates if the option in **In The Money**.
    * IV : Implied volatility
    * Delta, Vega, Theta, Rho, Epsilon, Gamma : The greeks.
    * Nd1, Nd2 : The probability of the event that the underlying price is over the strike price ($S_t≥K$) in the risk-neutral
      world.
* All function are vectorized.
* An advanced search and selection routine.

# Examples

Import OptionPy and other packages.

````python 
import optionpy as opy 
import numpy as np

`````

Generate random values.

````python
kind = np.random.choice([-1, 1], 1000)
s0 = np.random.uniform(low=50, high=150, size=(1000,))
k = np.random.uniform(low=50, high=150, size=(1000,)).round(2)
r = 0.01
sigma = np.random.uniform(low=0.05, high=0.5, size=(1000,)).round(2)
t = np.random.randint(1, 28, 1000)
````

Initialize the option class.

````python
option = opy.Option(kind=kind, s0=s0, k=k, r=r, sigma=sigma, t=t)
option
````

````console
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

````

Select where the Delta is greater than 0.3 and smaller than 0.35:

````python
option.select_where("Delta", ">=0.30", "<=0.35")
````
