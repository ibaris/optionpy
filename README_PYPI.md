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
