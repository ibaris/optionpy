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

Calculate the quantities

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
* Nd1, Nd2 : The probability of the event that the underlying price is over the strike price ($S_t≥K$) in the risk-neutral world.

and display it as a `DataFrame` object.

# Examples
