import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import *

# Download data
prices = load_prices(tickers)

# Calculate daily returns
returns = compute_returns(prices)

# Calculate annual returns
mean_returns = returns.mean() * 252

# Calculate Beta for each Asset
market_return = returns[ex_return_market]
betas = compute_betas(returns, market_return)

# Expected return of the market
market_avg_return = mean_returns[ex_return_market]

# Calculate expected returns via CAPM
expected_returns = compute_expected_returns(betas, market_avg_return)

# Plot the Security Market Line (SML)
plt.figure(figsize=(10, 10))
beta_range = np.linspace(0, 2, 100)
sml_y = risk_free_rate + beta_range * (market_avg_return - risk_free_rate)
plt.plot(beta_range, sml_y, label='SML', color='blue')

# Scatter plot of each asset with its beta and expected return
for ticker in betas:
    plt.scatter(betas[ticker], mean_returns[ticker], label=ticker)
    plt.text(betas[ticker], mean_returns[ticker], ticker)
plt.xlabel('Beta')
plt.ylabel('Expected Return (Annual)')
plt.title('Security Market Line (SML) - CAPM')
plt.legend()
plt.grid(True)
plt.show()