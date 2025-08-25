from scipy.optimize import minimize
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from data_utils import *

# Define tickers and download data from Yahoo Finance
prices = load_prices(investable_tickers)

# Calculate daily returns
returns = compute_returns(prices)

# Calculate annual returns
mean_returns = compute_anual_returns(returns)

# Calculate the standard deviation (volatility)
std_dev = compute_anual_volatility(returns)

# Calculate the correlation matrix
correlation_matrix = compute_correlation_matrix(returns)

# Calculate the covariance matrix
cov_matrix = np.outer(std_dev, std_dev) * correlation_matrix

# Inverse of the covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Constraints: sum of weights = 1 (full investment)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Initialize with equal weights
w0 = np.ones(len(mean_returns)) / len(mean_returns)

# Optimize the portfolio using the negative Sharpe ratio
opt_result = minimize(negative_sharpe_ratio, w0,
                      args=(mean_returns, cov_matrix),
                      method='SLSQP',
                      constraints=constraints)

# Get the optimal weights and the maximum Sharpe ratio
w_opt = opt_result.x
sharpe_opt = -opt_result.fun

# Calculate the return and volatility of the optimal portfolio
ret_opt = np.dot(w_opt, mean_returns)
vol_opt = np.sqrt(w_opt.T @ cov_matrix @ w_opt)

# Replot everything including the tangency portfolio
plt.figure(figsize=(10, 6))

# Plot the Tangency Portfolio
plt.scatter(vol_opt, ret_opt, color='red', marker='*', s=200, label='Tangency Portfolio')

# Plot the Capital Market Line (CML)
plt.plot([0, vol_opt], [risk_free_rate, ret_opt], color='green', linestyle='--', linewidth=2, label='Capital Market Line')
plt.xlabel('Annual Volatility')
plt.ylabel('Annual Expected Return')
plt.title('Efficient Frontier with Tangency Portfolio')
plt.legend()
plt.grid(True)
plt.xlim(0, None)
plt.show()