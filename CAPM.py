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

# CAPM Analysis (Capital Asset Pricing Model)
benchmark = yf.download("^GSPC", start="2018-01-01", end="2025-05-15", auto_adjust=True)
benchmark_ret = benchmark['Close'].pct_change().dropna()

# Calculate portfolio returns using the optimal weights
portfolio_returns = (returns @ w_opt)

# Align the portfolio returns with the benchmark returns
common_index = benchmark_ret.index.intersection(returns.index)
portfolio_returns = portfolio_returns.loc[common_index]
benchmark_ret = benchmark_ret.loc[common_index]

# Perform regression to calculate alpha and beta using statsmodels
X = sm.add_constant(benchmark_ret)
model = sm.OLS(portfolio_returns, X).fit()

# Extract beta and alpha from the model
beta = model.params[1]
alpha = model.params[0]

# Plot CAPM regression
plt.figure(figsize=(8, 5))
plt.scatter(benchmark_ret, portfolio_returns, alpha=0.5)
plt.plot(benchmark_ret, model.predict(X), color='red', label='Security Market Line')
plt.xlabel("Benchmark Return")
plt.ylabel("Portfolio Return")
plt.title("CAPM Regression")
plt.legend()
plt.grid(True)
plt.show()

# Robust Regression
model_rlm = sm.RLM(portfolio_returns, X).fit()
plt.figure(figsize=(8, 5))
plt.scatter(benchmark_ret, portfolio_returns, alpha=0.4)
plt.plot(benchmark_ret, model_rlm.predict(), color='orange')
plt.title('Robust CAPM  (RLM)')
plt.xlabel('Market Return ')
plt.ylabel('Portfolio Return')
plt.legend()
plt.grid(True)
plt.show()

#FF Model

ff = pd.read_csv("F-F_Research_Data_Factors_daily.CSV",
                 names=["Date", "Mkt-RF", "SMB", "HML", "RF"], skiprows=1)
ff["Date"] = pd.to_datetime(ff['Date'], format='%Y%m%d')
ff.set_index('Date', inplace=True)
ff = ff.loc["2018-01-01":"2025-05-15"]

common_dates = ff.index.intersection(portfolio_returns.index)
rp = portfolio_returns.loc[common_dates]
ff = ff.loc[common_dates]

ff["Mkt-RF"] = ff["Mkt-RF"] / 100
ff["SMB"] = ff["SMB"] / 100
ff["HML"] = ff["HML"] / 100
ff["RF"] = ff["RF"] / 100

rp_excess = rp - ff["RF"]
X = ff[["Mkt-RF"]]

# Rolling window
def run_capm_rolling(Y, X, window_size=252):
    alphas, betas, t_alphas, t_betas = [], [], [], []
    dates = []

    for i in range(window_size, len(Y)):
        y_subset = Y.iloc[i - window_size:i]
        X_subset = X.iloc[i - window_size:i]

        model = sm.OLS(y_subset, sm.add_constant(X_subset)).fit()
        robust = model.get_robustcov_results(cov_type='HAC', maxlags=30)

        alphas.append(model.params[0])
        betas.append(model.params[1])
        t_alphas.append(robust.tvalues[0])
        t_betas.append(robust.tvalues[1])
        dates.append(Y.index[i])

    return pd.DataFrame({
        'alpha': alphas,
        'beta': betas,
        't_alpha': t_alphas,
        't_beta': t_betas
    }, index=dates)

rolling_capm = run_capm_rolling(rp_excess, X, window_size=252)

# Compute realized annual return
realized_annual_return = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
expected_growth = (1 + ret_opt) ** (np.arange(len(portfolio_returns)) / 252)
portfolio_growth = (1 + portfolio_returns).cumprod()

plt.figure(figsize=(10, 5))
plt.plot(portfolio_growth.index, portfolio_growth, label='Realized Cumulative Return')
plt.plot(portfolio_growth.index, expected_growth[:len(portfolio_growth)], 'r--', label='Expected Growth Trajectory')
plt.title("Expected vs. Realized Portfolio Performance")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (base=1)")
plt.legend()
plt.grid(True)
plt.show()

# Beta
fig, ax1 = plt.subplots(figsize=(10, 5))

# Left Axis: beta coefficient
ax1.plot(rolling_capm.index, rolling_capm['beta'], color='blue', label='Beta')
ax1.set_ylabel('Beta', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Right Axis: t-stat of beta
ax2 = ax1.twinx()
ax2.plot(rolling_capm.index, rolling_capm['t_beta'], color='orange', label='t-Beta')
ax2.set_ylabel('t-Beta', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.axhline(2, color='red', linestyle='--', linewidth=0.8)
ax2.axhline(-2, color='red', linestyle='--', linewidth=0.8)

# Title
fig.suptitle('Rolling Beta and t-Statistic (CAPM)', fontsize=14)
fig.tight_layout()
plt.grid(True)
plt.show()

fig, ax1 = plt.subplots(figsize=(10, 5))

# Left Axis: Alpha
ax1.plot(rolling_capm.index, rolling_capm['alpha'], color='red', label='Alpha')
ax1.set_ylabel('Alpha', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Right Axis: t-Alpha
ax2 = ax1.twinx()
ax2.plot(rolling_capm.index, rolling_capm['t_alpha'], color='purple', label='t-Alpha')
ax2.set_ylabel('t-Alpha', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')
ax2.axhline(2, color='black', linestyle='--', linewidth=0.8)
ax2.axhline(-2, color='black', linestyle='--', linewidth=0.8)

# Títle
fig.suptitle('Rolling Alpha and t-Statistic (CAPM)', fontsize=14)
fig.tight_layout()
plt.grid(True)
plt.show()
