from scipy.optimize import minimize
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import *
from Simulation_Portfolios import simulation_portfolio

# Download data
prices = load_prices(investable_tickers)

# Calculate log returns
returns = np.log(prices / prices.shift(1))
returns = returns.dropna()

# Calculate annualized returns and standard deviation
mu = compute_anual_returns(returns)  # Annualized mean returns


# Calculate the correlation matrix
correlation_matrix = compute_correlation_matrix(returns)

# Covariance matrix (annualized)
Cov_matrix = returns.cov().values * 252

n = len(mu)  # Number of assets

# Constraints: sum of the weights = 1 (fully invested portfolio)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Bounds for no short-selling (weights between 0 and 1)
bounds = [(0, 1) for _ in range(len(mu))]

# Initial weights: equal distribution across assets
w0 = np.ones(len(mu)) / len(mu)

# Optimization to find the tangency portfolio (maximizing Sharpe ratio)
opt_result = minimize(negative_sharpe_ratio,
                      w0,
                      args=(mu, Cov_matrix),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

# Extract the optimal weights and the maximum Sharpe ratio
w_opt = opt_result.x
sharpe_opt = -opt_result.fun

# Calculate the return and volatility of the tangency portfolio
ret_opt = np.dot(w_opt, mu)
vol_opt = np.sqrt(w_opt.T @ Cov_matrix @ w_opt)

# Function to calculate portfolio variance
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

# Function to solve for the unconstrained efficient frontier
def solve_unconstrained_frontier(mu, Cov_matrix, n_points=100):
    mu_targets = np.linspace(min(mu) * 0.8, max(mu) * 1.2, n_points)
    mu_vals, sigma_vals = [], []

    for mu_target in mu_targets:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights must be 1
            {'type': 'eq', 'fun': lambda w: w @ mu - mu_target}  # Expected return must match target
        ]
        result = minimize(portfolio_variance, x0=np.ones(n) / n,
                          args=(Cov_matrix,), method='SLSQP',
                          constraints=constraints)
        if result.success:
            w = result.x
            mu_vals.append(np.dot(w, mu))
            sigma_vals.append(np.sqrt(w @ Cov_matrix @ w))
        else:
            mu_vals.append(np.nan)
            sigma_vals.append(np.nan)

    return np.array(mu_vals), np.array(sigma_vals)


# Function to solve for the constrained efficient frontier (no more than 50% weight per asset)
def solve_constrained_frontier(mu, Cov_matrix, n_points=100):
    mu_targets = np.linspace(min(mu) * 0.8, max(mu) * 1.2, n_points)
    mu_vals, sigma_vals = [], []
    bounds = [(0, 0.5) for _ in range(n)]

    for mu_target in mu_targets:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights must be 1
            {'type': 'eq', 'fun': lambda w: w @ mu - mu_target}  # Expected return must match target
        ]
        result = minimize(portfolio_variance,
                          x0=np.ones(n) / n,
                          args=(Cov_matrix,),
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
        if result.success:
            w = result.x
            mu_vals.append(np.dot(w, mu))
            sigma_vals.append(np.sqrt(w @ Cov_matrix @ w))
        else:
            mu_vals.append(np.nan)
            sigma_vals.append(np.nan)

    return np.array(mu_vals), np.array(sigma_vals)


# Solve for the unconstrained and constrained frontiers
mu_uncon, sigma_uncon = solve_unconstrained_frontier(mu, Cov_matrix)
mu_con, sigma_con = solve_constrained_frontier(mu, Cov_matrix)

# Simulate portfolios


results_df = simulation_portfolio(mu, Cov_matrix, risk_free_rate=0.03, n_portfolios= 10000)

# Convert results to DataFrame for easier analysis

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(sigma_uncon, mu_uncon, label='Unconstrained Frontier', lw=2)
plt.plot(sigma_con, mu_con, label='Constrained Frontier', lw=2)
scatter = plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Sharpe Ratio')

# Plot the tangency portfolio
plt.scatter(vol_opt, ret_opt, color='red', marker='*', s=200, label='Tangency Portfolio')

#Plot the Capital Market Line (CML)
plt.plot([0, vol_opt], [risk_free_rate, ret_opt], color='green', linestyle='--', linewidth=2, label='Capital Market Line')
plt.xlabel('Annual Volatility')
plt.ylabel('Annual Expected Return')
plt.legend()
plt.grid(True)
plt.xlim(0, 0.2)
plt.ylim(0, 0.25)
plt.show() ###