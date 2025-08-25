from scipy.ndimage import standard_deviation
from scipy.optimize import minimize
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import *

# Download data
prices = load_prices(investable_tickers)

# Calculate log daily returns
returns = np.log(prices / prices.shift(1))
returns = returns.dropna()

# Calculate annualized returns and standard deviation
mu = compute_anual_returns(returns)  # Annualized mean returns

# Calculate correlation and covariance matrices
correlation_matrix = compute_correlation_matrix(returns)
cov_matrix = returns.cov().values * 252

# Simulate portfolios
def simulation_portfolio(mu, cov_matrix, risk_free_rate, n_portfolios):
    weights_record = []
    results = np.zeros((3, n_portfolios))
    np.random.seed(42)
    for i in range(n_portfolios):
        weights = np.random.random(len(mu))
        weights /= np.sum(weights)  # Normalize weights to sum to 1
        weights_record.append(weights)

         # Portfolio return and risk
        port_return = np.dot(weights, mu)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility

        results[0, i] = port_return
        results[1, i] = port_volatility
        results[2, i] = sharpe_ratio

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'])
    return results_df

results_df = simulation_portfolio(mu, cov_matrix, risk_free_rate=0.03, n_portfolios= 10000)
# Plot Efficient Frontier
plt.figure(figsize=(12, 7))
scatter = plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'],
                      cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Volatility (Risk)')
plt.xlim(0,0.5)
plt.ylim(0,0.4)
plt.ylabel('Expected Return')
plt.title('Portfolio Simulation')
plt.grid(True)
plt.show()

