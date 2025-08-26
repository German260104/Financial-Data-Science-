import numpy as np
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
import os

# Define tickers and get data
tickers = ["NVDA", "VNQ", "PLTR"]
data = yf.download(tickers, period="100d", auto_adjust=True)
prices = data["Close"]

# Define return matrix for the three stocks
returns_matrix = ((prices/prices.shift(1)) - 1).dropna()

# Define risk tolerance
risk_tolerance = 0.015

# Define ambiguity radius
ambiguity_radius = 0.05

# Compute estimated return for each stock
estimated_return = returns_matrix.mean()

# Define parameters for optimization
T, n = returns_matrix.shape
mu = estimated_return.values
R_centered = (returns_matrix - mu).values

# Optimization variables
w = cp.Variable(n, nonneg=True)
u = cp.Variable(T) # Variable to represent the absolute value in the MAD definition

# MAD definition
MAD = (1/T) * cp.sum(u)

# Define Constraints
constraints = [
    cp.sum(w) == 1,
    u >= R_centered @ w,
    u >= -(R_centered @ w),
    MAD <= risk_tolerance
]

# Define objective function and optimization problem
objective_function = cp.Maximize(mu @ w - ambiguity_radius * MAD)
problem = cp.Problem(objective_function, constraints)
problem.solve()

# Compute portfolio's MAD with optimized weights
portfolio_deviation = R_centered @ w.value
optimized_MAD = np.mean(np.abs(portfolio_deviation))

# Compute portfolio's expected return with optimized weights
expected_portfolio_returns = mu @ w.value

# Print all the results
print("Optimized weights: ", w.value)
print("MAD with optimized weights: ", optimized_MAD)
print("Expected portfolio returns: ",  expected_portfolio_returns)
print(os.getcwd())

# Plot optimal weights and MAD vs Expected Return
plt.figure(figsize=(6, 4))
plt.bar(tickers, w.value)
plt.title("Optimized Portfolio Weights")
plt.ylabel("Weight")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig("optimized_weights.png", dpi=300)
plt.show()

metrics = ["MAD", "Expected Return"]
values = [optimized_MAD, expected_portfolio_returns]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values)
plt.title("Portfolio Metrics")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("portfolio_metrics.png", dpi=300)
plt.show()