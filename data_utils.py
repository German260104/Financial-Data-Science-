import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tukeyplot import results

# Global parameters
tickers = ['MSFT', 'DE', 'COST', 'BYDDY', 'AMD', 'GLD', '^GSPC']
investable_tickers = [t for t in tickers if t != '^GSPC']
print(investable_tickers)
start_date = '2018-01-01'
end_date = '2025-05-15'
risk_free_rate = 0.03
ex_return_market = '^GSPC'

# Define tickers and download data from Yahoo Finance
def load_prices(tickers):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    return data['Adj Close'].dropna()

# Calculate simple returns
def compute_returns(prices):
    returns = (prices - prices.shift()) / prices.shift()
    returns = returns.dropna()
    return returns

# Calculate anual returns
def compute_anual_returns(returns):
    anual_returns = returns.mean() * 252
    return anual_returns

# Calculate the standard deviation (volatility)
def compute_anual_volatility(returns):
    std_dev = returns.std() * np.sqrt(252)
    return std_dev

# Calculate the correlation matrix
def compute_correlation_matrix(returns):
    correlation_matrix = returns.corr()
    return correlation_matrix

# Jarque-Bera test statistic function
def jarque_bera_testStatistic(sample):
    n = len(sample)
    mean = np.mean(sample)
    standard_deviation = np.std(sample)
    skewness = np.mean(((sample - mean) / standard_deviation) ** 3)
    kurtosis = np.mean(((sample - mean) / standard_deviation) ** 4)
    jb_value = (n / 6) * (skewness ** 2 + ((kurtosis - 3) ** 2) / 4)
    return jb_value, skewness, kurtosis

# Calculate Beta for each Asset
def compute_betas(returns, market_return):
        betas = {}
        for ticker in investable_tickers:
            cov = np.cov(returns[ticker], market_return)
            beta = cov[0, 1] / cov[1, 1]
            betas[ticker] = beta

        return betas

# Calculate expected returns via CAPM
def compute_expected_returns(betas, market_avg_return):
    expected_returns = {}
    for ticker in investable_tickers:
        beta = betas[ticker]
        expected = risk_free_rate + beta * (market_avg_return - risk_free_rate)
        expected_returns[ticker] = expected

    return expected_returns

# Returns negative sharp ratio
def negative_sharpe_ratio(w, mu, cov_matrix):
    port_return = np.dot(w, mu)
    port_volatility = np.sqrt(w.T @ cov_matrix @ w)
    return - (port_return - risk_free_rate) / port_volatility


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