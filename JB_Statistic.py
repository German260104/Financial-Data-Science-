import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from data_utils import jarque_bera_testStatistic, load_prices, compute_returns, investable_tickers

# Download data
prices = load_prices(investable_tickers)

# Calculate daily returns
simple_returns = compute_returns(prices)

# Set up the plot
plt.figure(figsize=(15, 10))
for i, ticker in enumerate(investable_tickers):
    serie = simple_returns[ticker].dropna()
    jb_value, skewness, kurtosis = jarque_bera_testStatistic(serie)

    # Plot histogram for each ticker
    plt.subplot(3, 2, i + 1)
    plt.hist(serie, bins=30, color='lightblue', edgecolor='black', density=True)
    plt.title(f"Return Histogram: {ticker}")
    plt.xlabel('Return')
    plt.ylabel('Frequency')

    # Display Jarque-Bera statistics on the plot
    plt.text(0.05, 0.95, f"JB = {jb_value:.2f}\nSkew = {skewness:.2f}\nKurt = {kurtosis:.2f}",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

# Adjust layout and show the plot
plt.tight_layout()
plt.show()