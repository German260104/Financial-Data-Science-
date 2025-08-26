import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import matplotlib.dates as mdates

# Load the CSV with the calculated moments
file_path = "results_task1.csv"
df = pd.read_csv(file_path)

# Ensure proper datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Separate SPX and other symbols
spx_df = df[df["Symbol"] == "SPX"]
sp500_df = df[df["Symbol"] != "SPX"]

# Set the style
sns.set(style="whitegrid")

# Plot 1: Line plot of moments for SPX
fig1, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
fig1.suptitle("Daily Moments of SPX (Feb 2023)", fontsize=16)

axs[0, 0].plot(spx_df["Date"], spx_df["μ"], label="Mean (μ)")
axs[0, 0].set_ylabel("Mean")

axs[0, 1].plot(spx_df["Date"], spx_df["σ²"], label="Variance (σ²)", color="orange")
axs[0, 1].set_ylabel("Variance")

axs[1, 0].plot(spx_df["Date"], spx_df["Skew"], label="Skewness", color="green")
axs[1, 0].set_ylabel("Skewness")

axs[1, 1].plot(spx_df["Date"], spx_df["Kurtosis"], label="Kurtosis", color="red")
axs[1, 1].set_ylabel("Kurtosis")

for ax in axs.flat:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot 2: Boxplot of Skewness over time for SP500 constituents
plt.figure(figsize=(16, 6))
sns.boxplot(x="Date", y="Skew", data=sp500_df)
plt.title("Cross-Sectional Distribution of Skewness (S&P500)")
plt.xticks(rotation=45)
plt.ylabel("Skewness")
plt.tight_layout()
plt.show()

# Plot 3: SPX Skewness vs. Mean SP500 Skewness
avg_skew_df = sp500_df.groupby("Date")["Skew"].mean().reset_index(name="Avg_SP500_Skew")
merged_skew = pd.merge(spx_df[["Date", "Skew"]], avg_skew_df, on="Date")
merged_skew.rename(columns={"Skew": "SPX_Skew"}, inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(merged_skew["Date"], merged_skew["SPX_Skew"], label="SPX Skewness", color="red")
plt.plot(merged_skew["Date"], merged_skew["Avg_SP500_Skew"], label="Average S&P500 Skewness", color="blue")
plt.legend()
plt.title("SPX vs. Avg SP500 Skewness (Feb 2023)")
plt.xticks(rotation=45)
plt.ylabel("Skewness")
plt.tight_layout()
plt.show()

# Plot 4: Scatter plot of Skewness vs Kurtosis for S&P500 constituents
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Skew", y="Kurtosis", data=sp500_df, hue=sp500_df["Date"].dt.strftime('%b %d'), palette="tab20", legend=False)
plt.title("Skewness vs. Kurtosis (S&P500 Constituents, Feb 2023)")
plt.xlabel("Skewness")
plt.ylabel("Kurtosis")
plt.tight_layout()
plt.show()
