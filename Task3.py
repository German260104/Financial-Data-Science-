import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("results_task1.csv")
df["Date"] = pd.to_datetime(df["Date"])


index = df[df["Symbol"] == "SPX"]
stocks = df[df["Symbol"] != "SPX"]


avg_skew_stocks = stocks.groupby("Date")["Skew"].mean().reset_index(name="Avg_Skew_Stocks")

index_skew = index[["Date", "Skew"]].rename(columns={"Skew": "SPX_Skew"})
comparison = pd.merge(avg_skew_stocks, index_skew, on="Date")


comparison.plot(x="Date", y=["SPX_Skew", "Avg_Skew_Stocks"], figsize=(10, 5), title="Skewness: SPX vs. Avg Stocks")
plt.axhline(0, color="black", linestyle="--", alpha=0.5)
plt.ylabel("Implied Skewness")
plt.show()


print(comparison.mean(numeric_only=True))
