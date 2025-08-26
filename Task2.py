import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Task 1 results
df = pd.read_csv("results_task1.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Filter out SPX from constituents
constituents = df[df["Symbol"] != "SPX"]
index = df[df["Symbol"] == "SPX"]

# Group by date and calculate mean and std of skewness
grouped = constituents.groupby("Date")["Skew"].agg(["mean", "std"]).reset_index()

# Merge with index skewness
index_skew = index[["Date", "Skew"]].rename(columns={"Skew": "SPX_Skew"})
merged = pd.merge(grouped, index_skew, on="Date")

# Compute systematic and idiosyncratic skewness
merged["Systematic_Skew"] = merged["SPX_Skew"]
merged["Idiosyncratic_Skew"] = merged["std"]

plt.figure(figsize=(10, 6))
plt.plot(merged["Date"], merged["Systematic_Skew"], label="Systematic Skew (SPX)", color="blue")
plt.plot(merged["Date"], merged["Idiosyncratic_Skew"], label="Idiosyncratic Skew (std across stocks)", color="orange")
plt.title("Decomposition of Option-Implied Skewness (Feb 2023)")
plt.xlabel("Date")
plt.ylabel("Skewness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()