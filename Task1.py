import numpy as np
import pandas as pd
from scipy.integrate import trapz

# Define the implied volatility surface for S&P500 and SPX and the risk-free rate for each day
ivs_sp500 = pd.read_parquet("sp500_merged_ivs_2023-02parquet.sec")
ivs_spX = pd.read_parquet("spx_ivs_2023-02parquet.sec")
risk_free_df = pd.read_csv("riskfree_30d_2023-02 (1).csv")
risk_free_df['date'] = pd.to_datetime(risk_free_df['date'])
risk_free_df.set_index('date', inplace=True)

# Convert date column
ivs_sp500['loctimestamp'] = pd.to_datetime(ivs_sp500['loctimestamp'])
ivs_spX['loctimestamp'] = pd.to_datetime(ivs_spX['loctimestamp'])

# Convert percent to decimal
risk_free_df['r'] = risk_free_df['yld_pct_annual'] / 100

# Define all the trading dates of February
all_dates = ivs_sp500['loctimestamp'].dt.normalize().unique()
feb_dates = [d for d in all_dates if d.month == 2 and d.year == 2023]

# Function to compute price of the volatility, cubic and the quartic contracts (V,W,X)
def compute_vwx(dataframe, spot_col='underlyingprice'):

    dataframe = dataframe.dropna(subset=['implPrice', 'strike', spot_col, 'putcall'])
    S = dataframe[spot_col].iloc[0]

    # Divide call and put options
    calls = dataframe[(dataframe['putcall'].str.upper() == 'C') & (dataframe['strike'] >= S)].copy()
    puts = dataframe[(dataframe['putcall'].str.upper() == 'P') & (dataframe['strike'] < S)].copy()

    # Sort strikes
    calls = calls.sort_values('strike')
    puts = puts.sort_values('strike')

    # Define Strikes and prices (Call and Put)
    Kc, C = calls['strike'].values, calls['implPrice'].values
    Kp, P = puts['strike'].values, puts['implPrice'].values

    # Prevent errors with few data
    if len(Kc) < 3 or len(Kp) < 3:
        return np.nan, np.nan, np.nan

    # Compute V(t, τ)
    integrand_call_V = (2 * (1 - np.log(Kc / S))) / (Kc**2) * C
    integrand_put_V = (2 * (1 + np.log(S / Kp))) / (Kp**2) * P
    V = trapz(integrand_call_V, Kc) + trapz(integrand_put_V, Kp)

    # Compute W(t, τ)
    integrand_call_W = ((6 * np.log(Kc / S) - 3 * np.log(Kc / S) ** 2) / Kc ** 2) * C
    integrand_put_W = ((6 * np.log(S / Kp) + 3 * np.log(S / Kp) ** 2) / Kp ** 2) * P
    W = trapz(integrand_call_W, Kc) - trapz(integrand_put_W, Kp)

    # Compute X(t, τ)
    integrand_call_X = ((12 * np.log(Kc / S) ** 2 - 4 * np.log(Kc / S) ** 3) / Kc ** 2) * C
    integrand_put_X = ((12 * np.log(S / Kp) ** 2 + 4 * np.log(S / Kp) ** 3) / Kp ** 2) * P
    X = trapz(integrand_call_X, Kc) + trapz(integrand_put_X, Kp)

    return V, W, X

# Function to compute the mean(t, τ): First moment
def compute_mean(V_value, W_value, X_value, r_value, tau_value):
    e = np.exp(r_value * tau_value)
    mu = e - 1 - (e / 2) * V_value - (e / 6) * W_value - (e / 24) * X_value
    return mu

# Function to compute the variance(t, τ): Second central moment
def compute_variance(V_value, mean_value):
    variance = V_value - mean_value ** 2
    return variance

# Function to compute the skewness: third standardized moment
def compute_skewness(V_value, W_value, mean_value, r_value, tau_value):
    e = np.exp(r_value * tau_value)
    skew = ((e * W_value - 3 * mean_value * e * V_value + 2 * mean_value ** 3) /
            ((e * V_value - mean_value ** 2) ** (3/2)))
    return skew

# Function to compute the kurtosis: fourth standardized moment
def compute_kurtosis(V_value, W_value, X_value, mean_value, r_value, tau_value):
    e = np.exp(r_value * tau_value)
    kurtosis = (e * X_value - 4 * mean_value * e * W_value + 6 * e * (mean_value ** 2) * V_value -
                 3 * (mean_value ** 4)) / (e * V_value - (mean_value ** 2)) ** 2
    return kurtosis

# Filter maturity options and compute the four moments of the underlying return for Feb 2023 (S&P500 and SPX)

results = []

for t in feb_dates:

    # Define risk-free rate for the day t
    try:
        r = risk_free_df.loc[t, 'r']
    except KeyError:
        r = 0.05


    # Filter implied volatility surface of S&P500 and SPX
    ivs_sp500_filtered = ivs_sp500[(ivs_sp500['loctimestamp'] == t) &
                                   (ivs_sp500['daystomaturity'].between(27, 33))]

    ivs_spX_filtered = ivs_spX[(ivs_spX['loctimestamp'] == t) &
                               (ivs_spX['daystomaturity'].between(27, 33))]

    tau_days = ivs_sp500_filtered['daystomaturity'].mean()
    tau = tau_days / 365 if not np.isnan(tau_days) else 30 / 365

    # Compute BKM Moments for S&P500 constituents
    for symbol, group in ivs_sp500_filtered.groupby('Symbol'):
        try:
            group = group.dropna(subset=['implPrice', 'strike', 'underlyingprice', 'putcall'])
            if group.empty:
                continue

            S = group['underlyingprice'].iloc[0]
            calls = group[(group['putcall'].str.upper() == 'C') & (group['strike'] >= S)]
            puts = group[(group['putcall'].str.upper() == 'P') & (group['strike'] < S)]

            if len(calls) < 3 or len(puts) < 3:
                continue

            V, W, X = compute_vwx(group)
            mu = compute_mean(V, W, X, r, tau)
            variance = compute_variance(V, mu)
            skewness = compute_skewness(V, W, mu, r, tau)
            kurtosis = compute_kurtosis(V, W, X, mu, r, tau)

            results.append({
                'Date': t,
                'Symbol': symbol,
                'μ': mu,
                'σ²': variance,
                'Skew': skewness,
                'Kurtosis': kurtosis,
                'V': V,
                'W': W,
                'X': X,
            })
        except Exception as e:
            print(f"Failed for {symbol} on {t.date()}: {e}")

    # Compute the values of the four moments for SPX Index
    try:
        V_spx, W_spx, X_spx = compute_vwx(ivs_spX_filtered)
        mu_spx = compute_mean(V_spx, W_spx, X_spx, r, tau)
        variance_spx = compute_variance(V_spx, mu_spx)
        skewness_spx = compute_skewness(V_spx, W_spx, mu_spx, r, tau)
        kurtosis_spx = compute_kurtosis(V_spx, W_spx, X_spx, mu_spx, r, tau)

        results.append({
            'Date': t,
            'Symbol': 'SPX',
            'μ': mu_spx,
            'σ²': variance_spx,
            'Skew': skewness_spx,
            'Kurtosis': kurtosis_spx,
            'V': V_spx,
            'W': W_spx,
            'X': X_spx,
        })
    except Exception as e:
        print(f"Failed for SPX on {t.date()}: {e}")


df_results = pd.DataFrame(results)
df_results.to_csv("results_task1.csv", index=False)

# Print values of all the four moments for all the constituents of S&P500 and SPX Index
# print(df_results.to_string(index=False))
