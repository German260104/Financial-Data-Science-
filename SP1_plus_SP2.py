"""
Full pipeline for processing combined SPY1+SPY2 intraday data, merging with news events,
creating dummy variables, estimating variance regressions, and computing bootstrap standard errors.

Author: Rodrigo Castilla
Date: 2025-07-13
"""

# --- IMPORTS ---
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- STEP 1: Load and clean SPY1 and SPY2 data ---
# Load and clean SPY1
spy1 = pd.read_csv('../data/SPY_15min_2002-01_to_2006-08.csv')
spy1['timestamp'] = pd.to_datetime(spy1['Unnamed: 0'], utc=True).dt.tz_convert('America/New_York')
spy1 = spy1.sort_values('timestamp').reset_index(drop=True)
spy1['log_close'] = np.log(spy1['close'])
spy1['return'] = spy1['log_close'].diff()

# Load and clean SPY2
spy2 = pd.read_csv('../data/SPY_15min_2020-01_to_2022-01.csv')
spy2['timestamp'] = pd.to_datetime(spy2['Unnamed: 0'], utc=True).dt.tz_convert('America/New_York')
spy2 = spy2.sort_values('timestamp').reset_index(drop=True)
spy2['log_close'] = np.log(spy2['close'])
spy2['return'] = spy2['log_close'].diff()

# Combine datasets
spy = pd.concat([spy1, spy2], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
print(f"Combined SPY1+SPY2 with {len(spy)} rows.")

# --- STEP 2: Create calendar dummies ---
spy['day_of_week'] = spy['timestamp'].dt.dayofweek
spy['hour_of_day'] = spy['timestamp'].dt.hour

dow_dummies = pd.get_dummies(spy['day_of_week'], prefix='day').drop('day_0', axis=1)
hour_dummies = pd.get_dummies(spy['hour_of_day'], prefix='hour').drop('hour_10', axis=1, errors='ignore')

spy['overnight'] = ((spy['hour_of_day'] < 9) | (spy['hour_of_day'] > 16)).astype(int)

spy = pd.concat([spy, dow_dummies, hour_dummies], axis=1)
print("Calendar dummies created.")

# --- STEP 3: Merge with news events (FILTERED TO COMBINED RANGE) ---
events = pd.read_csv('../data/WhatMovesMarkets_eventdatabase.csv')
events['eventstart'] = pd.to_datetime(events['eventstart'], errors='coerce', format='%d.%m.%Y %H:%M:%S')
events = events.dropna(subset=['eventstart'])
events['eventstart'] = events['eventstart'].dt.tz_localize('Europe/Berlin').dt.tz_convert('America/New_York')
events['event_bin'] = events['eventstart'].dt.round('15min')

min_ts = spy['timestamp'].min()
max_ts = spy['timestamp'].max()
events = events[(events['event_bin'] >= min_ts) & (events['event_bin'] <= max_ts)]
print(f"Filtered events to {len(events)} rows matching SPY1+SPY2 date range.")
print("Number of events by type after filter:")
print(events['type'].value_counts())

events = events[(events['event_bin'].dt.hour >= 9) & (events['event_bin'].dt.hour <= 16)]

event_types = ['Macro Release', 'Geopolitical', 'Earnings', 'Central Bank', 'Ad Hoc']
for event_type in event_types:
    dummy_name = f'news_{event_type.replace(" ", "_")}'
    spy[dummy_name] = spy['timestamp'].isin(events.loc[events['type'] == event_type, 'event_bin']).astype(int)

print("Event dummies created.")

# --- STEP 4: Prepare regression data ---
spy['ret_sq'] = spy['return'] ** 2

calendar_cols = [col for col in spy.columns if col.startswith('day_') or col.startswith('hour_') or col == 'overnight']
news_cols = [col for col in spy.columns if col.startswith('news_')]

print("Proportion of 1s in each news dummy:")
for col in news_cols:
    print(f"{col}: {spy[col].mean():.4f}")

X = spy[calendar_cols + news_cols]
X = sm.add_constant(X).astype(float)
y = spy['ret_sq'].astype(float)

mask = y.notnull()
X_clean = X.loc[mask].copy()
y_clean = y.loc[mask].copy()

model = sm.OLS(y_clean, X_clean).fit()
print(model.summary())

# --- STEP 5: Diagnostics and summaries ---
print(f"R-squared: {model.rsquared:.4f}")
print(f"Condition Number: {model.condition_number:.2e}")
print("-" * 60)

pvals = model.pvalues
sig_news = {level: (pvals[news_cols] < level).sum() for level in [0.01, 0.05, 0.1]}
print("Significant 'news' coefficients by level:")
for level, count in sig_news.items():
    print(f"p < {level}: {count}")

coef_news = model.params[news_cols]
top5 = coef_news.abs().sort_values(ascending=False).head(5).index
print("Top 5 news coefficients:")
for var in top5:
    print(f"{var:30s}: coef = {model.params[var]:.4e}, p = {pvals[var]:.4f}")

mean_ret_sq = y_clean.mean()
omega_k = {}

print("\n=== Omega_k (point estimates BEFORE bootstrap) ===")
for k in news_cols:
    beta_k = model.params[k]
    pk = X_clean[k].mean()
    omega = (beta_k * pk) / mean_ret_sq
    omega_k[k] = omega
    print(f"{k:30s}: Omega = {omega:.4e}")

# --- STEP 6: Bootstrap standard errors ---
block_length = 15
n_bootstrap = 1000
beta_boot, omega_boot = [], []

n_obs = len(X_clean)
n_blocks = int(np.ceil(n_obs / block_length))

Xb = X_clean.reset_index(drop=True)
yb = y_clean.reset_index(drop=True)

for _ in tqdm(range(n_bootstrap)):
    starts = np.random.randint(0, n_obs - block_length + 1, size=n_blocks)
    indices = np.hstack([np.arange(s, s + block_length) for s in starts])
    indices = indices[indices < n_obs]

    try:
        m_b = sm.OLS(yb.iloc[indices], Xb.iloc[indices]).fit()
        b_b = m_b.params[news_cols]
        pk_b = Xb.iloc[indices][news_cols].mean()
        omega_b_val = (b_b * pk_b) / yb.iloc[indices].mean()

        beta_boot.append(b_b)
        omega_boot.append(omega_b_val)
    except:
        continue

beta_boot_df = pd.DataFrame(beta_boot)
omega_boot_df = pd.DataFrame(omega_boot)

se_beta = beta_boot_df.std()
se_omega = omega_boot_df.std()

print("\nBootstrap standard errors:")
for k in news_cols:
    print(f"{k:30s}: SE(beta) = {se_beta[k]:.4e}, SE(Omega) = {se_omega[k]:.4e}")

print("\n=== Bootstrap standard errors (for beta and Omega_k) ===")
for k in news_cols:
    print(f"{k:30s}: SE(beta) = {se_beta[k]:.4e}, SE(Omega) = {se_omega[k]:.4e}")

omega_mean_bootstrap = omega_boot_df.mean()
print("\n=== Mean Omega_k from bootstrap samples ===")
for k in news_cols:
    print(f"{k:30s}: mean Omega = {omega_mean_bootstrap[k]:.4e}")


# --- STEP 7: Visualization ---
plt.figure(figsize=(8, 5))
plt.bar(omega_k.keys(), omega_k.values())
plt.axhline(0, color='gray', linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Omega (variance share)')
plt.title('Variance decomposition by news category (SPY1+SPY2)')
plt.tight_layout()
plt.show()

spy['explained'] = model.fittedvalues
spy['unexplained'] = model.resid

plt.figure(figsize=(10, 5))
plt.plot(spy.loc[mask, 'timestamp'], spy.loc[mask, 'explained'].rolling(100).mean(), label='Explained variance', color='blue')
plt.plot(spy.loc[mask, 'timestamp'], spy.loc[mask, 'unexplained'].rolling(100).mean(), label='Unexplained variance', color='red')
plt.legend()
plt.title('Explained vs unexplained variance (rolling mean) - SPY1+SPY2')
plt.xlabel('Time')
plt.ylabel('Variance')
plt.tight_layout()
plt.show()
