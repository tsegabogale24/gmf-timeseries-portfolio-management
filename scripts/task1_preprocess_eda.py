# gmf_eda.py
"""
GMF Investments - Task 1 Modular EDA Functions
Author: Your Name
Date: 2025-08-09
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from scipy.stats import zscore
import os

plt.style.use("seaborn-v0_8")
sns.set_theme()

# -------------------------------
# 1. Data Fetching
# -------------------------------
def fetch_data(tickers, start_date, end_date, save_raw=True):
    """Fetch historical data for given tickers using yfinance (one by one to avoid MultiIndex)."""
    data = {}
    for ticker in tickers:
        # Fetch with standard OHLCV + 'Adj Close' and simple columns
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            group_by="column",
            progress=False,
        )
        df.reset_index(inplace=True)
        df["Ticker"] = ticker
        if save_raw:
            # Save under project data folder regardless of CWD
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
            os.makedirs(data_dir, exist_ok=True)
            df.to_csv(os.path.join(data_dir, f"{ticker}_raw.csv"), index=False)
        data[ticker] = df
    return data


# -------------------------------
# 2. Preprocessing
# -------------------------------
def preprocess_data(df):
    # Robustly materialize the datetime index as a 'Date' column if needed
    if "Date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.rename_axis("Date").reset_index()

    # Normalize the date column name
    if "Date" not in df.columns:
        if "index" in df.columns:
            df.rename(columns={"index": "Date"}, inplace=True)
        elif "date" in df.columns:
            df.rename(columns={"date": "Date"}, inplace=True)
        else:
            # Try to infer from columns or index
            datetime_like_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
            if datetime_like_cols:
                df.rename(columns={datetime_like_cols[0]: "Date"}, inplace=True)
            else:
                # Last resort: if the index looks like dates, promote it to a column
                index_as_dt = pd.to_datetime(df.index, errors="coerce")
                # If index looks datetime-like for at least some rows, promote it
                if getattr(index_as_dt, "notna", pd.Series([False]*len(df))).any():
                    df = df.copy()
                    df.insert(0, "Date", index_as_dt)
                    df.reset_index(drop=True, inplace=True)

    if "Date" not in df.columns:
        raise ValueError(f"'Date' column not found. Columns: {df.columns}")

    df["Date"] = pd.to_datetime(df["Date"])  # ensure dtype

    # Handle missing Adj Close
    if "Adj Close" not in df.columns:
        if "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        else:
            raise ValueError(f"No 'Adj Close' or 'Close' column found. Columns: {df.columns}")

    # Fill missing values
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # Feature Engineering
    df["Daily_Return"] = df["Adj Close"].pct_change()
    df["Rolling_Std_20"] = df["Daily_Return"].rolling(window=20).std()
    df["Rolling_Mean_50"] = df["Adj Close"].rolling(window=50).mean()

    return df



# -------------------------------
# 3. Risk Metrics
# -------------------------------
def calculate_risk_metrics(df, ticker):
    """Calculate VaR, Sharpe Ratio, and stationarity tests."""
    var_95 = np.percentile(df["Daily_Return"].dropna(), 5)
    sharpe_ratio = (df["Daily_Return"].mean() / df["Daily_Return"].std()) * np.sqrt(252)
    
    adf_close = adfuller(df["Adj Close"].dropna())
    adf_return = adfuller(df["Daily_Return"].dropna())
    
    return {
        "Asset": ticker,
        "Mean Return": df["Daily_Return"].mean(),
        "Volatility": df["Daily_Return"].std(),
        "VaR 95%": var_95,
        "Sharpe Ratio": sharpe_ratio,
        "Stationary (Close)": adf_close[1] < 0.05,
        "Stationary (Returns)": adf_return[1] < 0.05
    }

# -------------------------------
# 4. Visualization
# -------------------------------
def plot_price_trend(df, ticker, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Adj Close"], label="Adj Close")
    plt.plot(df["Date"], df["Rolling_Mean_50"], label="50-day MA", linestyle="--")
    plt.title(f"{ticker} - Closing Price & Rolling Mean")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.savefig(f"{save_dir}/{ticker}_price_trend.png")
    plt.close()

def plot_returns_hist(df, ticker, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Daily_Return"].dropna(), bins=50, kde=True)
    plt.title(f"{ticker} - Daily Returns Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.savefig(f"{save_dir}/{ticker}_returns_hist.png")
    plt.close()

def plot_volatility(df, ticker, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Rolling_Std_20"])
    plt.title(f"{ticker} - 20-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.savefig(f"{save_dir}/{ticker}_rolling_volatility.png")
    plt.close()

def plot_outliers(df, ticker, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    s = df["Daily_Return"]
    # Compute z-score aligned to the full index to avoid length mismatch
    z = (s - s.mean()) / s.std(ddof=0)
    df["Return_Z"] = z
    outliers = df[np.abs(df["Return_Z"]) > 3]
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Daily_Return"], label="Daily Return")
    plt.scatter(outliers["Date"], outliers["Daily_Return"], color="red", label="Outliers")
    plt.title(f"{ticker} - Outliers in Daily Returns")
    plt.legend()
    plt.savefig(f"{save_dir}/{ticker}_outliers.png")
    plt.close()

# -------------------------------
# 5. Full Workflow
# -------------------------------
def run_full_eda(tickers, start_date, end_date):
    """Fetch, preprocess, analyze, and plot EDA for given tickers."""
    raw_data = fetch_data(tickers, start_date, end_date)
    summaries = []
    for ticker, df in raw_data.items():
        # Handle potential MultiIndex columns defensively, preserving 'Date' if it's a column
        if isinstance(df.columns, pd.MultiIndex):
            try:
                preserved_date = None
                if "Date" in df.columns:
                    preserved_date = df["Date"].copy()
                    df_wo_date = df.drop(columns=["Date"])
                else:
                    df_wo_date = df

                # Prefer selecting the ticker level if present
                if ticker in df_wo_date.columns.get_level_values(-1):
                    df = df_wo_date.xs(ticker, axis=1, level=-1)
                else:
                    df = df_wo_date
                    df.columns = df.columns.get_level_values(0)

                if preserved_date is not None and "Date" not in df.columns:
                    df.insert(0, "Date", preserved_date.values)
            except Exception:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        df = preprocess_data(df)
        summary = calculate_risk_metrics(df, ticker)
        summaries.append(summary)
        
        # Plots
        plot_price_trend(df, ticker)
        plot_returns_hist(df, ticker)
        plot_volatility(df, ticker)
        plot_outliers(df, ticker)
    
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv("../data/summary_metrics.csv", index=False)
    return summary_df
