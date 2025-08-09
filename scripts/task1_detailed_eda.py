# task1_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import numpy as np

def basic_stats(df):
    """Calculate basic statistics of numeric columns."""
    stats = df.describe().T
    stats['skew'] = df.skew()
    stats['kurtosis'] = df.kurtosis()
    return stats

def plot_closing_price(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'])
    plt.title(f'{ticker} Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.grid(True)
    plt.show()

def plot_daily_returns(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Daily_Return'])
    plt.title(f'{ticker} Daily Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.grid(True)
    plt.show()

def plot_rolling_stats(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Rolling_Mean_50'], label='50-day Rolling Mean')
    plt.plot(df['Date'], df['Rolling_Std_20'], label='20-day Rolling Std Dev')
    plt.title(f'{ticker} Rolling Statistics')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.show()

def detect_outliers(df):
    """Detect outliers using z-score on daily returns."""
    returns = df['Daily_Return'].dropna()
    z_scores = (returns - returns.mean()) / returns.std()
    outliers = returns[np.abs(z_scores) > 3]
    return outliers

def adf_test(series, series_name="Series"):
    """Perform Augmented Dickey-Fuller test and print results."""
    print(f"\nADF Test on {series_name}")
    result = adfuller(series.dropna())
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    print(f"#Lags Used: {result[2]}")
    print(f"Number of Observations: {result[3]}")
    for key, val in result[4].items():
        print(f"Critical Value {key}: {val:.4f}")
    if result[1] < 0.05:
        print(f"Result: {series_name} is stationary (reject H0)")
    else:
        print(f"Result: {series_name} is non-stationary (fail to reject H0)")

def calculate_var(df, confidence_level=0.05):
    """Calculate historical Value at Risk (VaR) at specified confidence level."""
    daily_returns = df['Daily_Return'].dropna()
    var = daily_returns.quantile(confidence_level)
    return var

def calculate_sharpe_ratio(df, risk_free_rate=0):
    """Calculate Sharpe Ratio using daily returns."""
    daily_returns = df['Daily_Return'].dropna()
    excess_returns = daily_returns - risk_free_rate / 252  # Assuming annual risk-free rate, scaled to daily
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def run_eda(df, ticker):
    print(f"=== EDA for {ticker} ===\n")
    
    # Basic stats
    stats = basic_stats(df[['Close', 'Daily_Return']])
    print("Basic Statistics:")
    print(stats)
    
    # Plots
    plot_closing_price(df, ticker)
    plot_daily_returns(df, ticker)
    plot_rolling_stats(df, ticker)
    
    # Outlier detection
    outliers = detect_outliers(df)
    print(f"\nDetected {len(outliers)} outliers (|z| > 3) in daily returns.")
    if not outliers.empty:
        print(outliers)
    
    # Stationarity tests
    adf_test(df['Close'], 'Closing Price')
    adf_test(df['Daily_Return'], 'Daily Return')
    
    # Risk metrics
    var_95 = calculate_var(df, 0.05)
    sharpe = calculate_sharpe_ratio(df)
    print(f"\nValue at Risk (5% quantile): {var_95:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    
    print("\n====================\n")
    return stats, outliers, var_95, sharpe
