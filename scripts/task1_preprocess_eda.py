"""
Usage:
     import run_full_eda_enhanced
    run_full_eda_enhanced(['TSLA','BND','SPY'], '2015-01-01', '2025-08-11')

Outputs:
- cleaned CSVs under ./data/cleaned
- plots under ./results/plots
- consolidated metrics CSV at ./data/summary_metrics_enhanced.csv

"""

import os
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

plt.style.use("seaborn-v0_8")
sns.set_theme()

RESULTS_DIR = os.path.abspath("../results")
DATA_DIR = os.path.abspath("../data")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
for d in [RESULTS_DIR, DATA_DIR, CLEANED_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# -------------------------------
# Data fetching (robust)
# -------------------------------
def fetch_data(tickers, start_date, end_date, auto_adjust=False):
    data = {}
    for t in tickers:
        df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=auto_adjust)
        if df.empty:
            print(f"Warning: no data for {t} (empty DataFrame)")
            continue
        df = df.reset_index()
        df['Ticker'] = t
        data[t] = df
    return data

# -------------------------------
# Preprocessing + feature engineering
# -------------------------------

def preprocess_df(df):
    df = df.copy()
    # Normalize Date
    if 'Date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.rename_axis('Date').reset_index()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        # try to infer
        df.insert(0, 'Date', pd.to_datetime(df.index))
        df.reset_index(drop=True, inplace=True)

    # Ensure Adj Close exists
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            raise ValueError('No Close or Adj Close column found')

    # Set index to Date for time series ops
    df.set_index('Date', inplace=True)

    # Fill missing data
    df = df.sort_index()
    df[['Adj Close']] = df[['Adj Close']].fillna(method='ffill').fillna(method='bfill')

    # Feature engineering
    df['Daily_Return'] = df['Adj Close'].pct_change()
    df['Rolling_Std_20'] = df['Daily_Return'].rolling(window=20, min_periods=1).std()
    df['Rolling_Mean_50'] = df['Adj Close'].rolling(window=50, min_periods=1).mean()
    df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

    return df

# -------------------------------
# Additional analysis utilities
# -------------------------------

def save_cleaned(df, ticker):
    path = os.path.join(CLEANED_DIR, f"{ticker}_cleaned.csv")
    df.to_csv(path)
    return path


def plot_normalized_comparison(data_dict, start_date=None, end_date=None, savepath=None):
    plt.figure(figsize=(12,6))
    for t, df in data_dict.items():
        ser = df['Adj Close'].copy()
        if start_date:
            ser = ser[ser.index >= pd.to_datetime(start_date)]
        if end_date:
            ser = ser[ser.index <= pd.to_datetime(end_date)]
        norm = ser / ser.iloc[0]
        plt.plot(norm.index, norm, label=t)
    plt.title('Normalized Price (Start = 1.0)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    if savepath is None:
        savepath = os.path.join(PLOTS_DIR, 'normalized_comparison.png')
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    return savepath


def compute_returns_df(data_dict):
    rets = {}
    for t, df in data_dict.items():
        rets[t] = df['Daily_Return']
    rets_df = pd.DataFrame(rets)
    return rets_df


def plot_correlation_matrix(returns_df, savepath=None):
    corr = returns_df.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0)
    plt.title('Correlation Matrix of Daily Returns')
    if savepath is None:
        savepath = os.path.join(PLOTS_DIR, 'returns_correlation.png')
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    return savepath, corr


def seasonal_decompose_plot(df, ticker, period=252, model='additive'):
    outpath = os.path.join(PLOTS_DIR, f'{ticker}_seasonal_decompose.png')
    try:
        series = df['Adj Close'].dropna()
        if len(series) < period * 2:
            series = df['Daily_Return'].dropna()
            period = 5 * 1  # weekly-ish for returns if needed
        res = seasonal_decompose(series, period=period, model=model, extrapolate_trend='freq')
        fig = res.plot()
        fig.set_size_inches(10,8)
        fig.suptitle(f'{ticker} Seasonal Decompose (period={period})')
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
        return outpath
    except Exception as e:
        print(f"Seasonal decomposition failed for {ticker}: {e}")
        return None


def top_extreme_days(df, n=5):
    s = df['Daily_Return'].dropna()
    top_up = s.nlargest(n)
    top_down = s.nsmallest(n)
    return top_up, top_down

# -------------------------------
# Risk & summary
# -------------------------------

def historical_var(returns, alpha=0.05):
    return returns.dropna().quantile(alpha)

def parametric_var(returns, alpha=0.05):
    mu = returns.mean()
    sigma = returns.std()
    from scipy.stats import norm
    z = norm.ppf(alpha)
    return mu + z * sigma

def sharpe_ratio(returns, risk_free_rate=0.0, annualization=252):
    excess = returns - risk_free_rate / annualization
    return (excess.mean() / excess.std()) * np.sqrt(annualization)

def adf_stationarity(series, signif=0.05):
    res = adfuller(series.dropna())
    return {'p_value': res[1], 'stationary': res[1] < signif, 'test_stat': res[0]}

def make_stationary(series, max_diff=3, signif=0.05):
    """
    Apply differencing up to max_diff times until series becomes stationary
    based on ADF test p-value < signif.
    Returns:
      stationary_series (pd.Series),
      number_of_differences (int),
      adf_result (dict)
    """
    diff_series = series.dropna()
    n_diff = 0
    adf_res = None

    while n_diff <= max_diff:
        adf_res = adf_stationarity(diff_series, signif=signif)
        if adf_res['stationary']:
            break
        diff_series = diff_series.diff().dropna()
        n_diff += 1

    if not adf_res['stationary']:
        print(f"Warning: Series is non-stationary after {max_diff} differences.")

    return diff_series, n_diff, adf_res

def generate_summary_metrics(data_dict):
    rows = []
    for t, df in data_dict.items():
        r = df['Daily_Return'].dropna()
        if r.empty:
            continue

        # Close price stationarity with differencing
        close_series = df['Adj Close']
        stationary_close, n_diff_close, adf_close = make_stationary(close_series)

        # Daily_Returns stationarity as is (usually returns are stationary)
        stationary_ret, n_diff_ret, adf_ret = make_stationary(r)

        row = {
            'Ticker': t,
            'Start': df.index.min().strftime('%Y-%m-%d'),
            'End': df.index.max().strftime('%Y-%m-%d'),
            'Mean Return (daily)': r.mean(),
            'Volatility (daily std)': r.std(),
            'VaR_95_hist': historical_var(r, 0.05),
            'VaR_95_param': parametric_var(r, 0.05),
            'Sharpe (ann)': sharpe_ratio(r),
            'ADF_Close_pval': adf_close['p_value'],
            'ADF_Returns_pval': adf_ret['p_value'],
            'Close_stationary': adf_close['stationary'],
            'Returns_stationary': adf_ret['stationary'],
            'Close_diff_order': n_diff_close,
            'Returns_diff_order': n_diff_ret
        }
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(DATA_DIR, 'summary_metrics_enhanced.csv')
    summary_df.to_csv(summary_path, index=False)
    return summary_df, summary_path

# -------------------------------
# Full workflow
# -------------------------------

def run_full_eda(tickers, start_date, end_date, save_clean=True):
    raw = fetch_data(tickers, start_date, end_date)
    cleaned = {}
    for t, df in raw.items():
        try:
            pdf = preprocess_df(df)
            cleaned[t] = pdf
            if save_clean:
                save_cleaned(pdf, t)
        except Exception as e:
            print(f"Preprocessing failed for {t}: {e}")

    # Combined analyses
    plot_normalized_comparison(cleaned, start_date, end_date)
    rets = compute_returns_df(cleaned)
    corr_path, corr = plot_correlation_matrix(rets)

    for t, df in cleaned.items():
        seasonal_decompose_plot(df, t)

    extremes = {}
    for t, df in cleaned.items():
        up, down = top_extreme_days(df, n=5)
        extremes[t] = {'top_up': up, 'top_down': down}

    primary = tickers[0]
    if primary in cleaned:
        df = cleaned[primary]
        plt.figure(figsize=(12,6))
        plt.plot(df.index, df['Adj Close'], label='Adj Close')
        up_idx = extremes[primary]['top_up'].index
        down_idx = extremes[primary]['top_down'].index
        plt.scatter(up_idx, df.loc[up_idx]['Adj Close'], marker='^', color='g', label='Top Up')
        plt.scatter(down_idx, df.loc[down_idx]['Adj Close'], marker='v', color='r', label='Top Down')
        plt.title(f'{primary} Close with Extreme Days Annotated')
        plt.legend()
        path = os.path.join(PLOTS_DIR, f'{primary}_extremes_annotated.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    summary_df, summary_path = generate_summary_metrics(cleaned)

    print('\nEDA complete.')
    print(f'Cleaned files saved to: {CLEANED_DIR}')
    print(f'Plots saved to: {PLOTS_DIR}')
    print(f'Summary metrics CSV: {summary_path}')

    return {
        'cleaned': cleaned,
        'summary_df': summary_df,
        'plots_dir': PLOTS_DIR,
        'summary_csv': summary_path,
        'correlation_matrix': corr if 'corr' in locals() else None,
        'extremes': extremes
    }


if __name__ == '__main__':
    tickers = ['TSLA', 'BND', 'SPY']
    start = '2015-01-01'
    end = datetime.now().strftime('%Y-%m-%d')
    run_full_eda(tickers, start, end)
