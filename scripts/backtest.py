import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Config / Paths
# ---------------------------
DATA_DIR = "../data/cleaned"          # expects cleaned CSVs from Task 1 (e.g., TSLA_cleaned.csv)
RESULTS_DIR = "../results"
BT_DIR = os.path.join(RESULTS_DIR, "backtests")
os.makedirs(BT_DIR, exist_ok=True)

# ---------------------------
# Helpers: Loading & Prep
# ---------------------------
def load_prices(tickers, start_date, end_date):
    """Load Adj Close from cleaned CSVs produced in Task 1 and align by date."""
    prices = []
    for t in tickers:
        path = os.path.join(DATA_DIR, f"{t}_cleaned.csv")
        # Skip rows 1 and 2, keep header row 0 as column names
        df = pd.read_csv(path, skiprows=[1, 2], parse_dates=[0])
        
        # Rename first column to 'Date' for clarity
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        
        # Set Date as index and sort
        df = df.set_index("Date").sort_index()
        
        # Check for 'Adj Close' column
        if "Adj Close" not in df.columns:
            raise ValueError(f"{t}: 'Adj Close' column missing in {path}")
        
        # Extract Adj Close series named by ticker
        ser = df["Adj Close"].rename(t)
        prices.append(ser)
    
    # Combine all tickers into one DataFrame, drop rows where all tickers have NaN
    prices = pd.concat(prices, axis=1).dropna(how="all")
    
    # Filter by backtest window
    prices = prices.loc[(prices.index >= pd.to_datetime(start_date)) & (prices.index <= pd.to_datetime(end_date))]
    
    # Forward/backward fill gaps
    prices = prices.fillna(method="ffill").fillna(method="bfill")
    return prices




def daily_returns_from_prices(prices):
    return prices.pct_change().dropna(how="all")

# ---------------------------
# Portfolio Simulation
# ---------------------------
def simulate_portfolio_buy_and_hold(prices, weights, initial_capital=1_000_000.0):
    """
    Buy-and-hold simulation:
    - Allocate initial_capital using target weights at the first date's close
    - Hold constant shares; no rebalancing
    """
    w = pd.Series(weights).reindex(prices.columns).fillna(0.0)
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    start_price = prices.iloc[0]
    shares = (initial_capital * w) / start_price.replace(0, np.nan)
    # portfolio value over time
    port_val = (prices * shares).sum(axis=1)
    port_ret = port_val.pct_change().fillna(0.0)
    return port_val, port_ret

def simulate_portfolio_monthly_rebalance(prices, weights, initial_capital=1_000_000.0):
    """
    Monthly rebalance at month-end close to target weights.
    Uses daily Adj Close; ignores transaction costs/slippage.
    """
    w = pd.Series(weights).reindex(prices.columns).fillna(0.0)
    w = w / (w.sum() if w.sum() != 0 else 1.0)

    # Rebalance dates = last trading day of each month within window
    month_ends = prices.resample("M").last().index
    # Start
    dates = prices.index
    port_val = pd.Series(index=dates, dtype="float64")
    shares = None
    current_val = initial_capital

    for i, d in enumerate(dates):
        if (d in month_ends) or (i == 0):  # rebalance at start and month-ends
            px = prices.loc[d]
            # update shares to match target weights
            alloc = current_val * w
            shares = alloc / px.replace(0, np.nan)
        # compute portfolio value for the day
        current_val = float((prices.loc[d] * shares).sum())
        port_val.loc[d] = current_val

    port_ret = port_val.pct_change().fillna(0.0)
    return port_val, port_ret

# ---------------------------
# Benchmark (60/40 SPY/BND)
# ---------------------------
def simulate_benchmark(prices, rebalance="monthly"):
    """
    60% SPY / 40% BND benchmark (monthly rebalance by default).
    If SPY/BND not both present, raises.
    """
    for req in ["SPY", "BND"]:
        if req not in prices.columns:
            raise ValueError(f"Benchmark requires {req} in your tickers.")
    weights_6040 = {"SPY": 0.60, "BND": 0.40}
    bench_prices = prices[["SPY", "BND"]].copy()
    if rebalance == "hold":
        b_val, b_ret = simulate_portfolio_buy_and_hold(bench_prices, weights_6040)
    else:
        b_val, b_ret = simulate_portfolio_monthly_rebalance(bench_prices, weights_6040)
    return b_val, b_ret

# ---------------------------
# Metrics
# ---------------------------
def perf_metrics(returns, rf_annual=0.0, freq=252):
    """
    returns: pd.Series of daily returns
    rf_annual: annual risk-free (e.g., 0.03 for 3%)
    """
    r = returns.dropna()
    if len(r) == 0:
        return {"total_return": np.nan, "cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}

    total_return = (1 + r).prod() - 1
    ann_factor = freq / max(len(r), 1)
    cagr = (1 + total_return) ** ann_factor - 1

    vol = r.std() * np.sqrt(freq)
    rf_daily = rf_annual / freq
    sharpe = (r.mean() - rf_daily) / (r.std() + 1e-12) * np.sqrt(freq)

    # Max drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    max_dd = dd.min()

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
    }

# ---------------------------
# Plotting & Saving
# ---------------------------
def plot_cumulative(strategy_ret, bench_ret, title, savepath):
    plt.figure(figsize=(12,6))
    strat_cum = (1 + strategy_ret).cumprod()
    bench_cum = (1 + bench_ret).cumprod()
    plt.plot(strat_cum.index, strat_cum.values, label="Strategy")
    plt.plot(bench_cum.index, bench_cum.values, label="Benchmark 60/40")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=130)
    plt.close()

def save_series(df_or_ser, filename):
    path = os.path.join(BT_DIR, filename)
    if isinstance(df_or_ser, pd.Series):
        df_or_ser.to_frame(name="value_or_return").to_csv(path, index=True)
    else:
        df_or_ser.to_csv(path, index=True)
    return path

# ---------------------------
# Main Backtest Runner
# ---------------------------
def run_backtest(
    tickers,
    strategy_weights,
    start_date="2024-08-01",
    end_date="2025-07-31",
    strategy_rebalance="monthly",   # "monthly" or "hold"
    benchmark_rebalance="monthly",  # "monthly" or "hold"
    risk_free_annual=0.0
):
    """
    tickers: list like ["TSLA","SPY","BND"]
    strategy_weights: dict like {"TSLA":0.30,"SPY":0.50,"BND":0.20}  (from Task 4)
    """
    # 1) Load prices
    prices = load_prices(tickers, start_date, end_date)

    # 2) Strategy sim
    if strategy_rebalance == "hold":
        s_val, s_ret = simulate_portfolio_buy_and_hold(prices, strategy_weights)
    else:
        s_val, s_ret = simulate_portfolio_monthly_rebalance(prices, strategy_weights)

    # 3) Benchmark sim (60/40 SPY/BND)
    b_val, b_ret = simulate_benchmark(prices, rebalance=benchmark_rebalance)

    # Align (just in case)
    s_ret, b_ret = s_ret.align(b_ret, join="inner")

    # 4) Metrics
    s_metrics = perf_metrics(s_ret, rf_annual=risk_free_annual)
    b_metrics = perf_metrics(b_ret, rf_annual=risk_free_annual)

    # 5) Outputs
    plot_path = os.path.join(BT_DIR, "cumulative_strategy_vs_benchmark.png")
    plot_cumulative(s_ret, b_ret, "Strategy vs Benchmark (Growth of $1)", plot_path)

    # Save time series
    save_series(s_val, "strategy_portfolio_value.csv")
    save_series(b_val, "benchmark_portfolio_value.csv")
    save_series(s_ret.rename("strategy_ret"), "strategy_daily_returns.csv")
    save_series(b_ret.rename("benchmark_ret"), "benchmark_daily_returns.csv")

    # Save metrics
    metrics_df = pd.DataFrame([{"Portfolio": "Strategy", **s_metrics},
                               {"Portfolio": "Benchmark 60/40", **b_metrics}])
    metrics_csv = save_series(metrics_df, "backtest_metrics.csv")

    # Console summary
    print("\n=== Backtest Summary ===")
    print(f"Window: {s_ret.index.min().date()} → {s_ret.index.max().date()}")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved plot: {plot_path}")
    print(f"Saved metrics CSV: {metrics_csv}")
    print(f"Data dir: {BT_DIR}")

    # Return dict for programmatic use
    return {
        "strategy_returns": s_ret,
        "benchmark_returns": b_ret,
        "strategy_metrics": s_metrics,
        "benchmark_metrics": b_metrics,
        "plot": plot_path,
        "metrics_csv": metrics_csv,
    }

# ---------------------------
# Example run (edit weights!)
# ---------------------------
if __name__ == "__main__":

    strat_weights = {"TSLA": 0.30, "SPY": 0.50, "BND": 0.20}

    results = run_backtest(
        tickers=["TSLA", "SPY", "BND"],
        strategy_weights=strat_weights,
        start_date="2024-08-01",
        end_date="2025-07-31",
        strategy_rebalance="monthly",     
        benchmark_rebalance="monthly",    
        risk_free_annual=0.0             
    )
