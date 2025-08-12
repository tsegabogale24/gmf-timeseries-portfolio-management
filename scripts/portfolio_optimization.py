import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------------------
# 1. Calculate Expected Returns
# ------------------------------
def get_expected_returns(tsla_forecast_prices, bnd_prices, spy_prices):
    # Ensure numeric data
    tsla_prices = pd.Series(tsla_forecast_prices).astype(float)
    bnd_prices = pd.Series(bnd_prices).astype(float)
    spy_prices = pd.Series(spy_prices).astype(float)

    # Calculate daily returns
    tsla_returns = tsla_prices.pct_change().dropna()
    tsla_annualized = tsla_returns.mean() * 252

    bnd_annualized = bnd_prices.pct_change().dropna().mean() * 252
    spy_annualized = spy_prices.pct_change().dropna().mean() * 252

    return {
        "TSLA": tsla_annualized,
        "BND": bnd_annualized,
        "SPY": spy_annualized
    }

# ------------------------------
# 2. Covariance Matrix
# ------------------------------
def get_covariance_matrix(tsla_hist_prices, bnd_prices, spy_prices):
    """
    Uses historical returns to compute covariance matrix.
    """
    returns_df = pd.DataFrame({
        "TSLA": tsla_hist_prices.pct_change(),
        "BND": bnd_prices.pct_change(),
        "SPY": spy_prices.pct_change()
    }).dropna()
    return returns_df.cov() * 252  # annualized

# ------------------------------
# 3. Portfolio Performance
# ------------------------------
def portfolio_performance(weights, expected_returns, cov_matrix):
    """
    Given weights, compute annualized return, volatility, Sharpe ratio.
    """
    weights = np.array(weights)
    port_return = np.dot(weights, expected_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = port_return / port_volatility
    return port_return, port_volatility, sharpe_ratio

# ------------------------------
# 4. Optimization
# ------------------------------
def max_sharpe_ratio(expected_returns, cov_matrix):
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix)

    def neg_sharpe(weights):
        return -portfolio_performance(weights, expected_returns, cov_matrix)[2]

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(neg_sharpe, num_assets * [1. / num_assets],
                      bounds=bounds, constraints=constraints)
    return result

def min_volatility(expected_returns, cov_matrix):
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix)

    def portfolio_vol(weights):
        return portfolio_performance(weights, expected_returns, cov_matrix)[1]

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(portfolio_vol, num_assets * [1. / num_assets],
                      bounds=bounds, constraints=constraints)
    return result

# ------------------------------
# 5. Efficient Frontier
# ------------------------------
def efficient_frontier(expected_returns, cov_matrix, num_points=100):
    results = {"returns": [], "volatilities": [], "sharpe": [], "weights": []}
    num_assets = len(expected_returns)

    for ret_target in np.linspace(min(expected_returns), max(expected_returns), num_points):
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - ret_target}
        ]
        bounds = tuple((0, 1) for _ in range(num_assets))
        result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                          num_assets * [1. / num_assets],
                          bounds=bounds, constraints=constraints)
        if result.success:
            vol = portfolio_performance(result.x, expected_returns, cov_matrix)[1]
            results["returns"].append(ret_target)
            results["volatilities"].append(vol)
            results["weights"].append(result.x)
            results["sharpe"].append(ret_target / vol)

    return pd.DataFrame(results)

# ------------------------------
# 6. Plot Efficient Frontier
# ------------------------------
def plot_efficient_frontier(frontier_df, max_sharpe_res, min_vol_res, expected_returns, cov_matrix, asset_labels):
    plt.figure(figsize=(10, 6))
    plt.plot(frontier_df['volatilities'], frontier_df['returns'], 'b--', label='Efficient Frontier')

    # Max Sharpe
    max_sharpe_perf = portfolio_performance(max_sharpe_res.x, expected_returns, cov_matrix)
    plt.scatter(max_sharpe_perf[1], max_sharpe_perf[0], marker='*', color='g', s=200, label='Max Sharpe')

    # Min Vol
    min_vol_perf = portfolio_performance(min_vol_res.x, expected_returns, cov_matrix)
    plt.scatter(min_vol_perf[1], min_vol_perf[0], marker='*', color='r', s=200, label='Min Volatility')

    plt.title("Efficient Frontier")
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.grid(True)
    plt.show()
