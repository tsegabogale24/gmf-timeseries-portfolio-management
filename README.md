# Financial Time Series Forecasting and Portfolio Optimization

## Project Overview

This repository contains a complete workflow for financial data preprocessing, time series forecasting, portfolio optimization, and strategy backtesting using historical market data for TSLA, BND, and SPY. The project covers tasks from raw data extraction to evaluation of investment strategies, leveraging both classical and machine learning models.

---

## Folder Structure

.
├── data/
│ ├── raw/ # Raw downloaded CSV files from Yahoo Finance
│ ├── cleaned/ # Cleaned and preprocessed CSV files ready for analysis
│
├── notebooks/
│ ├── task1_data_preprocessing.ipynb # Data loading, cleaning, and exploratory analysis
│ ├── task2_forecasting_models.ipynb # Model development: ARIMA & LSTM
│ ├── task3_forecast_analysis.ipynb # Forecast generation and visualization
│ ├── task4_portfolio_optimization.ipynb # Portfolio optimization and Efficient Frontier plotting
│ └── task5_backtesting.ipynb # Strategy backtesting and comparison with benchmark
│
├── results/
│ ├── eda/ # EDA visualizations and reports
│ ├── forecasts/ # Forecast plots and evaluation metrics
│ ├── portfolio/ # Efficient Frontier charts and portfolio weights
│ ├── backtests/ # Backtest performance graphs and summary CSVs
│ └── logs/ # Logs of model training and backtest runs
│
├── scripts/
│ ├── data_preprocessing.py # Functions for loading, cleaning, and preparing data
│ ├── forecasting.py # Functions and classes for ARIMA and LSTM models
│ ├── portfolio.py # Portfolio optimization utilities
│ ├── backtest.py # Backtesting engine and metrics calculations
│ └── utils.py # Helper functions (e.g., plotting, metrics)
│
├── requirements.txt # Python package dependencies
├── README.md # Project overview and instructions (this file)
└── LICENSE # License file (if applicable)

yaml
Copy
Edit

---

## Description of Key Folders & Files

### `data/`
- **raw/**: Contains original downloaded CSV files from Yahoo Finance for TSLA, BND, and SPY.
- **cleaned/**: Contains processed CSV files after cleaning, missing value handling, and feature engineering ready for modeling.

### `notebooks/`
Interactive Jupyter notebooks used for exploratory data analysis, model training, forecasting, portfolio optimization, and backtesting, organized by task.

### `results/`
Outputs and visualizations generated during the project including plots for EDA, forecasting results, portfolio optimization charts, and backtest performance summaries.

### `scripts/`
Modular Python scripts with reusable functions and classes to automate tasks such as data loading, model building, portfolio calculations, and backtesting simulations.

### `requirements.txt`
Lists all Python dependencies needed to run the project smoothly.

---

## How to Use This Repository

1. **Install dependencies:**

```bash
pip install -r requirements.txt
Download raw data:

Download historical price data from Yahoo Finance for the assets and place the CSV files into data/raw/.

Run preprocessing:

Use scripts/data_preprocessing.py or the Task 1 notebook to clean and prepare the data saved into data/cleaned/.

Build and evaluate models:

Run Task 2 notebooks or scripts to train and evaluate ARIMA and LSTM forecasting models.

Generate forecasts:

Use Task 3 notebooks to create future price forecasts and visualize predictions.

Optimize portfolios:

Use Task 4 notebooks/scripts to compute efficient portfolios and analyze risk-return tradeoffs.

Backtest strategies:

Run Task 5 notebooks to simulate strategy performance against benchmark portfolios.
