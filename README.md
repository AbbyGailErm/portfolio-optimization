# Time Series Forecasting & Portfolio Optimization

## 📌 Project Overview
**Client:** GMF Investments  
**Goal:** Forecast Tesla (TSLA) stock prices and construct an optimal investment portfolio.  
**Methods:** Time Series Analysis (ARIMA), Sentiment Analysis proxy (Volatility), Modern Portfolio Theory (MPT).

This project analyzes historical data for **Tesla (TSLA)**, **Vanguard Total Bond Market ETF (BND)**, and **S&P 500 ETF (SPY)** to recommend a risk-adjusted investment strategy. By combining high-growth assets with stable market indices, we aim to maximize returns while minimizing volatility.

---

## 📂 Project Structure

```text
portfolio-optimization/
├── data/                   # Raw and processed market data
│   ├── raw/                # Original data from YFinance
│   └── processed/          # Cleaned data for analysis
├── models/                 # Saved ARIMA models (.pkl)
├── notebooks/              # Jupyter Notebooks for analysis
│   ├── 01_eda.ipynb        # Exploratory Data Analysis & Preprocessing
│   ├── 02_arima.ipynb      # Time Series Forecasting (ARIMA)
│   ├── 03_optimization.ipynb # Portfolio Optimization (Monte Carlo Simulation)
│   └── 04_report.ipynb     # Final Client Report & Recommendations
├── src/                    # Source code modules
│   ├── data_loader.py      # Data fetching (YFinance)
│   ├── model_arima.py      # ARIMA training & evaluation logic
│   └── portfolio_optimizer.py # Optimization math (Sharpe Ratio, Efficient Frontier)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies