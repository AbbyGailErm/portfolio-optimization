import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def calculate_daily_returns(df):
    """Calculates logarithmic returns for better statistical properties."""
    # Use log returns (standard in finance) or simple pct_change
    # Handling both DataFrame and Series
    return np.log(df / df.shift(1)).dropna()

def check_stationarity(timeseries):
    """
    Performs Augmented Dickey-Fuller test.
    """
    # handle potential missing values at start
    timeseries = timeseries.dropna()
    
    result = adfuller(timeseries, autolag='AIC')
    index = ['Test Statistic', 'p-value', '# Lags Used', '# Observations Used']
    out = pd.Series(result[0:4], index=index)
    
    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
        
    out['Is Stationary (5%)'] = result[1] < 0.05
    return out

def calculate_risk_metrics(returns_df):
    """
    Calculates VaR (95%) and Sharpe Ratio.
    """
    metrics = {}
    
    # Value at Risk (VaR) at 95% confidence level
    var_95 = returns_df.quantile(0.05)
    
    # Annualized Sharpe Ratio (assuming 252 trading days)
    mean_return = returns_df.mean() * 252
    std_dev = returns_df.std() * np.sqrt(252)
    sharpe_ratio = mean_return / std_dev
    
    metrics['VaR_95'] = var_95
    metrics['Sharpe_Ratio'] = sharpe_ratio
    
    return pd.DataFrame(metrics)

def detect_outliers(returns_df, threshold=3):
    """
    Detects outliers > 3 standard deviations.
    """
    z_scores = np.abs((returns_df - returns_df.mean()) / returns_df.std())
    outliers = returns_df[z_scores > threshold]
    return outliers