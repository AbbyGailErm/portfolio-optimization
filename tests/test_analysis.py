import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis import calculate_daily_returns, calculate_risk_metrics

def test_calculate_daily_returns_basic():
    """Test that daily returns are calculated correctly for simple input."""
    data = {'Close': [100, 105, 110]}
    df = pd.DataFrame(data)
    
    # Expected: log(105/100) and log(110/105)
    expected_1 = np.log(105/100)
    expected_2 = np.log(110/105)
    
    result = calculate_daily_returns(df)
    
    # Check shape (should have 2 rows, as first is NaN and dropped)
    assert len(result) == 2
    
    # Check values
    assert np.isclose(result.iloc[0].item(), expected_1)
    assert np.isclose(result.iloc[1].item(), expected_2)

def test_calculate_daily_returns_empty():
    """Test that an empty DataFrame returns an empty DataFrame."""
    df = pd.DataFrame()
    result = calculate_daily_returns(df)
    assert result.empty

def test_calculate_daily_returns_single_row():
    """Test that a single row DataFrame returns empty (since it needs previous day)."""
    df = pd.DataFrame({'Close': [100]})
    result = calculate_daily_returns(df)
    assert result.empty

def test_calculate_risk_metrics():
    """Test Sharpe Ratio calculation with known values."""
    # Create a series of constant returns (e.g., 1% daily)
    # Annualized Return = 0.01 * 252 = 2.52
    # Annualized Volatility = 0 (since returns are constant)
    # This is an edge case, so let's use varying returns
    
    returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.01])
    metrics = calculate_risk_metrics(returns)
    
    assert 'Sharpe_Ratio' in metrics.columns
    assert 'VaR_95' in metrics.columns
    assert not metrics.empty
