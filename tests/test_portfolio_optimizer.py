import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.portfolio_optimizer import (
    TRADING_DAYS,
    calculate_annualized_metrics,
    portfolio_performance,
    negative_sharpe,
)


# ---------------------------------------------------------------------------
# calculate_annualized_metrics tests
# ---------------------------------------------------------------------------

def test_calculate_annualized_metrics_return():
    """Annualized return should equal mean daily return * TRADING_DAYS."""
    daily_returns = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005])
    annual_return, _ = calculate_annualized_metrics(daily_returns)
    expected = daily_returns.mean() * TRADING_DAYS
    assert annual_return == pytest.approx(expected)


def test_calculate_annualized_metrics_volatility():
    """Annualized volatility should equal std of daily returns * sqrt(TRADING_DAYS)."""
    daily_returns = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005])
    _, annual_volatility = calculate_annualized_metrics(daily_returns)
    expected = daily_returns.std() * np.sqrt(TRADING_DAYS)
    assert annual_volatility == pytest.approx(expected)


def test_calculate_annualized_metrics_constant_returns():
    """Constant daily returns should produce zero volatility."""
    daily_returns = pd.Series([0.01] * 10)
    _, annual_volatility = calculate_annualized_metrics(daily_returns)
    assert annual_volatility == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# portfolio_performance tests
# ---------------------------------------------------------------------------

def test_portfolio_performance_equal_weights():
    """With equal weights the portfolio return should be the mean asset return annualised."""
    mean_returns = pd.Series([0.001, 0.001])
    cov_matrix = pd.DataFrame([[0.0004, 0.0], [0.0, 0.0004]])
    weights = np.array([0.5, 0.5])
    std, ret = portfolio_performance(weights, mean_returns, cov_matrix)
    expected_return = mean_returns.mean() * TRADING_DAYS
    assert ret == pytest.approx(expected_return)
    assert std > 0


def test_portfolio_performance_volatility():
    """Verify volatility calculation with a known covariance matrix."""
    mean_returns = pd.Series([0.001, 0.002])
    cov_matrix = pd.DataFrame([[0.0001, 0.0], [0.0, 0.0001]])
    weights = np.array([0.5, 0.5])
    std, _ = portfolio_performance(weights, mean_returns, cov_matrix)
    # Variance = w^T * cov * w = 0.5^2*0.0001 + 0.5^2*0.0001 = 0.00005
    expected_std = np.sqrt(0.00005) * np.sqrt(TRADING_DAYS)
    assert std == pytest.approx(expected_std)


# ---------------------------------------------------------------------------
# negative_sharpe tests
# ---------------------------------------------------------------------------

def test_negative_sharpe_is_negative():
    """With positive expected returns the negative Sharpe ratio should be negative."""
    mean_returns = pd.Series([0.001, 0.002])
    cov_matrix = pd.DataFrame([[0.0001, 0.0], [0.0, 0.0001]])
    weights = np.array([0.5, 0.5])
    result = negative_sharpe(weights, mean_returns, cov_matrix)
    assert result < 0


def test_negative_sharpe_zero_risk_free():
    """With risk_free_rate=0 the negative Sharpe should equal -(return/volatility)."""
    mean_returns = pd.Series([0.001, 0.002])
    cov_matrix = pd.DataFrame([[0.0001, 0.0], [0.0, 0.0001]])
    weights = np.array([0.5, 0.5])
    std, ret = portfolio_performance(weights, mean_returns, cov_matrix)
    expected = -ret / std
    result = negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0)
    assert result == pytest.approx(expected)


def test_negative_sharpe_with_risk_free_rate():
    """A higher risk-free rate should increase (make less negative) the negative Sharpe."""
    mean_returns = pd.Series([0.001, 0.002])
    cov_matrix = pd.DataFrame([[0.0001, 0.0], [0.0, 0.0001]])
    weights = np.array([0.5, 0.5])
    sharpe_0 = negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0)
    sharpe_rfr = negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.01)
    assert sharpe_rfr > sharpe_0
