import numpy as np
import pandas as pd
import scipy.optimize as optimize
from typing import Tuple

TRADING_DAYS = 252

def calculate_annualized_metrics(
    returns: pd.Series,
) -> Tuple[float, float]:
    """Calculates annualized return and volatility from daily returns.

    Args:
        returns: A pandas Series (or DataFrame) of daily returns.

    Returns:
        A tuple of ``(annual_return, annual_volatility)`` where each element
        is scaled to an annual basis using :data:`TRADING_DAYS`.
    """
    # Annual Return: Average daily return * TRADING_DAYS
    annual_return = returns.mean() * TRADING_DAYS
    
    # Annual Volatility: Standard deviation * square root of TRADING_DAYS
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    
    return annual_return, annual_volatility

def portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> Tuple[float, float]:
    """Calculates the expected volatility and return of a portfolio.

    The portfolio return is the weighted sum of individual mean returns
    annualized by :data:`TRADING_DAYS`.  The portfolio volatility is derived
    from the covariance matrix:

    .. math::

        \\sigma_p = \\sqrt{w^T \\Sigma w \\cdot T}

    Args:
        weights: Array of asset weights (must sum to 1).
        mean_returns: Series of mean daily returns for each asset.
        cov_matrix: Covariance matrix of daily returns.

    Returns:
        A tuple of ``(volatility, return)`` both annualized.
    """
    returns = np.sum(mean_returns * weights) * TRADING_DAYS
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(TRADING_DAYS)
    return std, returns

def negative_sharpe(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> float:
    """Returns the negative Sharpe ratio for a given set of portfolio weights.

    Optimization algorithms minimize functions; returning the *negative* Sharpe
    ratio allows a minimizer to effectively *maximize* the Sharpe ratio.

    Args:
        weights: Array of asset weights (must sum to 1).
        mean_returns: Series of mean daily returns for each asset.
        cov_matrix: Covariance matrix of daily returns.
        risk_free_rate: Annualised risk-free rate (default ``0.0``).

    Returns:
        The negative Sharpe ratio: ``-(return - risk_free_rate) / volatility``.
    """
    p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def optimize_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> optimize.OptimizeResult:
    """Finds the portfolio weights that maximize the Sharpe ratio.

    Uses the SLSQP method with the constraint that weights sum to 1 and the
    bound that each weight lies in ``[0, 1]`` (no short-selling).

    Args:
        mean_returns: Series of mean daily returns for each asset.
        cov_matrix: Covariance matrix of daily returns.

    Returns:
        The :class:`~scipy.optimize.OptimizeResult` from
        :func:`scipy.optimize.minimize`.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    
    # Constraints: Weights must sum to 1 (100%)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: Each weight must be between 0 and 1 (0% to 100% allocation)
    # This prevents "shorting" stocks (negative weights)
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # Initial Guess: Equal distribution (e.g., 33% each)
    initial_guess = num_assets * [1. / num_assets,]
    
    # The Optimization Algorithm (SLSQP is standard for this)
    result = optimize.minimize(negative_sharpe, initial_guess, args=args,
                               method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result