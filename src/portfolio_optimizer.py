import numpy as np
import pandas as pd
import scipy.optimize as optimize

def calculate_annualized_metrics(returns):
    """
    Calculates the annualized return and volatility for a set of returns.
    Assumes 252 trading days in a year.
    """
    # Annual Return: Average daily return * 252 days
    annual_return = returns.mean() * 252
    
    # Annual Volatility: Standard deviation * square root of 252
    annual_volatility = returns.std() * np.sqrt(252)
    
    return annual_return, annual_volatility

def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculates the expected return and volatility of a portfolio given specific weights.
    
    Math:
    - Return = Sum(Weight * Mean_Return)
    - Variance = Weight_Transpose * Covariance_Matrix * Weight
    - Volatility = Sqrt(Variance)
    """
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Calculates the negative Sharpe ratio.
    Optimization algorithms try to MINIMIZE functions. 
    Since we want to MAXIMIZE Sharpe, we minimize Negative Sharpe.
    """
    p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def optimize_portfolio(mean_returns, cov_matrix):
    """
    Finds the portfolio weights that maximize the Sharpe Ratio.
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