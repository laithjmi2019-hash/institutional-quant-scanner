import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the covariance matrix of returns.
    """
    return returns_df.cov() * 252 # Annualized covariance

def calculate_portfolio_risk(weights: np.array, cov_matrix: np.array) -> float:
    """
    Calculates portfolio risk (standard deviation).
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def calculate_risk_contribution(weights: np.array, cov_matrix: np.array) -> np.array:
    """
    RC_i = w_i * (Cov_Matrix * w)_i / sigma_p
    """
    port_risk = calculate_portfolio_risk(weights, cov_matrix)
    marginal_risk_contrib = np.dot(cov_matrix, weights)
    risk_contrib = np.multiply(weights, marginal_risk_contrib) / port_risk
    return risk_contrib

def risk_parity_objective(weights: np.array, cov_matrix: np.array) -> float:
    """
    Objective function for Risk Parity optimization (Equal Risk Contribution).
    Minimizes the variance of the risk contributions.
    """
    risk_contrib = calculate_risk_contribution(weights, cov_matrix)
    # Mean risk contribution
    mean_rc = np.mean(risk_contrib)
    # Sum of squared errors from mean RC
    sum_sq_errors = np.sum(np.square(risk_contrib - mean_rc))
    return sum_sq_errors

def optimize_portfolio(returns_df: pd.DataFrame) -> dict:
    """
    Generates optimal weights using Equal Risk Contribution (Risk Parity).
    """
    if returns_df.empty or returns_df.shape[1] < 2:
        return {col: 1.0/returns_df.shape[1] for col in returns_df.columns}
        
    cov_matrix = calculate_covariance_matrix(returns_df).values
    num_assets = len(cov_matrix)
    
    # Initial guess (equal weights)
    init_weights = np.array(num_assets * [1. / num_assets])
    
    # Constraints (weights sum to 1, all weights >= 0)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Optimization
    opt_result = minimize(
        risk_parity_objective,
        init_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = opt_result.x
    
    return {returns_df.columns[i]: round(optimal_weights[i], 4) for i in range(num_assets)}
