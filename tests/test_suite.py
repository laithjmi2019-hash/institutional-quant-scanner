import sys
import os
import pytest
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.business_quality import compute_business_quality
from engines.dynamic_valuation import compute_dcf_valuation
from engines.technical_structure import compute_technical_structure
from engines.portfolio_engine import calculate_portfolio_risk, calculate_risk_contribution
from engines.split_adjuster import adjust_for_splits

def test_business_quality_format():
    res = compute_business_quality("AAPL")
    assert 'roic' in res
    assert 'op_margin' in res
    assert 'fcf_yield' in res

def test_dcf_valuation_format():
    res = compute_dcf_valuation("AAPL")
    assert 'intrinsic_value' in res
    assert 'margin_of_safety' in res

def test_portfolio_math():
    weights = np.array([0.5, 0.5])
    cov = np.array([[0.04, 0.01], [0.01, 0.04]])
    risk = calculate_portfolio_risk(weights, cov)
    assert risk > 0
    rc = calculate_risk_contribution(weights, cov)
    assert len(rc) == 2

def test_split_adjuster():
    df = pd.DataFrame({
        'Open': [100, 20],
        'High': [110, 22],
        'Low': [90, 18],
        'Close': [100, 20],
        'Volume': [1000, 5000],
        'Stock Splits': [0, 5]
    })
    adj_df = adjust_for_splits(df)
    # The price on day 1 should be divided by 5 (100 -> 20)
    assert adj_df['Close'].iloc[0] == 20
    assert adj_df['Volume'].iloc[0] == 5000
