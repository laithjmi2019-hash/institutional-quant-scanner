import pandas as pd
import numpy as np

def adjust_for_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backwards-propagated split/corporate action normalization.
    Divides historical OHLC by the cumulative product of splits; multiplies volume.
    Requires a 'Stock Splits' column in the DataFrame (from yfinance).
    """
    if 'Stock Splits' not in df.columns:
        return df
        
    df = df.copy()
    
    # 1. Reverse the splits array (to calculate backward cumulative product)
    # yfinance 'Stock Splits' are > 0 on the day of the split. (e.g. 10 for a 10-for-1 split).
    # Replace 0s with 1s for the multiplier.
    splits = df['Stock Splits'].replace(0, 1.0)
    
    # We want to divide past prices by the split ratio. 
    # Since prices are chronological, a 10-for-1 split today means yesterday's price should be divided by 10.
    # We reverse, cumprod, then reverse back to apply multipliers backward in time.
    split_multipliers = splits[::-1].cumprod()[::-1]
    
    # Shift by 1 because the split happens ON that day, affecting PREVIOUS days' prices.
    # Actually, yfinance adjusts the close of the previous day if auto_adjust=False but usually yfinance
    # provides split data. We apply the multiplier to the days BEFORE the split.
    # To be precise, if split is on Day T, prices on Day T are already split-adjusted.
    # Prices on T-1 need to be divided by the split ratio.
    split_factors = split_multipliers.shift(-1).fillna(1.0)
    
    # Apply factors
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col] / split_factors
            
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'] * split_factors
        
    return df
