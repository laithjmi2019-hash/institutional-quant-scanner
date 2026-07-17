import pandas as pd
import numpy as np
import datetime

def compute_technical_structure(df: pd.DataFrame) -> dict:
    """
    RSI, AVWAP, and Hybrid RVOL (Relative Volume).
    """
    if df is None or len(df) < 20:
        return {'rsi': 0.0, 'avwap_distance': 0.0, 'rvol': 0.0}
        
    df = df.copy()
    
    # 1. RSI (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # 2. QTD AVWAP
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = df['Typical_Price'] * df['Volume']
    df['Quarter_Start'] = df.index.to_period('Q').start_time
    
    grouped = df.groupby('Quarter_Start')
    df['Cum_TP_Vol'] = grouped['TP_Vol'].cumsum()
    df['Cum_Vol'] = grouped['Volume'].cumsum()
    df['AVWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
    
    current_close = df['Close'].iloc[-1]
    current_avwap = df['AVWAP'].iloc[-1]
    avwap_distance = ((current_close - current_avwap) / current_avwap) * 100
    
    # 3. Hybrid RVOL
    # RVOL = V_today / (V_20D_Avg * Session Elapsed Ratio)
    v_today = df['Volume'].iloc[-1]
    v_20d_avg = df['Volume'].rolling(20, min_periods=1).mean().iloc[-2] # using iloc[-2] so today doesn't skew average
    
    # Approximate Session Elapsed Ratio
    # If during market hours (9:30 AM to 4:00 PM EST)
    now = datetime.datetime.now()
    # For EOD scanner, assume 1.0. If live, calculate ratio.
    # To keep it robust without timezone complexities, we'll assume 1.0 for End-of-Day data.
    session_elapsed_ratio = 1.0 
    
    rvol = v_today / (v_20d_avg * session_elapsed_ratio) if v_20d_avg > 0 else 0
    
    return {
        'rsi': round(current_rsi, 2) if not np.isnan(current_rsi) else 0.0,
        'avwap_distance': round(avwap_distance, 2) if not np.isnan(avwap_distance) else 0.0,
        'rvol': round(rvol, 2) if not np.isnan(rvol) else 0.0
    }
