import yfinance as yf
import pandas as pd

def compute_business_quality(ticker: str) -> dict:
    """
    Computes ROIC, Operating Margins, and FCF Yield using yfinance fundamental data.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        
        # 1. ROIC = (Operating Income * (1 - Tax Rate)) / (Total Debt + Total Equity - Cash)
        # We try to extract from info or fallback to financials
        op_income = info.get('operatingMargins', 0) * info.get('totalRevenue', 0)
        tax_rate = 0.21 # Default corporate tax rate if not available
        nopat = op_income * (1 - tax_rate)
        
        debt = info.get('totalDebt', 0)
        equity = info.get('totalStockholderEquity', 0)
        if equity == 0:
            # Try to get market cap if equity is missing
            equity = info.get('marketCap', 0)
            
        cash = info.get('totalCash', 0)
        
        invested_capital = debt + equity - cash
        if invested_capital <= 0:
            invested_capital = 1 # Div-by-zero protection
            
        roic = (nopat / invested_capital) * 100
        
        # 2. Operating Margin
        op_margin = info.get('operatingMargins', 0) * 100
        
        # 3. FCF Yield = FCF / Market Cap
        fcf = info.get('freeCashflow', 0)
        market_cap = info.get('marketCap', 1) # Div-by-zero protection
        fcf_yield = (fcf / market_cap) * 100
        
        return {
            'roic': round(roic, 2),
            'op_margin': round(op_margin, 2),
            'fcf_yield': round(fcf_yield, 2)
        }
        
    except Exception as e:
        return {'roic': 0.0, 'op_margin': 0.0, 'fcf_yield': 0.0}
