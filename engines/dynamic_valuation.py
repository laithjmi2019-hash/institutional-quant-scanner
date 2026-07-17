import yfinance as yf

def compute_dcf_valuation(ticker: str) -> dict:
    """
    Multi-stage DCF and intrinsic value modeling.
    5-year FCF projection, WACC discount, 2% terminal growth.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        
        fcf = info.get('freeCashflow', 0)
        shares = info.get('sharesOutstanding', 1)
        current_price = info.get('currentPrice', 0)
        
        if fcf <= 0 or shares == 1 or current_price == 0:
            return {'intrinsic_value': 0.0, 'margin_of_safety': 0.0}
            
        # Hardcoded assumptions for rapid DCF
        wacc = 0.09 # 9% Discount Rate
        terminal_growth = 0.02 # 2% Terminal Growth
        fcf_growth_rate = 0.08 # 8% FCF Growth for next 5 years
        
        # 5-Year Projection
        projected_fcf = []
        current_fcf = fcf
        for i in range(1, 6):
            current_fcf *= (1 + fcf_growth_rate)
            discounted_fcf = current_fcf / ((1 + wacc) ** i)
            projected_fcf.append(discounted_fcf)
            
        sum_pv_fcf = sum(projected_fcf)
        
        # Terminal Value
        terminal_value = (current_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
        pv_terminal_value = terminal_value / ((1 + wacc) ** 5)
        
        enterprise_value = sum_pv_fcf + pv_terminal_value
        
        # Equity Value = EV + Cash - Debt
        cash = info.get('totalCash', 0)
        debt = info.get('totalDebt', 0)
        equity_value = enterprise_value + cash - debt
        
        intrinsic_value_per_share = equity_value / shares
        
        if intrinsic_value_per_share <= 0:
            return {'intrinsic_value': 0.0, 'margin_of_safety': 0.0}
            
        margin_of_safety = ((intrinsic_value_per_share - current_price) / intrinsic_value_per_share) * 100
        
        return {
            'intrinsic_value': round(intrinsic_value_per_share, 2),
            'margin_of_safety': round(margin_of_safety, 2)
        }
        
    except Exception as e:
        return {'intrinsic_value': 0.0, 'margin_of_safety': 0.0}
