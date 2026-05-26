import sys
import argparse
import requests
from datetime import datetime, timedelta, timezone
import concurrent.futures
import pandas as pd
import numpy as np
import yfinance as yf
import tickers

# ---------------------------------------------------------
# ARGPARSE & HEADLESS MODE SETUP
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Quantitative Trading Scanner")
parser.add_argument("--headless", action="store_true", help="Run in headless mode without Streamlit UI")
parser.add_argument("--strategy", type=str, default="Deep Value Reversion", help="Strategy to use in headless mode")
args, unknown = parser.parse_known_args()

HEADLESS_MODE = args.headless
DEFAULT_STRATEGY = args.strategy

if HEADLESS_MODE:
    from unittest.mock import MagicMock
    st = MagicMock()
    
    # Mock Streamlit decorators so they return the original function instead of a MagicMock
    def mock_decorator(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper
    
    st.cache_data = mock_decorator
else:
    import streamlit as st

try:
    WEBHOOK_URL = st.secrets["WEBHOOK_URL"]
except Exception:
    WEBHOOK_URL = "LOCAL_PLACEHOLDER"

# ---------------------------------------------------------
# GLOBAL MARKET REGIME FILTER
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_regime():
    """
    Returns (vix_val, spy_price, spy_sma20, is_risk_on)
    """
    try:
        data = yf.download(["^VIX", "SPY"], period="40d", interval="1d", group_by="ticker", threads=True, auto_adjust=False)
        vix_close = data["^VIX"]["Close"].dropna()
        spy_close = data["SPY"]["Close"].dropna()
        
        if vix_close.empty or spy_close.empty:
            print("Warning: Market regime data empty. Defaulting to Risk-On.")
            return 0.0, 0.0, 0.0, True
            
        vix_val = vix_close.iloc[-1]
        spy_price = spy_close.iloc[-1]
        spy_sma20 = spy_close.tail(20).mean()
        
        is_risk_on = not (vix_val > 25 or spy_price < spy_sma20)
        return vix_val, spy_price, spy_sma20, is_risk_on
    except Exception as e:
        print(f"Warning: Exception fetching market regime data ({e}). Defaulting to Risk-On.")
        return 0.0, 0.0, 0.0, True

vix_val, spy_price, spy_sma20, is_risk_on = get_market_regime()

if not HEADLESS_MODE:
    st.set_page_config(layout="wide", page_title="Institutional Quant Scanner")
    
    if is_risk_on:
        st.success(f"**Risk-On Regime.** VIX: {vix_val:.2f} | SPY Price: {spy_price:.2f} (SMA20: {spy_sma20:.2f}). Momentum and Pullback strategies optimized.")
    else:
        st.error(f"**WARNING: Risk-Off Regime detected.** VIX: {vix_val:.2f} | SPY Price: {spy_price:.2f} (SMA20: {spy_sma20:.2f}). Breakout strategies carry high failure probability. Prioritize Deep Value or Bearish Distribution.")
        
    st.sidebar.header("The 5-Strategy Institutional Playbook")
    STRATEGY = st.sidebar.selectbox(
        "Select Trading Strategy",
        [
            "Deep Value Reversion",
            "Structural Pullback",
            "Accumulation Spring",
            "Momentum Breakout",
            "Bearish Distribution (Shorting)"
        ]
    )
else:
    STRATEGY = DEFAULT_STRATEGY

# ---------------------------------------------------------
# SMC ENGINE & RVOL VALIDATION
# ---------------------------------------------------------
def analyze_smc(df):
    """
    Computes RVOL, SMC flags (Sweep, MSS, FVG), and Exits (TP1, TP2).
    """
    if df is None or len(df) < 25:
        return False, False, False, {}, None, None
        
    df['Swing_High_20'] = df['High'].shift(1).rolling(window=20).max()
    df['Swing_Low_20'] = df['Low'].shift(1).rolling(window=20).min()
    
    # 1. Volume Validation (RVOL)
    if 'Volume' in df.columns:
        df['Volume_SMA_20'] = df['Volume'].shift(1).rolling(window=20).mean()
        df['RVOL'] = np.where(df['Volume_SMA_20'] > 0, df['Volume'] / df['Volume_SMA_20'], 1.0)
    else:
        df['RVOL'] = 1.0
        
    # Liquidity Sweep
    df['Bearish_Sweep'] = (df['High'] > df['Swing_High_20']) & (df['Close'] < df['Swing_High_20'])
    df['Bullish_Sweep'] = (df['Low'] < df['Swing_Low_20']) & (df['Close'] > df['Swing_Low_20'])
    sweep_triggered = df['Bearish_Sweep'].iloc[-1] or df['Bullish_Sweep'].iloc[-1]
    
    # Market Structure Shift (MSS) with RVOL Validation
    df['Bullish_MSS'] = (df['Close'] > df['Swing_High_20']) & (df['RVOL'] > 1.5)
    df['Bearish_MSS'] = (df['Close'] < df['Swing_Low_20']) & (df['RVOL'] > 1.5)
    mss_triggered = df['Bullish_MSS'].iloc[-1] or df['Bearish_MSS'].iloc[-1]
    mss_rvol = df['RVOL'].iloc[-1] if mss_triggered else 0
    
    # FVG Detector
    df['Bullish_FVG_Gap'] = df['Low'] > df['High'].shift(2)
    df['Bearish_FVG_Gap'] = df['High'] < df['Low'].shift(2)
    
    in_fvg_zone = False
    fvg_details = {}
    current_price = df['Close'].iloc[-1]
    
    tp1, tp2 = None, None
    
    for i in range(1, min(10, len(df))):
        idx = -i
        if df['Bullish_FVG_Gap'].iloc[idx]:
            gap_bottom = df['High'].iloc[idx - 2]
            gap_top = df['Low'].iloc[idx]
            if gap_bottom <= current_price <= gap_top:
                in_fvg_zone = True
                fvg_details = {'Type': 'Bullish', 'Top': gap_top, 'Bottom': gap_bottom}
                
                # Exits: Nearest two un-swept swing highs above current price
                highs = df['Swing_High_20'].dropna().unique()
                valid_highs = sorted([h for h in highs if h > current_price])
                if len(valid_highs) > 0: tp1 = valid_highs[0]
                if len(valid_highs) > 1: tp2 = valid_highs[1]
                break
                
        elif df['Bearish_FVG_Gap'].iloc[idx]:
            gap_top = df['Low'].iloc[idx - 2]
            gap_bottom = df['High'].iloc[idx]
            if gap_bottom <= current_price <= gap_top:
                in_fvg_zone = True
                fvg_details = {'Type': 'Bearish', 'Top': gap_top, 'Bottom': gap_bottom}
                
                # Exits: Nearest two un-swept swing lows below current price
                lows = df['Swing_Low_20'].dropna().unique()
                valid_lows = sorted([l for l in lows if l < current_price], reverse=True)
                if len(valid_lows) > 0: tp1 = valid_lows[0]
                if len(valid_lows) > 1: tp2 = valid_lows[1]
                break

    return bool(sweep_triggered), bool(mss_triggered), bool(in_fvg_zone), fvg_details, tp1, tp2

# ---------------------------------------------------------
# FUNDAMENTALS & COMPOSITE SCORING
# ---------------------------------------------------------
def score_fundamentals(info):
    scores = {'Solvency': 0, 'Profitability': 0, 'Growth': 0, 'Valuation': 0}
    if not info: return scores
    
    try:
        if info.get('currentRatio', 0) > 1.5: scores['Solvency'] += 10
    except: pass
    
    try:
        de = info.get('debtToEquity', 999)
        val = de / 100 if de > 10 else de
        if val < 1.0: scores['Solvency'] += 10
    except: pass

    try:
        fcf_yield = info.get('freeCashflow', 0) / info.get('marketCap', 1)
        if fcf_yield > 0.05: scores['Profitability'] += 10
    except: pass
    
    try:
        if info.get('operatingMargins', 0) > 0.15: scores['Profitability'] += 10
    except: pass

    try:
        if info.get('revenueGrowth', 0) > 0: scores['Growth'] += 10
    except: pass
    
    try:
        if info.get('earningsGrowth', 0) > 0: scores['Growth'] += 10
    except: pass

    try:
        if info.get('trailingPE', 999) < 25: scores['Valuation'] += 10
    except: pass

    return scores

def calculate_strategy_score(strategy, tech_flags, rsi, rs_score, fund_scores, info):
    """
    Dynamic 100-point scoring based on the 5-Strategy Institutional Playbook.
    """
    sweep, mss, fvg = tech_flags
    solvency = fund_scores['Solvency']
    profitability = fund_scores['Profitability']
    growth = fund_scores['Growth']
    valuation = fund_scores['Valuation']
    fcf = info.get('freeCashflow', 0)
    current_ratio = info.get('currentRatio', 0)
    
    score = 0
    
    if strategy == "Deep Value Reversion":
        if rsi < 35: score += 30
        elif rsi > 35: score -= 40
        if valuation == 10: score += 20
        if fcf > 0: score += 20
        if sweep: score += 30
        
    elif strategy == "Structural Pullback":
        score += min(30, rs_score * 1.2)  # High RS
        if growth > 0: score += 20
        if fvg: score += 50
        
    elif strategy == "Accumulation Spring":
        if mss: score += 50
        if current_ratio > 1.5: score += 30
        score += solvency
        
    elif strategy == "Momentum Breakout":
        fifty_two_high = info.get('fiftyTwoWeekHigh', 999999)
        current_price = info.get('currentPrice', 0) or info.get('previousClose', 0)
        # Within 5% of 52-week high
        if current_price > 0:
            if (fifty_two_high - current_price) / current_price < 0.05:
                score += 40
            else:
                score -= 40
        else:
            score -= 40
            
        if mss: score += 30
        if fvg: score += 30
        
    elif strategy == "Bearish Distribution (Shorting)":
        if rsi > 70: score += 30
        elif rsi < 60: score -= 40
        if fcf < 0: score += 20
        if mss: score += 30  # Assuming mss logic flags bearish if we passed it correctly
        if fvg: score += 20
        
    return min(100, score)

# ---------------------------------------------------------
# TIER 1 & TIER 2 PIPELINE
# ---------------------------------------------------------
def run_tier_1():
    st.write("Running Tier 1 Filter (Volume & RS & RSI)...")
    universe = tickers.ALL_TICKERS
    
    try:
        data = yf.download(universe + ['SPY'], period='90d', interval='1d', group_by='ticker', threads=True, auto_adjust=False)
    except Exception as e:
        st.error(f"Error downloading bulk data: {e}")
        return pd.DataFrame()

    results = []
    
    spy_data = data['SPY'] if 'SPY' in data else None
    spy_ret = 0
    if spy_data is not None and not spy_data['Close'].dropna().empty:
        spy_close = spy_data['Close'].dropna()
        spy_ret = (spy_close.iloc[-1] - spy_close.iloc[0]) / spy_close.iloc[0]

    for t in universe:
        if t not in data: continue
        t_data = data[t]
        if t_data['Close'].dropna().empty or t_data['Volume'].dropna().empty:
            continue
            
        t_close = t_data['Close'].dropna()
        t_vol = t_data['Volume'].dropna()
        
        if len(t_close) < 20: continue
            
        adv_20 = t_vol.tail(20).mean()
        if adv_20 < 1_000_000: continue
            
        t_ret = (t_close.iloc[-1] - t_close.iloc[0]) / t_close.iloc[0]
        rs_metric = t_ret - spy_ret 
        
        # Calculate Daily RSI
        delta = t_close.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        rs = gain.ewm(com=13, adjust=False).mean() / loss.ewm(com=13, adjust=False).mean()
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        results.append({'Ticker': t, 'RS': rs_metric, 'RSI': current_rsi})
        
    df_res = pd.DataFrame(results)
    if df_res.empty: return df_res
        
    rs_threshold = df_res['RS'].quantile(0.3)
    df_res = df_res[df_res['RS'] >= rs_threshold]
    df_res = df_res.sort_values(by='RS', ascending=False).head(100)
    df_res['RS_Base_Score'] = np.linspace(25, 5, len(df_res))
    
    return df_res

def fetch_tier2_data(row, strategy):
    ticker_str = row['Ticker']
    rs_base_score = row['RS_Base_Score']
    rsi = row['RSI']
    
    tk = yf.Ticker(ticker_str)
    
    try: info = tk.info
    except: info = {}
        
    # Earnings Risk Penalty Filter
    earnings_risk = False
    earnings_date = None
    try:
        # Check calendar dict first
        cal = tk.calendar
        if isinstance(cal, dict) and 'Earnings Date' in cal:
            dates = cal['Earnings Date']
            if len(dates) > 0: earnings_date = dates[0]
    except: pass

    if earnings_date is None:
        try:
            edates = tk.get_earnings_dates()
            if edates is not None and not edates.empty:
                future_dates = edates[edates.index > datetime.now(timezone.utc)]
                if not future_dates.empty: earnings_date = future_dates.index.min()
        except: pass
        
    if earnings_date is not None:
        try:
            days_to_earnings = (earnings_date.replace(tzinfo=None) - datetime.now()).days
            if 0 <= days_to_earnings <= 7:
                earnings_risk = True
        except: pass

    fund_scores = score_fundamentals(info)
    
    try:
        hist_1h = tk.history(period="60d", interval="1h")
        sweep_1h, mss_1h, fvg_1h, fvg_det, tp1, tp2 = analyze_smc(hist_1h)
    except:
        sweep_1h, mss_1h, fvg_1h, fvg_det, tp1, tp2 = False, False, False, {}, None, None
        
    tech_flags = (sweep_1h, mss_1h, fvg_1h)
    
    comp_score = calculate_strategy_score(strategy, tech_flags, rsi, rs_base_score, fund_scores, info)
    if earnings_risk:
        comp_score -= 30
        
    rec = "AVOID"
    if comp_score >= 80: rec = "STRONG BUY"
    elif comp_score >= 65: rec = "BUY"
    elif comp_score >= 40: rec = "HOLD"
    
    return {
        'Ticker': ticker_str,
        'Company Name': info.get('longName', ticker_str),
        'Sector': info.get('sector', 'Unknown'),
        'Composite Score': round(comp_score, 1),
        'Recommendation': rec,
        'Fund: Solvency': fund_scores['Solvency'],
        'Fund: Profitability': fund_scores['Profitability'],
        'Fund: Growth': fund_scores['Growth'],
        'Fund: Valuation': fund_scores['Valuation'],
        'Tech: RS Base': round(rs_base_score, 1),
        'Swept Liquidity': sweep_1h,
        'MSS Triggered': mss_1h,
        'In FVG Zone': fvg_1h,
        'Earnings Risk': earnings_risk,
        'RSI': round(rsi, 1),
        'TP1': tp1,
        'TP2': tp2,
        'FVG_Details': fvg_det
    }

# ---------------------------------------------------------
# WEBHOOK AUTOMATION
# ---------------------------------------------------------
def trigger_webhook(payload):
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        if not HEADLESS_MODE: st.error(f"Webhook failed: {e}")
        else: print(f"Webhook failed: {e}")

def run_headless_pipeline():
    print(f"--- HEADLESS MODE INITIATED ---")
    print(f"Strategy: {STRATEGY}")
    tier1_df = run_tier_1()
    if tier1_df.empty:
        print("Tier 1 returned empty.")
        return
        
    rows = tier1_df.to_dict('records')
    print(f"Executing Tier 2 for {len(rows)} tickers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_tier2_data, row, STRATEGY): row for row in rows}
        tickers_fired = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res['Composite Score'] >= 80 and res['In FVG Zone']:
                    # Build Webhook Payload
                    # Position sizing based on default 50k / 1%
                    fvg = res['FVG_Details']
                    entry = fvg.get('Bottom') if fvg.get('Type') == 'Bullish' else fvg.get('Top')
                    stop = fvg.get('Top') if fvg.get('Type') == 'Bullish' else fvg.get('Bottom')
                    risk_amount = 50000 * 0.01
                    shares = 0
                    if entry and stop and abs(entry - stop) > 0:
                        shares = int(risk_amount / abs(entry - stop))
                        
                    payload = {
                        "Ticker": res['Ticker'],
                        "Strategy": STRATEGY,
                        "Score": res['Composite Score'],
                        "FVG_Bounds": fvg,
                        "TP1": res['TP1'],
                        "TP2": res['TP2'],
                        "Suggested_Shares": shares
                    }
                    trigger_webhook(payload)
                    print(f"[Webhook Fired] Payload sent successfully for {res['Ticker']}")
                    tickers_fired += 1
            except Exception as e:
                pass
                
    if tickers_fired == 0:
        print("[Scan Complete] 0 tickers met the strict webhook criteria for entry.")
    else:
        print(f"[Scan Complete] {tickers_fired} tickers met criteria and fired webhooks.")

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
if __name__ == "__main__":
    if HEADLESS_MODE:
        run_headless_pipeline()
        sys.exit(0)
        
    # Standard UI Path
    tab1, tab2 = st.tabs(["Tab 1: Market Scanner", "Tab 2: Ticker Deep-Dive"])

    with tab1:
        st.title("Institutional Market Scanner")
        if st.button("Run Market Scan"):
            with st.spinner("Executing Tier 1 Filter..."):
                tier1_df = run_tier_1()
                
            if tier1_df.empty:
                st.error("Tier 1 failed or returned no tickers.")
            else:
                tier2_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                rows = tier1_df.to_dict('records')
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(fetch_tier2_data, row, STRATEGY): row for row in rows}
                    completed = 0
                    for future in concurrent.futures.as_completed(futures):
                        try: tier2_results.append(future.result())
                        except: pass
                        completed += 1
                        progress_bar.progress(completed / len(rows))
                        status_text.text(f"Processing Tier 2: {completed}/{len(rows)}")
                        
                if tier2_results:
                    final_df = pd.DataFrame(tier2_results)
                    final_df = final_df.sort_values(by='Composite Score', ascending=False)
                    # We drop internal FVG_Details dict from dataframe view
                    display_df = final_df.drop(columns=['FVG_Details'])
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("No results from Tier 2.")

    with tab2:
        st.title("Ticker Deep-Dive & Position Sizing")
        ticker_options = [f"{k} - {v}" for k, v in tickers.TICKER_MAPPING.items()]
        selected_option = st.selectbox(
            "Search Ticker or Company Name...",
            options=ticker_options,
            index=None,
            placeholder="Search Ticker or Company Name..."
        )
        
        search_ticker = None
        if selected_option:
            search_ticker = selected_option.split(" - ")[0].strip()
        
        if search_ticker:
            # Dynamic Position Sizing Inputs
            col_ps1, col_ps2 = st.columns(2)
            portfolio_size = col_ps1.number_input("Total Portfolio Size ($)", value=50000.0, step=1000.0)
            risk_pct = col_ps2.number_input("Risk Per Trade (%)", value=1.0, step=0.1)
            
            with st.spinner(f"Analyzing {search_ticker} against {STRATEGY}..."):
                row = {'Ticker': search_ticker, 'RS_Base_Score': 20.0, 'RSI': 50.0}
                res = fetch_tier2_data(row, STRATEGY)
                
                if res['Earnings Risk']:
                    st.error(f"⚠️ EARNINGS RISK: {search_ticker} reports earnings within the next 7 days. -30 Penalty Applied.")
                
                st.subheader(f"{res['Company Name']} ({res['Ticker']}) - {res['Sector']}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Composite Score", f"{res['Composite Score']} / 100")
                col2.metric("Recommendation", res['Recommendation'])
                col3.metric("Swept Liquidity", "Yes" if res['Swept Liquidity'] else "No")
                col4.metric("MSS Triggered", "Yes" if res['MSS Triggered'] else "No")
                
                with st.expander("Institutional Exits & Position Sizing"):
                    fvg = res['FVG_Details']
                    if res['In FVG Zone'] and fvg:
                        st.write(f"**Active FVG Detected ({fvg['Type']})**")
                        entry = fvg['Top'] if fvg['Type'] == 'Bullish' else fvg['Bottom']
                        stop = fvg['Bottom'] if fvg['Type'] == 'Bullish' else fvg['Top']
                        st.write(f"- FVG Entry Boundary: {entry:.2f}")
                        st.write(f"- FVG Invalidation (Stop Loss): {stop:.2f}")
                        
                        risk_amt = portfolio_size * (risk_pct / 100.0)
                        risk_per_share = abs(entry - stop)
                        if risk_per_share > 0:
                            allowed_shares = int(risk_amt / risk_per_share)
                            st.success(f"**Allowed Position Size:** {allowed_shares} shares (Risking ${risk_amt:.2f})")
                        
                        st.write("---")
                        st.write("**Take-Profit Targets (Un-swept Liquidity):**")
                        st.write(f"- **TP1:** {res['TP1']:.2f}" if res['TP1'] else "- **TP1:** None found")
                        st.write(f"- **TP2:** {res['TP2']:.2f}" if res['TP2'] else "- **TP2:** None found")
                    else:
                        st.write("No active Fair Value Gap zone detected. Position sizing requires an FVG boundary.")
