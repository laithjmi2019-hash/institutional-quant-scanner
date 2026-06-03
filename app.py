import sys
import argparse
import requests
from datetime import datetime, timedelta, timezone
import concurrent.futures
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
import tickers

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
WEBHOOK_URL = "LOCAL_PLACEHOLDER"
try:
    if "WEBHOOK_URL" in st.secrets:
        WEBHOOK_URL = st.secrets["WEBHOOK_URL"]
except Exception:
    pass

st.set_page_config(layout="wide", page_title="Institutional Quant Scanner")

# Argparse headless setup
parser = argparse.ArgumentParser()
parser.add_argument('--headless', action='store_true', help='Run headless background scan')
args, unknown = parser.parse_known_args()
HEADLESS_MODE = args.headless

# ---------------------------------------------------------
# GLOBAL MARKET REGIME
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_market_regime():
    try:
        spy_df = yf.download("SPY", period="1y", interval="1d", auto_adjust=False, progress=False)
        vix_df = yf.download("^VIX", period="10d", interval="1d", auto_adjust=False, progress=False)
        
        if spy_df.empty or vix_df.empty: return 0.0, 0.0, 0.0, True
        
        vix_val = float(vix_df['Close'].iloc[-1].iloc[0]) if isinstance(vix_df['Close'], pd.DataFrame) else float(vix_df['Close'].iloc[-1])
        spy_price = float(spy_df['Close'].iloc[-1].iloc[0]) if isinstance(spy_df['Close'], pd.DataFrame) else float(spy_df['Close'].iloc[-1])
        spy_sma20 = float(spy_df['Close'].rolling(20).mean().iloc[-1].iloc[0]) if isinstance(spy_df['Close'], pd.DataFrame) else float(spy_df['Close'].rolling(20).mean().iloc[-1])
        
        is_risk_on = (vix_val <= 25) and (spy_price >= spy_sma20)
        return vix_val, spy_price, spy_sma20, is_risk_on
    except Exception as e:
        print(f"Market regime fetch failed: {e}")
        return 0.0, 0.0, 0.0, True

vix_val, spy_price, spy_sma20, is_risk_on = get_market_regime()

# ---------------------------------------------------------
# VECTORIZED MATH INDICATORS
# ---------------------------------------------------------
def calc_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean(), true_range

def calc_bb(df, period=20, std=2):
    sma = df['Close'].rolling(period).mean()
    std_dev = df['Close'].rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def calc_kc(df, period=20, atr_mult=1.5):
    ema = df['Close'].ewm(span=period, adjust=False).mean()
    atr, _ = calc_atr(df, period)
    upper = ema + (atr_mult * atr)
    lower = ema - (atr_mult * atr)
    return upper, lower

def calc_adx(df, period=14):
    atr, tr = calc_atr(df, period)
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    pos_dm_ser = pd.Series(pos_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    neg_dm_ser = pd.Series(neg_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    atr_ewm = tr.ewm(alpha=1/period, adjust=False).mean()
    
    pdi = 100 * (pos_dm_ser / atr_ewm)
    ndi = 100 * (neg_dm_ser / atr_ewm)
    
    dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx, pdi, ndi

# ---------------------------------------------------------
# STRATEGY 1: VOLUME ABSORPTION
# ---------------------------------------------------------
def run_strategy_1_absorption(df_1d):
    if df_1d is None or len(df_1d) < 30: return 0, False, "Not enough data"
    vol_sma = df_1d['Volume'].rolling(30).mean()
    atr, tr = calc_atr(df_1d, 30)
    
    current_vol = df_1d['Volume'].iloc[-1]
    current_vol_sma = vol_sma.iloc[-1]
    current_tr = tr.iloc[-1]
    
    tr_30 = tr.iloc[-30:]
    tr_25th = np.percentile(tr_30.dropna(), 25)
    
    score = 0
    is_absorbing = False
    details = ""
    
    if current_vol > (3 * current_vol_sma) and current_tr <= tr_25th:
        is_absorbing = True
        score = 100
        details = "MASSIVE Volume Absorption Detected (Vol > 3x MA, TR in bottom 25th percentile)"
    elif current_vol > (2 * current_vol_sma) and current_tr <= tr_25th:
        score = 70
        details = "Moderate Volume Absorption"
    elif current_vol > current_vol_sma:
        score = 40
        details = "Normal Volume Activity"
    else:
        score = 10
        details = "Low Volume"
        
    return score, is_absorbing, details

# ---------------------------------------------------------
# STRATEGY 2: VOLATILITY COMPRESSION (SQUEEZE)
# ---------------------------------------------------------
def run_strategy_2_squeeze(df_4h):
    if df_4h is None or len(df_4h) < 20: return 0, False, "Not enough data"
    bb_up, bb_low = calc_bb(df_4h, 20, 2)
    kc_up, kc_low = calc_kc(df_4h, 20, 1.5)
    
    bbw = (bb_up - bb_low) / df_4h['Close'].rolling(20).mean()
    
    squeeze_active = (bb_up.iloc[-1] < kc_up.iloc[-1]) and (bb_low.iloc[-1] > kc_low.iloc[-1])
    bbw_min = bbw.rolling(150).min().iloc[-1] if len(bbw) > 150 else bbw.min()
    
    score = 0
    details = "No Squeeze"
    if squeeze_active:
        score = 80
        details = "SQUEEZE ACTIVE (BB inside KC)"
        if bbw.iloc[-1] <= (bbw_min * 1.1):
            score = 100
            details = "MAXIMUM SQUEEZE ACTIVE (BBW at multi-month low)"
    else:
        if bbw.iloc[-1] > kc_up.iloc[-1]:
            score = 50
            details = "Squeeze Fired / Expansion Phase"
            
    return score, squeeze_active, details

# ---------------------------------------------------------
# STRATEGY 3: HIGHER-TIMEFRAME SMC ENGINE
# ---------------------------------------------------------
def run_strategy_3_smc(df_1h):
    if df_1h is None or len(df_1h) < 10: return 0, False, {}, "Not enough data"
    
    score = 0
    sweep_choch = False
    fvg_det = {}
    details = ""
    
    highs = df_1h['High'].values
    lows = df_1h['Low'].values
    closes = df_1h['Close'].values
    
    swing_lows = []
    for i in range(2, len(lows)-2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append((i, lows[i]))
            
    recent_lows = lows[-10:]
    recent_closes = closes[-10:]
    if len(swing_lows) > 0:
        last_swing_low = swing_lows[-1][1]
        for i in range(len(recent_lows)):
            if recent_lows[i] < last_swing_low and recent_closes[i] > last_swing_low:
                sweep_choch = True
                score += 50
                details += "Bullish Liquidity Sweep (CHoCH). "
                break
                
    fvgs = []
    for i in range(1, 4):
        if len(df_1h) < i + 3: continue
        c1_h = df_1h['High'].iloc[-(i+2)]
        c3_l = df_1h['Low'].iloc[-i]
        if c1_h < c3_l:
            fvg_det = {'Type': 'Bullish', 'Top': c3_l, 'Bottom': c1_h}
            score += 50
            details += "Bullish FVG Spotted. "
            break
            
    if score == 0:
        score = 20
        details = "No major SMC setups"
        
    return min(100, score), sweep_choch, fvg_det, details

# ---------------------------------------------------------
# STRATEGY 4: MACRO TREND EXPANSION
# ---------------------------------------------------------
def run_strategy_4_trend(df_1d):
    if df_1d is None or len(df_1d) < 200: return 0, False, "Not enough data"
    
    ema20 = df_1d['Close'].ewm(span=20, adjust=False).mean()
    ema50 = df_1d['Close'].ewm(span=50, adjust=False).mean()
    ema100 = df_1d['Close'].ewm(span=100, adjust=False).mean()
    ema200 = df_1d['Close'].ewm(span=200, adjust=False).mean()
    
    adx, pdi, ndi = calc_adx(df_1d, 14)
    
    c = df_1d['Close'].iloc[-1]
    e20 = ema20.iloc[-1]
    e50 = ema50.iloc[-1]
    e100 = ema100.iloc[-1]
    e200 = ema200.iloc[-1]
    
    cur_adx = adx.iloc[-1]
    cur_pdi = pdi.iloc[-1]
    cur_ndi = ndi.iloc[-1]
    
    is_trend = False
    score = 0
    details = "Sideways or Bearish"
    
    if c > e20 and e20 > e50 and e50 > e100 and e100 > e200:
        if cur_adx > 25 and cur_pdi > cur_ndi:
            is_trend = True
            score = 100
            details = f"Strong Bullish Expansion (ADX: {cur_adx:.1f}, +DI > -DI, Perfect EMA Ribbon)"
        else:
            score = 70
            details = "Bullish EMA Ribbon but weak ADX/Momentum"
    elif c < e20 and e20 < e50 and e50 < e100 and e100 < e200:
        score = 0
        details = "Bearish Expansion Ribbon"
    else:
        if cur_adx < 20:
            score = 40
            details = "Chop / Range Bound"
            
    return score, is_trend, details

# ---------------------------------------------------------
# STRATEGY 5: FUNDAMENTAL SCRAPER
# ---------------------------------------------------------
class FundamentalScraper:
    def scrape(self, ticker, news_data):
        raise NotImplementedError

class YahooFinanceScraper(FundamentalScraper):
    def scrape(self, ticker, news_data):
        keywords = ["grant", "dod", "contract", "fda", "procurement", "award", "partnership", "phase"]
        found_catalysts = []
        
        for article in news_data:
            title = article.get('title', '').lower()
            for kw in keywords:
                if kw in title:
                    found_catalysts.append(article.get('title'))
                    break
                    
        score = 20
        details = "No Institutional Footprint Detected"
        if len(found_catalysts) > 0:
            score = 100
            details = f"Institutional Footprint: {found_catalysts[0]}"
            
        return score, len(found_catalysts) > 0, details

def run_strategy_5_fundamental(ticker, news_data):
    scraper = YahooFinanceScraper()
    return scraper.scrape(ticker, news_data)

# ---------------------------------------------------------
# AGGREGATION & DATA PIPELINE
# ---------------------------------------------------------
def aggregate_alpha_score(scores, is_risk_on):
    if is_risk_on:
        weights = {'Volume': 0.25, 'Squeeze': 0.15, 'SMC': 0.15, 'Trend': 0.35, 'Fund': 0.10}
    else:
        weights = {'Volume': 0.20, 'Squeeze': 0.30, 'SMC': 0.35, 'Trend': 0.05, 'Fund': 0.10}
        
    alpha = 0
    for k, v in weights.items():
        alpha += scores[k] * v
    return round(alpha, 1)

def fetch_multi_tf_data(ticker):
    tk = yf.Ticker(ticker)
    try:
        df_1h = tk.history(period="60d", interval="1h", auto_adjust=False)
        df_4h = df_1h.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        df_1d = tk.history(period="1y", interval="1d", auto_adjust=False)
        
        info = tk.info
        news = tk.news
        return df_1h, df_4h, df_1d, info, news
    except:
        return None, None, None, {}, []

def evaluate_ticker_pipeline(ticker):
    df_1h, df_4h, df_1d, info, news = fetch_multi_tf_data(ticker)
    if df_1d is None or len(df_1d) < 200: return None
    
    s1_score, s1_flag, s1_det = run_strategy_1_absorption(df_1d)
    s2_score, s2_flag, s2_det = run_strategy_2_squeeze(df_4h)
    s3_score, s3_flag, fvg_det, s3_str = run_strategy_3_smc(df_1h)
    s4_score, s4_flag, s4_det = run_strategy_4_trend(df_1d)
    s5_score, s5_flag, s5_det = run_strategy_5_fundamental(ticker, news)
    
    scores = {'Volume': s1_score, 'Squeeze': s2_score, 'SMC': s3_score, 'Trend': s4_score, 'Fund': s5_score}
    alpha = aggregate_alpha_score(scores, is_risk_on)
    
    rec = "AVOID"
    if alpha >= 80: rec = "STRONG BUY"
    elif alpha >= 65: rec = "BUY"
    elif alpha >= 40: rec = "HOLD"
    
    return {
        'Ticker': ticker,
        'Company Name': info.get('longName', ticker),
        'Sector': info.get('sector', 'Unknown'),
        'Alpha Score': alpha,
        'Recommendation': rec,
        'Vol Absorption': s1_flag,
        'Squeeze Active': s2_flag,
        'SMC Sweep/CHoCH': s3_flag,
        'Trend Expansion': s4_flag,
        'Catalyst': s5_flag,
        'Details': {'Vol': s1_det, 'Squeeze': s2_det, 'SMC': s3_str, 'Trend': s4_det, 'Fund': s5_det, 'FVG': fvg_det},
        'Scores': scores
    }

# ---------------------------------------------------------
# TIER 1 BATCH SCANNER (PRE-FILTER)
# ---------------------------------------------------------
def run_tier_1():
    st.write("Running Tier 1 Filter (Volume & Momentum)...")
    universe = tickers.ALL_TICKERS
    
    try:
        data = yf.download(universe, period="1y", interval="1d", group_by='ticker', auto_adjust=False, progress=False)
        spy_data = yf.download("SPY", period="1y", interval="1d", auto_adjust=False, progress=False)
    except:
        return []
        
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_close = spy_data[('Close', 'SPY')]
    else:
        spy_close = spy_data['Close']
        
    spy_ret = spy_close.pct_change(60).iloc[-1]
    
    tier1_results = []
    for ticker in universe:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[ticker]
            else:
                if len(universe) == 1: df = data
                else: continue
                
            if df.empty or len(df) < 60: continue
            
            c = df['Close']
            v = df['Volume']
            
            stock_ret = c.pct_change(60).iloc[-1]
            rs_score = ((stock_ret - spy_ret) / abs(spy_ret)) * 100 if spy_ret != 0 else 0
            avg_vol = v.rolling(20).mean().iloc[-1]
            
            if avg_vol > 500000:
                tier1_results.append({'Ticker': ticker, 'RS': rs_score, 'AvgVol': avg_vol})
        except:
            pass
            
    tier1_df = pd.DataFrame(tier1_results)
    if tier1_df.empty: return []
    
    tier1_df = tier1_df.sort_values(by='RS', ascending=False).head(100)
    return tier1_df['Ticker'].tolist()

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
    print(f"--- HEADLESS QUANT SCAN INITIATED ---")
    tier2_tickers = run_tier_1()
    print(f"Tier 1 completed. Processing {len(tier2_tickers)} tickers in Tier 2...")
    
    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(evaluate_ticker_pipeline, t): t for t in tier2_tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            try:
                res = future.result()
                if res and res['Alpha Score'] >= 80:
                    final_results.append(res)
                    print(f"INSTITUTIONAL SETUP DETECTED: {res['Ticker']} (Alpha: {res['Alpha Score']})")
                    trigger_webhook(res)
            except Exception as e:
                pass
                
    print("--- HEADLESS QUANT SCAN COMPLETED ---")
    sys.exit(0)

if HEADLESS_MODE:
    run_headless_pipeline()

# ---------------------------------------------------------
# STREAMLIT UI (TAB 1, TAB 2 & TAB 3)
# ---------------------------------------------------------
# Market Regime UI
if is_risk_on:
    st.success(f"**Risk-On Regime.** VIX: {vix_val:.2f} | SPY Price: {spy_price:.2f} (SMA20: {spy_sma20:.2f}). Trend Expansion highly weighted.")
else:
    st.error(f"**WARNING: Risk-Off Regime detected.** VIX: {vix_val:.2f} | SPY Price: {spy_price:.2f} (SMA20: {spy_sma20:.2f}). SMC and Compression heavily weighted.")

tab1, tab2, tab3 = st.tabs(["Tab 1: Market Scanner", "Tab 2: Ticker Deep-Dive", "Tab 3: Risk Sandbox"])

with st.sidebar:
    st.title("Tier-1 Scanner Engine")
    st.write("The Tier-1 scanner evaluates all tickers against 5 algorithmic engines concurrently.")
    if st.button("Run Full Market Scan"):
        st.session_state['run_scan'] = True

if 'run_scan' in st.session_state and st.session_state['run_scan']:
    with tab1:
        tier2_tickers = run_tier_1()
        if not tier2_tickers:
            st.warning("Tier 1 failed or no tickers found.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            final_results = []
            completed = 0
            total = len(tier2_tickers)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_ticker = {executor.submit(evaluate_ticker_pipeline, t): t for t in tier2_tickers}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    completed += 1
                    status_text.text(f"Processing Tier 2: {completed}/{total}")
                    progress_bar.progress(completed / total)
                    try:
                        res = future.result()
                        if res: final_results.append(res)
                    except:
                        pass
                        
            if final_results:
                final_df = pd.DataFrame(final_results)
                final_df = final_df.sort_values(by='Alpha Score', ascending=False)
                
                for col in ['Vol Absorption', 'Squeeze Active', 'SMC Sweep/CHoCH', 'Trend Expansion', 'Catalyst']:
                    final_df[col] = final_df[col].apply(lambda x: "✅" if x else "❌")
                
                display_df = final_df.drop(columns=['Details', 'Scores'])
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No results from Tier 2.")

with tab2:
    st.title("Institutional Grade Alpha Deep-Dive")
    ticker_options = [f"{k} - {v}" for k, v in tickers.TICKER_MAPPING.items()]
    selected_option = st.selectbox("Search Ticker...", options=ticker_options, index=None)
    
    if selected_option:
        search_ticker = selected_option.split(" - ")[0].strip()
        
        col_ps1, col_ps2 = st.columns(2)
        portfolio_size = col_ps1.number_input("Total Portfolio Size ($)", value=50000.0, step=1000.0)
        risk_pct = col_ps2.number_input("Risk Per Trade (%)", value=1.0, step=0.1)
        
        with st.spinner(f"Omni-Scanning {search_ticker} across all institutional engines..."):
            res = evaluate_ticker_pipeline(search_ticker)
            
            if not res:
                st.error("Failed to evaluate ticker. Not enough data.")
            else:
                st.subheader(f"{res['Company Name']} ({res['Ticker']}) - {res['Sector']}")
                st.metric("Institutional Alpha Score", f"{res['Alpha Score']} / 100", help="Weighted average of all 5 engines")
                
                st.markdown("### 🔍 5-Engine Matrix Breakdown")
                cols = st.columns(5)
                strat_names = ['Volume Absorption', 'Squeeze (4H)', 'SMC Structure', 'Macro Trend', 'Fundamental']
                strat_keys = ['Volume', 'Squeeze', 'SMC', 'Trend', 'Fund']
                
                for i in range(5):
                    with cols[i]:
                        st.metric(strat_names[i], f"{res['Scores'][strat_keys[i]]}/100")
                        st.caption(res['Details'][strat_keys[i]])
                
                # Passive Sector Velocity Sidebar Widget
                sector = res['Sector']
                SECTOR_ETF_MAP = {'Technology': 'XLK', 'Healthcare': 'XLV', 'Financial Services': 'XLF', 'Consumer Cyclical': 'XLY', 'Communication Services': 'XLC', 'Industrials': 'XLI', 'Consumer Defensive': 'XLP', 'Energy': 'XLE', 'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Basic Materials': 'XLB'}
                etf = SECTOR_ETF_MAP.get(sector)
                if etf:
                    try:
                        etf_data = yf.download(etf, period="20d", interval="1d", auto_adjust=False, progress=False)
                        etf_ret = (etf_data['Close'].iloc[-1] / etf_data['Close'].iloc[-10] - 1) * 100
                        etf_ret = float(etf_ret.iloc[0]) if isinstance(etf_ret, pd.Series) else float(etf_ret)
                        color = "green" if etf_ret > 0 else "red"
                        
                        st.sidebar.markdown("---")
                        st.sidebar.markdown("### 🌐 Macro Sector Context")
                        st.sidebar.markdown(f"**Sector:** {sector} ({etf})")
                        st.sidebar.markdown(f"**10-Day Capital Flow:** <span style='color:{color}; font-weight:bold'>{etf_ret:.2f}%</span>", unsafe_allow_html=True)
                        st.sidebar.caption("Note: Sector velocity does not impact the Alpha Score. Displayed for passive macro context only.")
                    except:
                        pass

                st.markdown("---")
                
                if res['Alpha Score'] < 70:
                    st.warning(f"**No Dominant Institutional Setup Detected.** Alpha Score: {res['Alpha Score']}/100. Position sizing restricted.")
                else:
                    st.success(f"**Dominant Setup Detected!** Alpha Score: {res['Alpha Score']}/100")
                    
                    with st.expander("Institutional Exits & Position Sizing", expanded=True):
                        fvg = res['Details']['FVG']
                        if res['SMC Sweep/CHoCH'] and fvg:
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
                        else:
                            st.write("No active Fair Value Gap zone detected for risk management.")
                            
                # Isolated Automated Backtester
                st.markdown("---")
                st.markdown("### 🤖 Automated Historical Backtester")
                st.write("Run a 2-year vectorized historical simulation to validate the statistical edge of this setup.")
                if st.button("Run Historical Simulation (2-Year Vectorized)"):
                    with st.spinner("Running vectorized historical backtest..."):
                        try:
                            bt_data = yf.download(search_ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
                            if len(bt_data) > 100:
                                bt_data['EMA50'] = bt_data['Close'].ewm(span=50, adjust=False).mean()
                                bt_data['VolSMA'] = bt_data['Volume'].rolling(30).mean()
                                
                                atr, _ = calc_atr(bt_data, 14)
                                adx, pdi, ndi = calc_adx(bt_data, 14)
                                
                                bb_up, bb_low = calc_bb(bt_data, 20, 2)
                                kc_up, kc_low = calc_kc(bt_data, 20, 1.5)
                                squeeze = (bb_up < kc_up) & (bb_low > kc_low)
                                
                                if isinstance(adx, pd.DataFrame):
                                    adx = adx.iloc[:, 0]
                                    pdi = pdi.iloc[:, 0]
                                    ndi = ndi.iloc[:, 0]
                                    
                                signal = (bt_data['Close'].iloc[:, 0] > bt_data['EMA50'].iloc[:, 0]) & (adx > 25) & (pdi > ndi) & ((bt_data['Volume'].iloc[:, 0] > 2 * bt_data['VolSMA'].iloc[:, 0]) | squeeze.iloc[:, 0])
                                
                                bt_data['Signal'] = signal
                                bt_data['Fwd_Ret_10d'] = bt_data['Close'].iloc[:, 0].shift(-10) / bt_data['Close'].iloc[:, 0] - 1
                                
                                trades = bt_data[bt_data['Signal'] & (~bt_data['Signal'].shift(1).fillna(False))].dropna(subset=['Fwd_Ret_10d'])
                                
                                if len(trades) > 0:
                                    win_rate = (trades['Fwd_Ret_10d'] > 0).mean() * 100
                                    avg_ret = trades['Fwd_Ret_10d'].mean() * 100
                                    max_dd = trades['Fwd_Ret_10d'].min() * 100
                                    
                                    st.info(f"**Historical Simulation Complete (2-Year)**\n\nIdentified **{len(trades)}** structural setups in the past 24 months matching these dynamics.\n- **Win Rate (10-Day Hold):** {win_rate:.1f}%\n- **Average Return per Trade:** {avg_ret:.2f}%\n- **Max Drawdown (Worst Trade):** {max_dd:.2f}%")
                                else:
                                    st.warning("No historical setups matched this strict Institutional Grade criteria in the last 2 years.")
                            else:
                                st.warning("Not enough historical data to run backtest.")
                        except Exception as e:
                            st.error(f"Backtest failed: {e}")

with tab3:
    st.title("Pre-Trade Portfolio Risk Sandbox")
    st.write("Analyze your current portfolio against staged scanner picks to identify critical systemic risk correlations.")
    
    col1, col2 = st.columns(2)
    active_port = col1.text_input("Active Portfolio Tickers (comma separated)", "AAPL, MSFT")
    
    stage_options = [f"{k} - {v}" for k, v in tickers.TICKER_MAPPING.items()]
    staged_picks = col2.multiselect("Stage Pending Scanner Picks", options=stage_options)
    
    if st.button("Run Risk Matrix"):
        with st.spinner("Calculating Covariance and Risk Parity..."):
            port_tickers = [t.strip().upper() for t in active_port.split(",") if t.strip()]
            staged_tickers = [t.split(" - ")[0].strip() for t in staged_picks]
            
            all_tickers = list(set(port_tickers + staged_tickers))
            
            if len(all_tickers) < 2:
                st.warning("Please enter at least 2 valid tickers to calculate correlation.")
            else:
                try:
                    data = yf.download(all_tickers, period="60d", interval="1d", group_by='ticker', auto_adjust=False, progress=False)
                    closes = pd.DataFrame()
                    
                    if len(all_tickers) == 1:
                        st.warning("Need 2+ tickers to correlate.")
                    else:
                        for t in all_tickers:
                            if isinstance(data.columns, pd.MultiIndex):
                                if t in data['Close']:
                                    closes[t] = data['Close'][t]
                            else:
                                pass # Should not happen with multiple tickers
                                
                        closes = closes.dropna()
                        log_returns = np.log(closes / closes.shift(1)).dropna()
                        
                        corr_matrix = log_returns.corr()
                        
                        st.subheader("Pearson Correlation Heatmap (60-Day)")
                        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect="auto")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        high_corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                if corr_matrix.iloc[i, j] > 0.85:
                                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                                    
                        if high_corr_pairs:
                            st.error("🚨 **SYSTEMIC RISK DETECTED** 🚨")
                            for p in high_corr_pairs:
                                st.write(f"- **{p[0]}** and **{p[1]}** have an extreme correlation of **{p[2]:.2f}**. Buying both provides zero diversification and doubles your downside risk.")
                        else:
                            st.success("✅ No extreme correlations (> 0.85) detected. Portfolio risk is distributed.")
                            
                        st.markdown("### Risk Parity Position Sizing")
                        st.write("Suggested capital allocation based on inverse volatility (targeting equal risk contribution):")
                        vols = log_returns.std() * np.sqrt(252)
                        inv_vols = 1.0 / vols
                        risk_parity_weights = inv_vols / inv_vols.sum()
                        
                        rp_df = pd.DataFrame({
                            'Annualized Volatility': vols.apply(lambda x: f"{x*100:.1f}%"),
                            'Suggested Allocation': risk_parity_weights.apply(lambda x: f"{x*100:.1f}%")
                        })
                        st.dataframe(rp_df.sort_values(by='Suggested Allocation', ascending=False))
                except Exception as e:
                    st.error(f"Failed to calculate risk matrix: {e}")
