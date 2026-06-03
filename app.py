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
# STRATEGY 1: RVOL (RELATIVE VOLUME) SPIKES
# ---------------------------------------------------------
def run_strategy_1_rvol(df_1d):
    if df_1d is None or len(df_1d) < 20: return 0, False, "Not enough data"
    vol_sma = df_1d['Volume'].rolling(20).mean()
    current_vol = df_1d['Volume'].iloc[-1]
    current_vol_sma = vol_sma.iloc[-1]
    
    score = 0
    is_absorbing = False
    details = "Normal Volume"
    
    if current_vol_sma > 0:
        rvol = current_vol / current_vol_sma
        if rvol >= 3.0:
            is_absorbing = True
            score = 100
            details = f"MASSIVE RVOL SPIKE: {rvol:.1f}x normal volume"
        elif rvol >= 1.5:
            score = 60
            details = f"Elevated Volume: {rvol:.1f}x normal"
        else:
            score = 10
            details = f"Normal Volume ({rvol:.1f}x)"
            
    return score, is_absorbing, details

# ---------------------------------------------------------
# STRATEGY 2: DAILY VOLATILITY SQUEEZE (BONUS MULTIPLIER)
# ---------------------------------------------------------
def run_strategy_2_squeeze(df_1d):
    if df_1d is None or len(df_1d) < 20: return 0, False, "Not enough data"
    bb_up, bb_low = calc_bb(df_1d, 20, 2)
    kc_up, kc_low = calc_kc(df_1d, 20, 1.5)
    
    bbw = (bb_up - bb_low) / df_1d['Close'].rolling(20).mean()
    
    squeeze_active = (bb_up.iloc[-1] < kc_up.iloc[-1]) and (bb_low.iloc[-1] > kc_low.iloc[-1])
    bbw_min = bbw.rolling(150).min().iloc[-1] if len(bbw) > 150 else bbw.min()
    
    score = 0
    details = "No Squeeze"
    if squeeze_active:
        score = 80
        details = "SQUEEZE ACTIVE (Daily BB inside KC)"
        if bbw.iloc[-1] <= (bbw_min * 1.1):
            score = 100
            details = "MAXIMUM SQUEEZE ACTIVE (Daily BBW at multi-month low)"
    else:
        if bbw.iloc[-1] > kc_up.iloc[-1]:
            score = 50
            details = "Squeeze Fired / Expansion Phase"
            
    return score, squeeze_active, details

# ---------------------------------------------------------
# STRATEGY 3: QTD ANCHORED VWAP
# ---------------------------------------------------------
def run_strategy_3_avwap(df_1d):
    if df_1d is None or len(df_1d) < 10: return 0, False, "Not enough data"
    
    df = df_1d.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = df['Typical_Price'] * df['Volume']
    
    # Identify the current quarter anchored to the first trading day of the quarter
    df['Quarter_Start'] = df.index.to_period('Q').start_time
    
    grouped = df.groupby('Quarter_Start')
    df['Cum_TP_Vol'] = grouped['TP_Vol'].cumsum()
    df['Cum_Vol'] = grouped['Volume'].cumsum()
    
    df['AVWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
    
    current_close = df['Close'].iloc[-1]
    current_avwap = df['AVWAP'].iloc[-1]
    
    score = 0
    is_above = False
    details = "Trading Below QTD AVWAP"
    
    if current_close > current_avwap:
        is_above = True
        distance = (current_close / current_avwap - 1) * 100
        score = 100
        details = f"Bullish: Price above QTD AVWAP (+{distance:.1f}%)"
    else:
        score = 0
        details = "Bearish: Price below QTD AVWAP"
        
    return score, is_above, details

# ---------------------------------------------------------
# STRATEGY 4: FAST MOMENTUM RIBBON
# ---------------------------------------------------------
def run_strategy_4_trend(df_1d):
    if df_1d is None or len(df_1d) < 25: return 0, False, "Not enough data"
    
    ema9 = df_1d['Close'].ewm(span=9, adjust=False).mean()
    ema21 = df_1d['Close'].ewm(span=21, adjust=False).mean()
    
    c = df_1d['Close'].iloc[-1]
    e9 = ema9.iloc[-1]
    e21 = ema21.iloc[-1]
    
    is_trend = False
    score = 0
    details = "Sideways or Bearish"
    
    if c > e9 and e9 > e21:
        is_trend = True
        score = 100
        details = "Fast Momentum Uptrend (Close > 9-EMA > 21-EMA)"
    elif c < e9 and e9 < e21:
        score = 0
        details = "Bearish Downtrend (Close < 9-EMA < 21-EMA)"
    else:
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
def aggregate_alpha_score(scores, flags_dict, is_risk_on):
    if is_risk_on:
        weights = {'Volume': 0.35, 'AVWAP': 0.25, 'Trend': 0.30, 'Fund': 0.10}
    else:
        weights = {'Volume': 0.30, 'AVWAP': 0.40, 'Trend': 0.10, 'Fund': 0.20}
        
    alpha = 0
    for k, v in weights.items():
        alpha += scores[k] * v
        
    # Bonus Multiplier for Squeeze + RVOL
    if flags_dict['Squeeze'] and flags_dict['Volume']:
        alpha += 15
        
    return min(100.0, round(alpha, 1))

def fetch_daily_data(ticker):
    tk = yf.Ticker(ticker)
    try:
        df_1d = tk.history(period="1y", interval="1d", auto_adjust=False)
        info = tk.info
        news = tk.news
        return df_1d, info, news
    except:
        return None, {}, []

def evaluate_ticker_pipeline(ticker, preloaded_df=None):
    if preloaded_df is not None:
        df_1d = preloaded_df.dropna()
        info = {}
        news = []
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            news = tk.news
        except:
            pass
    else:
        df_1d, info, news = fetch_daily_data(ticker)
        
    if df_1d is None or len(df_1d) < 200: return None
    
    s1_score, s1_flag, s1_det = run_strategy_1_rvol(df_1d)
    s2_score, s2_flag, s2_det = run_strategy_2_squeeze(df_1d)
    s3_score, s3_flag, s3_det = run_strategy_3_avwap(df_1d)
    s4_score, s4_flag, s4_det = run_strategy_4_trend(df_1d)
    s5_score, s5_flag, s5_det = run_strategy_5_fundamental(ticker, news)
    
    scores = {'Volume': s1_score, 'AVWAP': s3_score, 'Trend': s4_score, 'Fund': s5_score}
    flags_dict = {'Volume': s1_flag, 'Squeeze': s2_flag}
    alpha = aggregate_alpha_score(scores, flags_dict, is_risk_on)
    
    rec = "AVOID"
    if alpha >= 80: rec = "STRONG BUY"
    elif alpha >= 65: rec = "BUY"
    elif alpha >= 40: rec = "HOLD"
    
    atr_series, _ = calc_atr(df_1d, 14)
    current_atr = float(atr_series.iloc[-1])
    current_price = float(df_1d['Close'].iloc[-1])
    
    return {
        'Ticker': ticker,
        'Company Name': info.get('longName', ticker),
        'Sector': info.get('sector', 'Unknown'),
        'Alpha Score': alpha,
        'Recommendation': rec,
        'RVOL Spike': s1_flag,
        'Squeeze Active': s2_flag,
        'QTD AVWAP': s3_flag,
        'Trend Expansion': s4_flag,
        'Catalyst': s5_flag,
        'Details': {'Volume': s1_det, 'Squeeze': s2_det, 'AVWAP': s3_det, 'Trend': s4_det, 'Fund': s5_det},
        'Scores': {'Volume': s1_score, 'Squeeze': s2_score, 'AVWAP': s3_score, 'Trend': s4_score, 'Fund': s5_score},
        'ATR': current_atr,
        'Current Price': current_price
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
        return [], None
        
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
    if tier1_df.empty: return [], None
    
    tier1_df = tier1_df.sort_values(by='RS', ascending=False).head(100)
    return tier1_df['Ticker'].tolist(), data

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
    tier2_tickers, preloaded_data = run_tier_1()
    print(f"Tier 1 completed. Processing {len(tier2_tickers)} tickers in Tier 2...")
    
    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {}
        for t in tier2_tickers:
            df_preload = None
            if preloaded_data is not None:
                if isinstance(preloaded_data.columns, pd.MultiIndex):
                    df_preload = preloaded_data[t]
                else:
                    df_preload = preloaded_data
            future_to_ticker[executor.submit(evaluate_ticker_pipeline, t, df_preload)] = t
            
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
    st.error(f"**WARNING: Risk-Off Regime detected.** VIX: {vix_val:.2f} | SPY Price: {spy_price:.2f} (SMA20: {spy_sma20:.2f}). QTD AVWAP and Compression heavily weighted.")

tab1, tab2, tab3 = st.tabs(["Tab 1: Market Scanner", "Tab 2: Ticker Deep-Dive", "Tab 3: Risk Sandbox"])

with st.sidebar:
    st.title("Tier-1 Scanner Engine")
    st.write("The Tier-1 scanner evaluates all tickers against 5 daily momentum engines concurrently.")
    if st.button("Run Full Market Scan"):
        st.session_state['run_scan'] = True

if 'run_scan' in st.session_state and st.session_state['run_scan']:
    with tab1:
        tier2_tickers, preloaded_data = run_tier_1()
        if not tier2_tickers:
            st.warning("Tier 1 failed or no tickers found.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            final_results = []
            completed = 0
            total = len(tier2_tickers)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_ticker = {}
                for t in tier2_tickers:
                    df_preload = None
                    if preloaded_data is not None:
                        if isinstance(preloaded_data.columns, pd.MultiIndex):
                            df_preload = preloaded_data[t]
                        else:
                            df_preload = preloaded_data
                    future_to_ticker[executor.submit(evaluate_ticker_pipeline, t, df_preload)] = t
                    
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
                
                for col in ['RVOL Spike', 'Squeeze Active', 'QTD AVWAP', 'Trend Expansion', 'Catalyst']:
                    final_df[col] = final_df[col].apply(lambda x: "✅" if x else "❌")
                
                display_df = final_df.drop(columns=['Details', 'Scores', 'ATR', 'Current Price'])
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
                st.metric("Institutional Alpha Score", f"{res['Alpha Score']} / 100", help="Weighted average of all 5 daily engines")
                
                st.markdown("### 🔍 5-Engine Matrix Breakdown")
                cols = st.columns(5)
                strat_names = ['Relative Vol (RVOL)', 'Daily Squeeze (+15 Bonus)', 'QTD AVWAP', 'Fast Momentum', 'Fundamental']
                strat_keys = ['Volume', 'Squeeze', 'AVWAP', 'Trend', 'Fund']
                
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
                    st.warning(f"**No Dominant Setup Detected.** Alpha Score: {res['Alpha Score']}/100. Position sizing restricted.")
                else:
                    st.success(f"**Dominant Setup Detected!** Alpha Score: {res['Alpha Score']}/100")
                    
                    with st.expander("ATR Mathematical Exits & Position Sizing", expanded=True):
                        entry = res['Current Price']
                        atr = res['ATR']
                        tp = entry + (1.5 * atr)
                        sl = entry - (1 * atr)
                        
                        st.write(f"**14-Day ATR:** ${atr:.2f}")
                        st.write(f"**Entry Price:** ${entry:.2f}")
                        st.markdown(f"**🟢 Take-Profit (1.5x ATR):** ${tp:.2f}")
                        st.markdown(f"**🔴 Stop-Loss (1x ATR):** ${sl:.2f}")
                        
                        risk_amt = portfolio_size * (risk_pct / 100.0)
                        risk_per_share = abs(entry - sl)
                        if risk_per_share > 0:
                            allowed_shares = int(risk_amt / risk_per_share)
                            st.success(f"**Allowed Position Size:** {allowed_shares} shares (Risking ${risk_amt:.2f})")
                            
                # Isolated Automated Backtester
                st.markdown("---")
                st.markdown("### 🤖 Automated Historical Backtester")
                st.write("Run a 2-year vectorized historical simulation to validate the statistical edge of this setup.")
                if st.button("Run Historical Simulation (2-Year Vectorized)"):
                    with st.spinner("Running vectorized historical backtest..."):
                        try:
                            bt_data = yf.download(search_ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
                            if len(bt_data) > 100:
                                bt_data['EMA9'] = bt_data['Close'].ewm(span=9, adjust=False).mean()
                                bt_data['EMA21'] = bt_data['Close'].ewm(span=21, adjust=False).mean()
                                bt_data['VolSMA'] = bt_data['Volume'].rolling(20).mean()
                                
                                bb_up, bb_low = calc_bb(bt_data, 20, 2)
                                kc_up, kc_low = calc_kc(bt_data, 20, 1.5)
                                squeeze = (bb_up < kc_up) & (bb_low > kc_low)
                                
                                # Vectorized Signal proxy for the new daily engines
                                if isinstance(squeeze, pd.DataFrame):
                                    squeeze = squeeze.iloc[:, 0]
                                    
                                signal = (bt_data['Close'].iloc[:, 0] > bt_data['EMA9'].iloc[:, 0]) & (bt_data['EMA9'].iloc[:, 0] > bt_data['EMA21'].iloc[:, 0]) & ((bt_data['Volume'].iloc[:, 0] > 3.0 * bt_data['VolSMA'].iloc[:, 0]) | squeeze)
                                
                                bt_data['Signal'] = signal
                                bt_data['Fwd_Ret_5d'] = bt_data['Close'].iloc[:, 0].shift(-5) / bt_data['Close'].iloc[:, 0] - 1
                                
                                trades = bt_data[bt_data['Signal'] & (~bt_data['Signal'].shift(1).fillna(False))].dropna(subset=['Fwd_Ret_5d'])
                                
                                if len(trades) > 0:
                                    win_rate = (trades['Fwd_Ret_5d'] > 0).mean() * 100
                                    avg_ret = trades['Fwd_Ret_5d'].mean() * 100
                                    max_dd = trades['Fwd_Ret_5d'].min() * 100
                                    
                                    st.info(f"**Historical Simulation Complete (2-Year)**\n\nIdentified **{len(trades)}** structural setups in the past 24 months matching these fast momentum dynamics.\n- **Win Rate (5-Day Hold):** {win_rate:.1f}%\n- **Average Return per Trade:** {avg_ret:.2f}%\n- **Max Drawdown (Worst Trade):** {max_dd:.2f}%")
                                else:
                                    st.warning("No historical setups matched this strict Fast Momentum criteria in the last 2 years.")
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
