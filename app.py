import sys
import os

# Ensure all engine imports resolve correctly on Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import concurrent.futures

st.set_page_config(page_title="Project Antigravity X", layout="wide")

from engines.business_quality import compute_business_quality
from engines.dynamic_valuation import compute_dcf_valuation
from engines.technical_structure import compute_technical_structure
from engines.split_adjuster import adjust_for_splits
from engines.market_scanner import init_db, save_scan_results, get_latest_scans, load_sector_cache, generate_sector_cache
from engines.portfolio_engine import optimize_portfolio
from core.components.state_sync import fetch_portfolio_state, sync_portfolio_state
from core.utils.webhook_dispatcher import dispatch_trade_signal
from data.tickers import ALL_TICKERS


def run_scanner():
    st.info("Initializing DuckDB Scan Engine...")
    init_db()

    results = []

    sectors = load_sector_cache()
    if not sectors:
        generate_sector_cache(ALL_TICKERS)
        sectors = load_sector_cache()

    progress = st.progress(0)
    status = st.empty()

    def evaluate(ticker):
        try:
            df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False)
            if df.empty:
                return None

            # Flatten MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if len(df) < 60:
                return None

            df = adjust_for_splits(df)
            tech = compute_technical_structure(df)
            bq = compute_business_quality(ticker)
            dcf = compute_dcf_valuation(ticker)

            # Fundamental Quality Gate: must have positive ROIC (profitable capital allocation)
            if bq['roic'] <= 0:
                return None

            return {
                'scan_date': datetime.datetime.now().date(),
                'ticker': ticker,
                'close_price': round(float(df['Close'].iloc[-1]), 2),
                'volume': int(df['Volume'].iloc[-1]),
                'rvol': tech['rvol'],
                'avwap_distance': tech['avwap_distance'],
                'rsi': tech['rsi'],
                'roic': bq['roic'],
                'fcf_yield': bq['fcf_yield'],
                'dcf_valuation': dcf['intrinsic_value'],
                'margin_of_safety': dcf['margin_of_safety'],
                'rc_weight': 0.0
            }
        except Exception:
            return None

    completed = 0
    total = len(ALL_TICKERS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(evaluate, t): t for t in ALL_TICKERS}
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            progress.progress(completed / total)
            status.text(f"Scanning {futures[future]}... ({completed}/{total})")
            res = future.result()
            if res:
                results.append(res)

    if results:
        res_df = pd.DataFrame(results)

        st.info("Running Risk Parity Portfolio Optimization on viable candidates...")
        viable_tickers = res_df['ticker'].tolist()

        if len(viable_tickers) > 1:
            data = yf.download(viable_tickers, period="1y", interval="1d")['Close']
            returns = data.pct_change().dropna()
            optimal_weights = optimize_portfolio(returns)
            res_df['rc_weight'] = res_df['ticker'].map(optimal_weights)
        elif len(viable_tickers) == 1:
            res_df['rc_weight'] = 1.0

        save_scan_results(res_df)
        st.success(f"Scan complete. {len(results)} viable candidates found.")
    else:
        st.warning("No tickers passed the structural gates today.")


# ----------------------------------------------------------------
# MAIN UI
# ----------------------------------------------------------------
st.title("Project Antigravity X: 9-Volume Quantitative Platform")

tab1, tab2, tab3 = st.tabs(["Scanner Engine", "Deep-Dive Terminal", "Portfolio Sync & Execution"])

with tab1:
    st.header("DuckDB Market Scanner")
    if st.button("Run Full System Scan"):
        run_scanner()

    try:
        latest = get_latest_scans()
        if not latest.empty:
            st.subheader("Latest Viable Candidates (Optimized by Risk Parity)")
            st.dataframe(latest)
    except Exception:
        st.write("No scan data available. Please run the scanner.")

with tab2:
    st.header("Fundamental Deep-Dive")
    ticker_input = st.text_input("Enter Ticker for Deep-Dive Analysis:")
    if ticker_input:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Business Quality")
            bq = compute_business_quality(ticker_input.upper())
            st.write(bq)
        with col2:
            st.subheader("Dynamic Valuation")
            dcf = compute_dcf_valuation(ticker_input.upper())
            st.write(dcf)

with tab3:
    st.header("Portfolio Sync & Cloud Execution")
    st.write("Persists state via GitHub API and executes via n8n webhook pipelines.")

    state = fetch_portfolio_state()
    st.json(state)

    if st.button("Simulate Order Execution"):
        success = dispatch_trade_signal("AAPL", "BUY", 100, {"strategy": "Risk Parity AVWAP"})
        if success:
            st.success("Signal dispatched to n8n successfully.")
        else:
            st.error("Signal failed. Ensure n8n webhook URL is configured in secrets.toml")

        state['holdings']['AAPL'] = state['holdings'].get('AAPL', 0) + 100
        sync_portfolio_state(state)
        st.success("Portfolio state synced to GitHub.")
