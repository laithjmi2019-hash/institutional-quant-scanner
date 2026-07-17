import sys, os
sys.path.insert(0, '.')
import yfinance as yf
import pandas as pd
from engines.technical_structure import compute_technical_structure
from engines.split_adjuster import adjust_for_splits

test_tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'COST', 'AMD', 'MU', 'NFLX']

print("=" * 70)
print("STRUCTURAL GATE AUDIT - Project Antigravity X")
print("=" * 70)

passed = []
failed_rvol = []
failed_avwap = []
errors = []

for t in test_tickers:
    try:
        df = yf.download(t, period='1y', interval='1d', auto_adjust=False, progress=False)
        if df.empty:
            print(f"  {t}: EMPTY DATA")
            errors.append(t)
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = adjust_for_splits(df)
        tech = compute_technical_structure(df)

        rvol = tech['rvol']
        avwap = tech['avwap_distance']
        rsi = tech['rsi']

        gate1 = rvol >= 1.0
        gate2 = avwap >= 0

        if gate1 and gate2:
            status = "PASS"
            passed.append(t)
        elif not gate1:
            status = "FAIL (Low RVOL)"
            failed_rvol.append(t)
        else:
            status = "FAIL (Below AVWAP)"
            failed_avwap.append(t)

        print(f"  {t}: RVOL={rvol:.2f}, AVWAP_DIST={avwap:.2f}%, RSI={rsi:.1f} --> {status}")

    except Exception as e:
        print(f"  {t}: ERROR - {e}")
        errors.append(t)

print()
print("=" * 70)
print(f"PASSED: {len(passed)} | FAILED RVOL: {len(failed_rvol)} | FAILED AVWAP: {len(failed_avwap)} | ERRORS: {len(errors)}")
print(f"Passed tickers: {passed}")
print(f"Failed RVOL: {failed_rvol}")
print(f"Failed AVWAP: {failed_avwap}")
