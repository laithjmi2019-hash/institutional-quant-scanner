import duckdb
import pandas as pd
import json
import os

DB_PATH = "data/db/antigravity_scans.duckdb"

def init_db():
    if not os.path.exists("data/db"):
        os.makedirs("data/db")
    con = duckdb.connect(DB_PATH)
    # Create the master table for caching scans
    con.execute("""
        CREATE TABLE IF NOT EXISTS market_scans (
            scan_date DATE,
            ticker VARCHAR,
            close_price DOUBLE,
            volume DOUBLE,
            rvol DOUBLE,
            avwap_distance DOUBLE,
            roic DOUBLE,
            fcf_yield DOUBLE,
            dcf_valuation DOUBLE,
            rc_weight DOUBLE,
            PRIMARY KEY (scan_date, ticker)
        )
    """)
    con.close()

def save_scan_results(df: pd.DataFrame):
    """
    Saves a DataFrame of scan results to DuckDB using vectorized insertion.
    """
    con = duckdb.connect(DB_PATH)
    # Using duckdb's native pandas integration to upsert
    con.execute("INSERT OR REPLACE INTO market_scans SELECT * FROM df")
    con.close()
    
def get_latest_scans() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    df = con.execute("""
        SELECT * FROM market_scans 
        WHERE scan_date = (SELECT MAX(scan_date) FROM market_scans)
        ORDER BY rc_weight DESC
    """).df()
    con.close()
    return df

def generate_sector_cache(tickers):
    """
    Generates a local sector cache and saves to sectors.json
    """
    import yfinance as yf
    import concurrent.futures
    
    sector_map = {}
    
    def fetch_sector(t):
        try:
            return t, yf.Ticker(t).info.get('sector', 'Unknown')
        except:
            return t, 'Unknown'
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_sector, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            t, s = future.result()
            sector_map[t] = s
            
    with open('sectors.json', 'w') as f:
        json.dump(sector_map, f, indent=4)
        
def load_sector_cache():
    if not os.path.exists('sectors.json'):
        return {}
    with open('sectors.json', 'r') as f:
        return json.load(f)
