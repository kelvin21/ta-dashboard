import os
import sqlite3
import argparse
import json
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np
import requests

# Optional async dependencies
try:
    import asyncio
    import aiohttp
    HAS_ASYNC = True
except ImportError:
    asyncio = None
    aiohttp = None
    HAS_ASYNC = False

# Optional dependency for SFTP upload
try:
    import paramiko
except Exception:
    paramiko = None

# Import dividend adjuster
try:
    from dividend_adjuster import detect_dividend_adjustment, apply_dividend_adjustment, scan_all_tickers_for_dividends, confirm_and_apply_adjustments
    HAS_DIVIDEND_ADJUSTER = True
except ImportError:
    HAS_DIVIDEND_ADJUSTER = False
    print("⚠️ dividend_adjuster module not found. Dividend adjustment detection disabled.")

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default database paths relative to script location
NEW_DB_PATH = os.getenv("PRICE_DB_PATH", os.path.join(SCRIPT_DIR, "price_data.db"))
DEFAULT_LOCAL_DB = os.getenv("REF_DB_PATH", os.path.join(SCRIPT_DIR, "analysis_results.db"))

# For deployment subdirectory support
DEPLOY_DB_PATH = os.path.join(SCRIPT_DIR, "ta-dashboard-deploy", "data", "price_data.db")

# TCBS API endpoint
TCBS_URL = "https://apipubaws.tcbs.com.vn/stock-insight/v2/stock/bars-long-term"

CHUNK = 500

# NEW: lookback window used by median/ autoscaling helpers
LOOKBACK_DAYS = 60


def create_db(db_path=NEW_DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            source TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_data(ticker, date)")
    # NEW: table to remember TCBS scaling per ticker
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tcbs_scaling (
            ticker TEXT PRIMARY KEY,
            scale INTEGER,
            detected_by TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            note TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f"✅ Created/ensured schema in {db_path}")


def copy_existing_market_data(source_db=DEFAULT_LOCAL_DB, target_db=NEW_DB_PATH, limit=None):
    """Copy market_data from current analysis_results.db into new DB (initial load)."""
    if not os.path.exists(source_db):
        raise FileNotFoundError(f"Source DB not found: {source_db}")
    create_db(target_db)
    src_conn = sqlite3.connect(source_db)
    tgt_conn = sqlite3.connect(target_db)

    # attempt to read available columns from source
    query = "SELECT ticker, date, open, high, low, close, volume FROM market_data"
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, src_conn)
    src_conn.close()
    if df.empty:
        print("No market_data rows found in source DB")
        tgt_conn.close()
        return 0

    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "date"})
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['source'] = 'local_copy'
    df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'source']]

    # upsert in batches
    cursor = tgt_conn.cursor()
    total = 0
    insert_sql = """
        INSERT OR REPLACE INTO price_data
        (ticker, date, open, high, low, close, volume, source, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """
    for start in range(0, len(df), CHUNK):
        chunk = df.iloc[start:start + CHUNK]
        params = [(
            row.ticker, row.date, _safe(row.open), _safe(row.high),
            _safe(row.low), _safe(row.close), int(_safe_int(row.volume)),
            row.source
        ) for row in chunk.itertuples()]
        cursor.executemany(insert_sql, params)
        tgt_conn.commit()
        total += len(params)
        print(f"  ↳ Copied {total}/{len(df)}")
    tgt_conn.close()
    print(f"✅ Copied {total} rows into {target_db}")
    return total


def _safe(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return float(x) if x is not None else None


def _safe_int(x):
    try:
        if pd.isna(x):
            return 0
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        return 0


def fetch_historical_price(ticker: str, days: int = 365, resolution: str = "D", timeout=15) -> pd.DataFrame:
    """Fetch stock historical price and volume data from TCBS API.
    Returns DataFrame with columns: tradingDate(datetime), open, high, low, close, volume
    """
    # Calculate date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    
    # Format dates as strings
    start_date_str = from_date.strftime("%Y-%m-%d")
    end_date_str = to_date.strftime("%Y-%m-%d")
    
    # Convert to Unix timestamps using time.mktime (matches TCBS API expectations)
    to_timestamp = int(time.mktime(time.strptime(end_date_str, "%Y-%m-%d")))
    from_timestamp = int(time.mktime(time.strptime(start_date_str, "%Y-%m-%d")))

    # Determine type based on ticker
    ticker_upper = ticker.upper()
    ticker_type = "index" if ticker_upper == "VNINDEX" else "stock"
    
    params = {
        "ticker": ticker,
        "type": ticker_type,
        "resolution": resolution,
        "from": from_timestamp,
        "to": to_timestamp,
        "countBack": days
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://tcinvest.tcbs.com.vn/"
    }

    try:
        print(f"[{ticker}] Requesting {days} days from TCBS API ({start_date_str} to {end_date_str}, type={ticker_type})...")
        print(f"[{ticker}] Params: from={from_timestamp}, to={to_timestamp}, countBack={days}, type={ticker_type}")
        r = requests.get(TCBS_URL, params=params, headers=headers, timeout=timeout)
        
        # Check HTTP status
        if r.status_code != 200:
            print(f"[{ticker}] ⚠️ HTTP {r.status_code}: {r.reason}")
            print(f"[{ticker}] Response text: {r.text[:500]}")
            r.raise_for_status()
        
        payload = r.json()
        
        # Check if response has data
        data = payload.get('data') or payload.get('bars') or payload
        
        if not data:
            print(f"[{ticker}] ⚠️ API returned empty response (no 'data' or 'bars' field)")
            print(f"[{ticker}] Response keys: {list(payload.keys()) if isinstance(payload, dict) else 'Not a dict'}")
            return pd.DataFrame()
        
        if isinstance(data, list) and len(data) == 0:
            print(f"[{ticker}] ⚠️ API returned empty data list (ticker may not exist or no trading data)")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        print(f"[{ticker}] ✓ Received {len(df)} rows from TCBS")
        
        # Normalize trading date column
        if 'tradingDate' in df.columns:
            # tradingDate might be ISO string or epoch ms
            sample = df['tradingDate'].iloc[0]
            if isinstance(sample, str) and 'T' in sample:
                df['tradingDate'] = pd.to_datetime(df['tradingDate'])
            else:
                df['tradingDate'] = pd.to_datetime(df['tradingDate'], unit='ms', errors='coerce')
        else:
            # try common columns
            date_col_found = False
            for col in ('datetime', 'timestamp', 'date'):
                if col in df.columns:
                    try:
                        df['tradingDate'] = pd.to_datetime(df[col], unit='ms', errors='coerce') \
                            if np.issubdtype(df[col].dtype, np.number) else pd.to_datetime(df[col], errors='coerce')
                        date_col_found = True
                        break
                    except Exception:
                        continue
            
            if not date_col_found:
                print(f"[{ticker}] ⚠️ No date column found in response. Available columns: {list(df.columns)}")

        # keep relevant columns
        cols_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ('open', 'high', 'low', 'close', 'volume'):
                cols_map[c] = lc
        df = df.rename(columns=cols_map)
        
        if 'tradingDate' not in df.columns:
            print(f"[{ticker}] ⚠️ No date column after normalization - skipping")
            return pd.DataFrame()
        
        # Check if we have OHLCV data
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"[{ticker}] ⚠️ Missing required columns: {missing_cols}. Available: {list(df.columns)}")
        
        df = df[['tradingDate'] + [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]]
        df = df.dropna(subset=['tradingDate'])
        df = df.sort_values('tradingDate').reset_index(drop=True)
        
        if df.empty:
            print(f"[{ticker}] ⚠️ DataFrame empty after filtering (all dates were NaT)")
        else:
            date_range = f"{df['tradingDate'].min().strftime('%Y-%m-%d')} to {df['tradingDate'].max().strftime('%Y-%m-%d')}"
            print(f"[{ticker}] ✓ Processed {len(df)} rows, date range: {date_range}")
        
        return df
        
    except requests.Timeout:
        print(f"[{ticker}] ❌ Request timeout after {timeout}s")
        return pd.DataFrame()
    except requests.ConnectionError as e:
        print(f"[{ticker}] ❌ Connection error: {str(e)[:100]}")
        return pd.DataFrame()
    except requests.HTTPError as e:
        print(f"[{ticker}] ❌ HTTP error: {e}")
        return pd.DataFrame()
    except requests.RequestException as e:
        print(f"[{ticker}] ❌ Request error: {str(e)[:100]}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"[{ticker}] ❌ JSON decode error: {str(e)[:100]}")
        print(f"[{ticker}] Response text: {r.text[:200] if 'r' in locals() else 'N/A'}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[{ticker}] ❌ Unexpected error: {type(e).__name__}: {str(e)[:100]}")
        import traceback
        print(f"[{ticker}] Traceback: {traceback.format_exc()[:300]}")
        return pd.DataFrame()


# Only define async functions if aiohttp is available
if HAS_ASYNC:
    async def fetch_historical_price_async(session: aiohttp.ClientSession, ticker: str, days: int = 365, resolution: str = "D", timeout=15) -> pd.DataFrame:
        """Async version of fetch_historical_price for concurrent fetching.
        Returns DataFrame with columns: tradingDate(datetime), open, high, low, close, volume
        """
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Format dates as strings
        start_date_str = from_date.strftime("%Y-%m-%d")
        end_date_str = to_date.strftime("%Y-%m-%d")
        
        # Convert to Unix timestamps using time.mktime (matches TCBS API expectations)
        to_timestamp = int(time.mktime(time.strptime(end_date_str, "%Y-%m-%d")))
        from_timestamp = int(time.mktime(time.strptime(start_date_str, "%Y-%m-%d")))

        # Determine type based on ticker
        ticker_upper = ticker.upper()
        ticker_type = "index" if ticker_upper == "VNINDEX" else "stock"
        
        params = {
            "ticker": ticker,
            "type": ticker_type,
            "resolution": resolution,
            "from": from_timestamp,
            "to": to_timestamp,
            "countBack": days
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://tcinvest.tcbs.com.vn/"
        }

        try:
            print(f"[{ticker}] Requesting {days} days from TCBS API (async, type={ticker_type})...")
            async with session.get(TCBS_URL, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status != 200:
                    text = await response.text()
                    print(f"[{ticker}] ⚠️ HTTP {response.status}: {response.reason}")
                    print(f"[{ticker}] Response text: {text[:500]}")
                    return pd.DataFrame()
                
                payload = await response.json()
                
                # Check if response has data
                data = payload.get('data') or payload.get('bars') or payload
                
                if not data:
                    print(f"[{ticker}] ⚠️ API returned empty response")
                    return pd.DataFrame()
                
                if isinstance(data, list) and len(data) == 0:
                    print(f"[{ticker}] ⚠️ API returned empty data list")
                    return pd.DataFrame()

                df = pd.DataFrame(data)
                print(f"[{ticker}] ✓ Received {len(df)} rows from TCBS (async)")
                
                # Normalize trading date column
                if 'tradingDate' in df.columns:
                    sample = df['tradingDate'].iloc[0]
                    if isinstance(sample, str) and 'T' in sample:
                        df['tradingDate'] = pd.to_datetime(df['tradingDate'])
                    else:
                        df['tradingDate'] = pd.to_datetime(df['tradingDate'], unit='ms', errors='coerce')
                else:
                    # try common columns
                    for col in ('datetime', 'timestamp', 'date'):
                        if col in df.columns:
                            try:
                                df['tradingDate'] = pd.to_datetime(df[col], unit='ms', errors='coerce') \
                                    if np.issubdtype(df[col].dtype, np.number) else pd.to_datetime(df[col], errors='coerce')
                                break
                            except Exception:
                                continue

                # keep relevant columns
                cols_map = {}
                for c in df.columns:
                    lc = c.lower()
                    if lc in ('open', 'high', 'low', 'close', 'volume'):
                        cols_map[c] = lc
                df = df.rename(columns=cols_map)
                
                if 'tradingDate' not in df.columns:
                    print(f"[{ticker}] ⚠️ No date column after normalization")
                    return pd.DataFrame()
                
                df = df[['tradingDate'] + [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]]
                df = df.dropna(subset=['tradingDate'])
                df = df.sort_values('tradingDate').reset_index(drop=True)
                
                if not df.empty:
                    date_range = f"{df['tradingDate'].min().strftime('%Y-%m-%d')} to {df['tradingDate'].max().strftime('%Y-%m-%d')}"
                    print(f"[{ticker}] ✓ Processed {len(df)} rows, date range: {date_range}")
                
                return df
                
        except asyncio.TimeoutError:
            print(f"[{ticker}] ❌ Request timeout after {timeout}s")
            return pd.DataFrame()
        except Exception as e:
            print(f"[{ticker}] ❌ Error: {type(e).__name__}: {str(e)[:100]}")
            return pd.DataFrame()


    async def fetch_and_scale_async(session: aiohttp.ClientSession, ticker: str, days: int = 365, resolution: str = "D", timeout=15) -> pd.DataFrame:
        """Async version of fetch_and_scale for concurrent fetching."""
        df = await fetch_historical_price_async(session, ticker, days=days, resolution=resolution, timeout=timeout)
        if df is None or df.empty:
            return df

        t_up = (ticker or "").upper()
        # Exclude VNINDEX from scaling
        if t_up == "VNINDEX":
            return df

        # Apply 1000x scaling to all other tickers
        try:
            for col in ('open', 'high', 'low', 'close'):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 1000
            print(f"⚙️ Scaled {ticker} data by dividing by 1000")
        except Exception as e:
            print(f"⚠️ Error applying scale for {ticker}: {e}")

        return df


    async def update_tickers_async(tickers: list, days: int = 365, db_path=NEW_DB_PATH, max_concurrent=10):
        """Update multiple tickers concurrently using async/await.
        
        Args:
            tickers: List of ticker symbols to fetch
            days: Number of days of historical data to fetch
            db_path: Path to database
            max_concurrent: Maximum number of concurrent requests (default: 10)
        
        Returns:
            Tuple of (success_count, error_count, no_data_count)
        """
        connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for ticker in tickers:
                task = fetch_and_scale_async(session, ticker, days=days, resolution=resolution, timeout=timeout)
                tasks.append((ticker, task))
            
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            success_count = 0
            error_count = 0
            no_data_count = 0
            
            for (ticker, _), result in zip(tasks, results):
                try:
                    if isinstance(result, Exception):
                        print(f"[{ticker}] ❌ Exception: {result}")
                        error_count += 1
                        continue
                    
                    df = result
                    if df is None or df.empty:
                        print(f"[{ticker}] ⚠️ No data")
                        no_data_count += 1
                        continue
                    
                    if 'tradingDate' in df.columns:
                        df['date'] = pd.to_datetime(df['tradingDate']).dt.strftime('%Y-%m-%d')
                    elif 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    
                    upserted = upsert_prices_from_df(df.assign(ticker=ticker), db_path=db_path, ticker=ticker, source='tcbs')
                    if upserted > 0:
                        success_count += 1
                        print(f"[{ticker}] ✅ Upserted {upserted} rows")
                    
                except Exception as e:
                    print(f"[{ticker}] ❌ Error processing result: {e}")
                    error_count += 1
            
            return success_count, error_count, no_data_count


    def update_all_tickers_via_api_async(target_db=NEW_DB_PATH, source_db=DEFAULT_LOCAL_DB, days=365, max_concurrent=10):
        """
        Fetch historical prices for all tickers concurrently using async/await.
        Much faster than sequential fetching.
        
        Args:
            target_db: Target database path
            source_db: Source database to read tickers from
            days: Number of days to fetch
            max_concurrent: Maximum concurrent requests (default: 10)
        
        Returns:
            Number of tickers successfully processed
        """
        tickers = _get_distinct_tickers_from_db(source_db)
        if not tickers:
            print("No tickers found in source DB.")
            return 0

        create_db(target_db)
        n = len(tickers)
        print(f"Updating {n} tickers from {source_db} -> {target_db} (days={days}, max_concurrent={max_concurrent})")
        
        # Run async update
        success_count, error_count, no_data_count = asyncio.run(update_tickers_async(tickers, days=days, db_path=target_db, max_concurrent=max_concurrent))
        
        print(f"\n✅ Async update complete:")
        print(f"  - Success: {success_count}")
        print(f"  - No data: {no_data_count}")
        print(f"  - Errors: {error_count}")
        
        return success_count


def upsert_prices_from_df(df: pd.DataFrame, db_path=NEW_DB_PATH, ticker=None, source='api'):
    """Upsert normalized DataFrame into price_data table. df must have tradingDate, open, high, low, close, volume."""
    if df.empty:
        return 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    insert_sql = """
        INSERT OR REPLACE INTO price_data
        (ticker, date, open, high, low, close, volume, source, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """
    df = df.copy()
    if 'tradingDate' in df.columns:
        df['date'] = pd.to_datetime(df['tradingDate']).dt.strftime('%Y-%m-%d')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    else:
        raise ValueError("DataFrame missing date/tradingDate column")

    df['ticker'] = ticker if ticker else df.get('ticker', None)
    if df['ticker'].isnull().any():
        raise ValueError("Ticker not provided and not present in DataFrame")

    rows = []
    for row in df.itertuples(index=False):
        rows.append((
            row.ticker if hasattr(row, 'ticker') else ticker,
            row.date,
            _safe(getattr(row, 'open', None)),
            _safe(getattr(row, 'high', None)),
            _safe(getattr(row, 'low', None)),
            _safe(getattr(row, 'close', None)),
            _safe_int(getattr(row, 'volume', 0)),
            source
        ))
    total = 0
    for i in range(0, len(rows), CHUNK):
        batch = rows[i:i+CHUNK]
        cursor.executemany(insert_sql, batch)
        conn.commit()
        total += len(batch)
        print(f"  ↳ Upserted {total}/{len(rows)}")
    conn.close()
    return total


# NEW helper: get local DB median for a ticker (used by fetch_and_scale / autoscaling)
def _get_local_db_median(ticker, db_paths=None, lookback_days=LOOKBACK_DAYS):
    """
    Return median close for ticker from first available DB in db_paths using recent lookback.
    Default search order: NEW_DB_PATH (price_data.db), then DEFAULT_LOCAL_DB (analysis_results.db).
    """
    if db_paths is None:
        db_paths = [NEW_DB_PATH, DEFAULT_LOCAL_DB]
    for db_path in db_paths:
        if not os.path.exists(db_path):
            continue
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            # prefer market_data table if present, else price_data
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('market_data','price_data')")
            tables = [r[0] for r in cur.fetchall()]
            if 'market_data' in tables:
                q = "SELECT close FROM market_data WHERE ticker = ? AND date >= date('now', ? || ' day')"
            elif 'price_data' in tables:
                q = "SELECT close FROM price_data WHERE ticker = ? AND date >= date('now', ? || ' day')"
            else:
                conn.close()
                continue
            cur.execute(q, (ticker, f"-{lookback_days}"))
            rows = [r[0] for r in cur.fetchall() if r[0] is not None]
            conn.close()
            if rows:
                return float(pd.Series(rows).median())
        except Exception:
            # ignore DB errors and try next DB
            continue
    return None


def fetch_and_scale(ticker: str, days: int = 365, resolution: str = "D", timeout=15, db_path=NEW_DB_PATH) -> pd.DataFrame:
    """Fetch and scale TCBS data for a ticker.
    Always divides OHLC by 1000 except for VNINDEX (which is left unscaled).
    """
    df = fetch_historical_price(ticker, days=days, resolution=resolution, timeout=timeout)
    if df is None or df.empty:
        return df

    t_up = (ticker or "").upper()
    # Exclude VNINDEX from scaling
    if t_up == "VNINDEX":
        return df

    # Apply 1000x scaling to all other tickers
    try:
        for col in ('open', 'high', 'low', 'close'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') / 1000
        print(f"⚙️ Scaled {ticker} data by dividing by 1000")
    except Exception as e:
        print(f"⚠️ Error applying scale for {ticker}: {e}")

    return df


# NEW: helpers to persist/read saved scaling
def get_saved_scale(ticker, db_path=NEW_DB_PATH):
    """Return saved integer scale for ticker (e.g., 1000) or None."""
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT scale FROM tcbs_scaling WHERE ticker = ?", (ticker.upper(),))
        row = cur.fetchone()
        conn.close()
        return int(row[0]) if row and row[0] is not None else None
    except Exception:
        return None

def save_scale(ticker, scale, db_path=NEW_DB_PATH, detected_by='autoscale', note=None):
    """Insert or update scale for ticker."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO tcbs_scaling (ticker, scale, detected_by, detected_at, note)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
            ON CONFLICT(ticker) DO UPDATE SET scale=excluded.scale, detected_by=excluded.detected_by, detected_at=excluded.detected_at, note=excluded.note
        """, (ticker.upper(), int(scale), detected_by, note))
        conn.commit()
        conn.close()
        print(f"[tcbs_scaling] Saved scale {scale} for {ticker} in {db_path}")
    except Exception as e:
        print(f"[tcbs_scaling] Failed to save scale for {ticker}: {e}")


def update_from_api(tickers, days=365, db_path=NEW_DB_PATH, source='tcbs'):
    """Fetch + upsert with autoscaling/default-scaling for TCBS data."""
    create_db(db_path)
    total = 0
    for t in tickers:
        print(f"🔎 Fetching {t} ({days} days)...")
        # Pass db_path so fetched scale is read/saved in the correct DB
        df = fetch_and_scale(t, days=days, db_path=db_path)
        if df is None or df.empty:
            print(f"  ⚠️ No data for {t}")
            continue

        upserted = upsert_prices_from_df(df.assign(ticker=t), db_path=db_path, ticker=t, source=source)
        print(f"  ✅ {t}: upserted {upserted} rows")
        total += upserted
        time.sleep(0.5)
    print(f"✅ API update complete. Total rows upserted: {total}")
    return total


def update_all_tickers_via_api(target_db=NEW_DB_PATH, source_db=DEFAULT_LOCAL_DB, days=365, pause=0.25, start_index=0, check_dividends=True):
    """
    Fetch historical prices for all tickers found in source_db and upsert into target_db.
    Returns number of tickers processed.
    """
    tickers = _get_distinct_tickers_from_db(source_db)
    if not tickers:
        print("No tickers found in source DB.")
        return 0

    create_db(target_db)
    total = 0               # total rows upserted
    processed = 0           # tickers processed
    n = len(tickers)
    print(f"Updating {n} tickers from {source_db} -> {target_db} (days={days})")

    for idx, ticker in enumerate(tickers[start_index:], start=start_index+1):
        try:
            print(f"[{idx}/{n}] Fetching {ticker} ...")
            # Use fetch_and_scale with target_db so scale is saved there
            df = fetch_and_scale(ticker, days=days, db_path=target_db)
            if df is None or df.empty:
                print(f"[{idx}/{n}] {ticker}: no data")
            else:
                # ensure 'tradingDate' present
                if 'tradingDate' not in df.columns and 'date' in df.columns:
                    df = df.rename(columns={'date': 'tradingDate'})
                upserted = upsert_prices_from_df(df.assign(ticker=ticker), db_path=target_db, ticker=ticker, source='tcbs')
                print(f"[{idx}/{n}] {ticker}: upserted {upserted} rows")
                total += upserted

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"[{idx}/{n}] {ticker}: error {e}")

        time.sleep(pause)
    
    print(f"Finished updating {processed} tickers; {total} rows upserted.")
    
    # Check for dividend adjustments after all updates
    if check_dividends and HAS_DIVIDEND_ADJUSTER and total > 0:
        print("\n" + "="*80)
        print("Checking for dividend adjustments...")
        print("="*80)
        adjustments = scan_all_tickers_for_dividends(db_path=target_db, debug=False)
        if adjustments:
            confirm_and_apply_adjustments(adjustments, db_path=target_db, auto_confirm=False)
    
    return processed


def main():
    parser = argparse.ArgumentParser(description="Build / update price_data SQLite DB from API and CSVs")
    parser.add_argument("--create", action="store_true", help="Create new DB schema")
    parser.add_argument("--copy-existing", action="store_true", help="Copy market_data from existing analysis_results.db")
    parser.add_argument("--source-db", type=str, default=DEFAULT_LOCAL_DB, help="Existing DB to copy from")
    parser.add_argument("--db", type=str, default=NEW_DB_PATH, help="Target DB path")
    parser.add_argument("--update-api", nargs='?', const='', type=str,
                        help="Comma separated tickers to fetch from API. If flag provided without value, update all tickers found in source DB.")
    parser.add_argument("--api-days", type=int, default=365, help="Days to fetch per ticker from API")
    parser.add_argument("--days", type=int, default=None, help="Alias for --api-days (accepts same value)")
    parser.add_argument("--api-pause", type=float, default=0.25, help="Pause (s) between API calls")
    parser.add_argument("--upload-sftp", action="store_true", help="Upload DB to remote via SFTP")
    parser.add_argument("--sftp-host", type=str, default=None)
    parser.add_argument("--sftp-user", type=str, default=None)
    parser.add_argument("--sftp-pass", type=str, default=None)
    parser.add_argument("--sftp-key", type=str, default=None)
    parser.add_argument("--sftp-path", type=str, default='.', help="Remote path for uploaded DB")
    parser.add_argument("--update-all-api", action="store_true", help="Fetch and upsert historical prices for all tickers found in source DB")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async/concurrent fetching (faster)")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent requests when using --async (default: 10)")
    parser.add_argument("--clean-price-units", action="store_true",
                        help="Run price unit inconsistency scan on price_data.db (dry-run by default)")
    parser.add_argument("--apply-clean", action="store_true",
                        help="Apply fixes when running --clean-price-units")
    parser.add_argument("--autoclean", action="store_true",
                        help="Automatically run clean after data-updating operations and apply fixes if --apply-clean is provided")
    parser.add_argument("--clean-ref-db", type=str, default="analysis_results.db",
                        help="Reference DB for unit comparison (default: analysis_results.db)")
    parser.add_argument("--run-clean", action="store_true",
                        help="Run the cleaner immediately and exit (alias for --clean-price-units)")
    parser.add_argument("--fix-from-date", type=str, default=None,
                        help="When running cleaner, only inspect and (optionally) fix rows on/after this date (YYYY-MM-DD)")
    parser.add_argument("--debug-ticker", type=str, default=None,
                        help="Show detailed debug info for a specific ticker during cleaning")
    parser.add_argument("--remove-tcbs", action="store_true",
                        help="Remove all rows from price_data where source='tcbs'")
    parser.add_argument("--remove-tcbs-since", type=str, default=None,
                        help="Remove tcbs rows only on/after this date (YYYY-MM-DD)")
    parser.add_argument("--remove-tcbs-tickers", type=str, default=None,
                        help="Comma-separated tickers to restrict tcbs removal")
    # NEW: force-rescale option
    parser.add_argument("--force-rescale-tcbs", action="store_true",
                        help="Force-rescale TCBS data by a fixed factor (use with --scale)")
    parser.add_argument("--scale", type=int, default=1000,
                        help="Scale factor for force-rescale (default: 1000)")
    parser.add_argument("--rescale-since", type=str, default=None,
                        help="Force-rescale TCBS rows only on/after this date (YYYY-MM-DD)")
    parser.add_argument("--rescale-tickers", type=str, default=None,
                        help="Comma-separated tickers to restrict force-rescale")

    # NEW: CSV import/export commands
    parser.add_argument("--import-csv", type=str, help="Import CSV file into database")
    parser.add_argument("--export-csv", type=str, help="Export database to CSV file")
    parser.add_argument("--csv-source", type=str, default="csv_import", help="Source label for CSV import")
    parser.add_argument("--csv-tickers", type=str, help="Comma-separated tickers for CSV export")
    parser.add_argument("--csv-since", type=str, help="Export only data since this date (YYYY-MM-DD)")
    
    # NEW: Dividend adjustment options
    parser.add_argument("--check-dividends", action="store_true", default=True,
                        help="Check for dividend adjustments after updating (default: enabled)")
    parser.add_argument("--no-check-dividends", action="store_false", dest="check_dividends",
                        help="Skip dividend adjustment check")
    parser.add_argument("--scan-dividends", action="store_true",
                        help="Scan database for dividend adjustments and optionally apply")
    parser.add_argument("--apply-dividends", action="store_true",
                        help="Automatically apply dividend adjustments without confirmation")
    
    args = parser.parse_args()

    # Resolve DB_PATH with priority: CLI arg > env var > default
    if args.db:
        DB_PATH = os.path.abspath(args.db)
    else:
        DB_PATH = os.getenv("PRICE_DB_PATH", NEW_DB_PATH)
        DB_PATH = os.path.abspath(DB_PATH)
    
    print(f"Using database: {DB_PATH}")

    # map legacy --days -> api_days if present
    if getattr(args, "days", None) is not None:
        args.api_days = args.days

    target_db = DB_PATH  # Use resolved DB_PATH instead of args.db
    data_changed = False

    # If user requested immediate run-clean, perform now and exit
    if args.run_clean or args.clean_price_units:
        dry_run = not args.apply_clean
        print("Running price unit scan on target DB:", target_db)
        print(f"Dry run: {dry_run}. Reference DB: {args.clean_ref_db}. Since date: {args.fix_from_date}")
        fixes = scan_and_fix(db_path=target_db, ref_db=args.clean_ref_db, dry_run=dry_run, since_date=args.fix_from_date, debug_ticker=args.debug_ticker)
        if fixes and not dry_run:
            print(f"Applied fixes for {len(fixes)} tickers.")
        elif fixes:
            print(f"Detected {len(fixes)} potential fixes (dry-run). Run with --apply-clean to apply.")
        else:
            print("No unit inconsistencies detected.")
        return

    if args.create:
        create_db(target_db)

    if args.copy_existing:
        copied = copy_existing_market_data(source_db=args.source_db, target_db=target_db)
        if copied and copied > 0:
            data_changed = True

    # NEW: support optional --update-api
    if args.update_api is not None:
        # flag provided without value -> args.update_api == '' -> use tickers from source DB
        if args.update_api == '':
            tickers = _get_distinct_tickers_from_db(args.source_db)
            if not tickers:
                print(f"No tickers found in source DB ({args.source_db}). Nothing to update.")
            else:
                print(f"Updating {len(tickers)} tickers from source DB ({args.source_db}) via API...")
                processed = update_all_tickers_via_api(target_db, source_db=args.source_db, days=args.api_days, pause=args.api_pause)
                if processed:
                    data_changed = True
        else:
            # user provided comma-separated tickers
            tickers = [t.strip().upper() for t in args.update_api.split(",") if t.strip()]
            if tickers:
                updated = update_from_api(tickers, days=args.api_days, db_path=target_db)
                if updated:
                    data_changed = True
        # done with update-api processing
        # after update-api we may want to run cleaning if autoclean requested
        if args.autoclean and data_changed:
            print("Autoclean requested: running dry-run cleaning now...")
            fixes = scan_and_fix(db_path=target_db, ref_db=args.clean_ref_db, dry_run=True)
            if fixes and args.apply_clean:
                print("Applying fixes as requested...")
                scan_and_fix(db_path=target_db, ref_db=args.clean_ref_db, dry_run=False)
        return

    if args.update_all_api:
        if args.use_async:
            # Use async version (much faster)
            processed = update_all_tickers_via_api_async(
                target_db=target_db,
                source_db=args.source_db,
                days=args.api_days,
                max_concurrent=args.max_concurrent
            )
        else:
            # Use sequential version (original)
            processed = update_all_tickers_via_api(
                target_db=target_db,
                source_db=args.source_db,
                days=args.api_days,
                pause=args.api_pause,
                check_dividends=args.check_dividends
            )
        
        if processed:
            data_changed = True
        if args.autoclean and data_changed:
            fixes = scan_and_fix(db_path=target_db, ref_db=args.clean_ref_db, dry_run=True)
            if fixes and args.apply_clean:
                scan_and_fix(db_path=target_db, ref_db=args.clean_ref_db, dry_run=False)
        return

    # SAFELY handle optional update_csv flag (avoid AttributeError)
    if getattr(args, 'update_csv', None):
        up_csv_count = update_from_csv(args.update_csv, db_path=target_db)
        if up_csv_count:
            data_changed = True

    # After any manual update operations (copy, api, csv), if autoclean requested run the cleaner
    if args.autoclean and data_changed:
        print("Autoclean requested: running dry-run cleaning now...")
        fixes = scan_and_fix(db_path=target_db, ref_db=args.clean_ref_db, dry_run=True)
        if fixes:
            print(f"Detected {len(fixes)} fixable tickers.")
            if args.apply_clean:
                print("Applying fixes...")
                scan_and_fix(db_path=target_db, ref_db=args.clean_ref_db, dry_run=False)
            else:
                print("Run with --apply-clean to apply the fixes.")
        else:
            print("No unit inconsistency detected.")

    if args.upload_sftp:
        if not args.sftp_host or not args.sftp_user:
            print("SFTP host/user required (--sftp-host, --sftp-user)")
        else:
            upload_db_sftp(target_db, args.sftp_host, args.sftp_user, password=args.sftp_pass, keyfile=args.sftp_key, remote_path=args.sftp_path)

    # If cleaning requested, run and exit
    if args.clean_price_units:
        dry_run = not args.apply_clean
        print("Running price unit scan on price_data.db")
        print(f"Dry run: {dry_run}. Reference DB: {args.clean_ref_db}")
        scan_and_fix(db_path=args.db, ref_db=args.clean_ref_db, dry_run=dry_run)
        return

    # Remove tcbs data if requested
    if args.remove_tcbs:
        tickers = None
        if args.remove_tcbs_tickers:
            tickers = [t.strip().upper() for t in args.remove_tcbs_tickers.split(",") if t.strip()]
        remove_tcbs_data(db_path=args.db, since_date=args.remove_tcbs_since, tickers=tickers)
        return
    
    # NEW: Force-rescale TCBS data
    if args.force_rescale_tcbs:
        tickers = None
        if args.rescale_tickers:
            tickers = [t.strip().upper() for t in args.rescale_tickers.split(",") if t.strip()]
        force_rescale_tcbs(db_path=args.db, scale=args.scale, since_date=args.rescale_since, tickers=tickers)
        return

    # NEW: Handle CSV import
    if args.import_csv:
        print(f"Importing CSV: {args.import_csv}")
        inserted, errors = import_csv_to_db(args.import_csv, db_path=DB_PATH, source=args.csv_source)
        if inserted > 0:
            print(f"Import complete: {inserted} rows")
        return  # Changed from sys.exit(0)
    
    # NEW: Handle CSV export
    if args.export_csv:
        print(f"Exporting to CSV: {args.export_csv}")
        tickers = [t.strip().upper() for t in args.csv_tickers.split(",")] if args.csv_tickers else None
        success = export_db_to_csv(
            db_path=DB_PATH,
            output_csv=args.export_csv,
            tickers=tickers,
            since_date=args.csv_since
        )
        return  # Changed from sys.exit(0 if success else 1)

    # NEW: Dividend adjustment scan
    if args.scan_dividends:
        if not HAS_DIVIDEND_ADJUSTER:
            print("❌ Dividend adjuster module not available")
            return
        
        print("Scanning for dividend adjustments...")
        adjustments = scan_all_tickers_for_dividends(db_path=target_db, debug=False)
        
        if adjustments and args.apply_dividends:
            confirm_and_apply_adjustments(adjustments, db_path=target_db, auto_confirm=True)
        elif adjustments:
            confirm_and_apply_adjustments(adjustments, db_path=target_db, auto_confirm=False)
        return


if __name__ == "__main__":
    main()
