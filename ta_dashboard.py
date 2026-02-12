import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import asyncio
import sys

# Load .env file FIRST before any other imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed")

# Add the script directory to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Fix: Ensure DB_PATH is set correctly and used everywhere
# Set DB_PATH at the top after loading .env and checking for build_price_db
DB_PATH = os.getenv("PRICE_DB_PATH", os.path.join(SCRIPT_DIR, "price_data.db"))
DEFAULT_LOCAL_DB = os.getenv("REF_DB_PATH", os.path.join(SCRIPT_DIR, "analysis_results.db"))

# Auto-initialize database if it doesn't exist
try:
    from init_database import create_empty_database
    DB_PATH_CHECK = os.getenv("PRICE_DB_PATH", os.path.join(SCRIPT_DIR, "price_data.db"))
    DB_PATH_CHECK = "price_data.db"
    print(f"🔍 Checking database at: {DB_PATH_CHECK}")
    if not os.path.exists(DB_PATH_CHECK):
        create_empty_database(DB_PATH_CHECK)
except Exception as e:
    st.warning(f"⚠️ Could not initialize database: {e}")

# Try to import database adapter (supports both SQLite and MongoDB)
try:
    from db_adapter import get_db_adapter
    db = get_db_adapter()
    HAS_DB_ADAPTER = True
except ImportError:
    db = None
    HAS_DB_ADAPTER = False
    st.warning("⚠️ db_adapter not found. Using legacy SQLite mode.")

# OPTIMIZED: Make build_price_db optional and use environment variables
try:
    import build_price_db as bdb
    DB_PATH = os.getenv("PRICE_DB_PATH", bdb.NEW_DB_PATH)
    DEFAULT_LOCAL_DB = os.getenv("REF_DB_PATH", bdb.DEFAULT_LOCAL_DB)
    HAS_BDB = True
except ImportError as e:
    bdb = None
    DB_PATH = os.getenv("PRICE_DB_PATH", os.path.join(SCRIPT_DIR, "price_data.db"))
    DEFAULT_LOCAL_DB = os.getenv("REF_DB_PATH", os.path.join(SCRIPT_DIR, "analysis_results.db"))
    HAS_BDB = False
    # Only show warning in debug mode or if explicitly requested
    if os.getenv("SHOW_MODULE_WARNINGS", "false").lower() == "true":
        st.warning(f"⚠️ build_price_db module not found: {e}. TCBS refresh will be disabled.")

# OPTIMIZED: Make ticker_manager optional
try:
    import ticker_manager as tm
    HAS_TM = True
except ImportError:
    tm = None
    HAS_TM = False

# Import dividend adjuster
try:
    from dividend_adjuster import detect_dividend_adjustment, scan_all_tickers_for_dividends, apply_dividend_adjustment
    HAS_DIVIDEND_ADJUSTER = True
except ImportError:
    HAS_DIVIDEND_ADJUSTER = False

# Import intraday updater
try:
    from intraday_updater import (
        update_intraday_ohlcv, 
        update_multiple_tickers,
        update_intraday_with_adjustment_check,
        update_multiple_tickers_with_adjustment
    )
    HAS_INTRADAY_UPDATER = True
except ImportError:
    HAS_INTRADAY_UPDATER = False

st.set_page_config(page_title="MACD Reversal Dashboard", layout="wide", page_icon="📊")
st.markdown("#### MACD Histogram Reversal — Overview")

# Initialize session state for selected ticker
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None

# Export commonly used functions - updated to use db_adapter
@st.cache_resource(ttl=int(os.getenv("CACHE_TTL", "1800")))
def get_all_tickers(debug=False):
    if HAS_DB_ADAPTER:
        try:
            return db.get_all_tickers(debug=debug)
        except Exception as e:
            if debug:
                st.write(f"[DEBUG] DB adapter error: {e}")
            return []
    
    # Fallback to legacy SQLite
    if HAS_BDB:
        try:
            return bdb._get_distinct_tickers_from_db(DB_PATH, debug=debug)
        except Exception as e:
            if debug:
                st.write(f"[DEBUG] Error in get_all_tickers: {e}")
    
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT ticker FROM price_data WHERE ticker IS NOT NULL ORDER BY ticker")
        tickers = [r[0] for r in cur.fetchall()]
        conn.close()
        return tickers
    except Exception as e:
        if debug:
            st.write(f"[DEBUG] Fallback ticker query error: {e}")
        return []

@st.cache_resource(ttl=int(os.getenv("CACHE_TTL", "1800")))
def load_price_range(ticker, start_date, end_date):
    """
    Load price data for a single ticker within a date range.
    """
    if HAS_DB_ADAPTER:
        try:
            start_str = start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d")
            end_str = end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
            df = db.load_price_range(ticker, start_str, end_str)
            if not df.empty:
                df = df.sort_values('date', ascending=False)  # Sort by date in descending order
            return df
        except Exception as e:
            st.error(f"Database adapter error: {e}")
            return pd.DataFrame()
    
    # Fallback to legacy SQLite
    if not os.path.exists(DB_PATH):
        st.warning(f"Database file not found: {DB_PATH}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    try:
        start_str = start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d")
        end_str = end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
        
        query = """
            SELECT date, open, high, low, close, volume
            FROM price_data
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date DESC
        """
        df = pd.read_sql_query(query, conn, params=(ticker, start_str, end_str))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' is in datetime format
        return df
    except Exception as e:
        st.error(f"SQLite query error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_resource(ttl=int(os.getenv("CACHE_TTL", "1800")))
def load_price_range_multi(tickers, start_date, end_date, debug=False):
    """
    Load price data for multiple tickers at once.
    Returns a dict: {ticker: DataFrame}
    """
    if debug:
        st.write(f"[DEBUG] load_price_range_multi: tickers={tickers}, start_date={start_date}, end_date={end_date}")
    result = {}

    # Ensure tickers is a list and not empty
    if not isinstance(tickers, list) or not tickers:
        if debug:
            st.write("[DEBUG] load_price_range_multi: No tickers provided.")
        return {t: pd.DataFrame() for t in tickers}

    # Use DB adapter if available
    if HAS_DB_ADAPTER and hasattr(db, "load_price_range_multi"):
        try:
            start_str = start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d")
            end_str = end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
            
            # Split tickers into batches
            batch_size = int(os.getenv("BATCH_SIZE", 20))  # Default batch size is 50
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                if debug:
                    st.write(f"[DEBUG] Querying batch: {batch}")
                batch_result = db.load_price_range_multi(batch, start_str, end_str)
                result.update(batch_result)
                if debug:
                    st.write(f"[DEBUG] Batch result: {len(batch_result)} tickers loaded.")
            
            if debug:
                for t in tickers:
                    df = result.get(t, pd.DataFrame())
                    if not df.empty:
                        result[t] = df.sort_values('date', ascending=True)  # Sort by date in ascending order
                        st.write(f"[DEBUG] DB-adapter {t}: rows={len(df)}")
            return result
        except Exception as e:
            st.error(f"[DEBUG] Database adapter error: {e}")
            return {t: pd.DataFrame() for t in tickers}

    # Fallback: sequentially call load_price_range
    for t in tickers:
        try:
            df = load_price_range(t, start_date, end_date)
            if not df.empty:
                result[t] = df.sort_values('date', ascending=True)  # Sort by date in ascending order
            if debug:
                st.write(f"[DEBUG] Fallback {t}: rows={len(df)}")
        except Exception as e:
            if debug:
                st.write(f"[DEBUG] Error loading data for {t}: {e}")
            result[t] = pd.DataFrame()

    # Extra debug: show summary if all are empty
    if debug and all(df.empty for df in result.values()):
        st.write("[DEBUG] load_price_range_multi: All tickers returned empty DataFrames.")
        st.write(f"[DEBUG] DB_PATH: {DB_PATH}")
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT MIN(date), MAX(date) FROM price_data")
            minmax = cur.fetchone()
            st.write(f"[DEBUG] price_data date range in DB: {minmax}")
            cur.execute("SELECT COUNT(*) FROM price_data")
            total_rows = cur.fetchone()[0]
            st.write(f"[DEBUG] price_data total rows: {total_rows}")
            conn.close()
        except Exception as e:
            st.write(f"[DEBUG] Error querying DB for min/max date: {e}")

    return result

def macd_hist(close, fast=12, slow=26, signal=9):
    """
    Calculate MACD histogram matching AmiBroker's implementation.
    AmiBroker uses EMA for both MACD line and signal line.
    
    MACD Line = EMA(close, fast) - EMA(close, slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line
    """
    # Convert to float and ensure no NaN at start
    close = pd.Series(close).astype(float)
    
    # Calculate EMAs using adjust=False to match AmiBroker
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    
    # MACD Line
    macd_line = ema_fast - ema_slow
    
    # Signal Line - EMA of MACD line (not SMA!)
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    
    # Histogram
    hist = macd_line - macd_signal
    
    return macd_line, macd_signal, hist

def detect_stage(hist: pd.Series, lookback=20):
    """Return one of six stages for the latest bar with numeric prefix for sorting."""
    s = hist.dropna().reset_index(drop=True)
    if s.empty or len(s) < 3:
        return "N/A"
    last = float(s.iat[-1])
    prev = float(s.iat[-2])
    
    cross_up = (prev < 0 and last >= 0)
    cross_down = (prev > 0 and last <= 0)
    
    if cross_up:
        return "2. Confirmed Trough"
    if cross_down:
        return "5. Confirmed Peak"
    
    last_cross_idx = len(s) - 1
    for i in range(len(s)-2, max(0, len(s)-lookback-1), -1):
        if (s[i] < 0 and s[i+1] >= 0) or (s[i] > 0 and s[i+1] <= 0):
            last_cross_idx = i + 1
            break
    
    window_start = max(0, last_cross_idx)
    window = s.iloc[window_start:]
    
    if last < 0:
        if len(window) >= 3:
            min_idx_in_window = int(window.idxmin())
            min_val = float(window.min())
            if min_idx_in_window < len(s) - 1:
                recent_vals = s.iloc[min_idx_in_window:]
                if len(recent_vals) >= 2:
                    slope = np.polyfit(range(len(recent_vals)), recent_vals.values, 1)[0] if len(recent_vals) > 1 else (last - min_val)
                    if slope > 0:
                        return "1. Troughing"
        return "6. Falling below Zero"
    else:
        if len(window) >= 3:
            max_idx_in_window = int(window.idxmax())
            max_val = float(window.max())
            if max_idx_in_window < len(s) - 1:
                recent_vals = s.iloc[max_idx_in_window:]
                if len(recent_vals) >= 2:
                    slope = np.polyfit(range(len(recent_vals)), recent_vals.values, 1)[0] if len(recent_vals) > 1 else (last - max_val)
                    if slope < 0:
                        return "4. Peaking"
        return "3. Rising above Zero"

def stage_score(stage):
    # numeric scores: Confirmed Trough +3, Troughing +2, Rising +1, Peaking -2, Confirmed Peak -3, Falling -1
    # Now stages have prefixes, extract base name or use full match
    if "Confirmed Trough" in stage:
        return 3
    if "Troughing" in stage:
        return 2
    if "Rising above Zero" in stage:
        return 1
    if "Peaking" in stage:
        return -2
    if "Confirmed Peak" in stage:
        return -3
    if "Falling below Zero" in stage:
        return -1
    return 0

def _overview_row(t, df, lookback, debug):
    # df is already loaded for ticker t
    if df.empty:
        return None

    latest = df.iloc[-1]
    close = float(latest['close'])
    latest_date = latest['date']  # Extract the latest bar date
    
    current_vol = float(latest['volume'])
    close_series = df['close'].astype(float)

    _, _, histD = macd_hist(close_series)
    stageD = detect_stage(histD, lookback=lookback)
    histD_val = float(histD.iat[-1]) if not histD.empty and not pd.isna(histD.iat[-1]) else np.nan

    signal_note = ""
    if len(histD) >= 5:
        recent_hist = histD.tail(5).dropna()
        if len(recent_hist) >= 2:
            current = recent_hist.iloc[-1]
            prev = recent_hist.iloc[-2]
            momentum = current - prev
            cross_up = (prev < 0 and current >= 0)
            cross_down = (prev > 0 and current <= 0)
            if cross_up:
                signal_note = "↗0d"
            elif cross_down:
                signal_note = "↘0d"
            elif current < 0 and momentum > 0:
                days_to_cross = abs(current / momentum) if momentum > 0 else 999
                if days_to_cross <= 5 and days_to_cross >= 1:
                    signal_note = f"↗{int(days_to_cross)}d"
            elif current > 0 and momentum < 0:
                days_to_cross = abs(current / momentum) if momentum < 0 else 999
                if days_to_cross <= 5 and days_to_cross >= 1:
                    signal_note = f"↘{int(days_to_cross)}d"

    df_w_full = df.set_index('date').resample('W').agg({
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last',
        'volume':'sum'
    }).dropna(subset=['close'])

    if df_w_full.empty or len(df_w_full) < 3:
        stageW = "N/A"
        histW_val = np.nan
        if debug and len(df_w_full) > 0:
            st.write(f"[{t}] Weekly: {len(df_w_full)} weeks - insufficient")
    else:
        close_w = df_w_full['close'].astype(float)
        _, _, histW = macd_hist(close_w)
        stageW = detect_stage(histW, lookback=lookback)
        histW_val = float(histW.iat[-1]) if not histW.empty and not pd.isna(histW.iat[-1]) else np.nan

    df_m_full = df.set_index('date').resample('M').agg({
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last',
        'volume':'sum'
    }).dropna(subset=['close'])

    if df_m_full.empty or len(df_m_full) < 3:
        stageM = "N/A"
        histM_val = np.nan
        if debug:
            st.write(f"[{t}] Monthly: {len(df_m_full)} months - insufficient (need 3+)")
    else:
        close_m = df_m_full['close'].astype(float)
        _, _, histM = macd_hist(close_m)
        if histM.empty or pd.isna(histM.iat[-1]):
            stageM = "N/A"
            histM_val = np.nan
            if debug:
                st.write(f"[{t}] Monthly: {len(df_m_full)} months - MACD returned NaN")
        else:
            stageM = detect_stage(histM, lookback=lookback)
            histM_val = float(histM.iat[-1])
            if debug:
                st.write(f"[{t}] Monthly: {len(df_m_full)} months - MACD hist = {histM_val:.2f}, stage = {stageM}")

    df_hist = df[df['date'] < latest_date]
    if len(df_hist) >= 20:
        avg_vol = float(df_hist['volume'].tail(20).mean())
    elif len(df_hist) > 0:
        avg_vol = float(df_hist['volume'].mean())
    else:
        avg_vol = current_vol

    is_today = latest_date.date() == datetime.now().date()
    now = datetime.now()
    current_time = now.time()
    morning_start = datetime.strptime("09:00", "%H:%M").time()
    morning_end = datetime.strptime("11:30", "%H:%M").time()
    afternoon_start = datetime.strptime("13:00", "%H:%M").time()
    afternoon_end = datetime.strptime("14:45", "%H:%M").time()
    elapsed_trading_minutes = 0
    if morning_start <= current_time <= morning_end:
        elapsed_trading_minutes = (datetime.combine(datetime.today(), current_time) - datetime.combine(datetime.today(), morning_start)).seconds / 60
    elif afternoon_start <= current_time <= afternoon_end:
        morning_minutes = 150
        elapsed_afternoon = (datetime.combine(datetime.today(), current_time) - datetime.combine(datetime.today(), afternoon_start)).seconds / 60
        elapsed_trading_minutes = morning_minutes + elapsed_afternoon
    elif current_time > afternoon_end:
        elapsed_trading_minutes = 255
    total_trading_minutes = 255
    time_factor = elapsed_trading_minutes / total_trading_minutes if total_trading_minutes > 0 else 1.0

    if is_today and time_factor > 0 and time_factor < 1.0:
        adjusted_current_vol = current_vol / time_factor
    else:
        adjusted_current_vol = current_vol

    vol_ratio = adjusted_current_vol / avg_vol if avg_vol > 0 else 1.0

    score = 0.5*stage_score(stageD) + 0.3*stage_score(stageW) + 0.2*stage_score(stageM)
    
    # Calculate MACD slope (momentum indicator)
    macd_slope_3d = np.nan
    macd_slope_from_peak = np.nan
    
    if len(histD) >= 3:
        # Slope from last 3 days
        recent_3d = histD.tail(3).dropna()
        if len(recent_3d) >= 2:
            x = np.arange(len(recent_3d))
            y = recent_3d.values
            macd_slope_3d = np.polyfit(x, y, 1)[0]
    
    if len(histD) >= 5:
        # Slope from last significant peak/trough to now
        recent_hist = histD.tail(5).dropna()
        if len(recent_hist) >= 2:
            # Find the peak or trough in the recent window
            if stageD in ["1. Troughing", "6. Falling below Zero"]:
                # We're in a trough/falling phase - find the peak
                peak_idx = np.argmax(recent_hist.values)
            else:
                # We're in a peak/rising phase - find the trough
                peak_idx = np.argmin(recent_hist.values)
            
            if peak_idx < len(recent_hist) - 1:
                x = np.arange(len(recent_hist) - peak_idx)
                y = recent_hist.values[peak_idx:]
                macd_slope_from_peak = np.polyfit(x, y, 1)[0]
    
    return {
        "Ticker": t,
        "Close": f"{close:.1f}",
        "Latest Date": latest_date.strftime("%Y-%m-%d"),
        "Trend (Daily)": stageD,
        "Trend (Weekly)": stageW,
        "Trend (Monthly)": stageM,
        "Score": int(np.round(score)),
        "MACD_Hist_Daily": f"{histD_val:.2f}" if not np.isnan(histD_val) else "",
        "MACD_Hist_Weekly": f"{histW_val:.2f}" if not np.isnan(histW_val) else "",
        "MACD_Hist_Monthly": f"{histM_val:.2f}" if not np.isnan(histM_val) else "",
        "MACD_Slope_3d": f"{macd_slope_3d:.2f}" if not np.isnan(macd_slope_3d) else "",
        "MACD_Slope_Peak": f"{macd_slope_from_peak:.2f}" if not np.isnan(macd_slope_from_peak) else "",
        "Vol/AvgVol": f"{vol_ratio:.1f}x",
        "Signal": signal_note
    }

# Update build_overview to use the synchronous _overview_row
def build_overview(tickers, start_date, end_date, lookback=20, max_rows=200, debug=True):
    """
    Build an overview of tickers with MACD analysis for daily, weekly, and monthly timeframes.
    """
    if debug:
        st.write("[DEBUG] build_overview: Function called.")
        st.write(f"[DEBUG] Parameters: tickers={tickers}, start_date={start_date}, end_date={end_date}, lookback={lookback}, max_rows={max_rows}")

    rows = []

    # Defensive: ensure tickers is a list and not empty
    if not isinstance(tickers, list):
        try:
            tickers = list(tickers)
        except Exception:
            tickers = []
    if debug:
        st.write(f"[DEBUG] build_overview: tickers after conversion={tickers}")

    if not tickers:
        st.write("[DEBUG] build_overview: No tickers provided.")
        return pd.DataFrame()

    # Load price data for all tickers
    if debug:
        st.write("[DEBUG] Loading price data for tickers...")
    df_map = load_price_range_multi(tickers[:max_rows], start_date, end_date, debug=debug)
    if debug:
        st.write(f"[DEBUG] Loaded price data for {len(df_map)} tickers.")

    # Process each ticker
    for t in tickers[:max_rows]:
        if debug:
            st.write(f"[DEBUG] Processing ticker: {t}")
        df = df_map.get(t, pd.DataFrame())
        if df.empty:
            if debug:
                st.write(f"[DEBUG] {t}: No data available for this ticker.")
            continue

        try:
            # Generate the overview row for the ticker
            if debug:
                st.write(f"[DEBUG] Generating overview row for {t}...")
            row = _overview_row(t, df, lookback, debug)
            if row:
                rows.append(row)
                if debug:
                    st.write(f"[DEBUG] Successfully generated row for {t}: {row}")
            else:
                if debug:
                    st.write(f"[DEBUG] No row generated for {t}.")
        except Exception as e:
            if debug:
                st.write(f"[DEBUG] Error building overview row for {t}: {e}")

    if debug:
        st.write(f"[DEBUG] build_overview: Processed {len(rows)} tickers.")

    # Convert rows to a DataFrame
    if debug:
        st.write("[DEBUG] Converting rows to DataFrame...")
    df_out = pd.DataFrame(rows)

    # Sort by daily stage (1-6), then by MACD_Hist_Daily with stage-specific ordering
    if not df_out.empty:
        if debug:
            st.write("[DEBUG] Sorting DataFrame by stage and MACD histogram values...")
        # Extract numeric stage prefix for sorting (1-6)
        df_out['_stage_num'] = df_out['Trend (Daily)'].apply(lambda x: int(x.split('.')[0]) if '.' in str(x) else 999)

        # Convert MACD hist to numeric
        df_out['_macd_num'] = df_out['MACD_Hist_Daily'].replace('', np.nan).astype(float)

        # Create a custom sort key for MACD based on stage
        def get_macd_sort_key(row):
            stage = row['_stage_num']
            macd_val = row['_macd_num']

            if pd.isna(macd_val):
                return 999999  # Put NaN values at the end

            # Stages 1, 2, 6: descending (high to low) - negate the value
            if stage in [1, 2, 6]:
                return -macd_val
            # Stages 3, 4, 5: ascending (low to high) - keep positive
            else:
                return macd_val

        df_out['_macd_sort'] = df_out.apply(get_macd_sort_key, axis=1)

        # Sort by stage number first, then by the custom MACD sort key
        df_out = df_out.sort_values(['_stage_num', '_macd_sort'], ascending=[True, True])

        # Drop temporary columns
        df_out = df_out.drop(columns=['_stage_num', '_macd_num', '_macd_sort']).reset_index(drop=True)

        if debug:
            st.write("[DEBUG] DataFrame sorted successfully.")

    if debug:
        st.write("[DEBUG] build_overview completed.")
        st.write(f"[DEBUG] Final DataFrame:\n{df_out}")

    return df_out

def _get_db_stats(db_path):
    """Return diagnostic info for a sqlite DB: existence, size, mtime, table row counts and sample rows."""
    info = {
        "path": db_path,
        "exists": os.path.exists(db_path),
        "size_bytes": None,
        "modified": None,
        "tables": {},
        "errors": []
    }
    if not info["exists"]:
        return info
    try:
        info["size_bytes"] = os.path.getsize(db_path)
        info["modified"] = datetime.fromtimestamp(os.path.getmtime(db_path)).isoformat()
    except Exception as e:
        info["errors"].append(f"fs-error: {e}")

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        for tbl in ("price_data", "market_data", "tcbs_scaling"):
            if tbl in tables:
                try:
                    cur.execute(f"SELECT COUNT(1) FROM {tbl}")
                    cnt = cur.fetchone()[0]
                except Exception as e:
                    cnt = f"err:{e}"
                sample = []
                try:
                    cur.execute(f"SELECT * FROM {tbl} LIMIT 5")
                    cols = [d[0] for d in cur.description] if cur.description else []
                    rows = cur.fetchall()
                    sample = [dict(zip(cols, r)) for r in rows]
                except Exception as e:
                    sample = [f"sample-error: {e}"]
                info["tables"][tbl] = {"count": cnt, "sample": sample}
            else:
                info["tables"][tbl] = {"count": 0, "sample": []}
        conn.close()
    except Exception as e:
        info["errors"].append(f"db-error: {e}")
    return info

# NEW: Move style_stage_column here (top-level helper)
def style_stage_column(val):
    """Return CSS style for a stage cell based on numeric prefix (1-6)."""
    # Extract numeric prefix (e.g., "1. Troughing" -> 1)
    try:
        prefix = int(val.split('.')[0])
    except Exception:
        return ""
    # Green shades for stages 1-3, red shades for 4-6
    colors = {
        1: "background-color: #c8e6c9; color: black",        # Pale green (Troughing)
        2: "background-color: #39ff14; color: black",        # Neon green (Confirmed Trough)
        3: "background-color: #2e7d32; color: white",        # Dark green (Rising above Zero)
        4: "background-color: #ffccbc; color: black",        # Pale red/orange (Peaking)
        5: "background-color: #ff5252; color: white",        # Bright red (Confirmed Peak)
        6: "background-color: #c62828; color: white"         # Dark red (Falling below Zero)
    }
    return colors.get(prefix, "")

# NEW: Style functions for different columns
def style_vol_ratio(val):
    """Style Vol/AvgVol cell - green for high volume, gray for low."""
    try:
        ratio = float(val.rstrip('x'))
    except Exception:
        return ""
    # Green gradient for high volume (>1.5x), gray for low (<0.8x), white for normal
    if ratio >= 1.5:
        return "background-color: #66bb6a; color: white"  # Green
    elif ratio >= 1.2:
        return "background-color: #aed581; color: black"  # Light green
    elif ratio >= 1.0:
        return "background-color: #e8f5e9; color: black"  # Very light green
    elif ratio >= 0.8:
        return "background-color: #f5f5f5; color: black"  # Light gray
    else:
        return "background-color: #bdbdbd; color: black"  # Gray

def style_by_score(val, score):
    """Style Ticker/Close cells based on score value."""
    # Green for positive scores, red for negative
    if score >= 2:
        return "background-color: #66bb6a; color: white"  # Strong green
    elif score >= 1:
        return "background-color: #aed581; color: black"  # Light green
    elif score >= 0:
        return "background-color: #e8f5e9; color: black"  # Very light green
    elif score >= -1:
        return "background-color: #ffccbc; color: black"  # Light red
    elif score >= -2:
        return "background-color: #ff8a80; color: white"  # Medium red
    else:
        return "background-color: #ff5252; color: white"  # Strong red

def style_macd_by_trend(val, trend):
    """Style MACD hist cells based on their corresponding trend stage."""
    # Reuse stage_score colors but extract from trend string
    try:
        prefix = int(trend.split('.')[0])
    except Exception:
        return ""
    colors = {
        1: "background-color: #c8e6c9; color: black",
        2: "background-color: #39ff14; color: black",
        3: "background-color: #2e7d32; color: white",
        4: "background-color: #ffccbc; color: black",
        5: "background-color: #ff5252; color: white",
        6: "background-color: #c62828; color: white"
    }
    return colors.get(prefix, "")

def style_macd_slope(val):
    """Style MACD slope cell - green for positive, red for negative."""
    try:
        slope = float(val)
        if slope > 0.001:
            return "background-color: #66bb6a; color: white"  # Green for positive slope
        elif slope < -0.001:
            return "background-color: #ff5252; color: white"  # Red for negative slope
        else:
            return "background-color: #f5f5f5; color: black"  # Gray for near-zero
    except Exception:
        return ""

def plot_multi_tf_macd(ticker, start_date, end_date, lookback):
    """Plot candlestick + MACD histograms for daily/weekly/monthly in subplots."""
    df = load_price_range(ticker, start_date, end_date)
    
    # Debug: Check if data is loaded
    if df.empty:
        st.warning(f"No data available for ticker: {ticker}")
        return
    st.write(f"[DEBUG] Loaded data for {ticker}: {len(df)} rows")

    # Ensure data is sorted in ascending order for calculations
    df = df.sort_values('date', ascending=True).reset_index(drop=True)
    
    # Pre-convert close to float once
    close = df['close'].astype(float)
    
    # Compute daily MACD
    _, _, histD = macd_hist(close)
    stageD = detect_stage(histD, lookback=lookback)
    st.write(f"[DEBUG] Daily MACD stage for {ticker}: {stageD}")
    
    # Pre-calculate weekly resampling
    df_w = df.set_index('date').resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(subset=['close']).reset_index()
    
    # Debug: Check weekly data
    if df_w.empty:
        st.warning(f"No weekly data available for ticker: {ticker}")
        return
    st.write(f"[DEBUG] Weekly data for {ticker}: {len(df_w)} rows")
    
    # Weekly MACD
    close_w = df_w['close'].astype(float)
    _, _, histW = macd_hist(close_w)
    stageW = detect_stage(histW, lookback=lookback)
    st.write(f"[DEBUG] Weekly MACD stage for {ticker}: {stageW}")
    
    # Pre-calculate monthly resampling
    df_m = df.set_index('date').resample('M').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(subset=['close']).reset_index()
    
    # Debug: Check monthly data
    if df_m.empty:
        st.warning(f"No monthly data available for ticker: {ticker}")
        return
    st.write(f"[DEBUG] Monthly data for {ticker}: {len(df_m)} rows")
    
    # Monthly MACD
    close_m = df_m['close'].astype(float)
    _, _, histM = macd_hist(close_m)
    stageM = detect_stage(histM, lookback=lookback)
    st.write(f"[DEBUG] Monthly MACD stage for {ticker}: {stageM}")
    
    # Build subplots: candlestick + 3 MACD hists
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.16, 0.17, 0.17], vertical_spacing=0.02,
                        subplot_titles=("Price", f"MACD Hist (Daily) - {stageD}", f"MACD Hist (Weekly) - {stageW}", f"MACD Hist (Monthly) - {stageM}"))
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    
    # Daily MACD hist
    df['histD'] = histD.values
    fig.add_trace(go.Bar(x=df['date'], y=df['histD'], name='Daily', marker_color=['#1f77b4' if v >= 0 else '#ff7f0e' for v in df['histD']]), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=[0] * len(df), mode='lines', line=dict(color='black', width=1), showlegend=False), row=2, col=1)
    
    # Weekly MACD hist
    df_w['histW'] = histW.values
    fig.add_trace(go.Bar(x=df_w['date'], y=df_w['histW'], name='Weekly', marker_color=['#1f77b4' if v >= 0 else '#ff7f0e' for v in df_w['histW']]), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_w['date'], y=[0] * len(df_w), mode='lines', line=dict(color='black', width=1), showlegend=False), row=3, col=1)
    
    # Monthly MACD hist
    df_m['histM'] = histM.values
    fig.add_trace(go.Bar(x=df_m['date'], y=df_m['histM'], name='Monthly', marker_color=['#1f77b4' if v >= 0 else '#ff7f0e' for v in df_m['histM']]), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_m['date'], y=[0] * len(df_m), mode='lines', line=dict(color='black', width=1), showlegend=False), row=4, col=1)
    
    fig.update_layout(title=f"{ticker} — Multi-Timeframe MACD", xaxis_rangeslider_visible=False, template='plotly_white', height=900)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD Hist", row=2, col=1)
    fig.update_yaxes(title_text="MACD Hist", row=3, col=1)
    fig.update_yaxes(title_text="MACD Hist", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# --- UI layout ---------------------------------------------------------------
sidebar = st.sidebar

# --- Overview Controls -------------------------------------------------------
sidebar.header("Overview Controls")

# OPTIMIZED: Show DB path and module status
if sidebar.checkbox("Show system info", value=False):
    st.sidebar.markdown("---")
    st.sidebar.caption(f"**DB Path:** `{DB_PATH}`")
    st.sidebar.caption(f"**Ref DB:** `{DEFAULT_LOCAL_DB}`")
    st.sidebar.caption(f"**TCBS Module:** {'✓ Available' if HAS_BDB else '✗ Not Available'}")
    st.sidebar.caption(f"**Ticker Manager:** {'✓ Available' if HAS_TM else '✗ Not Available'}")

debug = sidebar.checkbox("Show debug info (DB diagnostics)", value=False)

# OPTIMIZED: Cache clear button
if sidebar.button("Clear cache & reload"):
    try:
        load_price_range.clear()
        get_all_tickers.clear()
        # Clear overview cache from database
        if HAS_DB_ADAPTER:
            db.clear_overview_cache()
            st.success("✅ Cache cleared (including database cache)")
        else:
            st.success("✅ Cache cleared")
    except Exception as e:
        st.error(f"Error clearing cache: {e}")

all_tickers = get_all_tickers(debug=debug)

# New: Load include/exclude tickers from database
if  not HAS_DB_ADAPTER:
    include_tickers_db = ""
    exclude_tickers_db = ""
else:
    include_tickers_db = db.get_setting("include_tickers")
    exclude_tickers_db = db.get_setting("exclude_tickers")

# New: Add text boxes for filtering tickers
include_tickers_input = sidebar.text_area(
    "Include Tickers (comma-separated)", 
    value=include_tickers_db,
    placeholder="e.g., FPT, VCB, DCM",
    help="Only display these tickers in the overview table. Leave empty to include all."
)
exclude_tickers_input = sidebar.text_area(
    "Exclude Tickers (comma-separated)", 
    value=exclude_tickers_db,
    placeholder="e.g., VIC, HPG",
    help="Exclude these tickers from the overview table. Leave empty to exclude none."
)

# Parse the include and exclude tickers
include_tickers = [t.strip().upper() for t in include_tickers_input.split(",") if t.strip()]
exclude_tickers = [t.strip().upper() for t in exclude_tickers_input.split(",") if t.strip()]

# New: Save settings button
if sidebar.button("💾 Save Ticker Filters"):
    if HAS_DB_ADAPTER:
        try:
            success1 = db.set_setting("include_tickers", include_tickers_input)
            success2 = db.set_setting("exclude_tickers", exclude_tickers_input)
            if success1 and success2:
                st.success("✅ Ticker filters saved!")
            else:
                st.error("❌ Failed to save ticker filters.")
                if not success1:
                    st.error("❌ Failed to save include_tickers.")
                if not success2:
                    st.error("❌ Failed to save exclude_tickers.")
                # Additional debug info
                st.write(f"[DEBUG] include_tickers_input: '{include_tickers_input}'")
                st.write(f"[DEBUG] exclude_tickers_input: '{exclude_tickers_input}'")
                st.write(f"[DEBUG] HAS_DB_ADAPTER: {HAS_DB_ADAPTER}")
                st.write(f"[DEBUG] db type: {type(db)}")
        except Exception as e:
            st.error(f"❌ Exception while saving ticker filters: {e}")
            st.write(f"[DEBUG] Exception details: {str(e)}")
    else:
        st.warning("⚠️ Database adapter not available. Settings not saved.")

days_back = sidebar.number_input("Days back for analysis", min_value=365, max_value=10000, value=3000)
lookback = sidebar.slider("Lookback (bars) for trough/peak detection", 5, 60, 20)

# date range
end_date = datetime.now().date()
start_date = end_date - timedelta(days=int(days_back))

if debug:
    st.sidebar.write(f"[DEBUG] Date range: {start_date} to {end_date}")
    st.sidebar.write(f"[DEBUG] Total tickers: {len(all_tickers)}")

sidebar.markdown("---")

# --- Intraday OHLCV Update Section (MOVED UP) --------------------------------
if HAS_INTRADAY_UPDATER:
    with sidebar.expander("📊 Update Intraday OHLCV", expanded=False):
        st.markdown("**Fetch today's intraday data and update database**")
        st.caption("Updates today's OHLCV data from TCBS intraday API")
        
        st.markdown("---")
        
        # Single ticker update
        st.markdown("**Update Single Ticker**")
        intraday_ticker = st.text_input("Ticker symbol", key="intraday_ticker", placeholder="e.g., VIC")
        
        # Price scaling option
        scale_intraday_price = st.checkbox("Scale prices (÷1000)", value=True, key="scale_price_single",
                                          help="TCBS intraday prices are in VND x1000, enable this to scale to actual VND")
        
        # Adjustment check option
        check_adjustment = st.checkbox("Auto-detect price adjustments", value=True, key="check_adj_single",
                                       help="Compare yesterday's price to detect dividends/splits and adjust historical data")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Update Today's Data", use_container_width=True):
                if intraday_ticker:
                    with st.spinner(f"Fetching intraday data for {intraday_ticker.upper()}..."):
                        success, message, rows, adj_applied = update_intraday_with_adjustment_check(
                            intraday_ticker.upper(), 
                            db_path=DB_PATH,
                            interval='1D',
                            source='intraday',
                            scale_price=scale_intraday_price,
                            check_adjustment=check_adjustment,
                            debug=False
                        )
                        
                        if success:
                            if adj_applied:
                                st.warning(f"⚠️ {message}")
                            else:
                                st.success(f"✅ {message}")
                            # Clear cache to show updated data
                            try:
                                load_price_range.clear()
                            except:
                                pass
                        else:
                            st.error(f"❌ {message}")
                else:
                    st.error("Enter a ticker symbol")
        
        with col2:
            st.caption("Updates OHLC for today using real-time tick data")
        
        st.markdown("---")
        
        # Bulk update - MANUAL BUTTON
        st.markdown("**Bulk Update (Manual)**")
        st.caption("Update today's data for all tickers in database")
        
        # Price scaling option for bulk
        scale_bulk_price = st.checkbox("Scale prices (÷1000)", value=True, key="scale_price_bulk",
                                       help="TCBS intraday prices are in VND x1000")
        
        # Adjustment check for bulk
        check_bulk_adjustment = st.checkbox("Auto-detect price adjustments", value=True, key="check_adj_bulk",
                                            help="Auto-adjust historical data for dividends/splits")
        
        bulk_confirm = st.checkbox("I confirm: update ALL tickers", key="bulk_intraday_confirm")
        
        if st.button("🔄 Update All Tickers (Manual)", use_container_width=True, disabled=not bulk_confirm):
            if bulk_confirm:
                with st.spinner("Updating intraday data for all tickers..."):
                    results = update_multiple_tickers_with_adjustment(
                        all_tickers,
                        db_path=DB_PATH,
                        interval='1D',
                        source='intraday',
                        scale_price=scale_bulk_price,
                        check_adjustment=check_bulk_adjustment,
                        debug=False
                    )
                    
                    successful = sum(1 for r in results.values() if r['success'])
                    failed = len(results) - successful
                    adjusted = sum(1 for r in results.values() if r.get('adjustment_applied', False))
                    
                    # Store results in session state
                    st.session_state['intraday_manual_update_done'] = True
                    st.session_state['intraday_update_results'] = results
                    
                    if successful > 0:
                        st.success(f"✅ Updated {successful}/{len(results)} tickers")
                    if adjusted > 0:
                        st.warning(f"⚠️ Applied price adjustments to {adjusted} ticker(s)")
                    if failed > 0:
                        st.error(f"❌ Failed: {failed}/{len(results)} tickers")
                    
                    # Show details in expander
                    with st.expander("📋 View Details"):
                        for ticker, result in results.items():
                            status = "✅" if result['success'] else "❌"
                            adj_icon = " 🔄" if result.get('adjustment_applied', False) else ""
                            st.caption(f"{status}{adj_icon} {ticker}: {result['message']}")
                    
                    # Clear cache
                    try:
                        load_price_range.clear()
                    except:
                        pass
        
        st.markdown("---")
        st.markdown("**ℹ️ About Intraday Updates**")
        st.caption("""
        This feature:
        - Fetches real-time tick data from TCBS API
        - Auto-detects price adjustments (dividends/splits)
        - Compares yesterday's DB price vs fresh fetch
        - Adjusts historical data automatically
        - Scales prices by ÷1000 (TCBS format)
        - Converts to daily OHLCV bars
        - Updates/inserts today's data
        - Uses async fetching for speed
        - **Auto-runs ONCE on first dashboard open during market hours (9AM-5PM)**
        - **After first run, use manual refresh button above**
        """)
else:
    sidebar.markdown("---")
    sidebar.info("📊 Intraday updater not available (missing intraday_updater.py)")

# --- Auto Intraday Update During Market Hours (ONLY ONCE) --------------------
# DISABLED: Auto-update turned off by user
if False and HAS_INTRADAY_UPDATER and 'intraday_auto_update_done' not in st.session_state:
    now = datetime.now()
    current_time = now.time()
    today_weekday = now.weekday()  # Monday=0, Sunday=6

    # Market hours: 9:00 AM to 5:00 PM, skip Sat/Sun (weekday 5,6)
    market_start = datetime.strptime("09:00", "%H:%M").time()
    market_end = datetime.strptime("17:00", "%H:%M").time()

    # Only auto-update if during market hours, not Sat/Sun, AND database already has data
    if market_start <= current_time <= market_end and today_weekday < 5 and len(all_tickers) > 0:
        # Check if today's data already exists in database
        today_str = datetime.now().strftime("%Y-%m-%d")
        has_today_data = False
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            # Check if any ticker has today's data
            cur.execute("SELECT COUNT(*) FROM price_data WHERE date = ?", (today_str,))
            count = cur.fetchone()[0]
            has_today_data = count > 0
            conn.close()
        except Exception as e:
            if debug:
                st.sidebar.write(f"[DEBUG] Error checking today's data: {e}")
        
        if has_today_data:
            # Today's data already exists - skip auto update
            st.session_state['intraday_auto_update_done'] = True
            st.sidebar.info(f"✅ Today's data ({today_str}) already in database. Use manual refresh if needed.")
        else:
            # No today's data - run auto update
            st.sidebar.info("🔄 First-time auto-update of intraday data...")
            
            with st.spinner("Fetching today's intraday data for all tickers..."):
                results = update_multiple_tickers_with_adjustment(
                    all_tickers,
                    db_path=DB_PATH,
                    interval='1D',
                    source='intraday',
                    scale_price=True,
                    check_adjustment=True,
                    debug=False
                )
                
                successful = sum(1 for r in results.values() if r['success'])
                failed = len(results) - successful
                
                # Store results in session state
                st.session_state['intraday_auto_update_done'] = True
                st.session_state['intraday_update_results'] = results
                
                if failed > 0:
                    failed_tickers = [t for t, r in results.items() if not r['success']]
                    st.sidebar.warning(f"⚠️ {failed} ticker(s) not updated via intraday: {', '.join(failed_tickers[:5])}" + 
                                      (f" and {len(failed_tickers)-5} more" if len(failed_tickers) > 5 else ""))
                else:
                    st.sidebar.success(f"✅ Auto-updated {successful} tickers via intraday")
                
                st.sidebar.caption("💡 Use manual refresh button for subsequent updates")
                
                # Clear cache to show updated data
                try:
                    load_price_range.clear()
                except:
                    pass
    else:
        # Mark as "done" even if not in market hours or on Sat/Sun to prevent repeated checks
        st.session_state['intraday_auto_update_done'] = True
        if today_weekday >= 5:
            st.sidebar.info("ℹ️ Weekend detected (Sat/Sun). Intraday update skipped.")
        elif len(all_tickers) == 0:
            st.sidebar.info("ℹ️ No tickers in database. Add tickers first.")

# Show intraday update status if available
if 'intraday_update_results' in st.session_state:
    results = st.session_state['intraday_update_results']
    failed = sum(1 for r in results.values() if not r['success'])
    if failed > 0:
        failed_tickers = [t for t, r in results.items() if not r['success']]
        with sidebar.expander(f"⚠️ Intraday Update Issues ({failed})", expanded=False):
            for ticker in failed_tickers:
                st.caption(f"❌ {ticker}: {results[ticker]['message']}")

sidebar.markdown("---")

# --- TCBS Refresh Section (MOVED BELOW INTRADAY, COLLAPSIBLE) ----------------
if HAS_BDB:
    with sidebar.expander("🔄 TCBS Historical Refresh", expanded=False):
        st.markdown("### TCBS refresh (all tickers)")
        st.caption("Use this for historical data updates. During market hours, intraday updates are preferred.")
        
        # Date range selector
        st.markdown("**Select Date Range**")
        col1, col2 = st.columns(2)
        with col1:
            refresh_start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=5),
                max_value=datetime.now().date(),
                key="tcbs_refresh_start"
            )
        with col2:
            refresh_end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                max_value=datetime.now().date(),
                key="tcbs_refresh_end"
            )
        
        # Show date range summary
        days_to_refresh = (refresh_end_date - refresh_start_date).days + 1
        st.caption(f"📅 Will refresh {days_to_refresh} days of data ({refresh_start_date} to {refresh_end_date})")
        
        st.markdown("---")
        
        pause_between = st.number_input("Pause between calls (s)", min_value=0.0, max_value=5.0, value=0.25, step=0.05, key="tcbs_pause")
        
        # Replace or append option
        replace_data = st.checkbox(
            "Replace existing data in range",
            value=True,
            help="If checked, will DELETE existing data in date range before inserting new data."
        )

        # Debug mode for refresh
        debug_refresh = st.checkbox(
            "Show detailed debug output",
            value=False,
            help="Display detailed information about each ticker's refresh process"
        )
        
        confirm_all = st.checkbox("I confirm: refresh ALL tickers from TCBS", value=False, key="tcbs_confirm")
        
        # Check which functions are available
        has_async = False
        has_sync_single = False
        
        try:
            import aiohttp
            has_async = hasattr(bdb, 'fetch_and_scale_async')
            has_sync_single = hasattr(bdb, 'fetch_ohlcv') or hasattr(bdb, 'fetch_price_data')
        except ImportError:
            pass
        
        # Show available methods
        if has_async:
            st.caption("✓ Async mode available")
        elif has_sync_single:
            st.caption("✓ Sync mode available")
        else:
            st.caption("⚠️ Limited refresh capability")
        
        # Show available bdb functions for debugging
        if debug_refresh:
            with st.expander("🔧 Available bdb functions", expanded=False):
                available_funcs = [func for func in dir(bdb) if not func.startswith('_')]
                st.code('\n'.join(available_funcs), language='python')
        
        if st.button("🔄 Force refresh ALL tickers now", use_container_width=True, disabled=not confirm_all):
            if confirm_all:
                # Create debug log container
                debug_container = st.container() if debug_refresh else None
                
                with st.spinner(f"Refreshing {len(all_tickers)} tickers from {refresh_start_date} to {refresh_end_date}..."):
                    success_count = 0
                    error_count = 0
                    error_details = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, ticker in enumerate(all_tickers):
                        ticker_debug = []
                        try:
                            status_text.text(f"Processing {ticker} ({idx+1}/{len(all_tickers)})...")
                            ticker_debug.append(f"[{ticker}] Starting refresh...")

                            

                            # Fetch new data
                            result_success = False
                            fetch_method = "unknown"

                            try:
                                if has_async:
                                    fetch_method = "async (fetch_and_scale_async)"
                                    ticker_debug.append(f"[{ticker}] Trying {fetch_method}...")

                                    import inspect
                                    sig = inspect.signature(bdb.fetch_and_scale_async)
                                    params = list(sig.parameters.keys())
                                    ticker_debug.append(f"[{ticker}] fetch_and_scale_async params: {params}")

                                    import aiohttp

                                    async def fetch_and_upsert():
                                        async with aiohttp.ClientSession() as session:
                                            # Call fetch_and_scale_async
                                            df = await bdb.fetch_and_scale_async(
                                                session=session,
                                                ticker=ticker,
                                                days=days_to_refresh if 'days' in params else None,
                                                resolution='D' if 'resolution' in params else None,
                                                timeout=30 if 'timeout' in params else None
                                            )
                                            # Upsert to DB if DataFrame and db_adapter available
                                            upserted = 0
                                            if isinstance(df, pd.DataFrame) and HAS_DB_ADAPTER:
                                                from db_adapter import get_db_adapter
                                                db_adapter = get_db_adapter()
                                                for row in df.itertuples(index=False):
                                                    ohlcv = {
                                                        'open': getattr(row, 'open', None),
                                                        'high': getattr(row, 'high', None),
                                                        'low': getattr(row, 'low', None),
                                                        'close': getattr(row, 'close', None),
                                                        'volume': getattr(row, 'volume', 0)
                                                    }
                                                    ok = db_adapter.insert_price_data(
                                                        ticker=ticker,
                                                        date=getattr(row, 'tradingDate', getattr(row, 'date', None)),
                                                        ohlcv=ohlcv,
                                                        source='tcbs'
                                                    )
                                                    if ok:
                                                        upserted += 1
                                            return df, upserted

                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    fetch_result, upserted = loop.run_until_complete(fetch_and_upsert())
                                    loop.close()

                                    if isinstance(fetch_result, pd.DataFrame):
                                        ticker_debug.append(f"[{ticker}] Fetched price data preview (last 10 rows):")
                                        for r in fetch_result.tail(10).itertuples(index=False):
                                            date = getattr(r, 'tradingDate', getattr(r, 'date', ''))
                                            close = getattr(r, 'close', '')
                                            open_ = getattr(r, 'open', '')
                                            ticker_debug.append(f"  {date} | open={open_} | close={close}")
                                        if len(fetch_result) > 10:
                                            ticker_debug.append(f"  ...{len(fetch_result)-10} earlier rows not shown")
                                        if not fetch_result.empty:
                                            result_success = True
                                            ticker_debug.append(f"[{ticker}] ✓ Async fetch returned non-empty DataFrame ({len(fetch_result)} rows)")
                                            ticker_debug.append(f"[{ticker}] ✓ Upserted {upserted} rows to DB")
                                        else:
                                            ticker_debug.append(f"[{ticker}] ✗ Async fetch returned empty DataFrame")
                                    elif fetch_result is not None and fetch_result is not False:
                                        ticker_debug.append(f"[{ticker}] Fetched price data type: {type(fetch_result)}")
                                        result_success = True
                                        ticker_debug.append(f"[{ticker}] ✓ Async fetch returned: {type(fetch_result)}")
                                    else:
                                        ticker_debug.append(f"[{ticker}] ✗ Async fetch returned False/None")
                                elif hasattr(bdb, 'fetch_ohlcv'):
                                    # Try fetch_ohlcv
                                    fetch_method = "sync (fetch_ohlcv)"
                                    ticker_debug.append(f"[{ticker}] Trying {fetch_method}...")
                                    
                                    # Check function signature
                                    import inspect
                                    sig = inspect.signature(bdb.fetch_ohlcv)
                                    params = sig.parameters.keys()
                                    ticker_debug.append(f"[{ticker}] fetch_ohlcv params: {list(params)}")
                                    
                                    data = bdb.fetch_ohlcv(ticker)
                                    # Show fetched price data (last 10 rows)
                                    if isinstance(data, pd.DataFrame):
                                        ticker_debug.append(f"[{ticker}] Fetched price data preview (last 10 rows):")
                                        for r in data.tail(10).itertuples(index=False):
                                            date = getattr(r, 'date', '')
                                            close = getattr(r, 'close', '')
                                            open_ = getattr(r, 'open', '')
                                            ticker_debug.append(f"  {date} | open={open_} | close={close}")
                                        if len(data) > 10:
                                            ticker_debug.append(f"  ...{len(data)-10} earlier rows not shown")
                                elif hasattr(bdb, 'fetch_price_data'):
                                    # Try fetch_price_data
                                    fetch_method = "sync (fetch_price_data)"
                                    ticker_debug.append(f"[{ticker}] Trying {fetch_method}...")
                                    
                                    # Check function signature
                                    import inspect
                                    sig = inspect.signature(bdb.fetch_price_data)
                                    params = sig.parameters.keys()
                                    ticker_debug.append(f"[{ticker}] fetch_price_data params: {list(params)}")
                                    
                                    data = bdb.fetch_price_data(ticker)
                                    # Show fetched price data (last 10 rows)
                                    if isinstance(data, pd.DataFrame):
                                        ticker_debug.append(f"[{ticker}] Fetched price data preview (last 10 rows):")
                                        for r in data.tail(10).itertuples(index=False):
                                            date = getattr(r, 'date', '')
                                            close = getattr(r, 'close', '')
                                            open_ = getattr(r, 'open', '')
                                            ticker_debug.append(f"  {date} | open={open_} | close={close}")
                                        if len(data) > 10:
                                            ticker_debug.append(f"  ...{len(data)-10} earlier rows not shown")
                                else:
                                    ticker_debug.append(f"[{ticker}] ✗ No fetch methods available")
                            except Exception as fetch_error:
                                import traceback
                                error_traceback = traceback.format_exc()
                                ticker_debug.append(f"[{ticker}] ❌ Fetch error using {fetch_method}: {fetch_error}")
                                ticker_debug.append(f"[{ticker}] Traceback: {error_traceback[-500:]}")  # Last 500 chars
                            
                            if result_success:
                                success_count += 1
                                ticker_debug.append(f"[{ticker}] ✅ SUCCESS")
                            else:
                                error_count += 1
                                ticker_debug.append(f"[{ticker}] ❌ FAILED")
                                error_details.append({
                                    'ticker': ticker,
                                    'method': fetch_method,
                                    'debug': ticker_debug
                                })
                            
                            # Show debug output if enabled
                            if debug_refresh and debug_container:
                                with debug_container:
                                    with st.expander(f"{ticker} - {'✅' if result_success else '❌'}", expanded=not result_success):
                                        for line in ticker_debug:
                                            st.text(line)
                            
                            # Update progress
                            progress_bar.progress((idx + 1) / len(all_tickers))
                            
                            # Pause between calls
                            if pause_between > 0:
                                time.sleep(pause_between)
                                
                        except Exception as e:
                            error_count += 1
                            ticker_debug.append(f"[{ticker}] ❌ EXCEPTION: {e}")
                            error_details.append({
                                'ticker': ticker,
                                'method': 'exception',
                                'debug': ticker_debug,
                                'error': str(e)
                            })
                            
                            if debug_refresh and debug_container:
                                with debug_container:
                                    with st.expander(f"{ticker} - ❌ EXCEPTION", expanded=True):
                                        for line in ticker_debug:
                                            st.text(line)
                            
                            time.sleep(0.1)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show results
                    if error_count > 0:
                        st.warning(f"⚠️ Completed with errors: {success_count} succeeded, {error_count} failed")
                        
                        # Show failed tickers summary
                        with st.expander(f"❌ Failed Tickers ({error_count})", expanded=True):
                            failed_tickers = [detail['ticker'] for detail in error_details]
                            st.error(f"**Failed:** {', '.join(failed_tickers)}")
                            
                            # Show detailed errors if debug mode
                            if debug_refresh:
                                for detail in error_details[:10]:  # Limit to first 10
                                    st.text(f"\n{detail['ticker']} ({detail['method']}):")
                                    for line in detail['debug'][-5:]:  # Last 5 lines
                                        st.text(f"  {line}")
                    else:
                        st.success(f"✅ Successfully refreshed {success_count} tickers")
                    
                    # Clear cache to show updated data
                    try:
                        load_price_range.clear()
                        get_all_tickers.clear()
                    except:
                        pass
                    
                    st.info("💡 Refresh complete! Data has been updated.")
        
        st.markdown("---")
        st.markdown("**ℹ️ About Historical Refresh**")
        st.caption("""
        This feature:
        - Fetches historical OHLCV data from TCBS API
        - Can replace existing data or append/update
        - Works with available fetch functions
        - Slower than intraday updates (use for historical data)
        - Enable debug mode to see detailed refresh information
        """)
else:
    with sidebar.expander("🔄 TCBS Historical Refresh", expanded=False):
        st.warning("TCBS refresh disabled (build_price_db not available)")

# --- Dividend Adjustments Section --------------------------------------------
if HAS_DIVIDEND_ADJUSTER:
    with sidebar.expander("💰 Dividend Adjustments", expanded=False):
        st.markdown("**Detect and apply dividend-related price adjustments**")
        st.caption("⚠️ This is a manual process. Run independently from other operations.")
        
        st.markdown("---")
        
        if st.button("🔍 Scan for Dividends", use_container_width=True):
            with st.spinner("Scanning for dividend adjustments..."):
                adjustments = scan_all_tickers_for_dividends(db_path=DB_PATH, debug=False)
                
                if adjustments:
                    st.warning(f"⚠️ Found {len(adjustments)} tickers with price discontinuities")
                    
                    # Display in a table with correct field names
                    adj_df = pd.DataFrame(adjustments)
                    display_df = adj_df[['ticker', 'adjustment_date', 'new_tcbs_price', 'old_db_price', 'db_source', 'adjustment_ratio', 'affected_rows', 'price_diff_pct']]
                    display_df.columns = ['Ticker', 'Date', 'New TCBS', 'Old DB', 'Source', 'Ratio', 'Rows', 'Diff %']
                    
                    # Format for better display
                    display_df['New TCBS'] = display_df['New TCBS'].apply(lambda x: f"{x:.2f}")
                    display_df['Old DB'] = display_df['Old DB'].apply(lambda x: f"{x:.2f}")
                    display_df['Ratio'] = display_df['Ratio'].apply(lambda x: f"{x:.6f}")
                    display_df['Diff %'] = display_df['Diff %'].apply(lambda x: f"{x:+.2f}%")
                    
                    st.dataframe(display_df, use_container_width=True, height=300)
                    
                    st.markdown("---")
                    st.markdown("**⚠️ Warning:** This will modify historical price data")
                    
                    # Store adjustments in session state for later use
                    st.session_state['pending_dividend_adjustments'] = adjustments
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Apply All Adjustments", use_container_width=True, type="primary"):
                            if 'pending_dividend_adjustments' in st.session_state:
                                applied = 0
                                progress = st.progress(0)
                                status = st.empty()
                                
                                for i, adj in enumerate(st.session_state['pending_dividend_adjustments']):
                                    status.text(f"Adjusting {adj['ticker']}...")
                                    rows = apply_dividend_adjustment(
                                        adj['ticker'],
                                        adj['adjustment_ratio'],
                                        adj['adjustment_date'],
                                        db_path=DB_PATH
                                    )
                                    if rows > 0:
                                        applied += 1
                                    progress.progress((i + 1) / len(st.session_state['pending_dividend_adjustments']))
                                
                                status.empty()
                                progress.empty()
                                
                                st.success(f"✓ Applied adjustments to {applied} tickers")
                                
                                # Clear pending adjustments
                                del st.session_state['pending_dividend_adjustments']
                                
                                # Clear caches to reflect updated data
                                try:
                                    load_price_range.clear()
                                    get_all_tickers.clear()
                                except:
                                    pass
                                
                                st.rerun()
                    
                    with col2:
                        if st.button("❌ Cancel", use_container_width=True):
                            # Clear pending adjustments
                            if 'pending_dividend_adjustments' in st.session_state:
                                del st.session_state['pending_dividend_adjustments']
                            st.info("Adjustments cancelled. Scan again to recheck.")
                else:
                    st.success("✓ No dividend adjustments needed")
        
        st.markdown("---")
        st.markdown("**ℹ️ About Dividend Adjustments**")
        st.caption("""
        This tool detects price discontinuities caused by dividends or stock splits:
        - Compares newly fetched TCBS data with existing database prices
        - Detects >5% price differences for the same date
        - Adjusts historical prices to maintain chart continuity
        - Only affects non-TCBS data sources (local_copy, csv, etc.)
        """)
else:
    sidebar.markdown("---")
    sidebar.info("💰 Dividend adjustment module not available")

sidebar.markdown("---")

# --- Admin: Manage Tickers (MOVED TO BOTTOM) ---------------------------------
with sidebar.expander("🔧 Admin: Manage Tickers", expanded=False):
    if not HAS_TM:
        st.warning("Ticker manager not available")
    else:
        st.markdown("**Add Ticker**")
        new_ticker = st.text_input("Ticker symbol", key="add_ticker_input", placeholder="e.g., VIC")
        new_source = st.selectbox("Data source", ["manual", "tcbs", "local_copy", "amibroker"], key="add_source")
        if st.button("➕ Add Ticker"):
            if new_ticker:
                if tm.add_ticker(new_ticker.upper(), db_path=DB_PATH, source=new_source):
                    st.success(f"✓ Added {new_ticker.upper()}")
                    try:
                        get_all_tickers.clear()
                    except:
                        pass
                else:
                    st.warning(f"Ticker {new_ticker.upper()} already exists")
            else:
                st.error("Enter a ticker symbol")
        
        st.markdown("---")
        st.markdown("**Remove Ticker**")
        
        current_tickers_df = tm.get_all_tickers(db_path=DB_PATH)
        if not current_tickers_df.empty:
            unique_tickers = sorted(current_tickers_df['ticker'].unique())
            remove_ticker = st.selectbox("Select ticker to remove", unique_tickers, key="remove_ticker_select")
            remove_source = st.selectbox("Source (or all)", ["all"] + list(current_tickers_df['source'].unique()), key="remove_source")
            confirm_remove = st.checkbox("I confirm deletion", key="confirm_remove")
            
            if st.button("🗑️ Remove Ticker"):
                if confirm_remove:
                    source_filter = None if remove_source == "all" else remove_source
                    deleted = tm.remove_ticker(remove_ticker, db_path=DB_PATH, source=source_filter, confirm=True)
                    if deleted > 0:
                        st.success(f"✓ Deleted {deleted} rows for {remove_ticker}")
                        try:
                            get_all_tickers.clear()
                            load_price_range.clear()
                        except:
                            pass
                        st.rerun()
                    else:
                        st.warning("No rows deleted")
                else:
                    st.error("Check confirmation box to delete")
        else:
            st.info("No tickers in database")

# --- Always build overview ---------------------------------------------------
with st.spinner("Building overview for tickers in DB..."):
    tickers = all_tickers

    # Apply include/exclude filters
    if include_tickers:
        tickers = [t for t in tickers if t in include_tickers]
    if exclude_tickers:
        tickers = [t for t in tickers if t not in exclude_tickers]

    # Use cached overview if available and date range matches
    cache_key = f"overview_{start_date}_{end_date}_{lookback}_{len(tickers)}"
    df_over = st.session_state.get(cache_key, None)
    
    if df_over is None:
        # Try to get from database cache
        if HAS_DB_ADAPTER:
            try:
                df_over = db.get_overview_cache(cache_key)
                if not df_over.empty:
                    if debug:
                        st.write(f"[DEBUG] Loaded overview from database cache: {len(df_over)} rows")
            except Exception as e:
                if debug:
                    st.write(f"[DEBUG] Error loading from database cache: {e}")
                df_over = None
        
        # If not in database cache, build it
        if df_over is None or df_over.empty:
            if not tickers:
                df_over = pd.DataFrame()
            else:
                df_over = build_overview(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), lookback=lookback, max_rows=len(tickers), debug=debug)
                if debug:
                    st.write(f"[DEBUG] Built overview with {len(df_over)} rows for {len(tickers)} tickers")
                
                # Save to database cache
                if HAS_DB_ADAPTER and not df_over.empty:
                    try:
                        success = db.save_overview_cache(cache_key, df_over)
                        if debug and success:
                            st.write(f"[DEBUG] Saved overview to database cache with key: {cache_key}")
                    except Exception as e:
                        if debug:
                            st.write(f"[DEBUG] Error saving to database cache: {e}")
        
        # Store in session state for current session
        st.session_state[cache_key] = df_over

# Main view: always show today/overview table with current date in header
current_date_str = datetime.now().strftime("%Y-%m-%d")  # <-- Add this line before using current_date_str

# Find latest bar date in DB for display
latest_bar_date = None
if not df_over.empty and "Ticker" in df_over.columns:
    try:
        # Get the latest date from the DataFrame
        latest_bar_date = df_over['Date'].idxmax()  # Ensure the latest date is displayed
    except Exception:
        latest_bar_date = None

if latest_bar_date:
    st.markdown(f"##### Overview — latest bar per ticker ({latest_bar_date})")
else:
    st.markdown(f"##### Overview — latest bar per ticker ({current_date_str})")

# Add collapsible list of tickers with signals - grouped by signal type
if not df_over.empty and 'Signal' in df_over.columns:
    tickers_with_signals = df_over[df_over['Signal'] != '']
    
    if not tickers_with_signals.empty:
        with st.expander(f"⚡ Tickers with Crossover Signals ({len(tickers_with_signals)})", expanded=False):
            # Group by signal type
            cross_up = tickers_with_signals[tickers_with_signals['Signal'] == '↗0d']['Ticker'].tolist()
            cross_down = tickers_with_signals[tickers_with_signals['Signal'] == '↘0d']['Ticker'].tolist()
            
            # Soon to cross (extract days from signal)
            soon_up = {}
            soon_down = {}
            for _, row in tickers_with_signals.iterrows():
                signal = row['Signal']
                ticker = row['Ticker']
                if signal.startswith('↗') and signal != '↗0d':
                    days = signal.replace('↗', '').replace('d', '')
                    if days not in soon_up:
                        soon_up[days] = []
                    soon_up[days].append(ticker)
                elif signal.startswith('↘') and signal != '↘0d':
                    days = signal.replace('↘', '').replace('d', '')
                    if days not in soon_down:
                        soon_down[days] = []
                    soon_down[days].append(ticker)
            
            # Build sentence parts
            parts = []
            
            if cross_up:
                parts.append(f"**Cross up:** {', '.join(cross_up)}")
            
            if cross_down:
                parts.append(f"**Cross down:** {', '.join(cross_down)}")
            
            # Group "soon to cross up" by days in single sentence
            if soon_up:
                up_parts = []
                for days in sorted(soon_up.keys(), key=int):
                    tickers_str = ', '.join(soon_up[days])
                    up_parts.append(f"{tickers_str} ({days}d)")
                parts.append(f"**Soon to cross up:** {', '.join(up_parts)}")
            
            # Group "soon to cross down" by days in single sentence
            if soon_down:
                down_parts = []
                for days in sorted(soon_down.keys(), key=int):
                    tickers_str = ', '.join(soon_down[days])
                    down_parts.append(f"{tickers_str} ({days}d)")
                parts.append(f"**Soon to cross down:** {', '.join(down_parts)}")
            
            # Display as sentence
            st.markdown('; '.join(parts))
    else:
        with st.expander("⚡ Tickers with Crossover Signals (0)", expanded=False):
            st.caption("No crossovers expected in next 5 days")

if df_over is None or df_over.empty:
    st.warning("No data available to build overview. Ensure price_data.db has rows.")
    # Debug info: show DB path, file existence, and row count
    st.markdown("#### Debug Info")
    st.write(f"DB Path: `{DB_PATH}`")
    db_exists = os.path.exists(DB_PATH)
    st.write(f"DB Exists: {db_exists}")
    if db_exists:
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM price_data")
            row_count = cur.fetchone()[0]
            st.write(f"price_data row count: {row_count}")
            cur.execute("SELECT DISTINCT ticker FROM price_data")
            tickers_in_db = [r[0] for r in cur.fetchall()]
            st.write(f"Tickers in DB: {tickers_in_db[:20]}{' ...' if len(tickers_in_db) > 20 else ''}")
            # Additional debug: show sample rows for first few tickers
            for sample_ticker in tickers_in_db[:3]:
                cur.execute("SELECT * FROM price_data WHERE ticker = ? ORDER BY date DESC LIMIT 5", (sample_ticker,))
                sample_rows = cur.fetchall()
                st.write(f"Sample rows for ticker {sample_ticker}:")
                for r in sample_rows:
                    st.write(r)
            # Show tickers requested for overview but not found in DB
           
            missing_tickers = [t for t in all_tickers if t not in tickers_in_db]
            if missing_tickers:
                st.write(f"Tickers requested but not found in DB: {missing_tickers[:10]}{' ...' if len(missing_tickers) > 10 else ''}")
            # Show tickers with no data in requested date range
            for t in all_tickers[:5]:
                cur.execute("SELECT COUNT(*) FROM price_data WHERE ticker = ? AND date >= ? AND date <= ?", (t, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
                cnt = cur.fetchone()[0]
                if cnt == 0:
                    st.write(f"[DEBUG] Ticker {t} has no rows in date range {start_date} to {end_date}")
            conn.close()
        except Exception as e:
            st.write(f"DB error: {e}")
    else:
        st.write("Database file not found.")
else:
    # Display main overview table
    display_cols = ["Ticker", "Close", "Trend (Daily)", "Trend (Weekly)", "Trend (Monthly)", "Score", "MACD_Hist_Daily", "MACD_Hist_Weekly", "MACD_Hist_Monthly", "Vol/AvgVol", "Signal"]
    
    # Add toggle option in the sidebar
    show_latest_date = sidebar.checkbox("Show Latest Date", value=False)
    show_macd_slope = sidebar.checkbox("Show MACD Slope (Strength)", value=False)

    if show_latest_date:
        display_cols.insert(2, "Latest Date")
    
    if show_macd_slope:
        # Insert slope columns after MACD histograms
        macd_idx = display_cols.index("MACD_Hist_Monthly") + 1
        display_cols.insert(macd_idx, "MACD_Slope_3d")
        display_cols.insert(macd_idx + 1, "MACD_Slope_Peak")

    styled = df_over[display_cols].style
    styled = styled.applymap(style_stage_column, subset=["Trend (Daily)", "Trend (Weekly)", "Trend (Monthly)"])
    styled = styled.applymap(style_vol_ratio, subset=["Vol/AvgVol"])
    
    def style_row_by_score(row):
        score = row["Score"]
        ticker_style = style_by_score(row["Ticker"], score)
        close_style = style_by_score(row["Close"], score)
        return pd.Series([ticker_style, close_style] + [''] * (len(display_cols) - 2), index=display_cols)
    
    styled = styled.apply(style_row_by_score, axis=1)
    
    def style_macd_by_trends(row):
        macd_d_style = style_macd_by_trend(row["MACD_Hist_Daily"], row["Trend (Daily)"])
        macd_w_style = style_macd_by_trend(row["MACD_Hist_Weekly"], row["Trend (Weekly)"])
        macd_m_style = style_macd_by_trend(row["MACD_Hist_Monthly"], row["Trend (Monthly)"])
        return pd.Series([''] * 6 + [macd_d_style, macd_w_style, macd_m_style] + [''] * (len(display_cols) - 9), index=display_cols)
    
    styled = styled.apply(style_macd_by_trends, axis=1)
    
    # Style MACD slope columns if visible
    if show_macd_slope:
        slope_cols = ["MACD_Slope_3d", "MACD_Slope_Peak"]
        for col in slope_cols:
            if col in display_cols:
                styled = styled.applymap(style_macd_slope, subset=[col])
    
    st.dataframe(styled, height=700, use_container_width=True)
    
    st.caption("📌 **Signal column:** ↗Nd = crossing up in N days | ↘Nd = crossing down in N days | ↗0d = just crossed up | ↘0d = just crossed down")
    if show_macd_slope:
        st.caption("📊 **MACD Slope columns:** 3d = slope from last 3 days | Peak = slope from peak/trough to now (higher = stronger momentum)")
    
    st.markdown("### 💡 Select a ticker to view detailed charts")
    selected_ticker_input = st.selectbox("Select ticker for detailed view", options=[""] + df_over['Ticker'].tolist(), index=0, key="ticker_selector")
    
    if selected_ticker_input:
        st.session_state.selected_ticker = selected_ticker_input
    
    st.download_button("Download overview CSV", data=df_over.to_csv(index=False).encode('utf-8'), file_name=f"macd_overview_{current_date_str}.csv", mime="text/csv")

# --- Detailed chart view (for selected ticker) --------------------------------
if st.session_state.selected_ticker:
    ticker = st.session_state.selected_ticker
    
    st.subheader(f"Detailed MACD Analysis — {ticker}")
    st.markdown("#### 📈 Multi-Timeframe MACD Analysis")
    plot_multi_tf_macd(ticker, start_date, end_date, lookback=lookback)
    
    
    if st.button("Back to Overview"):
        st.session_state.selected_ticker = None
        st.rerun()

# In build_overview and plot_multi_tf_macd, MACD histograms are calculated only if there is enough data.
# If you see no MACD hist for tickers like DCM to VTP, possible reasons are:
# 1. The DataFrame for that ticker is empty or missing recent data.
# 2. There are duplicate or missing dates, causing the MACD calculation to fail.
# 3. There are less than 3 valid data points after cleaning, so MACD cannot be computed.
# 4. Data was removed due to duplicate cleanup, or the database is missing historical bars.

# To debug:
# - Check if load_price_range(ticker, start_date, end_date) returns a non-empty DataFrame for those tickers.
# - Ensure there are no duplicate dates and the date range covers enough bars.
# - Confirm that after duplicate removal, at least 3 valid rows remain for each ticker.

# If you want to see why a ticker has no MACD hist, enable debug mode in the dashboard sidebar.
# This will print the latest date and row count for each ticker during overview building.
