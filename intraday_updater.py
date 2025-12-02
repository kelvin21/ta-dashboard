"""
Module to fetch today's OHLCV intraday data from TCBS and update the database.
Includes automatic price adjustment detection for dividends/splits.
"""

import sys
import os
import sqlite3
from datetime import datetime, timedelta
import asyncio

# Add script directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Check for required dependencies
try:
    import requests
    import pandas as pd
    from pandas import json_normalize
    HAS_REQUESTS = True
except ImportError as e:
    print(f"❌ Missing required packages. Please install: pip install requests pandas")
    raise

# Check for async support
try:
    import aiohttp
    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False

# Add db_adapter support
try:
    from db_adapter import get_db_adapter
    db_adapter = get_db_adapter()
    HAS_DB_ADAPTER = True
except Exception:
    db_adapter = None
    HAS_DB_ADAPTER = False


async def fetch_page_async(session, symbol, page_num, page_size, head_index):
    """Async function to fetch a single page of intraday data."""
    d = datetime.now()
    
    if d.weekday() > 4:  # weekend
        if head_index == -1:
            url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/{symbol}/his/paging?page={page_num}&size={page_size}&headIndex=-1'
        else:
            url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/{symbol}/his/paging?page={page_num}&size={page_size}&headIndex={head_index}'
    else:  # weekday
        if head_index == -1:
            url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/{symbol}/his/paging?page={page_num}&size={page_size}'
        else:
            url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/{symbol}/his/paging?page={page_num}&size={page_size}&headIndex={head_index}'
    
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status != 200:
                return (page_num, pd.DataFrame(), False)
            
            data = await response.json()
            
            if 'data' not in data or not data['data']:
                return (page_num, pd.DataFrame(), False)
            
            df = json_normalize(data['data']).rename(columns={'p': 'price', 'v': 'volume', 't': 'time'})
            
            today = datetime.now().date()
            df['datetime'] = pd.to_datetime(today.strftime('%Y-%m-%d') + ' ' + df['time'])
            
            return (page_num, df, True)
            
    except Exception as e:
        return (page_num, pd.DataFrame(), False)


async def fetch_intraday_data_async(symbol, page_size=100):
    """Fetch all intraday pages asynchronously."""
    # Step 1: Fetch first page for metadata
    d = datetime.now()
    if d.weekday() > 4:
        url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/{symbol}/his/paging?page=0&size={page_size}&headIndex=-1'
    else:
        url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/{symbol}/his/paging?page=0&size={page_size}'
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'data' not in data or not data['data']:
            return pd.DataFrame()
        
        total_items = data.get('total', 0)
        returned_head_index = data.get('headIndex', -1)
        max_pages = (total_items + page_size - 1) // page_size
        
        df_first = json_normalize(data['data']).rename(columns={'p': 'price', 'v': 'volume', 't': 'time'})
        today = datetime.now().date()
        df_first['datetime'] = pd.to_datetime(today.strftime('%Y-%m-%d') + ' ' + df_first['time'])
        
        all_data = [df_first]
        
        if max_pages <= 1:
            return df_first
        
        # Step 2: Fetch remaining pages
        async with aiohttp.ClientSession() as session:
            tasks = []
            for page_num in range(1, max_pages):
                if returned_head_index != -1:
                    head_index = returned_head_index + (page_num * page_size)
                else:
                    head_index = page_num * page_size
                
                task = fetch_page_async(session, symbol, page_num, page_size, head_index)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            for page_num, df, success in sorted(results, key=lambda x: x[0]):
                if success and not df.empty:
                    all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.drop_duplicates(subset=['datetime', 'price', 'volume'], keep='first')
            combined = combined.sort_values('datetime').reset_index(drop=True)
            return combined
        
        return pd.DataFrame()
            
    except Exception as e:
        return pd.DataFrame()


def convert_to_ohlcv(df, interval='1D', scale_price=True):
    """
    Convert tick data to OHLCV format.
    
    Args:
        df: DataFrame with datetime, price, volume columns
        interval: Resample interval ('1min', '5min', '15min', '30min', '1H', '1D')
        scale_price: If True, divide prices by 1000 (TCBS intraday prices are in VND x1000)
    
    Returns:
        DataFrame with OHLCV data
    """
    if df.empty or 'datetime' not in df.columns:
        return pd.DataFrame()
    
    try:
        df = df.sort_values('datetime')
        df = df.set_index('datetime')
        
        # Scale prices if needed (TCBS intraday prices are in thousands)
        price_column = df['price']
        if scale_price:
            price_column = price_column / 1000.0
        
        ohlcv = pd.DataFrame()
        ohlcv['open'] = price_column.resample(interval).first()
        ohlcv['high'] = price_column.resample(interval).max()
        ohlcv['low'] = price_column.resample(interval).min()
        ohlcv['close'] = price_column.resample(interval).last()
        ohlcv['volume'] = df['volume'].resample(interval).sum()
        
        ohlcv = ohlcv.dropna()
        ohlcv = ohlcv.reset_index()
        
        return ohlcv
        
    except Exception as e:
        print(f"Error converting to OHLCV: {e}")
        return pd.DataFrame()


def detect_price_adjustment(ticker, db_path, debug=False):
    """
    Compare yesterday's price in database with freshly fetched data to detect adjustments.
    Returns: (needs_adjustment: bool, adjustment_ratio: float, yesterday_date: str, old_close: float, new_close: float)
    """
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    if debug:
        print(f"Checking price adjustment for {ticker} on {yesterday_str}...")

    # Step 1: Get yesterday's price from database (existing old data)
    old_close = 0
    try:
        if HAS_DB_ADAPTER and getattr(db_adapter, "db_type", None) == "mongodb":
            df = db_adapter.load_price_range(ticker, yesterday_str, yesterday_str)
            if not df.empty:
                old_close = float(df.iloc[-1]['close'])
            else:
                if debug:
                    print(f"  No existing data for {yesterday_str}")
                return (False, 1.0, yesterday_str, 0, 0)
        else:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT close FROM price_data 
                WHERE ticker = ? AND date = ?
                ORDER BY ROWID DESC
                LIMIT 1
            """, (ticker, yesterday_str))
            result = cursor.fetchone()
            conn.close()
            if not result:
                if debug:
                    print(f"  No existing data for {yesterday_str}")
                return (False, 1.0, yesterday_str, 0, 0)
            old_close = float(result[0])
    except Exception as e:
        if debug:
            print(f"  Error reading from database: {e}")
        return (False, 1.0, yesterday_str, 0, 0)

    # Step 2: Fetch fresh data for yesterday from TCBS (do NOT query from database)
    try:
        # Use TCBS stock history API to get yesterday's data directly
        url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={ticker}&type=stock&resolution=D&from={int(yesterday.timestamp())}&to={int(yesterday.timestamp() + 86400)}'
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'data' not in data or not data['data']:
            if debug:
                print(f"  No fresh data available for {yesterday_str}")
            return (False, 1.0, yesterday_str, old_close, 0)
        
        # Get the close price from fresh data
        fresh_data = data['data'][0] if isinstance(data['data'], list) else data['data']
        new_close = float(fresh_data.get('close', 0))
        
        if new_close == 0:
            if debug:
                print(f"  Invalid close price in fresh data")
            return (False, 1.0, yesterday_str, old_close, 0)
        
    except Exception as e:
        if debug:
            print(f"  Error fetching fresh data: {e}")
        return (False, 1.0, yesterday_str, old_close, 0)
    
    # Step 3: Compare and calculate adjustment ratio
    price_diff_pct = ((new_close - old_close) / old_close * 100) if old_close > 0 else 0
    
    if debug:
        print(f"  Database close: {old_close:.2f}")
        print(f"  Fresh TCBS close: {new_close:.2f}")
        print(f"  Difference: {price_diff_pct:+.2f}%")
    
    # If difference is > 5%, there's likely been an adjustment
    if abs(price_diff_pct) > 5.0:
        adjustment_ratio = new_close / old_close
        if debug:
            print(f"  ⚠️ Price adjustment detected!")
            print(f"  Adjustment ratio: {adjustment_ratio:.6f}")
        return (True, adjustment_ratio, yesterday_str, old_close, new_close)
    else:
        if debug:
            print(f"  ✓ No significant price adjustment")
        return (False, 1.0, yesterday_str, old_close, new_close)


def apply_price_adjustment(ticker, adjustment_ratio, adjustment_date, db_path, debug=False):
    """
    Apply price adjustment to historical data before the adjustment date.
    Returns: Number of rows adjusted
    """
    if debug:
        print(f"Applying adjustment ratio {adjustment_ratio:.6f} to {ticker} before {adjustment_date}...")
    try:
        if HAS_DB_ADAPTER and getattr(db_adapter, "db_type", None) == "mongodb":
            # MongoDB: update all docs before adjustment_date
            result = db_adapter.price_data.update_many(
                {"ticker": ticker, "date": {"$lt": adjustment_date}},
                {"$mul": {"open": adjustment_ratio, "high": adjustment_ratio, "low": adjustment_ratio, "close": adjustment_ratio}}
            )
            rows_affected = result.modified_count
            if debug:
                print(f"  ✓ Adjusted {rows_affected} historical rows (MongoDB)")
            return rows_affected
        else:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE price_data 
                SET open = open * ?,
                    high = high * ?,
                    low = low * ?,
                    close = close * ?
                WHERE ticker = ? AND date < ?
            """, (adjustment_ratio, adjustment_ratio, adjustment_ratio, adjustment_ratio, ticker, adjustment_date))
            rows_affected = cursor.rowcount
            conn.commit()
            conn.close()
            if debug:
                print(f"  ✓ Adjusted {rows_affected} historical rows")
            return rows_affected
    except Exception as e:
        if debug:
            print(f"  ✗ Error applying adjustment: {e}")
        return 0


def update_intraday_ohlcv(ticker, db_path, interval='1D', source='intraday', scale_price=True, debug=False):
    """
    Fetch today's intraday data and update/insert into database.
    
    Args:
        ticker: Stock ticker symbol
        db_path: Path to SQLite database
        interval: OHLCV interval ('1D' for daily, '1H' for hourly, etc.)
        source: Data source identifier
        scale_price: If True, divide prices by 1000 (TCBS intraday format)
        debug: Print debug information
    
    Returns:
        Tuple of (success: bool, message: str, rows_affected: int)
    """
    if debug:
        print(f"Updating intraday OHLCV for {ticker}...")
        if scale_price:
            print(f"  Price scaling: ENABLED (divide by 1000)")
    
    # Fetch intraday data
    try:
        if HAS_ASYNC:
            df_ticks = asyncio.run(fetch_intraday_data_async(ticker))
        else:
            return (False, "Async support not available (install aiohttp)", 0)
    except Exception as e:
        return (False, f"Failed to fetch intraday data: {e}", 0)
    
    if df_ticks.empty:
        return (False, f"No intraday data available for {ticker}", 0)
    
    if debug:
        print(f"  Fetched {len(df_ticks)} ticks")
        print(f"  Raw price range: {df_ticks['price'].min():.0f} - {df_ticks['price'].max():.0f}")
    
    # Filter for today only
    today = datetime.now().date()
    df_ticks['date'] = df_ticks['datetime'].dt.date
    df_today = df_ticks[df_ticks['date'] == today].copy()
    
    if df_today.empty:
        return (False, f"No intraday data for today ({today})", 0)
    
    # Convert to OHLCV with price scaling
    ohlcv = convert_to_ohlcv(df_today, interval=interval, scale_price=scale_price)
    
    if ohlcv.empty:
        return (False, "Failed to convert to OHLCV", 0)
    
    if debug:
        print(f"  Converted to {len(ohlcv)} OHLCV bars")
        if scale_price:
            print(f"  Scaled price range: {ohlcv['low'].min():.1f} - {ohlcv['high'].max():.1f}")
    
    # Prepare data for database
    ohlcv['ticker'] = ticker
    ohlcv['source'] = source
    ohlcv['date'] = ohlcv['datetime'].dt.strftime('%Y-%m-%d')
    db_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'source']
    ohlcv_db = ohlcv[db_cols].copy()
    rows_affected = 0
    try:
        if HAS_DB_ADAPTER and getattr(db_adapter, "db_type", None) == "mongodb":
            for _, row in ohlcv_db.iterrows():
                ohlcv_dict = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': int(row['volume'])
                }
                ok = db_adapter.insert_price_data(
                    ticker=row['ticker'],
                    date=row['date'],
                    ohlcv=ohlcv_dict,
                    source=row['source']
                )
                if ok:
                    rows_affected += 1
                    if debug:
                        print(f"  Upserted {row['date']}: O={row['open']:.1f} H={row['high']:.1f} L={row['low']:.1f} C={row['close']:.1f} V={row['volume']:,.0f}")
            return (True, f"Updated {len(ohlcv_db)} OHLCV bar(s) for {ticker}", rows_affected)
        else:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            for _, row in ohlcv_db.iterrows():
                cursor.execute("""
                    SELECT COUNT(*) FROM price_data 
                    WHERE ticker = ? AND date = ? AND source = ?
                """, (row['ticker'], row['date'], row['source']))
                exists = cursor.fetchone()[0] > 0
                if exists:
                    cursor.execute("""
                        UPDATE price_data 
                        SET open = ?, high = ?, low = ?, close = ?, volume = ?
                        WHERE ticker = ? AND date = ? AND source = ?
                    """, (
                        row['open'], row['high'], row['low'], row['close'], row['volume'],
                        row['ticker'], row['date'], row['source']
                    ))
                    rows_affected += cursor.rowcount
                    if debug:
                        print(f"  Updated {row['date']}: O={row['open']:.1f} H={row['high']:.1f} L={row['low']:.1f} C={row['close']:.1f} V={row['volume']:,.0f}")
                else:
                    cursor.execute("""
                        INSERT INTO price_data (ticker, date, open, high, low, close, volume, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['ticker'], row['date'], row['open'], row['high'], row['low'], 
                        row['close'], row['volume'], row['source']
                    ))
                    rows_affected += cursor.rowcount
                    if debug:
                        print(f"  Inserted {row['date']}: O={row['open']:.1f} H={row['high']:.1f} L={row['low']:.1f} C={row['close']:.1f} V={row['volume']:,.0f}")
            conn.commit()
            conn.close()
            return (True, f"Updated {len(ohlcv_db)} OHLCV bar(s) for {ticker}", rows_affected)
    except Exception as e:
        return (False, f"Database error: {e}", 0)


def update_intraday_with_adjustment_check(ticker, db_path, interval='1D', source='intraday', 
                                          scale_price=True, check_adjustment=True, debug=False):
    """
    Fetch today's intraday data, check for price adjustments, and update database.
    
    Returns:
        Tuple of (success: bool, message: str, rows_affected: int, adjustment_applied: bool)
    """
    adjustment_applied = False
    
    # Step 1: Check for price adjustment if enabled
    if check_adjustment:
        needs_adj, adj_ratio, adj_date, old_close, new_close = detect_price_adjustment(ticker, db_path, debug)
        
        if needs_adj:
            if debug:
                print(f"  Price adjustment detected for {ticker}:")
                print(f"    Date: {adj_date}")
                print(f"    Old close: {old_close:.2f}")
                print(f"    New close: {new_close:.2f}")
                print(f"    Adjustment ratio: {adj_ratio:.6f}")
            
            # Apply adjustment to historical data
            rows_adjusted = apply_price_adjustment(ticker, adj_ratio, adj_date, db_path, debug)
            
            if rows_adjusted > 0:
                adjustment_applied = True
                if debug:
                    print(f"  ✓ Adjusted {rows_adjusted} historical rows")
    
    # Step 2: Update today's intraday data
    success, message, rows = update_intraday_ohlcv(ticker, db_path, interval, source, scale_price, debug)
    
    # Append adjustment info to message
    if adjustment_applied:
        message = f"{message} (Applied price adjustment: {adj_ratio:.4f})"
    
    return (success, message, rows, adjustment_applied)


def update_multiple_tickers(tickers, db_path, interval='1D', source='intraday', scale_price=True, debug=False):
    """Update intraday OHLCV for multiple tickers."""
    results = {}
    
    for ticker in tickers:
        success, message, rows = update_intraday_ohlcv(ticker, db_path, interval, source, scale_price, debug)
        results[ticker] = {
            'success': success,
            'message': message,
            'rows_affected': rows
        }
    
    return results


def update_multiple_tickers_with_adjustment(tickers, db_path, interval='1D', source='intraday', 
                                           scale_price=True, check_adjustment=True, debug=False):
    """Update intraday OHLCV for multiple tickers with adjustment check."""
    results = {}
    
    for ticker in tickers:
        success, message, rows, adj_applied = update_intraday_with_adjustment_check(
            ticker, db_path, interval, source, scale_price, check_adjustment, debug
        )
        
        results[ticker] = {
            'success': success,
            'message': message,
            'rows_affected': rows,
            'adjustment_applied': adj_applied
        }
    
    return results


if __name__ == "__main__":
    # Test with a single ticker
    import sys
    
    db_path = os.path.join(SCRIPT_DIR, "price_data.db")
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = "VIC"
    
    print(f"Testing intraday OHLCV update for {ticker}...")
    success, message, rows, adj_applied = update_intraday_with_adjustment_check(
        ticker, db_path, interval='1D', check_adjustment=True, debug=True
    )
    
    print(f"\nResult: {'✅ Success' if success else '❌ Failed'}")
    print(f"Message: {message}")
    print(f"Rows affected: {rows}")
    print(f"Adjustment applied: {adj_applied}")
