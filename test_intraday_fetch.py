"""
Test script for TCBS intraday data fetching.
This script demonstrates how to fetch real-time intraday trading data from TCBS API.
"""

import sys

# Check for required dependencies
try:
    import requests
except ImportError:
    print("""
    ❌ Missing required package: requests
    
    Please install it using:
        pip install requests
    
    Or install all requirements:
        pip install -r requirements_intraday.txt
    """)
    sys.exit(1)

try:
    import pandas as pd
    from pandas import json_normalize
except ImportError:
    print("""
    ❌ Missing required package: pandas
    
    Please install it using:
        pip install pandas
    
    Or install all requirements:
        pip install -r requirements_intraday.txt
    """)
    sys.exit(1)

from datetime import datetime
import time
import asyncio
import aiohttp

def stock_intraday_data(symbol, page_num, page_size):
    """
    This function returns the stock realtime intraday data.
    
    Returns tick-by-tick data with columns: price, volume, time
    NOT OHLCV format - this is trade-level data.
    """
    d = datetime.now()
    if d.weekday() > 4:  # today is weekend
        url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/{symbol}/his/paging?page={page_num}&size={page_size}&headIndex=-1'
    else:  # today is weekday
        url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/{symbol}/his/paging?page={page_num}&size={page_size}'
    
    print(f"Fetching from: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Debug: Print raw response structure
        print(f"Response keys: {data.keys()}")
        
        if 'data' not in data or not data['data']:
            print(f"⚠️ No data returned for {symbol}")
            return pd.DataFrame()
        
        # Check what fields are actually returned
        if data['data']:
            print(f"Sample data fields: {data['data'][0].keys()}")
        
        df = json_normalize(data['data']).rename(columns={'p': 'price', 'v': 'volume', 't': 'time'})
        
        # Check if we have OHLC fields or just tick data
        has_ohlc = all(col in df.columns for col in ['open', 'high', 'low', 'close'])
        print(f"Has OHLC format: {has_ohlc}")
        print(f"Actual columns: {list(df.columns)}")
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()


async def fetch_page_async(session, symbol, page_num, page_size, head_index):
    """
    Async function to fetch a single page of intraday data.
    
    Args:
        session: aiohttp ClientSession
        symbol: Stock ticker symbol
        page_num: Page number to fetch
        page_size: Number of items per page
        head_index: Head index for pagination
    
    Returns:
        Tuple of (page_num, DataFrame, success)
    """
    d = datetime.now()
    
    # Build URL
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
            
            # Convert to DataFrame
            df = json_normalize(data['data']).rename(columns={'p': 'price', 'v': 'volume', 't': 'time'})
            
            # Add datetime column
            today = datetime.now().date()
            df['datetime'] = pd.to_datetime(today.strftime('%Y-%m-%d') + ' ' + df['time'])
            
            return (page_num, df, True)
            
    except Exception as e:
        print(f"    ❌ Error fetching page {page_num}: {e}")
        return (page_num, pd.DataFrame(), False)


async def fetch_all_pages_async(symbol, page_size=100):
    """
    Async version: Fetch all pages concurrently after getting metadata from first page.
    
    Args:
        symbol: Stock ticker symbol
        page_size: Number of items per page
    
    Returns:
        DataFrame with all intraday tick data
    """
    print(f"Fetching all intraday data for {symbol} (async mode)...")
    
    # Step 1: Fetch first page to get metadata
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
            print(f"  ❌ No data returned for {symbol}")
            return pd.DataFrame()
        
        # Get metadata
        total_items = data.get('total', 0)
        returned_head_index = data.get('headIndex', -1)
        
        # Calculate max pages
        max_pages = (total_items + page_size - 1) // page_size
        print(f"  Total items: {total_items}, Max pages: {max_pages}")
        
        # Convert first page
        df_first = json_normalize(data['data']).rename(columns={'p': 'price', 'v': 'volume', 't': 'time'})
        today = datetime.now().date()
        df_first['datetime'] = pd.to_datetime(today.strftime('%Y-%m-%d') + ' ' + df_first['time'])
        
        all_data = [df_first]
        
        # If only one page, return immediately
        if max_pages <= 1:
            print(f"  ✅ Fetched all {total_items} items (1 page)")
            return df_first
        
        # Step 2: Fetch remaining pages asynchronously
        print(f"  Fetching pages 2-{max_pages} concurrently...")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for page_num in range(1, max_pages):
                # Calculate head_index for this page
                if returned_head_index != -1:
                    head_index = returned_head_index + (page_num * page_size)
                else:
                    head_index = page_num * page_size
                
                task = fetch_page_async(session, symbol, page_num, page_size, head_index)
                tasks.append(task)
            
            # Execute all tasks concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start_time
            
            print(f"  ✅ Fetched {len(results)} pages in {elapsed:.2f}s (async)")
            
            # Process results
            successful = 0
            for page_num, df, success in sorted(results, key=lambda x: x[0]):
                if success and not df.empty:
                    all_data.append(df)
                    successful += 1
            
            print(f"  ✅ Successfully processed {successful}/{len(results)} pages")
        
        # Combine all data
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"✅ Total rows fetched: {len(combined)} from {len(all_data)} page(s)")
            
            # Remove duplicates
            if 'datetime' in combined.columns and 'price' in combined.columns:
                before_dedup = len(combined)
                combined = combined.drop_duplicates(subset=['datetime', 'price', 'volume'], keep='first')
                after_dedup = len(combined)
                if before_dedup != after_dedup:
                    print(f"  Removed {before_dedup - after_dedup} duplicate rows")
            
            # Sort by datetime
            combined = combined.sort_values('datetime').reset_index(drop=True)
            
            return combined
        else:
            print(f"❌ No data collected")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_all_intraday_data(symbol, use_async=True):
    """
    Fetch ALL intraday data for a symbol (all pages).
    Properly handles TCBS API pagination using headIndex.
    
    Args:
        symbol: Stock ticker symbol
        use_async: If True, use async mode for faster fetching
    
    Returns:
        DataFrame with all intraday tick data
    """
    # Check if async is available
    if use_async:
        try:
            import aiohttp
            # Run async version
            return asyncio.run(fetch_all_pages_async(symbol))
        except ImportError:
            print("  ℹ️ aiohttp not available, using sequential mode")
            use_async = False
        except Exception as e:
            print(f"  ⚠️ Async mode failed: {e}, falling back to sequential")
            use_async = False
    
    # Sequential version (original implementation)
    all_data = []
    page_num = 0
    page_size = 100
    head_index = -1
    max_pages = None
    
    print(f"Fetching all intraday data for {symbol} (sequential mode)...")
    
    while True:
        d = datetime.now()
        
        # Build URL with proper parameters
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
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Check response structure
            if 'data' not in data or not data['data']:
                print(f"  Page {page_num}: No more data")
                break
            
            # Extract pagination info
            total_items = data.get('total', 0)
            num_items = data.get('numberOfItems', 0)
            returned_head_index = data.get('headIndex', -1)
            
            # Calculate max pages on first response
            if max_pages is None and total_items > 0:
                max_pages = (total_items + page_size - 1) // page_size
                print(f"  Total items: {total_items}, Max pages: {max_pages}")
            
            print(f"  Page {page_num + 1}/{max_pages if max_pages else '?'}: {num_items} rows (headIndex={returned_head_index})")
            
            # Convert data to DataFrame
            df = json_normalize(data['data']).rename(columns={'p': 'price', 'v': 'volume', 't': 'time'})
            
            # Add datetime column immediately after fetch
            today = datetime.now().date()
            df['datetime'] = pd.to_datetime(today.strftime('%Y-%m-%d') + ' ' + df['time'])
            
            all_data.append(df)
            
            # Check if we've fetched all data
            items_fetched = sum(len(d) for d in all_data)
            
            if items_fetched >= total_items:
                print(f"  ✅ Fetched all {total_items} items")
                break
            
            # Check if we've reached max pages
            if max_pages and page_num + 1 >= max_pages:
                print(f"  ✅ Reached max page {max_pages}")
                break
            
            # If we got less than page_size, we've reached the end
            if num_items < page_size:
                print(f"  ✅ Reached end of data (partial page)")
                break
            
            # Update headIndex for next page
            if returned_head_index != -1:
                head_index = returned_head_index + page_size
            else:
                head_index = page_size
            
            page_num += 1
            time.sleep(0.1)  # Rate limiting
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error on page {page_num}: {e}")
            break
        except Exception as e:
            print(f"  ❌ Unexpected error on page {page_num}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"✅ Total rows fetched: {len(combined)} from {len(all_data)} page(s)")
        
        # Remove duplicates if any (based on datetime and price)
        if 'datetime' in combined.columns and 'price' in combined.columns:
            before_dedup = len(combined)
            combined = combined.drop_duplicates(subset=['datetime', 'price', 'volume'], keep='first')
            after_dedup = len(combined)
            if before_dedup != after_dedup:
                print(f"  Removed {before_dedup - after_dedup} duplicate rows")
        
        # Sort by datetime
        combined = combined.sort_values('datetime').reset_index(drop=True)
        
        return combined
    else:
        print(f"❌ No data fetched")
        return pd.DataFrame()


def convert_tick_to_ohlcv(df, interval='1min'):
    """
    Convert tick-by-tick data to OHLCV format.
    
    Args:
        df: DataFrame with columns ['time', 'price', 'volume']
        interval: Resample interval (e.g., '1min', '5min', '15min', '1H')
    
    Returns:
        DataFrame with OHLCV format
    """
    if df.empty or 'time' not in df.columns:
        print("⚠️ Cannot convert to OHLCV: missing required columns")
        return pd.DataFrame()
    
    # Convert time to datetime
    # TCBS returns time as string like "09:22:19", combine with today's date
    today = datetime.now().date()
    df['datetime'] = pd.to_datetime(today.strftime('%Y-%m-%d') + ' ' + df['time'])
    df = df.sort_values('datetime')
    df = df.set_index('datetime')
    
    # Resample to create OHLCV
    ohlcv = pd.DataFrame()
    ohlcv['open'] = df['price'].resample(interval).first()
    ohlcv['high'] = df['price'].resample(interval).max()
    ohlcv['low'] = df['price'].resample(interval).min()
    ohlcv['close'] = df['price'].resample(interval).last()
    ohlcv['volume'] = df['volume'].resample(interval).sum()
    
    # Remove rows with no data
    ohlcv = ohlcv.dropna()
    
    return ohlcv.reset_index()


def test_single_ticker(symbol, page_size=100):
    """Test fetching intraday data for a single ticker and check data structure."""
    print(f"\n{'='*60}")
    print(f"Testing intraday data fetch for: {symbol}")
    print(f"{'='*60}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Day of week: {datetime.now().strftime('%A')}")
    print()
    
    # Fetch first page
    df = stock_intraday_data(symbol, page_num=0, page_size=page_size)
    
    if df.empty:
        print(f"❌ No data retrieved for {symbol}")
        return None
    
    print(f"\n✅ Successfully fetched {len(df)} rows")
    print(f"\nDataFrame Info:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Check if it's OHLCV or tick data
    has_ohlcv = all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    is_tick_data = all(col in df.columns for col in ['price', 'volume', 'time'])
    
    print(f"\n📊 Data Format:")
    print(f"  Has OHLCV format: {has_ohlcv}")
    print(f"  Has tick data (price/volume/time): {is_tick_data}")
    
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    
    print(f"\nData types:")
    print(df.dtypes)
    
    if not has_ohlcv and is_tick_data:
        print(f"\n🔄 Converting tick data to OHLCV (1-minute intervals)...")
        try:
            ohlcv_df = convert_tick_to_ohlcv(df, interval='1min')
            if not ohlcv_df.empty:
                print(f"✅ Converted to {len(ohlcv_df)} OHLCV bars")
                print(f"\nFirst 5 OHLCV bars:")
                print(ohlcv_df.head())
                
                # Save OHLCV data
                ohlcv_file = f"ohlcv_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
                ohlcv_df.to_csv(ohlcv_file, index=False)
                print(f"\n💾 OHLCV data saved to: {ohlcv_file}")
            else:
                print(f"⚠️ Conversion resulted in empty dataframe")
        except Exception as e:
            print(f"❌ Error converting to OHLCV: {e}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️ Missing values detected:")
        print(missing[missing > 0])
    else:
        print(f"\n✅ No missing values")
    
    return df


def get_today_ohlcv(symbol, interval='1min'):
    """
    Fetch today's intraday data and convert to OHLCV format.
    
    Args:
        symbol: Stock ticker symbol
        interval: OHLCV bar interval (e.g., '1min', '5min', '15min', '30min', '1H')
    
    Returns:
        DataFrame with OHLCV data for today
    """
    print(f"\n{'='*60}")
    print(f"Fetching today's OHLCV data for {symbol}")
    print(f"Interval: {interval}")
    print(f"{'='*60}\n")
    
    # Fetch all intraday data (already has datetime column)
    df = fetch_all_intraday_data(symbol)
    
    if df.empty:
        print(f"❌ No intraday data available for {symbol}")
        return pd.DataFrame()
    
    # Convert to OHLCV
    print(f"\nConverting {len(df)} ticks to OHLCV bars...")
    
    try:
        # datetime already created in fetch_all_intraday_data
        # Just filter for today
        today = datetime.now().date()
        df['date'] = df['datetime'].dt.date
        df_today = df[df['date'] == today].copy()
        
        if df_today.empty:
            print(f"⚠️ No data for today ({today})")
            return pd.DataFrame()
        
        print(f"  Today's data: {len(df_today)} ticks")
        print(f"  Time range: {df_today['datetime'].min()} to {df_today['datetime'].max()}")
        
        # Set datetime as index for resampling
        df_today = df_today.set_index('datetime')
        
        # Resample to create OHLCV
        ohlcv = pd.DataFrame()
        ohlcv['open'] = df_today['price'].resample(interval).first()
        ohlcv['high'] = df_today['price'].resample(interval).max()
        ohlcv['low'] = df_today['price'].resample(interval).min()
        ohlcv['close'] = df_today['price'].resample(interval).last()
        ohlcv['volume'] = df_today['volume'].resample(interval).sum()
        
        # Remove rows with no data
        ohlcv = ohlcv.dropna()
        
        # Reset index to get datetime as column
        ohlcv = ohlcv.reset_index()
        
        print(f"✅ Converted to {len(ohlcv)} OHLCV bars")
        print(f"\nOHLCV Statistics:")
        print(f"  Bars: {len(ohlcv)}")
        print(f"  Price range: {ohlcv['low'].min():.2f} - {ohlcv['high'].max():.2f}")
        print(f"  Total volume: {ohlcv['volume'].sum():,.0f}")
        print(f"\nFirst 5 bars:")
        print(ohlcv.head())
        print(f"\nLast 5 bars:")
        print(ohlcv.tail())
        
        # Save to CSV
        today_str = datetime.now().strftime('%Y%m%d')
        output_file = f"{symbol}_ohlcv_{interval}_{today_str}.csv"
        ohlcv.to_csv(output_file, index=False)
        print(f"\n💾 Data saved to: {output_file}")
        
        return ohlcv
        
    except Exception as e:
        print(f"❌ Error converting to OHLCV: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def test_today_ohlcv_multi_interval(symbol):
    """Test OHLCV conversion with multiple intervals."""
    print(f"\n{'='*60}")
    print(f"Testing multiple OHLCV intervals for {symbol}")
    print(f"{'='*60}\n")
    
    intervals = ['1min', '5min', '15min', '30min', '1H']
    results = {}
    
    # Fetch raw data once (already has datetime column)
    print("Fetching raw intraday data...")
    df_raw = fetch_all_intraday_data(symbol)
    
    if df_raw.empty:
        print(f"❌ No data available")
        return
    
    for interval in intervals:
        print(f"\n--- Converting to {interval} bars ---")
        
        try:
            # datetime column already exists, just filter for today
            df = df_raw.copy()
            
            # Filter for today
            today = datetime.now().date()
            df['date'] = df['datetime'].dt.date
            df_today = df[df['date'] == today].copy()
            
            if df_today.empty:
                print(f"  No data for today")
                continue
            
            # Set index and resample
            df_today = df_today.set_index('datetime')
            
            ohlcv = pd.DataFrame()
            ohlcv['open'] = df_today['price'].resample(interval).first()
            ohlcv['high'] = df_today['price'].resample(interval).max()
            ohlcv['low'] = df_today['price'].resample(interval).min()
            ohlcv['close'] = df_today['price'].resample(interval).last()
            ohlcv['volume'] = df_today['volume'].resample(interval).sum()
            ohlcv = ohlcv.dropna().reset_index()
            
            results[interval] = len(ohlcv)
            print(f"  ✅ {len(ohlcv)} bars created")
            
            # Save
            today_str = datetime.now().strftime('%Y%m%d')
            output_file = f"{symbol}_ohlcv_{interval}_{today_str}.csv"
            ohlcv.to_csv(output_file, index=False)
            print(f"  💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results[interval] = 0
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for interval, count in results.items():
        print(f"  {interval:8s}: {count:4d} bars")


def test_data_structure_comparison(tickers):
    """Compare data structure across multiple tickers."""
    print(f"\n{'='*60}")
    print(f"Testing data structure consistency")
    print(f"{'='*60}\n")
    
    results = {}
    
    for symbol in tickers:
        print(f"Checking {symbol}...", end=" ")
        
        df = stock_intraday_data(symbol, page_num=0, page_size=10)
        
        if df.empty:
            print(f"❌ No data")
            results[symbol] = {"columns": [], "has_ohlcv": False, "has_tick": False}
        else:
            has_ohlcv = all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
            has_tick = all(col in df.columns for col in ['price', 'volume', 'time'])
            
            results[symbol] = {
                "columns": list(df.columns),
                "has_ohlcv": has_ohlcv,
                "has_tick": has_tick,
                "rows": len(df)
            }
            print(f"✅ {len(df)} rows")
        
        time.sleep(0.25)
    
    print(f"\n{'='*60}")
    print("Data Structure Summary:")
    print(f"{'='*60}")
    
    # Check if all tickers have same structure
    all_columns = [set(r["columns"]) for r in results.values() if r["columns"]]
    
    if all_columns and all(cols == all_columns[0] for cols in all_columns):
        print(f"✅ All tickers have consistent data structure")
        print(f"   Common columns: {', '.join(sorted(all_columns[0]))}")
    else:
        print(f"⚠️ Inconsistent data structure across tickers")
        for symbol, info in results.items():
            print(f"   {symbol}: {', '.join(info['columns'])}")
    
    # Check format
    ohlcv_count = sum(1 for r in results.values() if r["has_ohlcv"])
    tick_count = sum(1 for r in results.values() if r["has_tick"])
    
    print(f"\n📊 Data Format Distribution:")
    print(f"   OHLCV format: {ohlcv_count}/{len(results)} tickers")
    print(f"   Tick format: {tick_count}/{len(results)} tickers")
    
    return results


def quick_test_intervals(symbol="VIC"):
    """
    Quick test to fetch and display sample OHLCV data for common intervals.
    Shows first and last few bars for each interval.
    """
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║           Quick Test: Common OHLCV Intervals               ║
    ║           Symbol: {symbol:40s}║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Fetch raw data once (already has datetime column)
    print("📥 Fetching intraday data...")
    df_raw = fetch_all_intraday_data(symbol)
    
    if df_raw.empty:
        print(f"❌ No data available for {symbol}")
        return
    
    # Filter for today
    today = datetime.now().date()
    df_raw['date'] = df_raw['datetime'].dt.date
    df_today = df_raw[df_raw['date'] == today].copy()
    
    if df_today.empty:
        print(f"⚠️ No data for today ({today})")
        return
    
    print(f"✅ Loaded {len(df_today)} ticks for {today}")
    print(f"   Time range: {df_today['datetime'].min().strftime('%H:%M:%S')} to {df_today['datetime'].max().strftime('%H:%M:%S')}\n")
    
    # Test intervals
    intervals = [
        ('1min', '1-Minute'),
        ('5min', '5-Minute'),
        ('15min', '15-Minute'),
        ('30min', '30-Minute'),
        ('1H', '1-Hour')
    ]
    
    for interval, label in intervals:
        print(f"\n{'='*60}")
        print(f"📊 {label} OHLCV")
        print(f"{'='*60}")
        
        try:
            # Set index and resample
            df_interval = df_today.set_index('datetime').copy()
            
            ohlcv = pd.DataFrame()
            ohlcv['open'] = df_interval['price'].resample(interval).first()
            ohlcv['high'] = df_interval['price'].resample(interval).max()
            ohlcv['low'] = df_interval['price'].resample(interval).min()
            ohlcv['close'] = df_interval['price'].resample(interval).last()
            ohlcv['volume'] = df_interval['volume'].resample(interval).sum()
            ohlcv = ohlcv.dropna().reset_index()
            
            if ohlcv.empty:
                print(f"  ⚠️ No bars created")
                continue
            
            # Format datetime for display
            ohlcv['time'] = ohlcv['datetime'].dt.strftime('%H:%M')
            
            print(f"\n  Total bars: {len(ohlcv)}")
            print(f"  Price range: {ohlcv['low'].min():.1f} - {ohlcv['high'].max():.1f}")
            print(f"  Total volume: {ohlcv['volume'].sum():,.0f}")
            
            # Show first 3 bars
            print(f"\n  First 3 bars:")
            display_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            print(ohlcv[display_cols].head(3).to_string(index=False))
            
            # Show last 3 bars
            if len(ohlcv) > 3:
                print(f"\n  Last 3 bars:")
                print(ohlcv[display_cols].tail(3).to_string(index=False))
            
            # Save to CSV
            today_str = datetime.now().strftime('%Y%m%d')
            output_file = f"{symbol}_ohlcv_{interval}_{today_str}.csv"
            ohlcv.to_csv(output_file, index=False)
            print(f"\n  💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ Quick test completed!")
    print(f"{'='*60}")


def test_current_day_summary(symbol="VIC"):
    """
    Test to show current day trading summary from OHLCV data.
    Displays daily statistics and intraday pattern.
    """
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║           Current Day Trading Summary                      ║
    ║           Symbol: {symbol:40s}║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Get 5-minute OHLCV for analysis
    print("📥 Fetching intraday data...")
    df_raw = fetch_all_intraday_data(symbol)
    
    if df_raw.empty:
        print(f"❌ No data available")
        return
    
    # datetime column already created by fetch_all_intraday_data, just filter for today
    today = datetime.now().date()
    df_raw['date'] = df_raw['datetime'].dt.date
    df_today = df_raw[df_raw['date'] == today].copy()
    
    if df_today.empty:
        print(f"⚠️ No data for today")
        return
    
    # Create 5-minute bars
    df_5min = df_today.set_index('datetime')
    ohlcv_5min = pd.DataFrame()
    ohlcv_5min['open'] = df_5min['price'].resample('5min').first()
    ohlcv_5min['high'] = df_5min['price'].resample('5min').max()
    ohlcv_5min['low'] = df_5min['price'].resample('5min').min()
    ohlcv_5min['close'] = df_5min['price'].resample('5min').last()
    ohlcv_5min['volume'] = df_5min['volume'].resample('5min').sum()
    ohlcv_5min = ohlcv_5min.dropna().reset_index()
    
    if ohlcv_5min.empty:
        print(f"⚠️ No OHLCV data created")
        return
    
    # Calculate daily statistics
    day_open = ohlcv_5min.iloc[0]['open']
    day_high = ohlcv_5min['high'].max()
    day_low = ohlcv_5min['low'].min()
    day_close = ohlcv_5min.iloc[-1]['close']
    day_volume = ohlcv_5min['volume'].sum()
    
    day_change = day_close - day_open
    day_change_pct = (day_change / day_open * 100) if day_open > 0 else 0
    
    # Time info
    first_time = ohlcv_5min.iloc[0]['datetime'].strftime('%H:%M:%S')
    last_time = ohlcv_5min.iloc[-1]['datetime'].strftime('%H:%M:%S')
    
    print(f"\n{'='*60}")
    print(f"📈 Daily Summary for {today}")
    print(f"{'='*60}")
    print(f"  Open:      {day_open:>10.1f}")
    print(f"  High:      {day_high:>10.1f}  (at {ohlcv_5min[ohlcv_5min['high'] == day_high].iloc[0]['datetime'].strftime('%H:%M')})")
    print(f"  Low:       {day_low:>10.1f}  (at {ohlcv_5min[ohlcv_5min['low'] == day_low].iloc[0]['datetime'].strftime('%H:%M')})")
    print(f"  Close:     {day_close:>10.1f}")
    print(f"  Change:    {day_change:>10.1f}  ({day_change_pct:+.2f}%)")
    print(f"  Volume:    {day_volume:>10,.0f}")
    print(f"  Range:     {day_high - day_low:>10.1f}  ({(day_high - day_low) / day_low * 100:.2f}%)")
    print(f"  Time:      {first_time} - {last_time}")
    print(f"  Bars:      {len(ohlcv_5min)} (5-minute intervals)")
    
    # Session breakdown (morning vs afternoon)
    morning_cutoff = pd.Timestamp(f"{today} 11:30:00")
    ohlcv_5min['session'] = ohlcv_5min['datetime'].apply(lambda x: 'Morning' if x < morning_cutoff else 'Afternoon')
    
    print(f"\n{'='*60}")
    print(f"📊 Session Breakdown")
    print(f"{'='*60}")
    
    for session in ['Morning', 'Afternoon']:
        session_data = ohlcv_5min[ohlcv_5min['session'] == session]
        if not session_data.empty:
            session_volume = session_data['volume'].sum()
            session_high = session_data['high'].max()
            session_low = session_data['low'].min()
            print(f"\n  {session} Session:")
            print(f"    Bars:      {len(session_data)}")
            print(f"    Volume:    {session_volume:>10,.0f}  ({session_volume/day_volume*100:.1f}%)")
            print(f"    High:      {session_high:>10.1f}")
            print(f"    Low:       {session_low:>10.1f}")
            print(f"    Range:     {session_high - session_low:>10.1f}")
    
    print(f"\n{'='*60}")


def test_ohlcv_converter_accuracy(symbol="VIC"):
    """
    Test OHLCV converter accuracy by validating:
    1. Open is first tick price in the interval
    2. High is maximum tick price in the interval
    3. Low is minimum tick price in the interval
    4. Close is last tick price in the interval
    5. Volume is sum of tick volumes in the interval
    """
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║           OHLCV Converter Accuracy Test                    ║
    ║           Symbol: {symbol:40s}║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Fetch raw tick data
    print("📥 Fetching tick data...")
    df_raw = fetch_all_intraday_data(symbol)
    
    if df_raw.empty:
        print(f"❌ No data available for {symbol}")
        return
    
    # datetime column already created by fetch_all_intraday_data
    # Filter for today
    today = datetime.now().date()
    df_raw['date'] = df_raw['datetime'].dt.date
    df_today = df_raw[df_raw['date'] == today].copy()
    
    if df_today.empty:
        print(f"⚠️ No data for today ({today})")
        return
    
    print(f"✅ Loaded {len(df_today)} ticks")
    
    # Test with 5-minute interval
    interval = '5min'
    print(f"\n{'='*60}")
    print(f"Testing {interval} OHLCV conversion accuracy")
    print(f"{'='*60}\n")
    
    # Create OHLCV using resample
    df_resampled = df_today.set_index('datetime')
    ohlcv = pd.DataFrame()
    ohlcv['open'] = df_resampled['price'].resample(interval).first()
    ohlcv['high'] = df_resampled['price'].resample(interval).max()
    ohlcv['low'] = df_resampled['price'].resample(interval).min()
    ohlcv['close'] = df_resampled['price'].resample(interval).last()
    ohlcv['volume'] = df_resampled['volume'].resample(interval).sum()
    ohlcv = ohlcv.dropna().reset_index()
    
    if ohlcv.empty:
        print("⚠️ No OHLCV bars created")
        return
    
    print(f"✅ Created {len(ohlcv)} OHLCV bars\n")
    
    # Validate random samples
    print("Validating OHLCV accuracy (checking random samples)...\n")
    
    errors = 0
    samples_to_check = min(5, len(ohlcv))
    
    # Check first, middle, and last bars
    indices_to_check = [0, len(ohlcv)//2, len(ohlcv)-1]
    if len(ohlcv) > 3:
        import random
        indices_to_check.extend(random.sample(range(1, len(ohlcv)-1), min(2, len(ohlcv)-2)))
    
    for idx in sorted(set(indices_to_check))[:samples_to_check]:
        bar = ohlcv.iloc[idx]
        bar_start = bar['datetime']
        bar_end = bar_start + pd.Timedelta(interval)
        
        # Get ticks within this bar's time range
        mask = (df_today['datetime'] >= bar_start) & (df_today['datetime'] < bar_end)
        ticks_in_bar = df_today[mask]
        
        if ticks_in_bar.empty:
            print(f"⚠️ Bar {idx+1}: No ticks found (should not happen)")
            errors += 1
            continue
        
        # Validate OHLCV
        expected_open = ticks_in_bar.iloc[0]['price']
        expected_high = ticks_in_bar['price'].max()
        expected_low = ticks_in_bar['price'].min()
        expected_close = ticks_in_bar.iloc[-1]['price']
        expected_volume = ticks_in_bar['volume'].sum()
        
        # Check each component
        bar_time_str = bar_start.strftime('%H:%M:%S')
        errors_in_bar = []
        
        if abs(bar['open'] - expected_open) > 0.01:
            errors_in_bar.append(f"Open: {bar['open']:.1f} ≠ {expected_open:.1f}")
        if abs(bar['high'] - expected_high) > 0.01:
            errors_in_bar.append(f"High: {bar['high']:.1f} ≠ {expected_high:.1f}")
        if abs(bar['low'] - expected_low) > 0.01:
            errors_in_bar.append(f"Low: {bar['low']:.1f} ≠ {expected_low:.1f}")
        if abs(bar['close'] - expected_close) > 0.01:
            errors_in_bar.append(f"Close: {bar['close']:.1f} ≠ {expected_close:.1f}")
        if abs(bar['volume'] - expected_volume) > 0:
            errors_in_bar.append(f"Volume: {bar['volume']:,.0f} ≠ {expected_volume:,.0f}")
        
        if errors_in_bar:
            print(f"❌ Bar {idx+1} ({bar_time_str}): {len(ticks_in_bar)} ticks")
            for err in errors_in_bar:
                print(f"   {err}")
            errors += 1
        else:
            print(f"✅ Bar {idx+1} ({bar_time_str}): {len(ticks_in_bar)} ticks - All values correct")
            print(f"   O: {bar['open']:.1f}, H: {bar['high']:.1f}, L: {bar['low']:.1f}, C: {bar['close']:.1f}, V: {bar['volume']:,.0f}")
    
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    print(f"  Total bars checked: {samples_to_check}")
    print(f"  Errors found: {errors}")
    
    if errors == 0:
        print(f"\n✅ All OHLCV conversions are accurate!")
    else:
        print(f"\n⚠️ Found {errors} bar(s) with conversion errors")
    
    print(f"{'='*60}")


def compare_ohlcv_methods(symbol="VIC", interval='5min'):
    """
    Compare different OHLCV conversion methods to ensure consistency.
    Tests manual calculation vs pandas resample.
    """
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║      OHLCV Conversion Methods Comparison                   ║
    ║      Symbol: {symbol:40s}║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Fetch raw data
    print(f"📥 Fetching intraday data for {symbol}...")
    df_raw = fetch_all_intraday_data(symbol)
    
    if df_raw.empty:
        print(f"❌ No data available")
        return
    
    # datetime column already created by fetch_all_intraday_data
    today = datetime.now().date()
    df_raw['date'] = df_raw['datetime'].dt.date
    df_today = df_raw[df_raw['date'] == today].copy()
    
    if df_today.empty:
        print(f"⚠️ No data for today")
        return
    
    print(f"✅ Loaded {len(df_today)} ticks\n")
    
    # Method 1: Using pandas resample (current method)
    print("Method 1: Pandas resample()...")
    start_time = time.time()
    
    df_resampled = df_today.set_index('datetime')
    ohlcv_method1 = pd.DataFrame()
    ohlcv_method1['open'] = df_resampled['price'].resample(interval).first()
    ohlcv_method1['high'] = df_resampled['price'].resample(interval).max()
    ohlcv_method1['low'] = df_resampled['price'].resample(interval).min()
    ohlcv_method1['close'] = df_resampled['price'].resample(interval).last()
    ohlcv_method1['volume'] = df_resampled['volume'].resample(interval).sum()
    ohlcv_method1 = ohlcv_method1.dropna().reset_index()
    
    method1_time = time.time() - start_time
    print(f"  ✅ Created {len(ohlcv_method1)} bars in {method1_time:.3f}s")
    
    # Method 2: Manual groupby (alternative)
    print("\nMethod 2: Manual groupby()...")
    start_time = time.time()
    
    df_today['interval'] = df_today['datetime'].dt.floor(interval)
    ohlcv_method2 = df_today.groupby('interval').agg({
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    })
    ohlcv_method2.columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv_method2 = ohlcv_method2.reset_index()
    ohlcv_method2 = ohlcv_method2.rename(columns={'interval': 'datetime'})
    
    method2_time = time.time() - start_time
    print(f"  ✅ Created {len(ohlcv_method2)} bars in {method2_time:.3f}s")
    
    # Compare results
    print(f"\n{'='*60}")
    print("Comparison Results:")
    print(f"{'='*60}")
    
    if len(ohlcv_method1) != len(ohlcv_method2):
        print(f"⚠️ Different number of bars: {len(ohlcv_method1)} vs {len(ohlcv_method2)}")
    else:
        print(f"✅ Same number of bars: {len(ohlcv_method1)}")
    
    # Compare values for matching rows
    differences = 0
    for i in range(min(len(ohlcv_method1), len(ohlcv_method2))):
        bar1 = ohlcv_method1.iloc[i]
        bar2 = ohlcv_method2.iloc[i]
        
        if (abs(bar1['open'] - bar2['open']) > 0.01 or
            abs(bar1['high'] - bar2['high']) > 0.01 or
            abs(bar1['low'] - bar2['low']) > 0.01 or
            abs(bar1['close'] - bar2['close']) > 0.01 or
            abs(bar1['volume'] - bar2['volume']) > 0):
            differences += 1
    
    if differences == 0:
        print(f"✅ All values match perfectly")
    else:
        print(f"⚠️ Found {differences} bars with different values")
    
    print(f"\nPerformance:")
    print(f"  Method 1 (resample): {method1_time:.3f}s")
    print(f"  Method 2 (groupby):  {method2_time:.3f}s")
    print(f"  Faster: {'Method 1' if method1_time < method2_time else 'Method 2'} " +
          f"({abs(method1_time - method2_time):.3f}s difference)")
    
    print(f"\n{'='*60}")


def main():
    """Run all tests."""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║      TCBS Intraday Data Structure Verification            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    symbol = "VIC"
    
    # Test 0: Quick interval samples (NEW - most useful for quick checks)
    print("\n" + "="*60)
    print("TEST 0: Quick Interval Samples (5min, 30min, 1H)")
    print("="*60)
    quick_test_intervals(symbol)
    
    # Test 0b: Current day summary (NEW)
    print("\n" + "="*60)
    print("TEST 0b: Current Day Trading Summary")
    print("="*60)
    test_current_day_summary(symbol)
    
    # Test 0c: OHLCV converter accuracy (NEW)
    print("\n" + "="*60)
    print("TEST 0c: OHLCV Converter Accuracy Validation")
    print("="*60)
    test_ohlcv_converter_accuracy(symbol)
    
    # Test 0d: Compare OHLCV conversion methods (NEW)
    print("\n" + "="*60)
    print("TEST 0d: OHLCV Conversion Methods Comparison")
    print("="*60)
    compare_ohlcv_methods(symbol, interval='5min')
    
    # Test 1: Fetch today's OHLCV data (main feature)
    print("\n" + "="*60)
    print("TEST 1: Fetch Today's OHLCV Data (1min)")
    print("="*60)
    
    ohlcv_1min = get_today_ohlcv(symbol, interval='1min')
    
    # Test 2: Multiple intervals (full test)
    if ohlcv_1min is not None and not ohlcv_1min.empty:
        print("\n" + "="*60)
        print("TEST 2: Multiple OHLCV Intervals (Full)")
        print("="*60)
        test_today_ohlcv_multi_interval(symbol)
    
    # Test 3: Single ticker detailed check (original test)
    print("\n" + "="*60)
    print("TEST 3: Single Ticker Detailed Check")
    print("="*60)
    df_vic = test_single_ticker("VIC", page_size=100)
    
    # Test 4: Data structure consistency check
    print("\n" + "="*60)
    print("TEST 4: Data Structure Consistency Check")
    print("="*60)
    test_tickers = ["VIC", "VHM", "FPT", "HPG"]
    results = test_data_structure_comparison(test_tickers)
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
