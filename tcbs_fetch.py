"""Simple wrapper for TCBS API fetching with proper parameters."""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_price_data(ticker: str, days_back: int = 365) -> pd.DataFrame:
    """
    Fetch price data from TCBS API.
    
    Args:
        ticker: Stock ticker symbol
        days_back: Number of days of historical data to fetch
    
    Returns:
        DataFrame with OHLCV data (prices scaled to match DB format)
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Convert to Unix timestamps (seconds)
        from_timestamp = int(start_date.timestamp())
        to_timestamp = int(end_date.timestamp())
        
        # TCBS API endpoint with required 'from' and 'to' parameters
        url = (
            f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
            f"?ticker={ticker}"
            f"&type=stock"
            f"&resolution=D"
            f"&from={from_timestamp}"
            f"&to={to_timestamp}"
        )
        
        # Make request with timeout
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if data exists
        if 'data' not in data or not data['data']:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Map TCBS column names to standard names
        # TCBS API may return different column names depending on endpoint
        column_mapping = {
            't': 'date',              # timestamp (seconds) - old format
            'tradingDate': 'date',    # YYYY-MM-DD string - new format
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }
        
        # Rename columns that exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Validate we have a date column now
        if 'date' not in df.columns:
            print(f"TCBS response missing date column for {ticker}. Columns: {list(df.columns)}")
            return pd.DataFrame()
        
        # Convert date to datetime
        # Handle both Unix timestamp (numeric) and date string formats
        if pd.api.types.is_numeric_dtype(df['date']):
            # Unix timestamp in seconds
            df['date'] = pd.to_datetime(df['date'], unit='s')
        else:
            # Date string (YYYY-MM-DD)
            df['date'] = pd.to_datetime(df['date'])
        
        # Scale OHLC prices by dividing by 1000 to match database format
        # TCBS API returns prices multiplied by 1000
        # EXCEPTION: VNINDEX prices are NOT scaled (already in correct format)
        price_cols = ['open', 'high', 'low', 'close']
        
        if ticker.upper() != 'VNINDEX':
            # For regular stocks: divide by 1000
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col] / 1000.0
        else:
            # For VNINDEX: keep as-is (already correctly scaled)
            if 'open' in df.columns:
                # Ensure numeric type
                for col in price_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns for {ticker}: {missing_cols}. Available: {list(df.columns)}")
            return pd.DataFrame()
        
        # Select and order columns
        df = df[required_cols]
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error fetching {ticker}: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return pd.DataFrame()
    
    except requests.exceptions.RequestException as e:
        print(f"Request Error fetching {ticker}: {e}")
        return pd.DataFrame()
    
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_price_data_with_retry(ticker: str, days_back: int = 365, max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch price data with retry logic.
    
    Args:
        ticker: Stock ticker symbol
        days_back: Number of days of historical data to fetch
        max_retries: Maximum number of retry attempts
    
    Returns:
        DataFrame with OHLCV data
    """
    for attempt in range(max_retries):
        try:
            df = fetch_price_data(ticker, days_back)
            if not df.empty:
                return df
            
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries} for {ticker}...")
                time.sleep(1)  # Wait 1 second between retries
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                time.sleep(1)
            else:
                print(f"All retry attempts failed for {ticker}")
    
    return pd.DataFrame()


if __name__ == "__main__":
    # Test the fetch function
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "VIC"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    print(f"Fetching {days} days of data for {ticker}...")
    df = fetch_price_data(ticker, days_back=days)
    
    if not df.empty:
        print(f"\n✓ Successfully fetched {len(df)} bars")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nLast 5 rows:")
        print(df.tail())
    else:
        print("✗ Failed to fetch data")
