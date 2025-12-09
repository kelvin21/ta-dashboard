"""
Script to print the 10 latest bars for a given ticker from the database using db_adapter.
Usage:
    python test_latest_bars.py TICKER
"""

import sys
import pandas as pd
from db_adapter import get_db_adapter

def print_latest_bars(ticker, limit=50):
    try:
        db = get_db_adapter()
    except ConnectionError as e:
        print(f"❌ Failed to connect to the database: {e}")
        print("Please check your MongoDB connection settings or ensure the database is accessible.")
        sys.exit(1)

    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d")  # 5 years back

    # Use load_price_range from db_adapter
    try:
        df = db.load_price_range(ticker, start_date, end_date)
    except Exception as e:
        print(f"❌ Failed to load data for ticker '{ticker}': {e}")
        sys.exit(1)

    if df.empty:
        print(f"No data found for ticker: {ticker}")
        return

    # Sort and limit the rows
    df = df.sort_values("date", ascending=False).head(limit)

    print(f"Latest {len(df)} bars for {ticker}:")
    print(df[["date", "open", "high", "low", "close", "volume"]].to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_latest_bars.py TICKER")
        sys.exit(1)
    ticker = sys.argv[1].upper()
    print_latest_bars(ticker)
