import os
import pandas as pd
from datetime import datetime
from db_adapter import get_db_adapter

# Initialize the database adapter
db = get_db_adapter()

try:
    # Debug query parameters
    tickers = ["FPT", "VCB", "DCM"]  # Replace with actual tickers to test
    start_date = "2023-01-01"
    end_date = "2025-11-30"

    print(f"🔍 Querying data for tickers: {tickers}, from {start_date} to {end_date}")

    # Use load_price_range_multi from db_adapter
    result = db.load_price_range_multi(tickers, start_date, end_date)

    # Debug output
    for ticker, df in result.items():
        print(f"🔍 Data for ticker {ticker}: {len(df)} rows.")
        if not df.empty:
            print(df.head())
        else:
            print(f"⚠️ No data found for ticker {ticker}.")

except Exception as e:
    print(f"❌ Error during query: {e}")
