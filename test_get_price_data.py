"""Test if get_price_data works correctly."""
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add parent to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

load_dotenv()

# Import db_async
from utils.db_async import get_sync_db_adapter

print("Testing get_price_data()")
print("="*60)

db = get_sync_db_adapter()

# Test with a known ticker
ticker = "AAA"
end_date = datetime(2025, 12, 8)
start_date = end_date - timedelta(days=30)

print(f"\nTesting: {ticker}")
print(f"Date range: {start_date.date()} to {end_date.date()}")

df = db.get_price_data(ticker, start_date, end_date)

print(f"\nResult:")
print(f"  Rows returned: {len(df)}")

if not df.empty:
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\n  Sample data (first 3 rows):")
    print(df.head(3)[['date', 'ticker', 'close', 'volume']])
else:
    print(f"  ❌ DataFrame is EMPTY!")
    
    # Check if data exists in MongoDB directly
    from pymongo import MongoClient
    mongo_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
    
    client = MongoClient(mongo_uri)
    mongo_db = client[db_name]
    
    # Direct MongoDB query
    print(f"\n  Checking MongoDB directly...")
    query = {
        "ticker": ticker.upper(),
        "date": {"$gte": start_date, "$lte": end_date}
    }
    count = mongo_db.price_data.count_documents(query)
    print(f"  MongoDB query result: {count} documents")
    
    if count > 0:
        sample = mongo_db.price_data.find_one(query)
        print(f"  Sample document: {sample}")
    else:
        # Check if ticker exists at all
        any_ticker = mongo_db.price_data.count_documents({"ticker": ticker.upper()})
        print(f"  Total documents for {ticker}: {any_ticker}")
        
        if any_ticker > 0:
            earliest = mongo_db.price_data.find_one({"ticker": ticker.upper()}, sort=[("date", 1)])
            latest = mongo_db.price_data.find_one({"ticker": ticker.upper()}, sort=[("date", -1)])
            print(f"  Date range for {ticker}: {earliest['date']} to {latest['date']}")
            print(f"  Requested range: {start_date} to {end_date}")
    
    client.close()

print("\n" + "="*60)
print("Testing with multiple tickers...")

tickers_to_test = ["AAA", "FPT", "VCB", "VNINDEX"]
for t in tickers_to_test:
    df = db.get_price_data(t, start_date, end_date)
    print(f"  {t}: {len(df)} rows")
