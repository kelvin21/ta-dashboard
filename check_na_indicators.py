"""Check for N/A indicators in the database."""
import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")

client = MongoClient(mongo_uri)
db = client[db_name]

# Check most recent date
recent_date = datetime(2025, 12, 8)

print(f"Checking indicators for date: {recent_date.date()}")
print("="*60)

# Count total and N/A
total_count = db.indicators.count_documents({"date": recent_date})
macd_na_count = db.indicators.count_documents({"date": recent_date, "macd_stage": "N/A"})

print(f"\nTotal tickers: {total_count}")
print(f"MACD stage = 'N/A': {macd_na_count} ({macd_na_count/total_count*100:.1f}%)")

# Find a sample ticker with N/A
sample_na = db.indicators.find_one({"date": recent_date, "macd_stage": "N/A"})
if sample_na:
    print(f"\nSample ticker with N/A: {sample_na.get('ticker')}")
    print(f"  RSI: {sample_na.get('rsi')}")
    print(f"  MACD: {sample_na.get('macd')}")
    print(f"  MACD Signal: {sample_na.get('macd_signal')}")
    print(f"  MACD Hist: {sample_na.get('macd_hist')}")
    print(f"  MACD Stage: {sample_na.get('macd_stage')}")
    
    # Check if this ticker has price data
    ticker = sample_na.get('ticker')
    price_count = db.ohlcv.count_documents({"ticker": ticker})
    print(f"  Price data records: {price_count}")
    
    # Check earliest price date
    earliest = db.ohlcv.find_one({"ticker": ticker}, sort=[("date", 1)])
    if earliest:
        print(f"  Earliest price date: {earliest['date']}")
        days_of_data = (recent_date - earliest['date']).days
        print(f"  Days of data: {days_of_data}")

# Find a sample ticker with valid MACD
sample_valid = db.indicators.find_one({"date": recent_date, "macd_stage": {"$ne": "N/A"}})
if sample_valid:
    print(f"\nSample ticker with valid MACD: {sample_valid.get('ticker')}")
    print(f"  MACD Stage: {sample_valid.get('macd_stage')}")
    ticker = sample_valid.get('ticker')
    price_count = db.ohlcv.count_documents({"ticker": ticker})
    print(f"  Price data records: {price_count}")

client.close()
