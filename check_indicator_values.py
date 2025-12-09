"""Check indicator values in database."""
import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")

client = MongoClient(mongo_uri)
db = client[db_name]

# Check a specific date and ticker
ticker = "AAA"
check_date = datetime(2025, 12, 8)

print(f"Checking indicators for {ticker} on {check_date.date()}")
print("="*60)

indicator_doc = db.indicators.find_one({"ticker": ticker, "date": check_date})

if indicator_doc:
    print("✅ Indicator document found!")
    print(f"\nFields:")
    for key, value in indicator_doc.items():
        if key != '_id':
            print(f"  {key}: {value}")
    
    # Check if values are actually NaN
    print(f"\n Analysis:")
    import math
    if indicator_doc.get('rsi'):
        if math.isnan(indicator_doc['rsi']):
            print(f"  RSI is NaN (not enough data)")
        else:
            print(f"  RSI is valid: {indicator_doc['rsi']:.2f}")
    else:
        print(f"  RSI is None/missing")
        
    if indicator_doc.get('macd_hist'):
        if math.isnan(indicator_doc['macd_hist']):
            print(f"  MACD Hist is NaN (not enough data)")
        else:
            print(f"  MACD Hist is valid: {indicator_doc['macd_hist']:.6f}")
    else:
        print(f"  MACD Hist is None/missing")
        
    print(f"  MACD Stage: {indicator_doc.get('macd_stage')}")
else:
    print(f"❌ No indicator document found for {ticker} on {check_date.date()}")

# Count valid vs N/A indicators
print(f"\n" + "="*60)
print(f"Indicator Statistics for {check_date.date()}")
print("="*60)

total = db.indicators.count_documents({"date": check_date})
print(f"Total indicator records: {total}")

# Check RSI
rsi_null = db.indicators.count_documents({"date": check_date, "rsi": None})
rsi_nan = db.indicators.count_documents({"date": check_date, "rsi": float('nan')})
print(f"\nRSI:")
print(f"  NULL: {rsi_null}")
print(f"  NaN: {rsi_nan}")
print(f"  Valid: {total - rsi_null}")

# Check MACD stage
macd_na = db.indicators.count_documents({"date": check_date, "macd_stage": "N/A"})
macd_valid = total - macd_na
print(f"\nMACD Stage:")
print(f"  N/A: {macd_na}")
print(f"  Valid: {macd_valid}")

if macd_valid > 0:
    # Show sample of valid MACD stages
    valid_samples = list(db.indicators.find(
        {"date": check_date, "macd_stage": {"$ne": "N/A"}},
        {"ticker": 1, "macd_stage": 1, "macd_hist": 1}
    ).limit(5))
    
    print(f"\n  Sample tickers with valid MACD:")
    for doc in valid_samples:
        print(f"    {doc['ticker']}: {doc['macd_stage']} (hist: {doc.get('macd_hist', 'N/A')})")

client.close()
