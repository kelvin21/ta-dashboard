"""Check price data availability."""
import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")

client = MongoClient(mongo_uri)
db = client[db_name]

print("Price Data Analysis")
print("="*60)

# Count total OHLCV records
total_price_records = db.ohlcv.count_documents({})
print(f"Total price records in ohlcv collection: {total_price_records}")

# Count unique tickers with price data
tickers_with_prices = db.ohlcv.distinct("ticker")
print(f"Tickers with price data: {len(tickers_with_prices)}")
print(f"Sample tickers: {tickers_with_prices[:5]}")

# Check a specific date
recent_date = datetime(2025, 12, 8)
price_on_date = db.ohlcv.count_documents({"date": recent_date})
print(f"\nPrice records on {recent_date.date()}: {price_on_date}")

# Check indicator data
print("\nIndicator Data Analysis")
print("="*60)
total_indicators = db.indicators.count_documents({})
print(f"Total indicator records: {total_indicators}")

tickers_with_indicators = db.indicators.distinct("ticker")
print(f"Tickers with indicators: {len(tickers_with_indicators)}")
print(f"Sample tickers: {tickers_with_indicators[:5]}")

# Check if there's a mismatch
print("\nMismatch Analysis")
print("="*60)
indicators_without_prices = [t for t in tickers_with_indicators if t not in tickers_with_prices]
print(f"Tickers with indicators but NO price data: {len(indicators_without_prices)}")
if indicators_without_prices:
    print(f"Examples: {indicators_without_prices[:10]}")

client.close()
