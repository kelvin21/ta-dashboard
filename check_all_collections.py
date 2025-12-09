"""Check all MongoDB collections for OHLCV data."""
import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")

client = MongoClient(mongo_uri)
db = client[db_name]

print("MongoDB Collections Analysis")
print("="*60)

# List all collections
collections = db.list_collection_names()
print(f"Available collections: {collections}")
print()

# Check each collection for data
for coll_name in collections:
    coll = db[coll_name]
    count = coll.count_documents({})
    print(f"Collection: {coll_name}")
    print(f"  Total documents: {count}")
    
    if count > 0:
        # Get sample document
        sample = coll.find_one()
        if sample:
            print(f"  Sample keys: {list(sample.keys())}")
            
            # Check if it looks like OHLCV data
            has_ohlcv = all(k in sample for k in ['open', 'high', 'low', 'close', 'volume'])
            has_ticker = 'ticker' in sample
            has_date = 'date' in sample
            
            if has_ohlcv and has_ticker and has_date:
                print(f"  ✅ This looks like OHLCV data!")
                print(f"     Sample ticker: {sample.get('ticker')}")
                print(f"     Sample date: {sample.get('date')}")
                print(f"     Sample close: {sample.get('close')}")
                
                # Count unique tickers
                unique_tickers = coll.distinct('ticker')
                print(f"     Unique tickers: {len(unique_tickers)}")
                
                # Check date range
                earliest = coll.find_one(sort=[('date', 1)])
                latest = coll.find_one(sort=[('date', -1)])
                if earliest and latest:
                    print(f"     Date range: {earliest['date']} to {latest['date']}")
    print()

# Check what db_adapter expects
print("\ndb_adapter.py Collections Mapping:")
print("="*60)
print("Expected collections:")
print("  • self.price_data = self.db.price_data")
print("  • self.market_data = self.db.market_data")
print("  • self.tcbs_scaling = self.db.tcbs_scaling")
print("  • self.user_settings = self.db.user_settings")
print()
print("Note: The 'ohlcv' collection vs 'price_data' collection")
print("      db_adapter uses 'price_data' but may have 'ohlcv' collection")

client.close()
