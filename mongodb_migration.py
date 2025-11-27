"""
Migrate data from SQLite to MongoDB.
Run this once when switching from local SQLite to cloud MongoDB.

Requirements: pip install pymongo dnspython
"""
import os
import sqlite3
from datetime import datetime
import pandas as pd

# Try to import pymongo
try:
    from pymongo import MongoClient
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False
    print("❌ pymongo not installed!")
    print("📦 Install it with: pip install pymongo dnspython")

def migrate_sqlite_to_mongodb(
    sqlite_path="price_data.db",
    mongo_uri=None,
    db_name="macd_reversal"
):
    """
    Migrate all data from SQLite to MongoDB.
    
    Args:
        sqlite_path: Path to SQLite database
        mongo_uri: MongoDB connection string (uses env var if None)
        db_name: MongoDB database name
    """
    if not HAS_PYMONGO:
        print("\n❌ Cannot migrate: pymongo not installed")
        print("📦 Run: pip install pymongo dnspython")
        return False
    
    print("🔄 Starting migration from SQLite to MongoDB...")
    
    # Connect to SQLite
    if not os.path.exists(sqlite_path):
        print(f"❌ SQLite database not found: {sqlite_path}")
        return False
    
    sqlite_conn = sqlite3.connect(sqlite_path)
    
    # Connect to MongoDB
    if mongo_uri is None:
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            print("❌ MONGODB_URI environment variable not set")
            return False
    
    try:
        mongo_client = MongoClient(mongo_uri)
        mongo_db = mongo_client[db_name]
        
        # Test connection
        mongo_client.admin.command('ping')
        print("✅ Connected to MongoDB")
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False
    
    # Migrate price_data
    print("\n📊 Migrating price_data...")
    try:
        df = pd.read_sql_query("SELECT * FROM price_data", sqlite_conn)
        if not df.empty:
            # Convert DataFrame to list of dicts
            records = df.to_dict('records')
            
            # Add created_at if not present
            for record in records:
                if 'created_at' not in record:
                    record['created_at'] = datetime.now()
            
            # Insert into MongoDB
            mongo_db.price_data.insert_many(records, ordered=False)
            print(f"   ✓ Migrated {len(records)} price data records")
        else:
            print("   ℹ️ No price data to migrate")
    except Exception as e:
        print(f"   ⚠️ Price data migration error: {e}")
    
    # Migrate market_data
    print("\n📈 Migrating market_data...")
    try:
        df = pd.read_sql_query("SELECT * FROM market_data", sqlite_conn)
        if not df.empty:
            records = df.to_dict('records')
            mongo_db.market_data.insert_many(records, ordered=False)
            print(f"   ✓ Migrated {len(records)} market data records")
        else:
            print("   ℹ️ No market data to migrate")
    except Exception as e:
        print(f"   ⚠️ Market data migration error: {e}")
    
    # Migrate tcbs_scaling
    print("\n⚖️ Migrating tcbs_scaling...")
    try:
        df = pd.read_sql_query("SELECT * FROM tcbs_scaling", sqlite_conn)
        if not df.empty:
            records = df.to_dict('records')
            mongo_db.tcbs_scaling.insert_many(records, ordered=False)
            print(f"   ✓ Migrated {len(records)} scaling records")
        else:
            print("   ℹ️ No scaling data to migrate")
    except Exception as e:
        print(f"   ⚠️ Scaling data migration error: {e}")
    
    # Create indexes
    print("\n🔍 Creating MongoDB indexes...")
    try:
        mongo_db.price_data.create_index([("ticker", 1), ("date", -1)])
        mongo_db.price_data.create_index([("ticker", 1), ("date", -1), ("source", 1)], unique=True)
        mongo_db.market_data.create_index([("date", -1)], unique=True)
        mongo_db.tcbs_scaling.create_index([("ticker", 1)], unique=True)
        print("   ✓ Indexes created")
    except Exception as e:
        print(f"   ⚠️ Index creation warning: {e}")
    
    # Close connections
    sqlite_conn.close()
    mongo_client.close()
    
    print("\n✅ Migration completed successfully!")
    print(f"📊 MongoDB database: {db_name}")
    print(f"💡 Set USE_MONGODB=true in your .env file to use MongoDB")
    
    return True

if __name__ == "__main__":
    if not HAS_PYMONGO:
        print("\n💡 To use MongoDB migration:")
        print("   1. Install dependencies: pip install pymongo dnspython")
        print("   2. Run this script again")
        import sys
        sys.exit(1)
    
    # Get MongoDB URI from command line or environment
    mongo_uri = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MONGODB_URI")
    
    if not mongo_uri:
        print("❌ Please provide MongoDB URI:")
        print("   python mongodb_migration.py 'mongodb+srv://...'")
        print("   OR set MONGODB_URI environment variable")
        sys.exit(1)
    
    migrate_sqlite_to_mongodb(mongo_uri=mongo_uri)
