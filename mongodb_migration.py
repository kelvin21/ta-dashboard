"""
Migrate data from SQLite to MongoDB.
Run this once when switching from local SQLite to cloud MongoDB.

Requirements: pip install pymongo dnspython
"""
import os
import sys
import sqlite3
from datetime import datetime
import pandas as pd

# HARDCODED CONNECTION STRING (Optional - only if .env doesn't work)
# Uncomment and set your MongoDB URI here
HARDCODED_MONGODB_URI = "mongodb+srv://longhalucky2111_db_user:abc123abc@cluster0.g4ndhy9.mongodb.net/?appName=Cluster0"

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables only")

# Try to import pymongo
try:
    from pymongo.mongo_client import MongoClient
    from pymongo.server_api import ServerApi
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
    # Priority: 1) parameter, 2) hardcoded, 3) command line, 4) environment
    if mongo_uri is None:
        if HARDCODED_MONGODB_URI:
            mongo_uri = HARDCODED_MONGODB_URI
            print("🔍 Using HARDCODED MongoDB URI from mongodb_migration.py")
        else:
            mongo_uri = os.getenv("MONGODB_URI")
            print("🔍 Using MongoDB URI from environment variable")
    else:
        print("🔍 Using MongoDB URI from function parameter")
    print(f"  URI length: {len(mongo_uri) if mongo_uri else 'None'} characters")
    if not mongo_uri:
        print("❌ MONGODB_URI not found")
        print("\nOptions to set MongoDB URI:")
        print("1. Hardcode in mongodb_migration.py:")
        print('   HARDCODED_MONGODB_URI = "mongodb+srv://..."')
        print("2. Set environment variable (PowerShell):")
        print('   $env:MONGODB_URI="mongodb+srv://..."')
        print("3. Add to .env file:")
        print("   MONGODB_URI=mongodb+srv://...")
        return False
    
    # Debug: Show connection info (hide password)
    print("\n🔍 Connection String Check:")
    if '@' in mongo_uri:
        safe_uri = mongo_uri.split('@')[0].split(':')[0] + ':****@' + mongo_uri.split('@')[1]
        
        print(f"  URI (masked): {safe_uri}")
    print(f"  Database: {db_name}")
    print(f"  URI length: {len(mongo_uri)} characters")
    
    try:
        print("\n🔗 Attempting MongoDB connection...")
        
        # Create MongoDB client with ServerApi
        mongo_client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        
        # Test connection
        mongo_client.admin.command('ping')
        print("✅ Pinged your deployment. You successfully connected to MongoDB!")
        
        mongo_db = mongo_client[db_name]
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\n🔍 Troubleshooting:")
        print("1. Check Network Access in MongoDB Atlas:")
        print("   → https://cloud.mongodb.com → Network Access → Add IP: 0.0.0.0/0")
        print("2. Verify connection string format:")
        print("   mongodb+srv://username:password@cluster.mongodb.net/?appName=...")
        print("3. Ensure password has no special characters or is URL-encoded")
        print("4. Wait 5 minutes after changing Network Access settings")
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
    print("=" * 60)
    print("MongoDB Migration Tool")
    print("=" * 60 + "\n")
    
    if not HAS_PYMONGO:
        print("\n💡 To use MongoDB migration:")
        print("   1. Install dependencies: pip install pymongo dnspython")
        print("   2. Run this script again")
        sys.exit(1)
    
    # Get MongoDB URI - priority to hardcoded value
    if HARDCODED_MONGODB_URI:
        mongo_uri = HARDCODED_MONGODB_URI
        print("✅ Using HARDCODED MongoDB URI")
    else:
        mongo_uri = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MONGODB_URI")
        print("🔍 Using MongoDB URI from command line or environment")
    
    # Debug: Show what we found
    print("🔍 Debug - URI Source:")
    if len(sys.argv) > 1:
        print("  ✓ Using URI from command line argument")
        print(f"  Argument length: {len(sys.argv[1])} characters")
    else:
        print("  ✓ Using URI from environment variable")
        if mongo_uri:
            print(f"  URI found, length: {len(mongo_uri)} characters")
        else:
            print("  ❌ No URI found in environment")
    
    if not mongo_uri:
        print("\n❌ Please provide MongoDB URI:")
        print("\nOption 1 - Command line:")
        print("   python mongodb_migration.py 'mongodb+srv://...'")
        print("\nOption 2 - Environment variable (PowerShell):")
        print('   $env:MONGODB_URI="mongodb+srv://..."')
        print("   python mongodb_migration.py")
        print("\nOption 3 - .env file:")
        print("   Add to .env: MONGODB_URI=mongodb+srv://...")
        print("   python mongodb_migration.py")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    migrate_sqlite_to_mongodb(mongo_uri=mongo_uri)
    print("=" * 60)
