"""
Verify MongoDB connection and migration status.
"""
import os
import sys

# Load environment from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def verify_mongodb_connection():
    """Check MongoDB connection and data."""
    
    print("🔍 Verifying MongoDB Setup...\n")
    
    # Check environment variables
    use_mongodb = os.getenv("USE_MONGODB", "false").lower()
    mongodb_uri = os.getenv("MONGODB_URI", "")
    
    print(f"USE_MONGODB: {use_mongodb}")
    print(f"MONGODB_URI: {'Set ✓' if mongodb_uri else 'Not set ✗'}\n")
    
    if use_mongodb != "true":
        print("⚠️ USE_MONGODB is not set to 'true'")
        print("Set it with: export USE_MONGODB=true")
        return False
    
    if not mongodb_uri:
        print("❌ MONGODB_URI not set")
        print("Set it with: export MONGODB_URI='mongodb+srv://...'")
        return False
    
    # Try to import pymongo
    try:
        from pymongo import MongoClient
        print("✓ pymongo installed")
    except ImportError:
        print("❌ pymongo not installed")
        print("Install with: pip install pymongo dnspython")
        return False
    
    # Try to connect
    try:
        print(f"\n🔗 Connecting to MongoDB...")
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        print("✓ MongoDB connection successful!")
        
        # Get database
        db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
        db = client[db_name]
        
        print(f"\n📊 Database: {db_name}")
        
        # Check collections
        collections = db.list_collection_names()
        print(f"Collections: {', '.join(collections) if collections else 'None'}")
        
        # Check price_data
        if "price_data" in collections:
            count = db.price_data.count_documents({})
            print(f"\n✓ price_data collection exists")
            print(f"  Records: {count:,}")
            
            if count > 0:
                # Get sample data
                sample = db.price_data.find_one()
                print(f"  Sample ticker: {sample.get('ticker', 'N/A')}")
                print(f"  Sample date: {sample.get('date', 'N/A')}")
                
                # Get unique tickers
                tickers = db.price_data.distinct("ticker")
                print(f"  Unique tickers: {len(tickers)}")
                print(f"  Tickers: {', '.join(sorted(tickers)[:10])}" + 
                      (f" and {len(tickers)-10} more..." if len(tickers) > 10 else ""))
                
                # Get date range
                pipeline = [
                    {"$group": {
                        "_id": None,
                        "min_date": {"$min": "$date"},
                        "max_date": {"$max": "$date"}
                    }}
                ]
                date_range = list(db.price_data.aggregate(pipeline))
                if date_range:
                    print(f"  Date range: {date_range[0]['min_date']} to {date_range[0]['max_date']}")
                
                print("\n✅ Migration appears COMPLETE!")
                print("   Your data is in MongoDB and ready to use.")
            else:
                print("\n⚠️ price_data collection is EMPTY")
                print("   Run migration: python mongodb_migration.py")
        else:
            print("\n❌ price_data collection does NOT exist")
            print("   Run migration: python mongodb_migration.py")
        
        # Check other collections
        if "market_data" in collections:
            count = db.market_data.count_documents({})
            print(f"\n✓ market_data: {count} records")
        
        if "tcbs_scaling" in collections:
            count = db.tcbs_scaling.count_documents({})
            print(f"✓ tcbs_scaling: {count} records")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"\n❌ MongoDB connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your MONGODB_URI is correct")
        print("2. Verify network access in MongoDB Atlas")
        print("3. Ensure database user has correct permissions")
        return False

def check_local_database():
    """Check local SQLite database for comparison."""
    import sqlite3
    
    db_path = "price_data.db"
    
    print(f"\n📁 Checking local SQLite database...")
    
    if not os.path.exists(db_path):
        print(f"❌ {db_path} not found")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Get counts
        cur.execute("SELECT COUNT(*) FROM price_data")
        count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT ticker) FROM price_data")
        ticker_count = cur.fetchone()[0]
        
        cur.execute("SELECT MIN(date), MAX(date) FROM price_data")
        date_range = cur.fetchone()
        
        conn.close()
        
        print(f"✓ SQLite database found")
        print(f"  Records: {count:,}")
        print(f"  Tickers: {ticker_count}")
        print(f"  Date range: {date_range[0]} to {date_range[1]}")
        
        return count
        
    except Exception as e:
        print(f"⚠️ Error reading SQLite: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("MongoDB Migration Verification Tool")
    print("=" * 60 + "\n")
    
    # Check local database first
    local_count = check_local_database()
    
    # Check MongoDB
    success = verify_mongodb_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ MongoDB is ready for deployment!")
        print("\nNext steps:")
        print("1. Commit and push to GitHub")
        print("2. Deploy to Streamlit Cloud")
        print("3. Add MongoDB secrets in Streamlit Cloud settings")
    else:
        print("❌ MongoDB setup incomplete")
        print("\nNext steps:")
        print("1. Fix the issues above")
        print("2. Run: python mongodb_migration.py")
        print("3. Run this script again to verify")
    print("=" * 60)
