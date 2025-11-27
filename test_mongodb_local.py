"""
Comprehensive MongoDB test - local and cloud ready
Tests connection, data operations, and deployment readiness
"""
import os
import sys
from datetime import datetime

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file\n")
except:
    print("⚠️ .env not loaded\n")

def test_mongodb_connection():
    """Test MongoDB connection with detailed diagnostics"""
    
    print("=" * 60)
    print("MongoDB Connection Test")
    print("=" * 60 + "\n")
    
    # Check pymongo
    try:
        from pymongo.mongo_client import MongoClient
        from pymongo.server_api import ServerApi
        print("✓ pymongo installed\n")
    except ImportError:
        print("❌ pymongo not installed")
        print("   Run: pip install pymongo dnspython\n")
        return False
    
    # Get connection details
    use_mongodb = os.getenv("USE_MONGODB", "false")
    mongodb_uri = os.getenv("MONGODB_URI", "")
    db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
    
    print("🔍 Configuration:")
    print(f"  USE_MONGODB: {use_mongodb}")
    print(f"  MONGODB_URI: {'Set ✓' if mongodb_uri else 'Not set ✗'}")
    print(f"  DB_NAME: {db_name}\n")
    
    if not mongodb_uri:
        print("❌ MONGODB_URI not set")
        print("\nTo fix:")
        print("1. Update .env file:")
        print("   MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/")
        print("2. Or set environment variable (PowerShell):")
        print('   $env:MONGODB_URI="mongodb+srv://..."')
        return False
    
    # Test connection
    print("🔗 Testing MongoDB connection...\n")
    try:
        client = MongoClient(mongodb_uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
        
        # Ping
        client.admin.command('ping')
        print("✅ SUCCESS! Connected to MongoDB\n")
        
        # Get database
        db = client[db_name]
        
        # List collections
        collections = db.list_collection_names()
        print(f"📊 Database: {db_name}")
        print(f"   Collections: {', '.join(collections) if collections else 'None (empty database)'}\n")
        
        # Test read/write
        print("🧪 Testing read/write operations...\n")
        
        test_collection = db.test_connection
        
        # Write test
        test_doc = {
            "test": "connection_test",
            "timestamp": datetime.now(),
            "message": "If you see this, MongoDB read/write works!"
        }
        result = test_collection.insert_one(test_doc)
        print(f"  ✓ Write test: Inserted document ID {result.inserted_id}")
        
        # Read test
        found = test_collection.find_one({"test": "connection_test"})
        if found:
            print(f"  ✓ Read test: Found document - {found['message']}")
        
        # Cleanup test
        test_collection.delete_many({"test": "connection_test"})
        print(f"  ✓ Delete test: Cleaned up test data\n")
        
        # Check price_data if exists
        if "price_data" in collections:
            count = db.price_data.count_documents({})
            print(f"📈 price_data collection:")
            print(f"   Records: {count:,}")
            
            if count > 0:
                # Get sample
                sample = db.price_data.find_one()
                print(f"   Sample ticker: {sample.get('ticker', 'N/A')}")
                print(f"   Sample date: {sample.get('date', 'N/A')}")
                
                # Get tickers
                tickers = db.price_data.distinct("ticker")
                print(f"   Unique tickers: {len(tickers)}")
                if tickers:
                    print(f"   Tickers: {', '.join(sorted(tickers)[:5])}" + 
                          (f" and {len(tickers)-5} more..." if len(tickers) > 5 else ""))
        
        client.close()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - MongoDB is ready!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}\n")
        print("🔍 Troubleshooting:")
        print("1. Check MongoDB Atlas → Network Access → Add IP: 0.0.0.0/0")
        print("2. Verify password is correct in connection string")
        print("3. Ensure cluster is not paused (should be green in Atlas)")
        print("4. Wait 5 minutes after changing Network Access settings")
        return False

def test_db_adapter():
    """Test database adapter integration"""
    
    print("\n" + "=" * 60)
    print("Database Adapter Test")
    print("=" * 60 + "\n")
    
    try:
        from db_adapter import get_db_adapter
        
        print("🔧 Testing db_adapter...\n")
        
        db = get_db_adapter()
        print(f"  ✓ Database adapter initialized")
        print(f"  ✓ Database type: {db.db_type}\n")
        
        # Test get_all_tickers
        tickers = db.get_all_tickers()
        print(f"  ✓ get_all_tickers() works")
        print(f"    Found {len(tickers)} tickers\n")
        
        if tickers:
            # Test load_price_range
            test_ticker = tickers[0]
            from datetime import date, timedelta
            end = date.today()
            start = end - timedelta(days=30)
            
            df = db.load_price_range(test_ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            print(f"  ✓ load_price_range() works")
            print(f"    Loaded {len(df)} rows for {test_ticker}\n")
        
        print("=" * 60)
        print("✅ Database Adapter Test PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Database adapter test failed: {e}")
        return False

def cloud_deployment_check():
    """Check if ready for cloud deployment"""
    
    print("\n" + "=" * 60)
    print("Cloud Deployment Readiness Check")
    print("=" * 60 + "\n")
    
    checks = []
    
    # Check 1: .gitignore exists
    if os.path.exists(".gitignore"):
        print("✓ .gitignore exists")
        checks.append(True)
    else:
        print("✗ .gitignore missing")
        checks.append(False)
    
    # Check 2: requirements.txt exists
    if os.path.exists("requirements.txt"):
        print("✓ requirements.txt exists")
        with open("requirements.txt") as f:
            content = f.read()
            if "pymongo" in content:
                print("  ✓ pymongo in requirements.txt")
            else:
                print("  ✗ pymongo NOT in requirements.txt")
                checks.append(False)
        checks.append(True)
    else:
        print("✗ requirements.txt missing")
        checks.append(False)
    
    # Check 3: .streamlit/config.toml exists
    if os.path.exists(".streamlit/config.toml"):
        print("✓ .streamlit/config.toml exists")
        checks.append(True)
    else:
        print("⚠️ .streamlit/config.toml missing (optional)")
    
    # Check 4: Database files not in git
    if os.path.exists(".gitignore"):
        with open(".gitignore") as f:
            gitignore = f.read()
            if "*.db" in gitignore:
                print("✓ Database files excluded from git")
                checks.append(True)
            else:
                print("⚠️ Database files might be tracked by git")
    
    # Check 5: init_database.py exists
    if os.path.exists("init_database.py"):
        print("✓ init_database.py exists (auto-init on startup)")
        checks.append(True)
    else:
        print("✗ init_database.py missing")
        checks.append(False)
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ Ready for cloud deployment!")
        print("\nStreamlit Cloud Secrets to add:")
        print("-" * 60)
        print('USE_MONGODB = "true"')
        print('MONGODB_URI = "mongodb+srv://user:pass@cluster.mongodb.net/..."')
        print('MONGODB_DB_NAME = "macd_reversal"')
        print('SHOW_MODULE_WARNINGS = "false"')
        print("-" * 60)
    else:
        print("⚠️ Some checks failed - review above")
    print("=" * 60)

if __name__ == "__main__":
    # Run all tests
    mongodb_ok = test_mongodb_connection()
    
    if mongodb_ok:
        adapter_ok = test_db_adapter()
    
    cloud_deployment_check()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"MongoDB Connection: {'✅ PASS' if mongodb_ok else '❌ FAIL'}")
    print("\nNext steps:")
    if mongodb_ok:
        print("1. ✅ MongoDB works locally")
        print("2. Run: git add . && git commit -m 'Final' && git push")
        print("3. Deploy to Streamlit Cloud with MongoDB secrets")
    else:
        print("1. Fix MongoDB connection issues above")
        print("2. Or deploy with SQLite (USE_MONGODB=false)")
    print("=" * 60)
