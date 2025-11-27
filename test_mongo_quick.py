"""Quick MongoDB connection test"""
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Your connection string
uri = "mongodb+srv://longhalucky2111_db_user:abc123abc@cluster0.g4ndhy9.mongodb.net/?appName=Cluster0"

print("Testing MongoDB connection...")
try:
    client = MongoClient(uri, server_api=ServerApi('1'))
    client.admin.command('ping')
    print("✅ SUCCESS! Pinged your deployment. You successfully connected to MongoDB!")
    
    # List databases
    print("\nDatabases:", client.list_database_names())
    
    client.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
