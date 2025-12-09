import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# MongoDB connection setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://longhalucky2111_db_user:abc123abc@cluster0.g4ndhy9.mongodb.net/?appName=Cluster0")
DB_NAME = os.getenv("MONGODB_DB_NAME", "macd_reversal")

try:
    # Connect to MongoDB
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    client.admin.command('ping')  # Test connection
    print("✅ MongoDB connection successful!")

    # Access the database and collection
    db = client[DB_NAME]
    price_data = db.price_data

    # Use updateMany with an aggregation pipeline to convert all 'date' fields to ISODate format
    result = price_data.update_many(
        {"date": {"$type": "string"}},  # Match documents where 'date' is a string
        [
            {
                "$set": {
                    "date": {
                        "$dateFromString": {
                            "dateString": "$date",  # Convert the 'date' string field
                            "format": "%Y-%m-%d"   # Adjust format if necessary (e.g., "2023-01-01")
                        }
                    }
                }
            }
        ]
    )

    # Output the result of the update
    print(f"✅ Updated {result.modified_count} documents with ISODate format for 'date' field.")

except Exception as e:
    print(f"❌ MongoDB connection or update failed: {e}")
finally:
    client.close()
