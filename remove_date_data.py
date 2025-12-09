import os
import sqlite3
from pymongo import MongoClient

def remove_sqlite_date_to_latest(db_path, date_str, table="price_data"):
    """Remove all rows from a specific date to the latest in SQLite price_data table."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {table} WHERE date >= ?", (date_str,))
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    print(f"✓ Removed {deleted} rows from date {date_str} to latest in {table} (SQLite)")

def remove_mongodb_date_to_latest(mongo_uri, date_str, db_name="macd_reversal", collection="price_data"):
    """Remove all docs from a specific date to the latest in MongoDB price_data collection."""
    client = MongoClient(mongo_uri)
    db = client[db_name]
    coll = db[collection]
    result = coll.delete_many({"date": {"$gte": date_str}})
    print(f"✓ Removed {result.deleted_count} docs from date {date_str} to latest in {collection} (MongoDB)")
    client.close()

if __name__ == "__main__":
    # Example usage:
    # Set the date to remove
    date_to_remove = input("Enter start date to remove (YYYY-MM-DD): ").strip()

   

    # MongoDB
    mongo_uri = "mongodb+srv://longhalucky2111_db_user:abc123abc@cluster0.g4ndhy9.mongodb.net/?appName=Cluster0"
    if mongo_uri:
        remove_mongodb_date_to_latest(mongo_uri, date_to_remove)
    else:
        print("MongoDB URI not set in environment. Skipping MongoDB removal.")
