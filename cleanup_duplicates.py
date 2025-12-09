import os
import sqlite3
import pandas as pd

def cleanup_sqlite_duplicates(db_path, table="price_data"):
    """Find and optionally delete duplicate (ticker, date) rows in SQLite price_data table."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Find all (ticker, date) pairs with duplicates
    cur.execute(f"""
        SELECT ticker, date, COUNT(*) as cnt
        FROM {table}
        GROUP BY ticker, date
        HAVING cnt > 1
    """)
    dup_rows = cur.fetchall()
    print(f"Found {len(dup_rows)} duplicate (ticker, date) pairs in {table} (SQLite).")
    # Show actual duplicate rows for consistency with test_latest_bars.py
    if dup_rows:
        print("Duplicate rows (first 10):")
        for ticker, date, _ in dup_rows[:10]:
            cur.execute(f"SELECT * FROM {table} WHERE ticker = ? AND date = ?", (ticker, date))
            rows = cur.fetchall()
            for r in rows:
                print(r)
        confirm = input("Delete all but one for each duplicate pair? (y/N): ").strip().lower()
        if confirm == "y":
            total_deleted = 0
            for ticker, date, _ in dup_rows:
                cur.execute(f"""
                    DELETE FROM {table}
                    WHERE ticker = ? AND date = ? AND rowid NOT IN (
                        SELECT MIN(rowid) FROM {table} WHERE ticker = ? AND date = ?
                    )
                """, (ticker, date, ticker, date))
                total_deleted += cur.rowcount
            conn.commit()
            cur.execute("VACUUM")
            print(f"✓ Removed {total_deleted} duplicate rows (kept one per ticker/date) in {table} (SQLite)")
        else:
            print("No rows deleted.")
    else:
        print("No duplicates found.")
    conn.close()

def cleanup_mongodb_duplicates(mongo_uri, db_name="macd_reversal", collection="price_data"):
    """
    Find and optionally delete duplicate (ticker, date) docs in MongoDB price_data collection.
    Uses bulk operations for efficiency.
    Handles both string and date types for 'date' field.
    """
    from pymongo import MongoClient
    from pymongo import DeleteOne
    client = MongoClient(mongo_uri)
    db = client[db_name]
    coll = db[collection]
    # Try to group by date string, fallback to raw value if error
    try:
        pipeline = [
            {
                '$group': {
                    '_id': {
                        'ticker': '$ticker',
                        'date_str': { '$dateToString': { 'format': '%Y-%m-%d', 'date': '$date' } }
                    },
                    'count': {'$sum': 1},
                    'ids': {'$push': '$_id'}
                }
            },
            {
                '$match': {
                    'count': {'$gt': 1}
                }
            }
        ]
        duplicates = list(coll.aggregate(pipeline, allowDiskUse=True))
    except Exception as e:
        print(f"Aggregation with $dateToString failed: {e}")
        print("Retrying with raw date value (no conversion)...")
        # Debug: check if there are any docs with non-date values in 'date'
        sample_docs = list(coll.find({}, {"date": 1, "ticker": 1}).limit(20))
        print("Sample 'date' field values from first 20 docs:")
        for doc in sample_docs:
            print(doc)
        pipeline = [
            {
                '$group': {
                    '_id': {
                        'ticker': '$ticker',
                        'date': '$date'
                    },
                    'count': {'$sum': 1},
                    'ids': {'$push': '$_id'}
                }
            },
            {
                '$match': {
                    'count': {'$gt': 1}
                }
            }
        ]
        duplicates = list(coll.aggregate(pipeline, allowDiskUse=True))
    print(f"Found {len(duplicates)} duplicate (ticker, date) groups in {collection} (MongoDB).")
    # Show actual duplicate docs for debug
    if duplicates:
        print("Duplicate docs (first 10 groups):")
        for group in duplicates[:10]:
            ticker = group['_id'].get('ticker')
            date_val = group['_id'].get('date_str', group['_id'].get('date'))
            ids = group['ids']
            docs = list(coll.find({'_id': {'$in': ids}}))
            for doc in docs:
                # Print all columns for debug
                print({k: doc.get(k) for k in doc.keys()})
        confirm = input("Delete all but one for each duplicate date group for each ticker? (y/N): ").strip().lower()
        if confirm == "y":
            bulk_operations = []
            count_deleted = 0
            for document in duplicates:
                duplicate_ids = document['ids']
                ids_to_remove = duplicate_ids[1:]
                for doc_id in ids_to_remove:
                    bulk_operations.append(DeleteOne({'_id': doc_id}))
                    count_deleted += 1
            if bulk_operations:
                print(f"Preparing to delete {count_deleted} duplicate documents...")
                result = coll.bulk_write(bulk_operations)
                print(f"Successfully deleted {result.deleted_count} documents.")
            else:
                print("No duplicates found to delete.")
        else:
            print("No documents deleted.")
    else:
        print("No duplicates found.")
    client.close()

def cleanup_sqlite_remove_nan_id_updated(db_path, table="price_data"):
    """Remove all rows in SQLite price_data table where id IS NULL and updated_at IS NULL."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {table} WHERE id IS NULL AND updated_at IS NULL")
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    print(f"✓ Removed {deleted} rows where id IS NULL and updated_at IS NULL in {table} (SQLite)")

def cleanup_mongodb_remove_nan_id_updated(mongo_uri, db_name="macd_reversal", collection="price_data"):
    """Remove all docs in MongoDB price_data collection where 'id' and 'updated_at' are missing or null."""
    from pymongo import MongoClient
    client = MongoClient(mongo_uri)
    db = client[db_name]
    coll = db[collection]
    # Remove docs where 'id' does not exist or is null AND 'updated_at' does not exist or is null
    result = coll.delete_many({
        "$and": [
            { "$or": [ { "id": { "$exists": False } }, { "id": None } ] },
            { "$or": [ { "updated_at": { "$exists": False } }, { "updated_at": None } ] }
        ]
    })
    print(f"✓ Removed {result.deleted_count} docs where id and updated_at are missing or null in {collection} (MongoDB)")
    client.close()

if __name__ == "__main__":
    # Example usage:
    # For SQLite
    db_path = os.getenv("PRICE_DB_PATH", "price_data.db")
    if os.path.exists(db_path):
        cleanup_sqlite_duplicates(db_path)
    else:
        print(f"SQLite DB not found at {db_path}")

    # For MongoDB
    mongo_uri = "mongodb+srv://longhalucky2111_db_user:abc123abc@cluster0.g4ndhy9.mongodb.net/?appName=Cluster0"

    if mongo_uri:
        cleanup_mongodb_duplicates(mongo_uri)
    else:
        print("MongoDB URI not set in environment. Skipping MongoDB cleanup.")
    db_path = os.getenv("PRICE_DB_PATH", "price_data.db")
    if os.path.exists(db_path):
        cleanup_sqlite_remove_nan_id_updated(db_path)
    else:
        print(f"SQLite DB not found at {db_path}")

    mongo_uri = "mongodb+srv://longhalucky2111_db_user:abc123abc@cluster0.g4ndhy9.mongodb.net/?appName=Cluster0"
    if mongo_uri:
        cleanup_mongodb_remove_nan_id_updated(mongo_uri)
    else:
        print("MongoDB URI not set in environment. Skipping MongoDB cleanup.")
