import sys
from pymongo import MongoClient, DeleteOne
import os

MONGODB_URI = "mongodb+srv://longhalucky2111_db_user:abc123abc@cluster0.g4ndhy9.mongodb.net/?appName=Cluster0"
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "macd_reversal")
COLLECTION_NAME = "price_data"

def remove_duplicates_by_ticker(ticker):
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB_NAME]
    coll = db[COLLECTION_NAME]
    # Group by ticker/date/source, find duplicates
    pipeline = [
        {'$match': {'ticker': ticker.upper()}},
        {
            '$group': {
                '_id': {
                    'date': '$date',
                    'source': '$source'
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
    print(f"Found {len(duplicates)} duplicate (date, source) groups for {ticker} in {COLLECTION_NAME}.")
    if duplicates:
        print("Duplicate docs (first 10 groups):")
        for group in duplicates[:10]:
            date_val = group['_id']['date']
            source_val = group['_id']['source']
            ids = group['ids']
            docs = list(coll.find({'_id': {'$in': ids}}))
            for doc in docs:
                print({k: doc.get(k) for k in doc.keys()})
        confirm = input("Delete all but one for each duplicate date/source group for this ticker? (y/N): ").strip().lower()
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_duplicates_by_ticker.py TICKER")
        sys.exit(1)
    ticker = sys.argv[1].upper()
    remove_duplicates_by_ticker(ticker)
