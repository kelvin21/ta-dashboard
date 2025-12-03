"""
Database adapter supporting both SQLite (local) and MongoDB (cloud deployment).
Automatically switches based on environment variables.
"""
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import sqlite3
from pathlib import Path

# Load .env file FIRST before checking environment variables
# Explicitly specify the .env file path
try:
    from dotenv import load_dotenv
    
    # Get the directory where this script is located
    SCRIPT_DIR = Path(__file__).parent
    ENV_PATH = SCRIPT_DIR / '.env'
    
    print(f"🔍 Looking for .env at: {ENV_PATH}")
    print(f"🔍 .env exists: {ENV_PATH.exists()}")
    
    # Load .env from the script directory
    loaded = load_dotenv(dotenv_path=ENV_PATH, verbose=True)
    print(f"✓ load_dotenv() result: {loaded}")
    
    # Verify what was loaded
    print(f"✓ After load_dotenv():")
    print(f"  USE_MONGODB: {os.getenv('USE_MONGODB', 'true')}")
    mongodb_uri = os.getenv('MONGODB_URI', '')
    if mongodb_uri:
        print(f"  MONGODB_URI: SET (length: {len(mongodb_uri)})")
    else:
        print(f"  MONGODB_URI: NOT SET")
    print(f"  MONGODB_DB_NAME: {os.getenv('MONGODB_DB_NAME', 'NOT SET')}")
    
except ImportError:
    print("⚠️ python-dotenv not installed - using system environment only")

# HARDCODED CONNECTION STRING (Optional - only if .env doesn't work)
# Uncomment and set your MongoDB URI here if environment variables aren't loading
HARDCODED_MONGODB_URI = "mongodb+srv://longhalucky2111_db_user:abc123abc@cluster0.g4ndhy9.mongodb.net/?appName=Cluster0"

# Try to import MongoDB
try:
    from pymongo.mongo_client import MongoClient
    from pymongo.server_api import ServerApi
    from pymongo import ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, DuplicateKeyError
    HAS_MONGO = True
except ImportError:
    HAS_MONGO = False

class DatabaseAdapter:
    """Universal database adapter for SQLite or MongoDB."""
    
    def __init__(self, db_type=None, connection_string=None):
        """
        Initialize database adapter.
        
        Args:
            db_type: 'sqlite' or 'mongodb'. Auto-detected if None.
            connection_string: MongoDB URI or SQLite path
        """
        # Auto-detect database type from environment
        if db_type is None:
            use_mongo_env = os.getenv("USE_MONGODB", "true").lower()
            use_mongo_env = "true"
            print(f"🔍 DatabaseAdapter init:")
            print(f"  USE_MONGODB from env: '{use_mongo_env}'")
            
            if use_mongo_env == "true":
                db_type = "mongodb"
            else:
                db_type = "sqlite"
            
            print(f"  Selected db_type: {db_type}")
        
        self.db_type = db_type
        
        if db_type == "mongodb":
            if not HAS_MONGO:
                raise ImportError("pymongo not installed. Run: pip install pymongo dnspython")
            
            # MongoDB setup with ServerApi
            mongo_uri = connection_string or HARDCODED_MONGODB_URI or os.getenv("MONGODB_URI")
            
            if not mongo_uri:
                raise ValueError(
                    "MongoDB URI not found!\n"
                    "Solutions:\n"
                    "1. Set MONGODB_URI in .env file\n"
                    "2. Set HARDCODED_MONGODB_URI in db_adapter.py\n"
                    "3. Pass connection_string to DatabaseAdapter()"
                )
            
            # Add timeout options to the URI
            if "?" in mongo_uri:
                mongo_uri += "&"
            else:
                mongo_uri += "?"
            mongo_uri += "socketTimeoutMS=60000&connectTimeoutMS=60000"  # Increase timeouts to 60 seconds
            
            try:
                # Retry logic for MongoDB connection
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"🔄 Attempting MongoDB connection (Attempt {attempt + 1}/{max_retries})...")
                        self.client = MongoClient(mongo_uri, server_api=ServerApi('1'))
                        self.client.admin.command('ping')  # Test connection
                        print("✅ MongoDB connection successful!")
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        print(f"⚠️ MongoDB connection failed (Attempt {attempt + 1}/{max_retries}): {e}")
                        print("Retrying...")
                
                db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
                self.db = self.client[db_name]
                
                # Collections
                self.price_data = self.db.price_data
                self.market_data = self.db.market_data
                self.tcbs_scaling = self.db.tcbs_scaling
                self.user_settings = self.db.user_settings
                
                # Create indexes
                self._create_mongo_indexes()
                
            except Exception as e:
                print(f"❌ MongoDB connection failed after {max_retries} attempts: {e}")
                raise ConnectionError(
                    f"MongoDB connection failed: {e}\n"
                    "Please check your MongoDB URI, network connectivity, and database permissions."
                )
        else:
            print(f"✓ Using SQLite mode")
            # SQLite setup
            self.db_path = connection_string or os.getenv(
                "PRICE_DB_PATH",
                "price_data.db"
            )
            self.conn = None
            
            # Auto-initialize database if it doesn't exist
            self._ensure_sqlite_initialized()
    
    def _create_mongo_indexes(self):
        """Create MongoDB indexes for performance and uniqueness."""
        try:
            # Check existing indexes
            existing_indexes = self.price_data.index_information()
            existing_index_names = set(existing_indexes.keys())

            # Define indexes to create
            indexes_to_create = [
                {
                    "keys": [("ticker", ASCENDING), ("date", DESCENDING)],
                    "unique": True,
                    "name": "unique_ticker_date"
                },
                {
                    "keys": [("ticker", ASCENDING), ("date", DESCENDING), ("source", ASCENDING)],
                    "unique": False,
                    "name": "ticker_date_source"
                }
            ]

            # Create indexes if they don't already exist
            for index in indexes_to_create:
                if index["name"] not in existing_index_names:
                    self.price_data.create_index(index["keys"], unique=index["unique"], name=index["name"])
                    print(f"✅ Created index: {index['name']}")
                else:
                    print(f"⚠️ Index already exists: {index['name']}")

        except Exception as e:
            print(f"❌ Failed to create indexes: {e}")
            raise
    
    def _ensure_sqlite_initialized(self):
        """Ensure SQLite database exists and has required tables."""
        try:
            from init_database import create_empty_database
            create_empty_database(self.db_path)
        except Exception as e:
            # If initialization fails, create minimal schema
            if not os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS price_data (
                        ticker TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open REAL, high REAL, low REAL, close REAL,
                        volume INTEGER,
                        source TEXT DEFAULT 'manual',
                        created_at TEXT,
                        PRIMARY KEY (ticker, date)
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS overview_cache (
                        cache_key TEXT PRIMARY KEY,
                        data TEXT,
                        created_at TEXT
                    )
                """)
                conn.commit()
                conn.close()
    
    def _get_sqlite_conn(self):
        """Get SQLite connection (lazy loading)."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def get_all_tickers(self, debug=False) -> List[str]:
        """Get list of all unique tickers."""
        if self.db_type == "mongodb":
            try:
                tickers = self.price_data.distinct("ticker")
                return sorted([t for t in tickers if t])
            except Exception as e:
                if debug:
                    print(f"MongoDB error: {e}")
                return []
        else:
            try:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT ticker FROM price_data WHERE ticker IS NOT NULL ORDER BY ticker")
                return [r[0] for r in cur.fetchall()]
            except Exception as e:
                if debug:
                    print(f"SQLite error: {e}")
                return []
    
    def load_price_range(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load price data for a ticker within a date ranload_price_rangege."""
        if self.db_type == "mongodb":
            try:
                query = {
                    "ticker": ticker.upper(),
                    "date": {"$gte": datetime.strptime(start_date, "%Y-%m-%d"), "$lte": datetime.strptime(end_date, "%Y-%m-%d")}
                }
                cursor = self.price_data.find(query).sort([("date", ASCENDING)])  # Fixed sorting syntax
                df = pd.DataFrame(list(cursor))
                
                if not df.empty:
                    # Remove MongoDB _id field
                    if '_id' in df.columns:
                        df = df.drop('_id', axis=1)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Ensure required columns exist
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col not in df.columns:
                            df[col] = 0
                
                return df
            except Exception as e:
                print(f"MongoDB load error: {e}")
                return pd.DataFrame()
        else:
            try:
                conn = self._get_sqlite_conn()
                query = """
                    SELECT date, open, high, low, close, volume
                    FROM price_data
                    WHERE ticker = ? AND date >= ? AND date <= ?
                    ORDER BY date ASC
                """
                df = pd.read_sql_query(query, conn, params=(ticker.upper(), start_date, end_date))
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                return df
            except Exception as e:
                print(f"SQLite load error: {e}")
                return pd.DataFrame()
    
    def insert_price_data(self, ticker: str, date: str, ohlcv: Dict, source: str = 'manual') -> bool:
        """Insert or update price data. Ensures no duplicate rows for ticker/date. Cleans up duplicates if found."""
        if self.db_type == "mongodb":
            try:
                # Ensure the 'date' field is a datetime object
                if isinstance(date, str):
                    date = datetime.strptime(date, "%Y-%m-%d")

                # Clean up any duplicate docs for ticker/date/source before upsert
                dup_query = {"ticker": ticker, "date": date}
                docs = list(self.price_data.find(dup_query))
                if len(docs) > 1:
                    # Keep the latest (by _id), remove others
                    ids_to_remove = [doc['_id'] for doc in sorted(docs, key=lambda d: d.get('created_at', datetime.min))[:-1]]
                    if ids_to_remove:
                        self.price_data.delete_many({"_id": {"$in": ids_to_remove}})
                doc = {
                    "ticker": ticker,
                    "date": date,
                    "open": float(ohlcv.get('open', 0)),
                    "high": float(ohlcv.get('high', 0)),
                    "low": float(ohlcv.get('low', 0)),
                    "close": float(ohlcv.get('close', 0)),
                    "volume": int(ohlcv.get('volume', 0)),
                    "source": source,
                    "created_at": datetime.now()
                }
                # Upsert (update if exists, insert if not)
                self.price_data.update_one(
                    {"ticker": ticker, "date": date},
                    {"$set": doc},
                    upsert=True
                )
                return True
            except Exception as e:
                print(f"MongoDB insert error: {e}")
                return False
        else:
            try:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                # Clean up any duplicate rows for ticker/date/source before insert
                cur.execute("""
                    DELETE FROM price_data
                    WHERE ticker = ? AND date = ? 
                """, (ticker, date))
                cur.execute("""
                    INSERT INTO price_data 
                    (ticker, date, open, high, low, close, volume, source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker, date,
                    ohlcv.get('open', 0),
                    ohlcv.get('high', 0),
                    ohlcv.get('low', 0),
                    ohlcv.get('close', 0),
                    ohlcv.get('volume', 0),
                    source,
                    datetime.now().isoformat()
                ))
                conn.commit()
                # Extra cleanup: remove any further duplicates (shouldn't happen, but for safety)
                cur.execute("""
                    DELETE FROM price_data
                    WHERE rowid NOT IN (
                        SELECT MIN(rowid) FROM price_data
                        GROUP BY ticker, date
                    )
                """)
                conn.commit()
                return True
            except Exception as e:
                print(f"SQLite insert error: {e}")
                return False
    
    def check_today_data_exists(self, today_str: str) -> Tuple[bool, int]:
        """Check if today's data exists. Returns (exists, count)."""
        if self.db_type == "mongodb":
            try:
                count = self.price_data.count_documents({"date": today_str})
                return count > 0, count
            except Exception as e:
                print(f"MongoDB check error: {e}")
                return False, 0
        else:
            try:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM price_data WHERE date = ?", (today_str,))
                count = cur.fetchone()[0]
                return count > 0, count
            except Exception as e:
                print(f"SQLite check error: {e}")
                return False, 0
    
    def get_tcbs_scale_factor(self, ticker: str) -> float:
        """Get scale factor for a ticker."""
        if self.db_type == "mongodb":
            try:
                doc = self.tcbs_scaling.find_one({"ticker": ticker})
                return doc.get('scale_factor', 1000.0) if doc else 1000.0
            except Exception:
                return 1000.0
        else:
            try:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("SELECT scale_factor FROM tcbs_scaling WHERE ticker = ?", (ticker,))
                row = cur.fetchone()
                return row[0] if row else 1000.0
            except Exception:
                return 1000.0
    
    def close(self):
        """Close database connection."""
        if self.db_type == "mongodb":
            self.client.close()
        else:
            if self.conn:
                self.conn.close()
                self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def delete_price_range(self, ticker: str, start_date: str, end_date: str) -> int:
        """
        Delete price data for a ticker within a date range.
        Returns number of deleted rows/documents.
        """
        if self.db_type == "mongodb":
            try:
                result = self.price_data.delete_many({
                    "ticker": ticker,
                    "date": {"$gte": start_date, "$lte": end_date}
                })
                return result.deleted_count
            except Exception as e:
                print(f"MongoDB delete error: {e}")
                return 0
        else:
            try:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("""
                    DELETE FROM price_data
                    WHERE ticker = ? AND date >= ? AND date <= ?
                """, (ticker, start_date, end_date))
                deleted_rows = cur.rowcount
                conn.commit()
                return deleted_rows
            except Exception as e:
                print(f"SQLite delete error: {e}")
                return 0
    
    def load_price_range_multi(self, tickers, start_date, end_date):
        """
        Load price data for multiple tickers at once.
        Returns a dict: {ticker: DataFrame}
        """
        result = {}
        if self.db_type == "mongodb":
            try:
                query = {
                    "ticker": {"$in": [t.upper() for t in tickers]},
                    "date": {"$gte": datetime.strptime(start_date, "%Y-%m-%d"), "$lte": datetime.strptime(end_date, "%Y-%m-%d")}
                }
                cursor = self.price_data.find(query).sort([("date", ASCENDING)]).max_time_ms(180000)  # Added maxTimeMS for 60 seconds timeout
                print(f"MongoDB multi-load query: {query}")
                df = pd.DataFrame(list(cursor))
                if not df.empty:
                    if '_id' in df.columns:
                        df = df.drop('_id', axis=1)
                    df['date'] = pd.to_datetime(df['date'])
                for t in tickers:
                    result[t] = df[df['ticker'] == t.upper()].copy() if not df.empty else pd.DataFrame()
                return result
            except Exception as e:
                print(f"MongoDB multi-load error: {e}")
                return {t: pd.DataFrame() for t in tickers}
        else:
            try:
                conn = self._get_sqlite_conn()
                q = f"""
                    SELECT ticker, date, open, high, low, close, volume
                    FROM price_data
                    WHERE ticker IN ({','.join(['?']*len(tickers))}) AND date >= ? AND date <= ?
                    ORDER BY ticker, date
                """
                params = [t for t in tickers] + [start_date, end_date]
                df = pd.read_sql_query(q, conn, params=params)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                for t in tickers:
                    result[t] = df[df['ticker'] == t].copy() if not df.empty else pd.DataFrame()
                return result
            except Exception as e:
                print(f"SQLite multi-load error: {e}")
                return {t: pd.DataFrame() for t in tickers}
    
    def get_setting(self, key: str) -> str:
        """Get a user setting value."""
        if self.db_type == "mongodb":
            try:
                doc = self.user_settings.find_one({"key": key})
                return doc.get("value", "") if doc else ""
            except Exception as e:
                print(f"MongoDB get_setting error: {e}")
                return ""
        else:
            try:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("SELECT value FROM user_settings WHERE key = ?", (key,))
                row = cur.fetchone()
                return row[0] if row else ""
            except Exception as e:
                print(f"SQLite get_setting error: {e}")
                return ""
    
    def set_setting(self, key: str, value: str) -> bool:
        """Set a user setting value."""
        if self.db_type == "mongodb":
            try:
                self.user_settings.update_one(
                    {"key": key},
                    {"$set": {"value": value, "updated_at": datetime.now()}},
                    upsert=True
                )
                return True
            except Exception as e:
                print(f"MongoDB set_setting error: {e}")
                return False
        else:
            try:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO user_settings (key, value)
                    VALUES (?, ?)
                """, (key, value))
                conn.commit()
                return True
            except Exception as e:
                print(f"SQLite set_setting error: {e}")
                return False
    
    def save_overview_cache(self, cache_key: str, df_data: pd.DataFrame) -> bool:
        """Save overview table data to cache."""
        if self.db_type == "mongodb":
            try:
                import json
                json_data = df_data.to_json(orient='records', date_format='iso')
                self.db.overview_cache.update_one(
                    {"cache_key": cache_key},
                    {"$set": {"data": json_data, "created_at": datetime.now()}},
                    upsert=True
                )
                return True
            except Exception as e:
                print(f"MongoDB save overview cache error: {e}")
                return False
        else:
            try:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                import json
                json_data = df_data.to_json(orient='records', date_format='iso')
                cur.execute("""
                    INSERT OR REPLACE INTO overview_cache (cache_key, data, created_at)
                    VALUES (?, ?, ?)
                """, (cache_key, json_data, datetime.now().isoformat()))
                conn.commit()
                return True
            except Exception as e:
                print(f"SQLite save overview cache error: {e}")
                return False
    
    def get_overview_cache(self, cache_key: str) -> pd.DataFrame:
        """Retrieve overview table data from cache."""
        if self.db_type == "mongodb":
            try:
                doc = self.db.overview_cache.find_one({"cache_key": cache_key})
                if doc and "data" in doc:
                    import json
                    data = json.loads(doc["data"])
                    df = pd.DataFrame(data)
                    # Convert date columns back to datetime
                    for col in df.columns:
                        if "date" in col.lower():
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    return df
                return pd.DataFrame()
            except Exception as e:
                print(f"MongoDB get overview cache error: {e}")
                return pd.DataFrame()
        else:
            try:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("SELECT data FROM overview_cache WHERE cache_key = ?", (cache_key,))
                row = cur.fetchone()
                if row:
                    import json
                    data = json.loads(row[0])
                    df = pd.DataFrame(data)
                    # Convert date columns back to datetime
                    for col in df.columns:
                        if "date" in col.lower():
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    return df
                return pd.DataFrame()
            except Exception as e:
                print(f"SQLite get overview cache error: {e}")
                return pd.DataFrame()
    
    def clear_overview_cache(self) -> bool:
        """Clear all overview cache entries."""
        if self.db_type == "mongodb":
            try:
                self.db.overview_cache.delete_many({})
                return True
            except Exception as e:
                print(f"MongoDB clear overview cache error: {e}")
                return False
        else:
            try:
                conn = self._get_sqlite_conn()
                cur = conn.cursor()
                cur.execute("DELETE FROM overview_cache")
                conn.commit()
                return True
            except Exception as e:
                print(f"SQLite clear overview cache error: {e}")
                return False

# Global instance (lazy-loaded)
_db_adapter = None

def get_db_adapter() -> DatabaseAdapter:
    """Get global database adapter instance."""
    global _db_adapter
    if _db_adapter is None:
        _db_adapter = DatabaseAdapter()
        # Verify the adapter has required methods and attributes
        required_methods = ['get_setting', 'set_setting', 'get_all_tickers', 'load_price_range', 'load_price_range_multi', 'save_overview_cache', 'get_overview_cache', 'clear_overview_cache']
        for method in required_methods:
            if not hasattr(_db_adapter, method):
                print(f"⚠️ WARNING: DatabaseAdapter missing method: {method}")
        
        # Verify db_type is set correctly
        print(f"✓ DatabaseAdapter initialized with db_type: {_db_adapter.db_type}")
    return _db_adapter
