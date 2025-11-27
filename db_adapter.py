"""
Database adapter supporting both SQLite (local) and MongoDB (cloud deployment).
Automatically switches based on environment variables.
"""
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import sqlite3

# Try to import MongoDB
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
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
            if os.getenv("USE_MONGODB", "false").lower() == "true":
                db_type = "mongodb"
            else:
                db_type = "sqlite"
        
        self.db_type = db_type
        
        if db_type == "mongodb":
            if not HAS_MONGO:
                raise ImportError("pymongo not installed. Run: pip install pymongo")
            
            # MongoDB setup
            mongo_uri = connection_string or os.getenv(
                "MONGODB_URI",
                "mongodb://localhost:27017/"
            )
            self.client = MongoClient(mongo_uri)
            db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
            self.db = self.client[db_name]
            
            # Collections
            self.price_data = self.db.price_data
            self.market_data = self.db.market_data
            self.tcbs_scaling = self.db.tcbs_scaling
            
            # Create indexes
            self._create_mongo_indexes()
            
        else:
            # SQLite setup
            self.db_path = connection_string or os.getenv(
                "PRICE_DB_PATH",
                "price_data.db"
            )
            self.conn = None
            
            # Auto-initialize database if it doesn't exist
            self._ensure_sqlite_initialized()
    
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
                        PRIMARY KEY (ticker, date, source)
                    )
                """)
                conn.commit()
                conn.close()
    
    def _create_mongo_indexes(self):
        """Create MongoDB indexes for performance."""
        # Price data indexes
        self.price_data.create_index([("ticker", ASCENDING), ("date", DESCENDING)])
        self.price_data.create_index([("ticker", ASCENDING), ("date", DESCENDING), ("source", ASCENDING)], unique=True)
        
        # Market data index
        self.market_data.create_index([("date", DESCENDING)], unique=True)
        
        # TCBS scaling index
        self.tcbs_scaling.create_index([("ticker", ASCENDING)], unique=True)
    
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
        """Load price data for a ticker within date range."""
        if self.db_type == "mongodb":
            try:
                query = {
                    "ticker": ticker,
                    "date": {"$gte": start_date, "$lte": end_date}
                }
                cursor = self.price_data.find(query).sort("date", ASCENDING)
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
                    ORDER BY date
                """
                df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                return df
            except Exception as e:
                print(f"SQLite load error: {e}")
                return pd.DataFrame()
    
    def insert_price_data(self, ticker: str, date: str, ohlcv: Dict, source: str = 'manual') -> bool:
        """Insert or update price data."""
        if self.db_type == "mongodb":
            try:
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
                    {"ticker": ticker, "date": date, "source": source},
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
                cur.execute("""
                    INSERT OR REPLACE INTO price_data 
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

# Global instance (lazy-loaded)
_db_adapter = None

def get_db_adapter() -> DatabaseAdapter:
    """Get global database adapter instance."""
    global _db_adapter
    if _db_adapter is None:
        _db_adapter = DatabaseAdapter()
    return _db_adapter
