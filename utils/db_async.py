"""
Async MongoDB operations for market breadth calculations.
Uses motor for async database access.
"""
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# motor not supported on Streamlit Cloud - use SyncDatabaseAdapter instead
HAS_MOTOR = False

# Try to import pymongo for synchronous fallback
try:
    from pymongo import MongoClient
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False

from dotenv import load_dotenv

# Load environment
load_dotenv()


class AsyncDatabaseAdapter:
    """Async MongoDB adapter for market breadth calculations."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize async database adapter.
        
        Args:
            connection_string: MongoDB URI (defaults to env variable)
        """
        raise ImportError(
            "AsyncDatabaseAdapter requires motor which is not supported on Streamlit Cloud. " +
            "Use SyncDatabaseAdapter instead: from utils.db_async import get_sync_db_adapter"
        )
        db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
        self.db = self.client[db_name]
        
        # Collections
        self.price_data = self.db.price_data
        self.indicators = self.db.indicators
        self.market_breadth = self.db.market_breadth
    
    async def get_all_tickers(self) -> List[str]:
        """Get list of all unique tickers."""
        try:
            tickers = await self.price_data.distinct("ticker")
            return sorted([t for t in tickers if t])
        except Exception as e:
            print(f"Error fetching tickers: {e}")
            return []
    
    async def get_price_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get price data for a ticker within date range.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            query = {
                "ticker": ticker.upper(),
                "date": {"$gte": start_date, "$lte": end_date}
            }
            cursor = self.price_data.find(query).sort("date", 1)
            docs = await cursor.to_list(length=None)
            
            if not docs:
                return pd.DataFrame()
            
            df = pd.DataFrame(docs)
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Ensure date is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
        except Exception as e:
            print(f"Error loading price data for {ticker}: {e}")
            return pd.DataFrame()
    
    async def save_indicators(self, ticker: str, date: datetime, indicators: Dict) -> bool:
        """
        Save calculated indicators to database.
        
        Args:
            ticker: Ticker symbol
            date: Date of the indicators
            indicators: Dictionary of indicator values
        
        Returns:
            True if successful
        """
        try:
            doc = {
                "ticker": ticker.upper(),
                "date": date,
                **indicators,
                "updated_at": datetime.now()
            }
            
            await self.indicators.update_one(
                {"ticker": ticker.upper(), "date": date},
                {"$set": doc},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving indicators for {ticker} on {date}: {e}")
            return False
    
    async def get_indicators(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get calculated indicators for a ticker within date range.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with indicator data
        """
        try:
            query = {
                "ticker": ticker.upper(),
                "date": {"$gte": start_date, "$lte": end_date}
            }
            cursor = self.indicators.find(query).sort("date", 1)
            docs = await cursor.to_list(length=None)
            
            if not docs:
                return pd.DataFrame()
            
            df = pd.DataFrame(docs)
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Ensure date is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
        except Exception as e:
            print(f"Error loading indicators for {ticker}: {e}")
            return pd.DataFrame()
    
    async def get_latest_date(self) -> Optional[datetime]:
        """Get the latest date available in price_data collection."""
        try:
            doc = await self.price_data.find_one(
                {},
                sort=[("date", -1)]
            )
            if doc and 'date' in doc:
                return pd.to_datetime(doc['date'])
            return None
        except Exception as e:
            print(f"Error getting latest date: {e}")
            return None
    
    async def get_indicators_for_date(self, date: datetime) -> pd.DataFrame:
        """
        Get all indicators for all tickers on a specific date.
        
        Args:
            date: Target date
        
        Returns:
            DataFrame with all tickers' indicators for that date
        """
        try:
            query = {"date": date}
            cursor = self.indicators.find(query)
            docs = await cursor.to_list(length=None)
            
            if not docs:
                return pd.DataFrame()
            
            df = pd.DataFrame(docs)
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            return df
        except Exception as e:
            print(f"Error loading indicators for date {date}: {e}")
            return pd.DataFrame()
    
    async def save_market_breadth(self, date: datetime, breadth_data: Dict) -> bool:
        """
        Save market breadth calculations to database.
        
        Args:
            date: Date of the breadth calculation
            breadth_data: Dictionary with breadth metrics
        
        Returns:
            True if successful
        """
        try:
            # Convert numpy types to Python native types
            clean_breadth_data = self._convert_numpy_types(breadth_data)
            
            doc = {
                "date": date,
                **clean_breadth_data,
                "updated_at": datetime.now()
            }
            
            await self.market_breadth.update_one(
                {"date": date},
                {"$set": doc},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving market breadth for {date}: {e}")
            return False
    
    def _convert_numpy_types(self, obj):
        """
        Convert numpy types to Python native types for MongoDB compatibility.
        
        Args:
            obj: Object that may contain numpy types
        
        Returns:
            Object with numpy types converted to Python native types
        """
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.to_pydatetime()
        else:
            return obj
    
    async def get_market_breadth(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get market breadth data within date range.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with breadth data
        """
        try:
            query = {
                "date": {"$gte": start_date, "$lte": end_date}
            }
            cursor = self.market_breadth.find(query).sort("date", 1)
            docs = await cursor.to_list(length=None)
            
            if not docs:
                return pd.DataFrame()
            
            df = pd.DataFrame(docs)
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Ensure date is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
        except Exception as e:
            print(f"Error loading market breadth: {e}")
            return pd.DataFrame()
    
    async def check_indicators_exist(self, date: datetime) -> bool:
        """
        Check if indicators exist for a specific date.
        
        Args:
            date: Date to check
        
        Returns:
            True if indicators exist
        """
        try:
            count = await self.indicators.count_documents({"date": date})
            return count > 0
        except Exception:
            return False
    
    async def get_trading_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Get all trading dates (dates with price data) within range.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            List of trading dates
        """
        try:
            dates = await self.price_data.distinct(
                "date",
                {"date": {"$gte": start_date, "$lte": end_date}}
            )
            return sorted([pd.to_datetime(d) for d in dates])
        except Exception as e:
            print(f"Error getting trading dates: {e}")
            return []
    
    def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()


# Synchronous fallback adapter (uses pymongo instead of motor)
class SyncDatabaseAdapter:
    """Synchronous MongoDB adapter for market breadth calculations."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize sync database adapter."""
        if not HAS_PYMONGO:
            raise ImportError("pymongo not installed. Run: pip install pymongo")
        
        mongo_uri = connection_string or os.getenv("MONGODB_URI")
        if not mongo_uri:
            raise ValueError("MongoDB URI not found in environment")
        
        # Add timeout options
        if "?" in mongo_uri:
            mongo_uri += "&"
        else:
            mongo_uri += "?"
        mongo_uri += "socketTimeoutMS=60000&connectTimeoutMS=60000"
        
        self.client = MongoClient(mongo_uri)
        db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
        self.db = self.client[db_name]
        
        # Collections
        self.price_data = self.db.price_data
        self.indicators = self.db.indicators
        self.market_breadth = self.db.market_breadth
    
    def get_all_tickers(self) -> List[str]:
        """Get list of all unique tickers."""
        try:
            tickers = self.price_data.distinct("ticker")
            return sorted([t for t in tickers if t])
        except Exception as e:
            print(f"Error fetching tickers: {e}")
            return []
    
    def get_price_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get price data for a ticker within date range."""
        try:
            query = {
                "ticker": ticker.upper(),
                "date": {"$gte": start_date, "$lte": end_date}
            }
            cursor = self.price_data.find(query).sort("date", 1)
            docs = list(cursor)
            
            if not docs:
                return pd.DataFrame()
            
            df = pd.DataFrame(docs)
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
        except Exception as e:
            print(f"Error loading price data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_latest_date(self) -> Optional[datetime]:
        """Get the latest date available in price_data collection."""
        try:
            doc = self.price_data.find_one({}, sort=[("date", -1)])
            if doc and 'date' in doc:
                return pd.to_datetime(doc['date'])
            return None
        except Exception as e:
            print(f"Error getting latest date: {e}")
            return None
    
    def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()


def get_async_db_adapter(connection_string: Optional[str] = None) -> AsyncDatabaseAdapter:
    """
    Get async database adapter instance.
    
    Args:
        connection_string: MongoDB URI (optional)
    
    Returns:
        AsyncDatabaseAdapter instance
    """
    return AsyncDatabaseAdapter(connection_string)


def get_sync_db_adapter(connection_string: Optional[str] = None) -> SyncDatabaseAdapter:
    """
    Get synchronous database adapter instance.
    
    Args:
        connection_string: MongoDB URI (optional)
    
    Returns:
        SyncDatabaseAdapter instance
    """
    return SyncDatabaseAdapter(connection_string)
