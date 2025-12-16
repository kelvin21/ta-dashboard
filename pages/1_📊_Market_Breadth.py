"""
Market Breadth Analysis Page
Comprehensive market breadth indicators for VN Market using MongoDB data.
Based on specification from MARKET_BREADTH_README.md
"""
import os
import sys
from pathlib import Path
import asyncio
from typing import List, Dict, Tuple, Optional
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import utilities
try:
    from utils.indicators import (
        calculate_all_indicators, categorize_rsi, check_price_above_ema,
        calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands,
        categorize_rsi_vectorized, check_price_above_ema_vectorized
    )
    from utils.macd_stage import (
        detect_macd_stage, categorize_macd_stage, get_all_macd_stages,
        get_macd_stage_color, get_macd_stage_display_name,
        categorize_macd_stage_vectorized, detect_macd_stage_vectorized
    )
    from utils.db_async import get_sync_db_adapter
    USE_UTILS = True
except ImportError as e:
    st.error(f"Failed to import utility modules: {e}")
    st.info("Please ensure utils/ directory exists with indicators.py, macd_stage.py, and db_async.py")
    st.stop()

# Use synchronous database adapter (motor not supported on Streamlit Cloud)
HAS_MOTOR = False

# Import from main dashboard (for compatibility)
try:
    from db_adapter import get_db_adapter
    HAS_DB_ADAPTER = True
except ImportError:
    HAS_DB_ADAPTER = False

# Note: Async functions below are disabled since motor is not supported on Streamlit Cloud
# All processing uses synchronous SyncDatabaseAdapter instead

# Page config
st.set_page_config(page_title="Market Breadth Analysis", layout="wide", page_icon="📊")

# Title
st.markdown("# 📊 Market Breadth by Indicators")
st.markdown("Analyze market breadth across multiple technical indicators for the Vietnamese stock market")

# Initialize session state
if 'calculation_status' not in st.session_state:
    st.session_state.calculation_status = None
if 'calculation_progress' not in st.session_state:
    st.session_state.calculation_progress = 0

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

@st.cache_resource(ttl=3600)
def get_db():
    """Get database adapter (cached)."""
    return get_sync_db_adapter()

@st.cache_data(ttl=1800)
def get_latest_date_from_db():
    """Get latest date from database."""
    try:
        db = get_db()
        latest = db.get_latest_date()
        return latest
    except Exception as e:
        st.error(f"Error getting latest date: {e}")
        return None

@st.cache_data(ttl=1800)
def get_all_tickers_cached():
    """Get all tickers from database (cached)."""
    try:
        db = get_db()
        return db.get_all_tickers()
    except Exception as e:
        st.error(f"Error getting tickers: {e}")
        return []

def load_price_data_for_ticker(ticker: str, start_date: datetime, end_date: datetime):
    """Load price data for a single ticker."""
    try:
        db = get_db()
        return db.get_price_data(ticker, start_date, end_date)
    except Exception as e:
        return pd.DataFrame()

def calculate_indicators_for_ticker(ticker: str, start_date: datetime, end_date: datetime, skip_macd_stage: bool = False) -> pd.DataFrame:
    """Calculate all indicators for a ticker.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date
        skip_macd_stage: If True, skip MACD stage calculation for faster processing
    
    Returns:
        DataFrame with indicators
    """
    try:
        # Add warmup period for indicators (200 days = ~290 calendar days)
        warmup_days = 365  # Use 1 year warmup to ensure all indicators have enough data
        warmup_start = start_date - timedelta(days=warmup_days)
        
        # Load data with warmup period
        df = load_price_data_for_ticker(ticker, warmup_start, end_date)
        
        if df.empty or 'close' not in df.columns:
            return pd.DataFrame()
        
        # Calculate indicators on full dataset (including warmup)
        df = calculate_all_indicators(df)
        
        # Calculate MACD stage for each row (vectorized - but still slowest part)
        if not skip_macd_stage and 'macd_hist' in df.columns:
            df['macd_stage'] = detect_macd_stage_vectorized(df['macd_hist'], lookback=20)
        elif skip_macd_stage and 'macd_hist' in df.columns:
            # Set placeholder for lazy loading
            df['macd_stage'] = 'N/A'
        
        # Trim to original date range (remove warmup period)
        df = df[df['date'] >= start_date].copy()
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators for {ticker}: {e}")
        return pd.DataFrame()

async def calculate_indicators_for_ticker_async(
    db,  # AsyncDatabaseAdapter not imported - motor not supported on Streamlit Cloud
    ticker: str,
    start_date: datetime,
    end_date: datetime
) -> Tuple[str, pd.DataFrame]:
    """
    Async version: Calculate all indicators for a ticker.
    Returns tuple of (ticker, dataframe) for batch processing.
    """
    try:
        # Add warmup period for indicators (200 days = ~290 calendar days)
        warmup_days = 365  # Use 1 year warmup to ensure all indicators have enough data
        warmup_start = start_date - timedelta(days=warmup_days)
        
        # Load price data with warmup period
        df = await db.get_price_data(ticker, warmup_start, end_date)
        
        if df.empty or 'close' not in df.columns:
            return (ticker, pd.DataFrame())
        
        # Calculate indicators on full dataset (including warmup)
        df = calculate_all_indicators(df)
        
        # Calculate MACD stage for each row (vectorized - much faster!)
        if 'macd_hist' in df.columns:
            df['macd_stage'] = detect_macd_stage_vectorized(df['macd_hist'], lookback=20)
        
        # Trim to original date range (remove warmup period)
        df = df[df['date'] >= start_date].copy()
        
        return (ticker, df)
    except Exception as e:
        print(f"Error calculating indicators for {ticker}: {e}")
        return (ticker, pd.DataFrame())

async def save_ticker_indicators_async(
    db,  # AsyncDatabaseAdapter not imported - motor not supported on Streamlit Cloud
    ticker: str,
    df: pd.DataFrame
) -> Tuple[str, bool]:
    """
    Async version: Save calculated indicators to database.
    Returns tuple of (ticker, success) for tracking.
    """
    try:
        if df.empty:
            return (ticker, False)
        
        # Save each date's indicators
        tasks = []
        for _, row in df.iterrows():
            if pd.notna(row.get('date')):
                indicators = {
                    'close': float(row.get('close', np.nan)),
                    'ema10': float(row.get('ema10', np.nan)),
                    'ema20': float(row.get('ema20', np.nan)),
                    'ema50': float(row.get('ema50', np.nan)),
                    'ema100': float(row.get('ema100', np.nan)),
                    'ema200': float(row.get('ema200', np.nan)),
                    'rsi': float(row.get('rsi', np.nan)),
                    'macd': float(row.get('macd', np.nan)),
                    'macd_signal': float(row.get('macd_signal', np.nan)),
                    'macd_hist': float(row.get('macd_hist', np.nan)),
                    'macd_stage': str(row.get('macd_stage', 'N/A')),
                    'bb_upper': float(row.get('bb_upper', np.nan)),
                    'bb_middle': float(row.get('bb_middle', np.nan)),
                    'bb_lower': float(row.get('bb_lower', np.nan)),
                }
                task = db.save_indicators(ticker, row['date'], indicators)
                tasks.append(task)
        
        # Save all dates for this ticker in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if all succeeded
        success = all(r is True for r in results if not isinstance(r, Exception))
        return (ticker, success)
        
    except Exception as e:
        print(f"Error saving indicators for {ticker}: {e}")
        return (ticker, False)

async def process_ticker_batch_async(
    db,  # AsyncDatabaseAdapter not imported - motor not supported on Streamlit Cloud
    tickers: List[str],
    start_date: datetime,
    end_date: datetime
) -> Tuple[int, int]:
    """
    Process a batch of tickers in parallel.
    Returns tuple of (success_count, failed_count).
    """
    # Step 1: Calculate indicators for all tickers in batch (parallel)
    calc_tasks = [
        calculate_indicators_for_ticker_async(db, ticker, start_date, end_date)
        for ticker in tickers
    ]
    calc_results = await asyncio.gather(*calc_tasks, return_exceptions=True)
    
    # Step 2: Save indicators for all tickers in batch (parallel)
    save_tasks = []
    for result in calc_results:
        if isinstance(result, Exception):
            continue
        ticker, df = result
        if not df.empty:
            save_tasks.append(save_ticker_indicators_async(db, ticker, df))
    
    save_results = await asyncio.gather(*save_tasks, return_exceptions=True)
    
    # Count successes and failures
    success_count = sum(1 for r in save_results if not isinstance(r, Exception) and r[1] is True)
    failed_count = len(tickers) - success_count
    
    return (success_count, failed_count)

async def calculate_and_save_breadth_for_date_async(
    db,  # AsyncDatabaseAdapter not imported - motor not supported on Streamlit Cloud
    date: datetime
) -> Tuple[datetime, bool]:
    """
    Calculate and save market breadth for a specific date.
    Returns tuple of (date, success).
    """
    try:
        # Get indicators for this date
        df_indicators = await db.get_indicators_for_date(date)
        
        if df_indicators.empty:
            return (date, False)
        
        # Calculate breadth (synchronous but fast)
        breadth = calculate_market_breadth(df_indicators)
        
        # Convert numpy types to Python native types
        clean_breadth = convert_numpy_types(breadth)
        
        # Save to database
        success = await db.save_market_breadth(date, clean_breadth)
        
        return (date, success)
    except Exception as e:
        print(f"Error calculating breadth for {date}: {e}")
        return (date, False)

async def process_date_batch_async(
    db,  # AsyncDatabaseAdapter not imported - motor not supported on Streamlit Cloud
    dates: List[datetime]
) -> Tuple[int, int]:
    """
    Process a batch of dates in parallel for breadth calculation.
    Returns tuple of (success_count, failed_count).
    """
    tasks = [
        calculate_and_save_breadth_for_date_async(db, date)
        for date in dates
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    success_count = sum(1 for r in results if not isinstance(r, Exception) and r[1] is True)
    failed_count = len(dates) - success_count
    
    return (success_count, failed_count)

def run_async_batch_calculation(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    ticker_batch_size: int = 10,
    progress_callback=None
) -> Tuple[int, int]:
    """
    Run async batch calculation with proper event loop handling.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for calculation
        end_date: End date for calculation
        ticker_batch_size: Number of tickers to process in parallel
        progress_callback: Optional callback for progress updates
    
    Returns:
        Tuple of (total_success, total_failed)
    """
    if not HAS_MOTOR:
        return (0, 0)  # Fall back to sync processing
    
    async def run():
        # AsyncDatabaseAdapter not available - this code path not used
        from utils.db_async import get_sync_db_adapter
        db = get_sync_db_adapter()  # Note: This won't work with async, but HAS_MOTOR=False prevents execution
        
        total_success = 0
        total_failed = 0
        
        # Process tickers in batches
        for i in range(0, len(tickers), ticker_batch_size):
            batch = tickers[i:i + ticker_batch_size]
            
            if progress_callback:
                progress_callback(i, len(tickers), f"Processing batch {i//ticker_batch_size + 1}")
            
            success, failed = await process_ticker_batch_async(db, batch, start_date, end_date)
            total_success += success
            total_failed += failed
        
        # Get all trading dates from database
        trading_dates = await db.get_trading_dates(start_date, end_date)
        
        if trading_dates:
            # Process dates in batches for breadth calculation
            date_batch_size = 20
            for i in range(0, len(trading_dates), date_batch_size):
                date_batch = trading_dates[i:i + date_batch_size]
                
                if progress_callback:
                    progress_callback(
                        len(tickers) + i,
                        len(tickers) + len(trading_dates),
                        f"Calculating breadth for {len(date_batch)} dates"
                    )
                
                await process_date_batch_async(db, date_batch)
        
        return (total_success, total_failed)
    
    # Run with proper event loop handling
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running (Jupyter/Streamlit), create new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run())
                return future.result()
        else:
            # If no loop is running, use asyncio.run
            return asyncio.run(run())
    except RuntimeError:
        # Fallback: create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run())
        finally:
            loop.close()

def calculate_macd_stages_for_ticker(ticker: str, start_date: datetime, end_date: datetime) -> bool:
    """Calculate only MACD stages for a ticker that already has indicators.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load with warmup for proper stage detection
        warmup_days = 365
        warmup_start = start_date - timedelta(days=warmup_days)
        
        df = load_price_data_for_ticker(ticker, warmup_start, end_date)
        
        if df.empty or 'close' not in df.columns:
            return False
        
        # Calculate only MACD
        df = calculate_all_indicators(df)
        
        if 'macd_hist' not in df.columns:
            return False
        
        # Calculate MACD stages (vectorized)
        df['macd_stage'] = detect_macd_stage_vectorized(df['macd_hist'], lookback=20)
        
        # Trim to target range
        df = df[df['date'] >= start_date].copy()
        
        # Update only the macd_stage field in database
        from pymongo import MongoClient
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            return False
            
        client = MongoClient(mongo_uri + "&socketTimeoutMS=60000&connectTimeoutMS=60000")
        db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
        collection = client[db_name].indicators
        
        for _, row in df.iterrows():
            if pd.notna(row.get('date')):
                date_value = row['date']
                if isinstance(date_value, pd.Timestamp):
                    date_value = date_value.to_pydatetime()
                
                collection.update_one(
                    {"ticker": ticker.upper(), "date": date_value},
                    {"$set": {
                        "macd_stage": str(row.get('macd_stage', 'N/A')),
                        "updated_at": datetime.now()
                    }}
                )
        
        client.close()
        return True
    except Exception as e:
        return False


def save_indicators_to_db(ticker: str, df: pd.DataFrame) -> bool:
    """Save calculated indicators to database."""
    try:
        from pymongo import MongoClient
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            return False
            
        client = MongoClient(mongo_uri + "&socketTimeoutMS=60000&connectTimeoutMS=60000")
        db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
        collection = client[db_name].indicators
        
        for _, row in df.iterrows():
            if pd.notna(row.get('date')):
                # Convert date to datetime if it's a Timestamp
                date_value = row['date']
                if isinstance(date_value, pd.Timestamp):
                    date_value = date_value.to_pydatetime()
                
                indicators = {
                    'ticker': ticker.upper(),
                    'date': date_value,
                    'close': float(row.get('close', np.nan)),
                    'ema10': float(row.get('ema10', np.nan)),
                    'ema20': float(row.get('ema20', np.nan)),
                    'ema50': float(row.get('ema50', np.nan)),
                    'ema100': float(row.get('ema100', np.nan)),
                    'ema200': float(row.get('ema200', np.nan)),
                    'rsi': float(row.get('rsi', np.nan)),
                    'macd': float(row.get('macd', np.nan)),
                    'macd_signal': float(row.get('macd_signal', np.nan)),
                    'macd_hist': float(row.get('macd_hist', np.nan)),
                    'macd_stage': str(row.get('macd_stage', 'N/A')),
                    'bb_upper': float(row.get('bb_upper', np.nan)),
                    'bb_middle': float(row.get('bb_middle', np.nan)),
                    'bb_lower': float(row.get('bb_lower', np.nan)),
                    'updated_at': datetime.now()
                }
                
                collection.update_one(
                    {"ticker": ticker.upper(), "date": date_value},
                    {"$set": indicators},
                    upsert=True
                )
        
        client.close()
        return True
    except Exception as e:
        st.error(f"Error saving indicators: {e}")
        return False

def get_indicators_for_date(target_date: datetime) -> pd.DataFrame:
    """Get all indicators for all tickers on a specific date."""
    try:
        from pymongo import MongoClient
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            return pd.DataFrame()
            
        client = MongoClient(mongo_uri + "&socketTimeoutMS=60000&connectTimeoutMS=60000")
        db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
        collection = client[db_name].indicators
        
        docs = list(collection.find({"date": target_date}))
        client.close()
        
        if not docs:
            return pd.DataFrame()
        
        df = pd.DataFrame(docs)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error loading indicators: {e}")
        return pd.DataFrame()

def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for MongoDB compatibility.
    
    Args:
        obj: Object that may contain numpy types
    
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
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

def calculate_market_breadth(df_indicators: pd.DataFrame) -> dict:
    """Calculate market breadth from indicators DataFrame."""
    if df_indicators.empty:
        return {}
    
    total_tickers = len(df_indicators)
    
    breadth = {
        'total_tickers': total_tickers,
        'date': df_indicators['date'].iloc[0] if 'date' in df_indicators.columns else None
    }
    
    # EMA breadth
    for period in [10, 20, 50, 100, 200]:
        col = f'ema{period}'
        if col in df_indicators.columns:
            # Vectorized: much faster than apply
            above = check_price_above_ema_vectorized(
                df_indicators['close'],
                df_indicators[col]
            ).sum()
            breadth[f'above_ema{period}'] = above
            breadth[f'above_ema{period}_pct'] = (above / total_tickers * 100) if total_tickers > 0 else 0
    
    # RSI breadth
    if 'rsi' in df_indicators.columns:
        # Vectorized: much faster than apply
        df_indicators['rsi_category'] = categorize_rsi_vectorized(df_indicators['rsi'])
        
        # Count valid RSI values (exclude N/A)
        rsi_valid_count = (df_indicators['rsi_category'] != 'N/A').sum()
        rsi_na_count = (df_indicators['rsi_category'] == 'N/A').sum()
        breadth['rsi_valid_tickers'] = rsi_valid_count
        breadth['rsi_na_tickers'] = rsi_na_count
        
        for category in ['oversold', '<50', '>50', 'overbought']:
            count = (df_indicators['rsi_category'] == category).sum()
            breadth[f'rsi_{category}'] = count
            # Use valid count as denominator (exclude N/A tickers)
            breadth[f'rsi_{category}_pct'] = (count / rsi_valid_count * 100) if rsi_valid_count > 0 else 0
    
    # MACD breadth
    if 'macd_stage' in df_indicators.columns:
        # Vectorized: much faster than apply
        df_indicators['macd_category'] = categorize_macd_stage_vectorized(df_indicators['macd_stage'])
        
        # Count valid MACD values (exclude N/A)
        macd_valid_count = (df_indicators['macd_category'] != 'N/A').sum()
        macd_na_count = (df_indicators['macd_category'] == 'N/A').sum()
        breadth['macd_valid_tickers'] = macd_valid_count
        breadth['macd_na_tickers'] = macd_na_count
        
        for category in ['troughing', 'confirmed_trough', 'rising', 'peaking', 'confirmed_peak', 'declining']:
            count = (df_indicators['macd_category'] == category).sum()
            breadth[f'macd_{category}'] = count
            # Use valid count as denominator (exclude N/A tickers)
            breadth[f'macd_{category}_pct'] = (count / macd_valid_count * 100) if macd_valid_count > 0 else 0
    
    return breadth

def save_market_breadth(date: datetime, breadth_data: dict) -> bool:
    """Save market breadth to database."""
    try:
        from pymongo import MongoClient
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            return False
            
        client = MongoClient(mongo_uri + "&socketTimeoutMS=60000&connectTimeoutMS=60000")
        db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
        collection = client[db_name].market_breadth
        
        # Convert numpy types to Python native types
        clean_breadth_data = convert_numpy_types(breadth_data)
        
        doc = {
            "date": date,
            **clean_breadth_data,
            "updated_at": datetime.now()
        }
        
        collection.update_one(
            {"date": date},
            {"$set": doc},
            upsert=True
        )
        
        client.close()
        return True
    except Exception as e:
        st.error(f"Error saving market breadth: {e}")
        return False

def get_market_breadth_history(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get market breadth history from database."""
    try:
        from pymongo import MongoClient
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            return pd.DataFrame()
            
        client = MongoClient(mongo_uri + "&socketTimeoutMS=60000&connectTimeoutMS=60000")
        db_name = os.getenv("MONGODB_DB_NAME", "macd_reversal")
        collection = client[db_name].market_breadth
        
        query = {"date": {"$gte": start_date, "$lte": end_date}}
        cursor = collection.find(query).sort("date", 1)
        docs = list(cursor)
        client.close()
        
        if not docs:
            return pd.DataFrame()
        
        df = pd.DataFrame(docs)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    except Exception as e:
        st.error(f"Error loading breadth history: {e}")
        return pd.DataFrame()

def plot_vnindex_chart(start_date: datetime, end_date: datetime):
    """Plot VNINDEX technical chart with all indicators."""
    try:
        df = load_price_data_for_ticker('VNINDEX', start_date, end_date)
        
        if df.empty:
            st.warning("No VNINDEX data available")
            return
        
        # Calculate indicators
        df = calculate_all_indicators(df)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('VNINDEX Price & Moving Averages', 'RSI(14)', 'MACD Histogram')
        )
        
        # Candlestick chart
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(
                go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='VNINDEX'
                ),
                row=1, col=1
            )
        
        # EMAs
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        for i, period in enumerate([10, 20, 50, 100, 200]):
            col = f'ema{period}'
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df[col],
                        name=f'EMA{period}',
                        line=dict(color=colors[i % len(colors)], width=1.5)
                    ),
                    row=1, col=1
                )
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df['bb_upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dot'),
                    opacity=0.5
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df['bb_lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dot'),
                    fill='tonexty',
                    opacity=0.3
                ),
                row=1, col=1
            )
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df['rsi'],
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
        
        # MACD Histogram
        if 'macd_hist' in df.columns:
            colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
            fig.add_trace(
                go.Bar(
                    x=df['date'], y=df['macd_hist'],
                    name='MACD Hist',
                    marker_color=colors
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='VNINDEX Technical Analysis',
            xaxis_title='Date',
            height=900,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error plotting VNINDEX chart: {e}")

def plot_breadth_chart(df_breadth: pd.DataFrame, metric: str, title: str):
    """Plot a single breadth metric over time."""
    if df_breadth.empty or metric not in df_breadth.columns:
        st.warning(f"No data available for {title}")
        return
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df_breadth['date'],
            y=df_breadth[metric],
            mode='lines+markers',
            name=title,
            fill='tozeroy',
            line=dict(width=2)
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Percentage (%)',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================

with st.sidebar:
    st.header("📅 Control Panel")
    
    # Get latest date
    latest_date = get_latest_date_from_db()
    
    if latest_date is None:
        st.error("No data found in database")
        st.stop()
    
    st.info(f"Latest data: {latest_date.strftime('%Y-%m-%d')}")
    
    # Date selector
    selected_date = st.date_input(
        "Select Date",
        value=latest_date.date(),
        max_value=latest_date.date()
    )
    selected_datetime = datetime.combine(selected_date, datetime.min.time())
    
    st.markdown("---")
    
    # Recalculation controls
    st.subheader("🔄 Recalculate Indicators")
    
    recalc_enabled = st.checkbox("Enable recalculation", value=False)
    
    if recalc_enabled:
        # Calculation mode
        calc_mode = st.radio(
            "Calculation Mode",
            options=["Missing dates only", "Full range replacement"],
            help="Missing dates: Only calculate dates without data. Full range: Recalculate entire range."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            lookback_days = st.number_input(
                "Trading days",
                min_value=10,
                max_value=1000,
                value=200,
                step=10
            )
        
        # Calculate date range
        calc_end_date = latest_date
        calendar_days = int(lookback_days * 365 / 252)
        calc_start_date = calc_end_date - timedelta(days=calendar_days)
        
        if calc_mode == "Missing dates only":
            st.info(f"📅 Range: {calc_start_date.strftime('%Y-%m-%d')} to {calc_end_date.strftime('%Y-%m-%d')} (missing only)")
        else:
            st.warning(f"📅 Range: {calc_start_date.strftime('%Y-%m-%d')} to {calc_end_date.strftime('%Y-%m-%d')} (full replacement)")
        
        # Batch size configuration
        ticker_batch_size = st.slider(
            "Ticker batch size",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Number of tickers to process in parallel. Higher = faster but more memory."
        )
        
        # Processing mode selection
        use_async = st.checkbox(
            "Use async batch processing",
            value=HAS_MOTOR,
            disabled=not HAS_MOTOR,
            help="Process tickers in parallel batches (requires motor library)"
        )
        
        if not HAS_MOTOR and use_async:
            st.warning("⚠️ Motor not installed. Install with: pip install motor")
        
        if st.button("▶️ Calculate Now", type="primary"):
            all_tickers = get_all_tickers_cached()
            
            if not all_tickers:
                st.error("No tickers found")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = datetime.now()
                
                if use_async and HAS_MOTOR:
                    # === ASYNC BATCH PROCESSING ===
                    status_text.text("🚀 Starting async batch processing...")
                    
                    def progress_callback(current, total, message):
                        progress = current / total if total > 0 else 0
                        progress_bar.progress(min(progress, 1.0))
                        status_text.text(f"{message} ({current}/{total})")
                    
                    try:
                        success_count, failed_count = run_async_batch_calculation(
                            all_tickers,
                            calc_start_date,
                            calc_end_date,
                            ticker_batch_size=ticker_batch_size,
                            progress_callback=progress_callback
                        )
                        
                        elapsed = (datetime.now() - start_time).total_seconds()
                        
                        st.success(
                            f"✅ Async calculation complete in {elapsed:.1f}s!\n\n"
                            f"- Success: {success_count} tickers\n"
                            f"- Failed: {failed_count} tickers\n"
                            f"- Speed: {len(all_tickers)/elapsed:.1f} tickers/sec"
                        )
                        
                    except Exception as e:
                        st.error(f"Async processing error: {e}")
                        st.info("Falling back to synchronous processing...")
                        use_async = False
                
                if not use_async or not HAS_MOTOR:
                    # === SYNCHRONOUS PROCESSING (FALLBACK) ===
                    success_count = 0
                    failed_count = 0
                    
                    for idx, ticker in enumerate(all_tickers):
                        status_text.text(f"Processing {ticker} ({idx + 1}/{len(all_tickers)})...")
                        
                        try:
                            df_indicators = calculate_indicators_for_ticker(
                                ticker,
                                calc_start_date,
                                calc_end_date
                            )
                            
                            if not df_indicators.empty:
                                if save_indicators_to_db(ticker, df_indicators):
                                    success_count += 1
                                else:
                                    failed_count += 1
                            else:
                                failed_count += 1
                        except Exception as e:
                            failed_count += 1
                            st.error(f"Error processing {ticker}: {e}")
                        
                        progress_bar.progress((idx + 1) / len(all_tickers))
                    
                    elapsed = (datetime.now() - start_time).total_seconds()
                    
                    st.success(
                        f"✅ Sync calculation complete in {elapsed:.1f}s!\n\n"
                        f"- Success: {success_count}\n"
                        f"- Failed: {failed_count}\n"
                        f"- Speed: {len(all_tickers)/elapsed:.1f} tickers/sec"
                    )
                
                status_text.empty()
                progress_bar.empty()
                
                # Clear cache
                st.cache_data.clear()
                st.rerun()
    
    st.markdown("---")
    
    # Debug mode toggle
    debug_mode = st.checkbox("🐛 Debug Mode", value=False, help="Show detailed metrics table and raw data")
    
    st.markdown("---")
    
    # Indicator selection
    st.subheader("📊 Filter Indicators")
    
    with st.expander("Moving Averages", expanded=True):
        selected_emas = st.multiselect(
            "Select EMAs",
            options=[10, 20, 50, 100, 200],
            default=[20, 50, 200]
        )
    
    with st.expander("RSI Groups", expanded=True):
        selected_rsi = st.multiselect(
            "Select RSI ranges",
            options=['oversold', '<50', '>50', 'overbought'],
            default=['oversold', 'overbought']
        )
    
    with st.expander("MACD Stages", expanded=True):
        selected_macd = st.multiselect(
            "Select MACD stages",
            options=['troughing', 'confirmed_trough', 'rising', 'peaking', 'confirmed_peak', 'declining'],
            default=['confirmed_trough', 'confirmed_peak']
        )

# =====================================================================
# MAIN CONTENT
# =====================================================================

# Load indicators for selected date
df_indicators = get_indicators_for_date(selected_datetime)

if df_indicators.empty:
    st.warning(f"⚠️ No indicator data found for {selected_date}.")
    
    # Auto-calculate missing data
    st.info("🔄 Automatically calculating indicators for missing date...")
    st.caption("📊 Phase 1: Fast indicators (EMA, RSI, MACD) | Phase 2: MACD stages (lazy load)")
    
    all_tickers = get_all_tickers_cached()
    
    if not all_tickers:
        st.error("No tickers found in database. Please check your data source.")
        st.stop()
    
    # Calculate date range (7 days warmup for this specific date)
    calc_start_date = selected_datetime - timedelta(days=10)
    calc_end_date = selected_datetime
    
    # === PHASE 1: Fast indicators without MACD stages ===
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    success_count = 0
    failed_count = 0
    ticker_times = []
    
    start_time = datetime.now()
    phase1_start = datetime.now()
    
    status_text.text("📊 Phase 1/2: Calculating fast indicators (EMA, RSI, MACD)...")
    
    for idx, ticker in enumerate(all_tickers):
        ticker_start = datetime.now()
        
        try:
            # Skip MACD stage for faster initial load
            df_indicators_calc = calculate_indicators_for_ticker(
                ticker,
                calc_start_date,
                calc_end_date,
                skip_macd_stage=True  # Fast mode!
            )
            
            if not df_indicators_calc.empty:
                if save_indicators_to_db(ticker, df_indicators_calc):
                    success_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
        
        ticker_elapsed = (datetime.now() - ticker_start).total_seconds()
        ticker_times.append(ticker_elapsed)
        avg_time = sum(ticker_times) / len(ticker_times)
        remaining = (len(all_tickers) - idx - 1) * avg_time
        
        progress_bar.progress((idx + 1) / len(all_tickers))
        status_text.text(
            f"Phase 1: {ticker} ({idx + 1}/{len(all_tickers)}) | "
            f"{ticker_elapsed:.1f}s | Avg: {avg_time:.1f}s | ETA: {remaining:.0f}s"
        )
    
    phase1_elapsed = (datetime.now() - phase1_start).total_seconds()
    
    # Show phase 1 results
    if success_count > 0:
        st.success(
            f"✅ Phase 1 complete in {phase1_elapsed:.1f}s! "
            f"({len(all_tickers)/phase1_elapsed:.1f} tickers/sec)\n\n"
            f"Success: {success_count} | Failed: {failed_count}"
        )
        
        # Reload indicators to show data immediately
        df_indicators = get_indicators_for_date(selected_datetime)
        
        if not df_indicators.empty:
            # Show data with placeholder MACD stages
            st.info("📈 Showing market breadth with basic indicators. MACD stages: N/A (calculate below if needed)")
            
            # === PHASE 2: Calculate MACD stages (optional, user-triggered) ===
            if st.button("🔄 Calculate MACD Stages Now", type="secondary", key="calc_macd_stages"):
                phase2_start = datetime.now()
                status_text.text("📊 Phase 2/2: Calculating MACD stages...")
                progress_bar.progress(0)
                
                macd_success = 0
                macd_failed = 0
                
                for idx, ticker in enumerate(all_tickers):
                    if calculate_macd_stages_for_ticker(ticker, calc_start_date, calc_end_date):
                        macd_success += 1
                    else:
                        macd_failed += 1
                    
                    progress_bar.progress((idx + 1) / len(all_tickers))
                    status_text.text(f"Phase 2: MACD stages for {ticker} ({idx + 1}/{len(all_tickers)})")
                
                phase2_elapsed = (datetime.now() - phase2_start).total_seconds()
                
                status_text.empty()
                progress_bar.empty()
                
                st.success(
                    f"✅ Phase 2 complete in {phase2_elapsed:.1f}s!\n\n"
                    f"MACD Success: {macd_success} | Failed: {macd_failed}"
                )
                
                # Reload with MACD stages
                st.cache_data.clear()
                st.rerun()
        else:
            st.error("No data after Phase 1. Please check your data source.")
            st.stop()
    else:
        st.error("Phase 1 failed. Please check your data source.")
        st.stop()
    
    # Don't continue to breadth calculation yet if MACD stages not calculated
    # User needs to click button to proceed

# Calculate breadth
breadth = calculate_market_breadth(df_indicators)

# Save breadth to database
save_market_breadth(selected_datetime, breadth)

# Labels for display
rsi_labels = {
    'oversold': 'RSI Oversold (<30)',
    '<50': 'RSI Below 50',
    '>50': 'RSI Above 50',
    'overbought': 'RSI Overbought (>70)'
}

macd_labels = {
    'troughing': 'MACD Troughing',
    'confirmed_trough': 'MACD Confirmed Trough',
    'rising': 'MACD Rising',
    'peaking': 'MACD Peaking',
    'confirmed_peak': 'MACD Confirmed Peak',
    'declining': 'MACD Declining'
}

# =====================================================================
# MAIN VIEW: MARKET BREADTH CHARTS
# =====================================================================

st.markdown("## 📈 Market Breadth Analysis")
st.markdown(f"**Date:** {selected_date} | **Total Tickers:** {breadth.get('total_tickers', 0)}")

# =====================================================================
# PIE CHARTS: Market Distribution
# =====================================================================

st.markdown("### 📊 Market Distribution (Current Date)")

# Create 3 columns for pie charts
col_ma, col_rsi, col_macd = st.columns(3)

with col_ma:
    st.markdown("**Moving Average Position**")
    # Get MA data for a representative period (e.g., EMA50 or most selected)
    if selected_emas:
        # Use the middle period for summary
        summary_period = 50 if 50 in selected_emas else selected_emas[len(selected_emas)//2]
        above_count = breadth.get(f'above_ema{summary_period}', 0)
        below_count = breadth.get('total_tickers', 0) - above_count
        
        ma_data = pd.DataFrame({
            'Category': [f'Above EMA{summary_period}', f'Below EMA{summary_period}'],
            'Count': [above_count, below_count],
            'Percentage': [
                breadth.get(f'above_ema{summary_period}_pct', 0),
                100 - breadth.get(f'above_ema{summary_period}_pct', 0)
            ]
        })
        
        fig_ma = go.Figure(data=[go.Pie(
            labels=ma_data['Category'],
            values=ma_data['Count'],
            hole=0.3,
            marker=dict(colors=['#00CC96', '#EF553B']),
            textinfo='label+percent',
            textposition='auto'
        )])
        fig_ma.update_layout(
            showlegend=True,
            height=300,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_ma, use_container_width=True)
    else:
        st.info("Select EMA periods in sidebar")

with col_rsi:
    st.markdown("**RSI Categories**")
    if 'rsi' in df_indicators.columns and breadth.get('rsi_valid_tickers', 0) > 0:
        rsi_categories = ['oversold', '<50', '>50', 'overbought']
        rsi_data = pd.DataFrame({
            'Category': [rsi_labels[cat] for cat in rsi_categories],
            'Count': [breadth.get(f'rsi_{cat}', 0) for cat in rsi_categories],
            'Percentage': [breadth.get(f'rsi_{cat}_pct', 0) for cat in rsi_categories]
        })
        # Filter out zero counts
        rsi_data = rsi_data[rsi_data['Count'] > 0]
        
        if not rsi_data.empty:
            fig_rsi = go.Figure(data=[go.Pie(
                labels=rsi_data['Category'],
                values=rsi_data['Count'],
                hole=0.3,
                marker=dict(colors=['#EF553B', '#FFA15A', '#00CC96', '#AB63FA']),
                textinfo='label+percent',
                textposition='auto'
            )])
            fig_rsi.update_layout(
                showlegend=True,
                height=300,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Show N/A count if any
            na_count = breadth.get('rsi_na_tickers', 0)
            if na_count > 0:
                st.caption(f"⚠️ {na_count} tickers with insufficient RSI data")
        else:
            st.info("No RSI data available")
    else:
        st.info("RSI data not available")

with col_macd:
    st.markdown("**MACD Stages**")
    if 'macd_stage' in df_indicators.columns and breadth.get('macd_valid_tickers', 0) > 0:
        macd_categories = ['troughing', 'confirmed_trough', 'rising', 'peaking', 'confirmed_peak', 'declining']
        macd_data = pd.DataFrame({
            'Category': [macd_labels[cat] for cat in macd_categories],
            'Count': [breadth.get(f'macd_{cat}', 0) for cat in macd_categories],
            'Percentage': [breadth.get(f'macd_{cat}_pct', 0) for cat in macd_categories]
        })
        # Filter out zero counts
        macd_data = macd_data[macd_data['Count'] > 0]
        
        if not macd_data.empty:
            # Color scheme: bullish (green shades) to bearish (red shades)
            macd_colors = {
                'MACD Confirmed Trough': '#00CC96',
                'MACD Troughing': '#66D9A6',
                'MACD Rising': '#99E6C2',
                'MACD Peaking': '#FFAA99',
                'MACD Confirmed Peak': '#EF553B',
                'MACD Declining': '#FF8866'
            }
            colors = [macd_colors.get(cat, '#636EFA') for cat in macd_data['Category']]
            
            fig_macd = go.Figure(data=[go.Pie(
                labels=macd_data['Category'],
                values=macd_data['Count'],
                hole=0.3,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textposition='auto'
            )])
            fig_macd.update_layout(
                showlegend=True,
                height=300,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # Show N/A count if any
            na_count = breadth.get('macd_na_tickers', 0)
            if na_count > 0:
                st.caption(f"⚠️ {na_count} tickers with insufficient MACD data")
        else:
            st.info("No MACD data available")
    else:
        st.info("MACD data not available")

st.markdown("---")

if debug_mode:
    # =====================================================================
    # DEBUG: MARKET BREADTH SUMMARY TABLE
    # =====================================================================
    
    st.markdown("### 🐛 Debug: Breadth Metrics Table")
    
    # Create summary table
    summary_rows = []
    
    # EMA breadth
    for period in selected_emas:
        count = breadth.get(f'above_ema{period}', 0)
        pct = breadth.get(f'above_ema{period}_pct', 0)
        summary_rows.append({
            'Indicator Group': f'Above EMA{period}',
            '% of Market': f"{pct:.1f}%",
            'Count': count,
            'Total': breadth.get('total_tickers', 0)
        })
    
    # RSI breadth
    for category in selected_rsi:
        count = breadth.get(f'rsi_{category}', 0)
        pct = breadth.get(f'rsi_{category}_pct', 0)
        summary_rows.append({
            'Indicator Group': rsi_labels.get(category, category),
            '% of Market': f"{pct:.1f}%",
            'Count': count,
            'Total': breadth.get('total_tickers', 0)
        })
    
    # MACD breadth
    for category in selected_macd:
        count = breadth.get(f'macd_{category}', 0)
        pct = breadth.get(f'macd_{category}_pct', 0)
        summary_rows.append({
            'Indicator Group': macd_labels.get(category, category),
            '% of Market': f"{pct:.1f}%",
            'Count': count,
            'Total': breadth.get('total_tickers', 0)
        })
    
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Select indicators in the sidebar to view breadth summary")
    
    st.markdown("---")

# =====================================================================
# BREADTH CHARTS BY INDICATOR GROUP (Main View)
# =====================================================================

# Prepare ticker lists for hover display
ticker_lists = {}

# EMA ticker lists
for period in selected_emas:
    col = f'ema{period}'
    if col in df_indicators.columns:
        # Vectorized: much faster than apply
        above_mask = check_price_above_ema_vectorized(
            df_indicators['close'],
            df_indicators[col]
        )
        tickers_above = df_indicators[above_mask]['ticker'].tolist()
        ticker_lists[f'ema{period}'] = sorted(tickers_above)

# RSI ticker lists
if 'rsi' in df_indicators.columns:
    # Vectorized: much faster than apply
    df_indicators['rsi_category'] = categorize_rsi_vectorized(df_indicators['rsi'])
    for category in selected_rsi:
        tickers_in_cat = df_indicators[df_indicators['rsi_category'] == category]['ticker'].tolist()
        ticker_lists[f'rsi_{category}'] = sorted(tickers_in_cat)

# MACD ticker lists
if 'macd_stage' in df_indicators.columns:
    # Vectorized: much faster than apply
    df_indicators['macd_category'] = categorize_macd_stage_vectorized(df_indicators['macd_stage'])
    for category in selected_macd:
        tickers_in_cat = df_indicators[df_indicators['macd_category'] == category]['ticker'].tolist()
        ticker_lists[f'macd_{category}'] = sorted(tickers_in_cat)

# Display ticker lists for current date
with st.expander("📋 View Ticker Lists (Current Date)", expanded=False):
    tab_ema, tab_rsi, tab_macd = st.tabs(["Moving Averages", "RSI", "MACD"])
    
    with tab_ema:
        for period in selected_emas:
            tickers = ticker_lists.get(f'ema{period}', [])
            st.markdown(f"**Above EMA{period}** ({len(tickers)} tickers)")
            if tickers:
                st.write(", ".join(tickers))
            else:
                st.write("_No tickers_")
    
    with tab_rsi:
        for category in selected_rsi:
            tickers = ticker_lists.get(f'rsi_{category}', [])
            st.markdown(f"**{rsi_labels.get(category, category)}** ({len(tickers)} tickers)")
            if tickers:
                st.write(", ".join(tickers))
            else:
                st.write("_No tickers_")
    
    with tab_macd:
        for category in selected_macd:
            tickers = ticker_lists.get(f'macd_{category}', [])
            st.markdown(f"**{macd_labels.get(category, category)}** ({len(tickers)} tickers)")
            if tickers:
                st.write(", ".join(tickers))
            else:
                st.write("_No tickers_")

# =====================================================================
# VNINDEX TECHNICAL CHART (with date axis)
# =====================================================================

st.markdown("## 📊 VNINDEX Technical Analysis")
st.caption("All charts share synchronized date axes for easy comparison")

chart_lookback = st.slider("Chart lookback (days)", 30, 365, 180)
chart_start = selected_datetime - timedelta(days=chart_lookback)

plot_vnindex_chart(chart_start, selected_datetime)

# =====================================================================
# MARKET BREADTH TRENDS (Synchronized with VNINDEX)
# =====================================================================

st.markdown("## 📈 Market Breadth Trends")
st.caption(f"Historical breadth data from {chart_start.strftime('%Y-%m-%d')} to {selected_datetime.strftime('%Y-%m-%d')}")

# Load breadth history for same period as VNINDEX chart
df_breadth_history = get_market_breadth_history(chart_start, selected_datetime)

if not df_breadth_history.empty:
    # EMA breadth charts
    if selected_emas:
        st.markdown("### Moving Average Breadth")
        cols = st.columns(min(len(selected_emas), 3))
        for i, period in enumerate(selected_emas):
            with cols[i % len(cols)]:
                metric = f'above_ema{period}_pct'
                plot_breadth_chart(df_breadth_history, metric, f'Above EMA{period}')
    
    # RSI breadth charts
    if selected_rsi:
        st.markdown("### RSI Breadth")
        cols = st.columns(min(len(selected_rsi), 2))
        for i, category in enumerate(selected_rsi):
            with cols[i % len(cols)]:
                metric = f'rsi_{category}_pct'
                plot_breadth_chart(df_breadth_history, metric, rsi_labels.get(category, category))
    
    # MACD breadth charts
    if selected_macd:
        st.markdown("### MACD Breadth")
        cols = st.columns(min(len(selected_macd), 2))
        for i, category in enumerate(selected_macd):
            with cols[i % len(cols)]:
                metric = f'macd_{category}_pct'
                plot_breadth_chart(df_breadth_history, metric, macd_labels.get(category, category))
else:
    st.warning("No historical breadth data available. Recalculate indicators to populate history.")

# =====================================================================
# DEBUG VIEW
# =====================================================================

with st.expander("🔍 Debug View: All Indicators", expanded=False):
    st.markdown(f"### Indicators for all tickers on {selected_date}")
    
    if not df_indicators.empty:
        # Select columns to display
        display_cols = ['ticker', 'close']
        
        for period in [10, 20, 50, 100, 200]:
            col = f'ema{period}'
            if col in df_indicators.columns:
                display_cols.append(col)
        
        if 'rsi' in df_indicators.columns:
            display_cols.append('rsi')
        
        if 'macd_stage' in df_indicators.columns:
            display_cols.append('macd_stage')
        
        # Filter and format
        df_debug = df_indicators[display_cols].copy()
        
        # Format numeric columns
        for col in df_debug.columns:
            if col not in ['ticker', 'macd_stage'] and col in df_debug.columns:
                df_debug[col] = df_debug[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(df_debug, use_container_width=True, hide_index=True)
        
        # Download button
        csv = df_indicators.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Data (CSV)",
            data=csv,
            file_name=f"indicators_{selected_date}.csv",
            mime="text/csv"
        )
    else:
        st.info("No data to display")

st.markdown("---")
st.markdown("_Market Breadth Analysis • Data from MongoDB • Built with utils/indicators.py, utils/macd_stage.py, utils/db_async.py_")
