"""
Initialize empty database with proper schema.
This will be called automatically on first run if database doesn't exist.
"""
import sqlite3
import os

def create_empty_database(db_path="price_data.db"):
    """Create empty database with correct schema."""
    
    # Check if database already exists and has tables
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_data'")
            if cur.fetchone():
                conn.close()
                return True  # Database already initialized
            conn.close()
        except:
            pass
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create price_data table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            source TEXT DEFAULT 'manual',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, date, source)
        )
    """)
    
    # Create market_data table (for market breadth)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            date TEXT PRIMARY KEY,
            advance INTEGER,
            decline INTEGER,
            unchanged INTEGER,
            above_ma50 INTEGER,
            below_ma50 INTEGER,
            above_ma200 INTEGER,
            below_ma200 INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create tcbs_scaling table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tcbs_scaling (
            ticker TEXT PRIMARY KEY,
            scale_factor REAL DEFAULT 1000.0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for performance
    cur.execute("CREATE INDEX IF NOT EXISTS idx_price_ticker ON price_data(ticker)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_price_date ON price_data(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_market_date ON market_data(date)")
    
    conn.commit()
    conn.close()
    
    return True

def add_sample_data(db_path="price_data.db"):
    """Add sample ticker data for demo purposes."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    conn = sqlite3.connect(db_path)
    
    # Add a few sample tickers
    sample_tickers = ["VIC", "VHM", "HPG", "VNM", "TCB"]
    
    for ticker in sample_tickers:
        # Generate 100 days of sample OHLCV data
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(100, 0, -1)]
        
        for i, date in enumerate(dates):
            # Simple random-walk price data
            base_price = 50 + i * 0.1
            data = {
                'ticker': ticker,
                'date': date,
                'open': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price * 1.01,
                'volume': 1000000 + i * 10000,
                'source': 'sample'
            }
            
            conn.execute("""
                INSERT OR IGNORE INTO price_data 
                (ticker, date, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (data['ticker'], data['date'], data['open'], data['high'], 
                  data['low'], data['close'], data['volume'], data['source']))
    
    conn.commit()
    conn.close()
    return True

if __name__ == "__main__":
    create_empty_database()
    # Uncomment to add sample data:
    # add_sample_data()
    print("✅ Database initialized successfully")
