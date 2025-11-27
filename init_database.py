"""
Initialize empty database with proper schema.
Run this to create a fresh database before using the dashboard.
"""
import sqlite3
import os

def create_empty_database(db_path="price_data.db"):
    """Create empty database with correct schema."""
    
    # Remove existing database if present
    if os.path.exists(db_path):
        print(f"⚠️ Database {db_path} already exists. Delete it manually if you want to recreate.")
        return False
    
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
    
    print(f"✅ Empty database created: {db_path}")
    print("📝 Next steps:")
    print("  1. Add tickers using the Admin panel in the dashboard")
    print("  2. Use TCBS Historical Refresh to fetch data")
    print("  3. Or use Intraday Update for today's data")
    return True

if __name__ == "__main__":
    create_empty_database()
