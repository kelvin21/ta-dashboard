"""
Ticker Management Module
Supports add/remove/list operations for price_data.db
Works standalone or via import for dashboard integration
"""
import sqlite3
import os
import pandas as pd
from datetime import datetime
import argparse

DEFAULT_DB_PATH = os.getenv("PRICE_DB_PATH", "price_data.db")

def get_all_tickers(db_path=DEFAULT_DB_PATH):
    """Get list of all tickers in database with row counts."""
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT ticker, 
                   COUNT(*) as rows,
                   MIN(date) as first_date,
                   MAX(date) as last_date,
                   source
            FROM price_data 
            WHERE ticker IS NOT NULL 
            GROUP BY ticker, source
            ORDER BY ticker, source
        """
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def add_ticker(ticker, db_path=DEFAULT_DB_PATH, source="manual"):
    """
    Add a new ticker to database (creates placeholder entry).
    Returns True if added, False if already exists.
    """
    ticker = ticker.strip().upper()
    
    conn = sqlite3.connect(db_path)
    try:
        # Check if ticker already exists
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM price_data WHERE ticker = ?", (ticker,))
        count = cur.fetchone()[0]
        
        if count > 0:
            print(f"Ticker {ticker} already exists with {count} rows.")
            return False
        
        # Insert placeholder row (no price data yet)
        today = datetime.now().strftime("%Y-%m-%d")
        cur.execute("""
            INSERT INTO price_data (ticker, date, source, created_at, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (ticker, today, source))
        conn.commit()
        print(f"✓ Added ticker {ticker} (placeholder entry with source={source})")
        return True
    except Exception as e:
        print(f"✗ Error adding ticker {ticker}: {e}")
        return False
    finally:
        conn.close()

def remove_ticker(ticker, db_path=DEFAULT_DB_PATH, source=None, confirm=False):
    """
    Remove ticker from database.
    If source specified, only removes rows with that source.
    If confirm=True, actually deletes; otherwise dry-run.
    Returns number of rows that would be/were deleted.
    """
    ticker = ticker.strip().upper()
    
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        
        # Count rows to be deleted
        if source:
            cur.execute("SELECT COUNT(*) FROM price_data WHERE ticker = ? AND source = ?", (ticker, source))
        else:
            cur.execute("SELECT COUNT(*) FROM price_data WHERE ticker = ?", (ticker,))
        count = cur.fetchone()[0]
        
        if count == 0:
            print(f"Ticker {ticker} not found" + (f" with source={source}" if source else ""))
            return 0
        
        if not confirm:
            print(f"[DRY RUN] Would delete {count} rows for {ticker}" + (f" (source={source})" if source else ""))
            return count
        
        # Actually delete
        if source:
            cur.execute("DELETE FROM price_data WHERE ticker = ? AND source = ?", (ticker, source))
        else:
            cur.execute("DELETE FROM price_data WHERE ticker = ?", (ticker,))
        conn.commit()
        print(f"✓ Deleted {count} rows for {ticker}" + (f" (source={source})" if source else ""))
        return count
    except Exception as e:
        print(f"✗ Error removing ticker {ticker}: {e}")
        return 0
    finally:
        conn.close()

def bulk_add_tickers(tickers_list, db_path=DEFAULT_DB_PATH, source="manual"):
    """Add multiple tickers from list. Returns (added_count, skipped_count)."""
    added = 0
    skipped = 0
    for ticker in tickers_list:
        if add_ticker(ticker, db_path, source):
            added += 1
        else:
            skipped += 1
    return added, skipped

def bulk_remove_tickers(tickers_list, db_path=DEFAULT_DB_PATH, source=None, confirm=False):
    """Remove multiple tickers. Returns total rows deleted."""
    total = 0
    for ticker in tickers_list:
        total += remove_ticker(ticker, db_path, source, confirm)
    return total

def import_tickers_from_csv(csv_path, db_path=DEFAULT_DB_PATH, source="csv_import"):
    """
    Import tickers from CSV file.
    CSV should have columns: ticker, date, open, high, low, close, volume
    Returns (rows_inserted, errors)
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return 0, 1
    
    try:
        df = pd.read_csv(csv_path)
        required_cols = {"ticker", "date", "open", "high", "low", "close", "volume"}
        
        if not required_cols.issubset(df.columns):
            print(f"CSV missing required columns: {required_cols - set(df.columns)}")
            return 0, 1
        
        conn = sqlite3.connect(db_path)
        inserted = 0
        errors = 0
        
        for _, row in df.iterrows():
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO price_data 
                    (ticker, date, open, high, low, close, volume, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    str(row['ticker']).upper(),
                    str(row['date']),
                    float(row['open']) if pd.notna(row['open']) else None,
                    float(row['high']) if pd.notna(row['high']) else None,
                    float(row['low']) if pd.notna(row['low']) else None,
                    float(row['close']) if pd.notna(row['close']) else None,
                    float(row['volume']) if pd.notna(row['volume']) else None,
                    source
                ))
                inserted += 1
            except Exception as e:
                print(f"Error inserting row {row['ticker']}/{row['date']}: {e}")
                errors += 1
        
        conn.commit()
        conn.close()
        print(f"✓ Imported {inserted} rows from {csv_path} ({errors} errors)")
        return inserted, errors
    except Exception as e:
        print(f"✗ Error reading CSV {csv_path}: {e}")
        return 0, 1

def main():
    parser = argparse.ArgumentParser(description="Manage tickers in price_data.db")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Database path")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all tickers")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add ticker(s)")
    add_parser.add_argument("tickers", nargs="+", help="Ticker symbols to add")
    add_parser.add_argument("--source", default="manual", help="Data source label")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove ticker(s)")
    remove_parser.add_argument("tickers", nargs="+", help="Ticker symbols to remove")
    remove_parser.add_argument("--source", default=None, help="Only remove specific source")
    remove_parser.add_argument("--confirm", action="store_true", help="Actually delete (otherwise dry-run)")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import from CSV")
    import_parser.add_argument("csv_file", help="CSV file path")
    import_parser.add_argument("--source", default="csv_import", help="Data source label")
    
    args = parser.parse_args()
    
    if args.command == "list":
        df = get_all_tickers(args.db)
        if df.empty:
            print("No tickers found in database.")
        else:
            print(f"\n{len(df)} ticker entries found:\n")
            print(df.to_string(index=False))
    
    elif args.command == "add":
        added, skipped = bulk_add_tickers(args.tickers, args.db, args.source)
        print(f"\nSummary: {added} added, {skipped} skipped")
    
    elif args.command == "remove":
        if not args.confirm:
            print("\n⚠️  DRY RUN MODE - use --confirm to actually delete\n")
        total = bulk_remove_tickers(args.tickers, args.db, args.source, args.confirm)
        print(f"\nSummary: {total} rows deleted" if args.confirm else f" would be deleted")
    
    elif args.command == "import":
        inserted, errors = import_tickers_from_csv(args.csv_file, args.db, args.source)
        print(f"\nSummary: {inserted} imported, {errors} errors")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
