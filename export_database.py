"""
Export database to CSV files for sharing.
"""
import sqlite3
import pandas as pd
import os

def export_database(db_path="price_data.db", export_dir="database_export"):
    """Export all tables to CSV files."""
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    # Create export directory
    os.makedirs(export_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    # Export price_data (last 90 days only for smaller file)
    print("📤 Exporting price_data...")
    df = pd.read_sql_query("""
        SELECT * FROM price_data 
        WHERE date >= date('now', '-90 days')
        ORDER BY ticker, date
    """, conn)
    df.to_csv(f"{export_dir}/price_data_recent.csv", index=False)
    print(f"   ✓ Exported {len(df)} rows")
    
    # Export market_data
    print("📤 Exporting market_data...")
    df = pd.read_sql_query("SELECT * FROM market_data ORDER BY date", conn)
    df.to_csv(f"{export_dir}/market_data.csv", index=False)
    print(f"   ✓ Exported {len(df)} rows")
    
    # Export tcbs_scaling
    print("📤 Exporting tcbs_scaling...")
    df = pd.read_sql_query("SELECT * FROM tcbs_scaling", conn)
    df.to_csv(f"{export_dir}/tcbs_scaling.csv", index=False)
    print(f"   ✓ Exported {len(df)} rows")
    
    # Export ticker list
    print("📤 Exporting ticker list...")
    df = pd.read_sql_query("SELECT DISTINCT ticker FROM price_data ORDER BY ticker", conn)
    df.to_csv(f"{export_dir}/tickers.csv", index=False)
    print(f"   ✓ Exported {len(df)} tickers")
    
    conn.close()
    
    print(f"\n✅ Database exported to: {export_dir}/")
    print("📦 You can share these CSV files instead of the database")
    return True

def import_database(export_dir="database_export", db_path="price_data.db"):
    """Import CSV files back to database."""
    
    if os.path.exists(db_path):
        response = input(f"⚠️ {db_path} exists. Overwrite? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Import cancelled")
            return False
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    
    # Import price_data
    print("📥 Importing price_data...")
    df = pd.read_csv(f"{export_dir}/price_data_recent.csv")
    df.to_sql("price_data", conn, if_exists="append", index=False)
    print(f"   ✓ Imported {len(df)} rows")
    
    # Import market_data
    print("📥 Importing market_data...")
    df = pd.read_csv(f"{export_dir}/market_data.csv")
    df.to_sql("market_data", conn, if_exists="append", index=False)
    print(f"   ✓ Imported {len(df)} rows")
    
    # Import tcbs_scaling
    print("📥 Importing tcbs_scaling...")
    df = pd.read_csv(f"{export_dir}/tcbs_scaling.csv")
    df.to_sql("tcbs_scaling", conn, if_exists="append", index=False)
    print(f"   ✓ Imported {len(df)} rows")
    
    conn.close()
    
    print(f"\n✅ Database imported to: {db_path}")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "import":
        import_database()
    else:
        export_database()
