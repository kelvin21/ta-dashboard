import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "price_data.db")

# Threshold for detecting price discontinuity (e.g., 5% difference)
DISCONTINUITY_THRESHOLD = 0.05  # 5%

def detect_dividend_adjustment(ticker, db_path=DB_PATH, debug=False):
    """
    Detect if there's a dividend adjustment by comparing newly fetched TCBS price
    for yesterday with the existing database price for yesterday.
    
    The newly fetched TCBS data comes from today's fetch_and_scale() call.
    We compare yesterday's newly fetched price with yesterday's existing DB price.
    
    Returns: dict with {
        'needs_adjustment': bool,
        'adjustment_date': date string,
        'new_tcbs_price': float (newly fetched from today's call),
        'old_db_price': float (existing in DB),
        'adjustment_ratio': float,
        'affected_rows': int
    }
    """
    if not os.path.exists(db_path):
        return None
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get the most recent TCBS entry (just inserted from today's fetch)
    cur.execute("""
        SELECT date, close, updated_at FROM price_data 
        WHERE ticker = ? AND source = 'tcbs' AND close IS NOT NULL
        ORDER BY date DESC, updated_at DESC LIMIT 1
    """, (ticker,))
    latest_row = cur.fetchone()
    
    if not latest_row:
        conn.close()
        return None  # No TCBS data
    
    latest_date, latest_close, latest_updated = latest_row
    
    # Get the second most recent date (yesterday) from TCBS
    cur.execute("""
        SELECT date, close, updated_at FROM price_data 
        WHERE ticker = ? AND source = 'tcbs' AND close IS NOT NULL AND date < ?
        ORDER BY date DESC, updated_at DESC LIMIT 1
    """, (ticker, latest_date))
    yesterday_row = cur.fetchone()
    
    if not yesterday_row:
        conn.close()
        return None  # Not enough TCBS data
    
    yesterday_date, yesterday_close_new, yesterday_updated = yesterday_row
    
    if debug:
        print(f"[{ticker}] New TCBS yesterday ({yesterday_date}): {yesterday_close_new:.2f} (updated: {yesterday_updated})")
    
    # Now get the OLD price for yesterday from database (any source including old TCBS)
    # We want the price that existed BEFORE today's fetch
    # Strategy: Get the oldest entry for yesterday's date, OR get from non-TCBS sources
    
    # First try: Get non-TCBS price for yesterday
    cur.execute("""
        SELECT close, source, updated_at FROM price_data 
        WHERE ticker = ? AND date = ? AND source != 'tcbs' AND close IS NOT NULL
        ORDER BY updated_at DESC LIMIT 1
    """, (ticker, yesterday_date))
    old_row = cur.fetchone()
    
    # If no non-TCBS data, get the OLDEST TCBS entry for yesterday (before today's update)
    if not old_row:
        cur.execute("""
            SELECT close, source, updated_at FROM price_data 
            WHERE ticker = ? AND date = ? AND source = 'tcbs' AND close IS NOT NULL
            ORDER BY updated_at ASC LIMIT 1
        """, (ticker, yesterday_date))
        old_row = cur.fetchone()
    
    if not old_row:
        conn.close()
        return None  # No old comparison data
    
    yesterday_close_old, old_source, old_updated = old_row
    
    if debug:
        print(f"[{ticker}] Old DB yesterday ({yesterday_date}, {old_source}): {yesterday_close_old:.2f} (updated: {old_updated})")
    
    # Skip comparison if the old and new are from the same update (same updated_at timestamp)
    if old_source == 'tcbs' and yesterday_updated == old_updated:
        if debug:
            print(f"[{ticker}] Skipping - old and new TCBS data are from same update")
        conn.close()
        return {'needs_adjustment': False, 'ticker': ticker}
    
    # Calculate price difference
    price_diff = abs(yesterday_close_new - yesterday_close_old)
    price_diff_pct = price_diff / yesterday_close_old if yesterday_close_old > 0 else 0
    
    if debug:
        print(f"[{ticker}] Price difference: {price_diff:.2f} ({price_diff_pct*100:.2f}%)")
    
    # Check if discontinuity exceeds threshold
    if price_diff_pct > DISCONTINUITY_THRESHOLD:
        # Count rows that need adjustment (all dates before yesterday, excluding TCBS source)
        cur.execute("""
            SELECT COUNT(*) FROM price_data 
            WHERE ticker = ? AND date < ? AND source != 'tcbs'
        """, (ticker, yesterday_date))
        affected_rows = cur.fetchone()[0]
        
        # Calculate adjustment ratio
        adjustment_ratio = yesterday_close_new / yesterday_close_old
        
        conn.close()
        
        return {
            'needs_adjustment': True,
            'adjustment_date': yesterday_date,
            'new_tcbs_price': float(yesterday_close_new),
            'old_db_price': float(yesterday_close_old),
            'db_source': old_source,
            'adjustment_ratio': float(adjustment_ratio),
            'affected_rows': affected_rows,
            'ticker': ticker,
            'price_diff_pct': float(price_diff_pct * 100),
            'old_updated_at': old_updated,
            'new_updated_at': yesterday_updated
        }
    
    conn.close()
    return {'needs_adjustment': False, 'ticker': ticker}


def apply_dividend_adjustment(ticker, adjustment_ratio, adjustment_date, db_path=DB_PATH, dry_run=False):
    """
    Apply dividend adjustment to all historical prices before the adjustment date.
    Multiplies OHLC by adjustment_ratio for all rows where date < adjustment_date.
    
    Returns: number of rows adjusted
    """
    if not os.path.exists(db_path):
        return 0
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Count affected rows
    cur.execute("""
        SELECT COUNT(*) FROM price_data 
        WHERE ticker = ? AND date < ? AND source != 'tcbs'
    """, (ticker, adjustment_date))
    affected_count = cur.fetchone()[0]
    
    if affected_count == 0:
        conn.close()
        return 0
    
    if dry_run:
        print(f"[DRY RUN] Would adjust {affected_count} rows for {ticker} by ratio {adjustment_ratio:.6f}")
        conn.close()
        return affected_count
    
    # Apply adjustment
    update_sql = """
        UPDATE price_data
        SET open = CASE WHEN open IS NOT NULL THEN open * ? ELSE NULL END,
            high = CASE WHEN high IS NOT NULL THEN high * ? ELSE NULL END,
            low = CASE WHEN low IS NOT NULL THEN low * ? ELSE NULL END,
            close = CASE WHEN close IS NOT NULL THEN close * ? ELSE NULL END,
            updated_at = CURRENT_TIMESTAMP
        WHERE ticker = ? AND date < ? AND source != 'tcbs'
    """
    
    cur.execute(update_sql, (adjustment_ratio, adjustment_ratio, adjustment_ratio, adjustment_ratio, ticker, adjustment_date))
    rows_affected = cur.rowcount
    conn.commit()
    conn.close()
    
    print(f"✓ Adjusted {rows_affected} historical rows for {ticker} by ratio {adjustment_ratio:.6f}")
    return rows_affected


def scan_all_tickers_for_dividends(db_path=DB_PATH, debug=False):
    """
    Scan all tickers in database for potential dividend adjustments.
    Returns list of tickers that need adjustment.
    """
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get all tickers with TCBS data
    cur.execute("SELECT DISTINCT ticker FROM price_data WHERE source = 'tcbs'")
    tickers = [r[0] for r in cur.fetchall()]
    conn.close()
    
    print(f"Scanning {len(tickers)} tickers for dividend adjustments...")
    
    adjustments_needed = []
    for ticker in tickers:
        result = detect_dividend_adjustment(ticker, db_path=db_path, debug=debug)
        if result and result.get('needs_adjustment'):
            adjustments_needed.append(result)
            db_source = result.get('db_source', 'unknown')
            print(f"⚠️ {ticker}: Price discontinuity detected ({result['price_diff_pct']:.2f}%)")
            print(f"   New TCBS: {result['new_tcbs_price']:.2f} (updated: {result.get('new_updated_at', 'N/A')})")
            print(f"   Old DB ({db_source}): {result['old_db_price']:.2f} (updated: {result.get('old_updated_at', 'N/A')})")
            print(f"   Adjustment ratio: {result['adjustment_ratio']:.6f}")
            print(f"   Affected rows: {result['affected_rows']}")
    
    return adjustments_needed


def confirm_and_apply_adjustments(adjustments, db_path=DB_PATH, auto_confirm=False):
    """
    Present adjustments to user for confirmation and apply if confirmed.
    
    Args:
        adjustments: List of adjustment dicts from scan_all_tickers_for_dividends
        db_path: Database path
        auto_confirm: If True, skip confirmation prompt
    
    Returns: Number of tickers adjusted
    """
    if not adjustments:
        print("No dividend adjustments needed.")
        return 0
    
    print(f"\n{'='*80}")
    print(f"DIVIDEND ADJUSTMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Found {len(adjustments)} tickers requiring dividend adjustment:\n")
    
    for adj in adjustments:
        print(f"  {adj['ticker']:10s} | Date: {adj['adjustment_date']} | "
              f"Ratio: {adj['adjustment_ratio']:.6f} | "
              f"Rows: {adj['affected_rows']:4d} | "
              f"Diff: {adj['price_diff_pct']:+.2f}%")
    
    print(f"\n{'='*80}")
    
    if not auto_confirm:
        response = input("\nApply all adjustments? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Adjustment cancelled.")
            return 0
    
    print("\nApplying adjustments...")
    adjusted_count = 0
    
    for adj in adjustments:
        rows = apply_dividend_adjustment(
            adj['ticker'],
            adj['adjustment_ratio'],
            adj['adjustment_date'],
            db_path=db_path,
            dry_run=False
        )
        if rows > 0:
            adjusted_count += 1
    
    print(f"\n✅ Successfully adjusted {adjusted_count} tickers.")
    return adjusted_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect and apply dividend adjustments")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Database path")
    parser.add_argument("--ticker", type=str, help="Check specific ticker only")
    parser.add_argument("--scan", action="store_true", help="Scan all tickers")
    parser.add_argument("--apply", action="store_true", help="Apply adjustments after scanning")
    parser.add_argument("--auto-confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    
    args = parser.parse_args()
    
    if args.ticker:
        # Check single ticker
        result = detect_dividend_adjustment(args.ticker, db_path=args.db, debug=args.debug)
        if result and result.get('needs_adjustment'):
            print(f"\n⚠️ Dividend adjustment needed for {args.ticker}")
            print(f"  Adjustment date: {result['adjustment_date']}")
            print(f"  TCBS price: {result['tcbs_price']:.2f}")
            print(f"  DB price: {result['db_price']:.2f}")
            print(f"  Adjustment ratio: {result['adjustment_ratio']:.6f}")
            print(f"  Affected rows: {result['affected_rows']}")
            
            if args.apply:
                if not args.auto_confirm:
                    response = input("\nApply adjustment? (yes/no): ").strip().lower()
                    if response not in ['yes', 'y']:
                        print("Adjustment cancelled.")
                        exit(0)
                
                apply_dividend_adjustment(
                    args.ticker,
                    result['adjustment_ratio'],
                    result['adjustment_date'],
                    db_path=args.db
                )
        else:
            print(f"✓ No dividend adjustment needed for {args.ticker}")
    
    elif args.scan:
        # Scan all tickers
        adjustments = scan_all_tickers_for_dividends(db_path=args.db, debug=args.debug)
        
        if args.apply and adjustments:
            confirm_and_apply_adjustments(adjustments, db_path=args.db, auto_confirm=args.auto_confirm)
    
    else:
        parser.print_help()
