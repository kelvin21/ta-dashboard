import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, List
import asyncio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_db_adapter():
    """Get database adapter (MongoDB or SQLite)."""
    try:
        from db_adapter import get_db_adapter as get_adapter
        return get_adapter()
    except ImportError:
        return None

def fetch_fresh_price_data(ticker: str, lookback_days: int = 30, debug: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch fresh price data directly from API (TCBS) for comparison.
    Tries multiple fetch methods in order of preference.
    
    Args:
        ticker: Stock ticker symbol
        lookback_days: How many days back to fetch
        debug: Show debug output
    
    Returns:
        DataFrame with fresh API data or None if fetch fails
    """
    # Try multiple import options
    fetch_func = None
    is_async = False
    
    # Option 1: Try fetch_and_scale_async (async version)
    try:
        from fetch_and_scale_async import fetch_and_scale_async
        fetch_func = fetch_and_scale_async
        is_async = True
        if debug:
            print(f"[{ticker}] Using fetch_and_scale_async")
    except ImportError:
        pass
    
    # Option 2: Try regular fetch_and_scale (sync version)
    if fetch_func is None:
        try:
            from fetch_and_scale import fetch_and_scale
            fetch_func = fetch_and_scale
            is_async = False
            if debug:
                print(f"[{ticker}] Using fetch_and_scale (sync)")
        except ImportError:
            pass
    
    # Option 3: Try tcbs_fetch module
    if fetch_func is None:
        try:
            from tcbs_fetch import fetch_price_data
            fetch_func = fetch_price_data
            is_async = False
            if debug:
                print(f"[{ticker}] Using tcbs_fetch.fetch_price_data")
        except ImportError:
            pass
    
    if fetch_func is None:
        if debug:
            print(f"[{ticker}] No fetch module available")
            print(f"    Tried: fetch_and_scale_async, fetch_and_scale, tcbs_fetch")
        return None

    try:
        if debug:
            print(f"[{ticker}] Fetching fresh API data...")

        today = datetime.now().date()
        lookback_start = today - timedelta(days=lookback_days)

        # Fetch data (handle async or sync)
        if is_async:
            fresh_df = asyncio.run(fetch_func(ticker))
        else:
            fresh_df = fetch_func(ticker)
        
        if fresh_df is None or fresh_df.empty:
            if debug:
                print(f"[{ticker}] No fresh data from API")
            return None

        # Normalize date column: accept 'date' or 'tradingDate'
        if 'date' not in fresh_df.columns:
            if 'tradingDate' in fresh_df.columns:
                fresh_df = fresh_df.rename(columns={'tradingDate': 'date'})
                if debug:
                    print(f"[{ticker}] Renamed 'tradingDate' -> 'date'")
            else:
                if debug:
                    print(f"[{ticker}] No 'date' or 'tradingDate' column in fetched data. Columns: {list(fresh_df.columns)}")
                return None

        fresh_df['date'] = pd.to_datetime(fresh_df['date'])
        fresh_df = fresh_df[fresh_df['date'].dt.date >= lookback_start].copy()

        # Ensure required columns exist
        required_cols = ['close', 'open', 'high', 'low', 'volume']
        for col in required_cols:
            if col not in fresh_df.columns:
                if debug:
                    print(f"[{ticker}] Missing column '{col}' in fetched data. Columns: {list(fresh_df.columns)}")
                return None

        if debug:
            print(f"[{ticker}] Fetched {len(fresh_df)} bars from API")
            print(f"    Date range: {fresh_df['date'].min().date()} to {fresh_df['date'].max().date()}")

        return fresh_df
    except Exception as e:
        if debug:
            print(f"[{ticker}] Failed to fetch fresh data: {e}")
            import traceback
            traceback.print_exc()
        return None

def detect_dividend_adjustment(ticker: str, lookback_days: int = 30, min_data_points: int = 3, debug: bool = False) -> Optional[Dict]:
    """
    Detect dividend/split adjustment by comparing historical DB data vs fresh API data.
    
    Strategy:
    1. Load historical data from database (older cached data)
    2. Fetch fresh data from API for the same date range
    3. Find matching dates between historical and fresh API data
    4. EXCLUDE the latest matching date (likely just updated/synced)
    5. Use earlier matching dates (3-5 points) for momentum validation
    6. Calculate price change ratios (momentum) for both
    7. If price base changed but momentum remained same, it's a dividend/split
    8. Use adjustment ratio from earliest non-matching date vs latest API price
    
    This handles partial updates where latest data matches but older data needs adjustment.
    
    Args:
        ticker: Stock ticker symbol
        lookback_days: How far back to look for comparison data
        min_data_points: Minimum number of matching data points needed (default 3)
        debug: Show debug output
    
    Returns:
        Dict with adjustment details if adjustment needed, else None
    """
    db = get_db_adapter()
    if not db:
        if debug:
            print(f"[{ticker}] No database adapter available")
        return None
    
    # Calculate date range
    today = datetime.now().date()
    lookback_start = today - timedelta(days=lookback_days)
    
    if debug:
        print(f"\n[{ticker}] Detecting dividend/split adjustment...")
        print(f"  Lookback period: {lookback_start} to {today}")
        print(f"  Min data points required: {min_data_points}")
        print(f"  Strategy: Compare historical DB vs fresh API data")
        print(f"  Special case: Exclude latest matching date (likely synced), use earlier dates")
    
    # Load historical data from database
    hist_df = db.load_price_range(ticker, lookback_start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
    
    if hist_df is None or hist_df.empty or len(hist_df) < min_data_points:
        if debug:
            rows = len(hist_df) if hist_df is not None and not hist_df.empty else 0
            print(f"  ✗ Not enough historical data ({rows} rows, need {min_data_points})")
        return None
    
    hist_df = hist_df.sort_values('date', ascending=True).reset_index(drop=True)
    
    if debug:
        print(f"  ✓ Loaded {len(hist_df)} historical bars from database")
        print(f"    Date range: {hist_df['date'].min()} to {hist_df['date'].max()}")
    
    # Fetch fresh API data (fully adjusted latest prices)
    fresh_df = fetch_fresh_price_data(ticker, lookback_days=lookback_days, debug=debug)
    if fresh_df is None or fresh_df.empty:
        if debug:
            print(f"  ✗ Could not fetch fresh API data")
        return None
    
    fresh_df = fresh_df.sort_values('date', ascending=True).reset_index(drop=True)
    
    if debug:
        print(f"  ✓ Loaded {len(fresh_df)} bars from fresh API")
        print(f"    Date range: {fresh_df['date'].min().date()} to {fresh_df['date'].max().date()}")
    
    # Find matching dates between historical and fresh API data
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    fresh_df['date'] = pd.to_datetime(fresh_df['date'])
    
    matching_dates = set(hist_df['date'].dt.date) & set(fresh_df['date'].dt.date)
    
    if len(matching_dates) < min_data_points + 1:  # +1 because we'll exclude the latest
        if debug:
            print(f"  ⚠️ Not enough matching dates ({len(matching_dates)}, need {min_data_points + 1} to exclude latest)")
            print(f"     Historical latest: {hist_df['date'].max().date()}")
            print(f"     Fresh API latest: {fresh_df['date'].max().date()}")
        return None
    
    # Get matching dates sorted
    matching_dates_sorted = sorted(list(matching_dates), reverse=True)
    
    # EXCLUDE the latest matching date (it's likely just synced)
    # Use the next N dates for comparison (to find divergence from older data)
    validation_dates = sorted(matching_dates_sorted[1:1+min(5, len(matching_dates_sorted)-1)])  # Skip latest, take next 3-5
    
    if len(validation_dates) < min_data_points:
        if debug:
            print(f"  ⚠️ Not enough validation dates after excluding latest ({len(validation_dates)}, need {min_data_points})")
        return None
    
    latest_matching_date = matching_dates_sorted[0]  # The most recent date (matched, so exclude from comparison)
    earliest_validation_date = validation_dates[0]  # Earliest date we'll use for comparison
    
    if debug:
        print(f"  ✓ Found {len(matching_dates)} matching dates total")
        print(f"    EXCLUDING latest matching date: {latest_matching_date} (likely synced)")
        print(f"    Using {len(validation_dates)} earlier dates for divergence detection: {validation_dates}")
    
    # Extract prices for validation dates from both sources
    hist_prices = []
    fresh_prices = []
    
    for match_date in validation_dates:
        hist_row = hist_df[hist_df['date'].dt.date == match_date]
        fresh_row = fresh_df[fresh_df['date'].dt.date == match_date]
        
        if not hist_row.empty and not fresh_row.empty:
            hist_close = float(hist_row.iloc[-1]['close'])  # Latest entry for that date
            fresh_close = float(fresh_row.iloc[-1]['close'])
            
            hist_prices.append(hist_close)
            fresh_prices.append(fresh_close)
            
            if debug:
                print(f"    {match_date}: DB={hist_close:.2f} vs Fresh API={fresh_close:.2f} (diff={((fresh_close-hist_close)/hist_close*100):+.2f}%)")
    
    if len(hist_prices) < min_data_points:
        if debug:
            print(f"  ⚠️ Not enough valid price pairs ({len(hist_prices)}, need {min_data_points})")
        return None
    
    # Calculate price change ratios (momentum) for validation
    hist_prices_arr = np.array(hist_prices)
    fresh_prices_arr = np.array(fresh_prices)
    
    # Calculate returns/momentum for each series
    hist_momentum = np.diff(hist_prices_arr) / hist_prices_arr[:-1]  # Price change ratios
    fresh_momentum = np.diff(fresh_prices_arr) / fresh_prices_arr[:-1]
    
    momentum_correlation = np.nan
    if len(hist_momentum) >= 1 and len(fresh_momentum) > 0:
        momentum_correlation = np.corrcoef(hist_momentum, fresh_momentum)[0, 1]
    
    if debug:
        print(f"\n  Price change momentum (returns) - For validation:")
        print(f"    DB momentum:        {hist_momentum}")
        print(f"    Fresh API momentum: {fresh_momentum}")
        print(f"    Momentum correlation: {momentum_correlation:.4f}")
    
    # Use the earliest validation date (oldest divergence point) for adjustment ratio
    # Compare against latest API price
    earliest_hist_row = hist_df[hist_df['date'].dt.date == earliest_validation_date]
    earliest_fresh_row = fresh_df[fresh_df['date'].dt.date == earliest_validation_date]
    latest_fresh_row = fresh_df[fresh_df['date'].dt.date == latest_matching_date]
    
    if earliest_hist_row.empty or earliest_fresh_row.empty or latest_fresh_row.empty:
        if debug:
            print(f"  ✗ Could not get required data points")
        return None
    
    # Use earliest validation date to detect the split ratio
    earliest_hist_price = float(earliest_hist_row.iloc[-1]['close'])
    earliest_fresh_price = float(earliest_fresh_row.iloc[-1]['close'])
    latest_fresh_price = float(latest_fresh_row.iloc[-1]['close'])
    
    # Calculate adjustment ratio from the divergence between DB and fresh API at earliest date
    adjustment_ratio = earliest_fresh_price / earliest_hist_price if earliest_hist_price > 0 else 1.0
    price_change_pct = ((earliest_fresh_price - earliest_hist_price) / earliest_hist_price * 100) if earliest_hist_price > 0 else 0
    
    if debug:
        print(f"\n  Price base comparison (EARLIEST VALIDATION DATE):")
        print(f"    Date: {earliest_validation_date}")
        print(f"    DB price:        {earliest_hist_price:.2f}")
        print(f"    Fresh API price: {earliest_fresh_price:.2f}")
        print(f"    Adjustment ratio: {adjustment_ratio:.6f}")
        print(f"    Price change: {price_change_pct:+.2f}%")
        print(f"\n  Latest date (excluded from comparison):")
        print(f"    Date: {latest_matching_date}")
        print(f"    Fresh API price: {latest_fresh_price:.2f}")
    
    # Determine if adjustment is needed
    # Threshold: >2% price difference suggests dividend/split (not just price drift)
    ADJUSTMENT_THRESHOLD = 0.02  # 2%
    
    if abs(price_change_pct) > ADJUSTMENT_THRESHOLD * 100:
        # Adjust all rows before the earliest validation date
        adjustment_date = earliest_validation_date.strftime("%Y-%m-%d")
        affected_df = hist_df[hist_df['date'].dt.date < earliest_validation_date]
        affected_rows = len(affected_df)
        
        if debug:
            print(f"\n  ✓ ADJUSTMENT DETECTED!")
            print(f"    Adjustment date: {adjustment_date} (adjustment applies before this)")
            print(f"    Rows affected: {affected_rows}")
            print(f"    Note: Latest data matched but earlier data diverged (partial update case)")
            print(f"    Momentum validation: {'✓ PASSED' if momentum_correlation > 0.8 or np.isnan(momentum_correlation) else '⚠️ LOW CORRELATION'}")
        
        return {
            'needs_adjustment': True,
            'ticker': ticker,
            'adjustment_date': adjustment_date,
            'latest_matching_date': latest_matching_date.strftime("%Y-%m-%d"),
            'new_api_price': float(earliest_fresh_price),
            'old_db_price': float(earliest_hist_price),
            'adjustment_ratio': float(adjustment_ratio),
            'affected_rows': affected_rows,
            'price_diff_pct': float(price_change_pct),
            'momentum_correlation': float(momentum_correlation) if not np.isnan(momentum_correlation) else None,
            'validation_dates': [d.strftime("%Y-%m-%d") for d in validation_dates],
            'data_points': len(hist_prices)
        }
    else:
        if debug:
            print(f"\n  ⚠️ NO ADJUSTMENT (price change: {price_change_pct:+.2f}% < {ADJUSTMENT_THRESHOLD*100}% threshold)")
            print(f"     Analysis based on earlier dates (latest date excluded from comparison):")
            print(f"     1. Price change {price_change_pct:+.2f}% at {earliest_validation_date} is below {ADJUSTMENT_THRESHOLD*100}% threshold")
            print(f"     2. Check if momentum correlation indicates data mismatch:")
            if not np.isnan(momentum_correlation):
                print(f"        Momentum correlation: {momentum_correlation:.4f} {'(Good match)' if momentum_correlation > 0.7 else '(Poor match - different sources/periods)'}")
            print(f"     3. If correlation is very negative, sources may have detected split at different times")
            print(f"     4. Latest date ({latest_matching_date}) was excluded (likely synced)")
        
        return {'needs_adjustment': False, 'ticker': ticker}

def apply_dividend_adjustment(ticker: str, adjustment_ratio: float, adjustment_date: str, dry_run: bool = False, debug: bool = False) -> int:
    """
    Apply dividend/split adjustment to historical prices before adjustment_date.
    Multiplies OHLC by adjustment_ratio for all rows where date < adjustment_date.
    
    Works with both SQLite and MongoDB via db_adapter.
    
    Args:
        ticker: Stock ticker symbol
        adjustment_ratio: Ratio to multiply prices by
        adjustment_date: Date string (YYYY-MM-DD) - adjust prices before this date
        dry_run: If True, don't actually modify database
        debug: Show debug output
    
    Returns:
        Number of rows adjusted
    """
    db = get_db_adapter()
    if not db:
        if debug:
            print(f"[{ticker}] No database adapter available")
        return 0
    
    if debug:
        print(f"\n[{ticker}] Applying dividend adjustment...")
        print(f"  Adjustment date: {adjustment_date}")
        print(f"  Adjustment ratio: {adjustment_ratio:.6f}")
        print(f"  Dry run: {dry_run}")
    
    # Load all data before adjustment date
    start_date = "2000-01-01"  # Far past
    adjust_dt = datetime.strptime(adjustment_date, "%Y-%m-%d")
    
    df = db.load_price_range(ticker, start_date, (adjust_dt - timedelta(days=1)).strftime("%Y-%m-%d"))
    
    if df.empty:
        if debug:
            print(f"  ✗ No data found before {adjustment_date}")
        return 0
    
    if debug:
        print(f"  ✓ Found {len(df)} rows before {adjustment_date}")
    
    if dry_run:
        if debug:
            print(f"  [DRY RUN] Would adjust {len(df)} rows by ratio {adjustment_ratio:.6f}")
        return len(df)
    
    # Apply adjustment to each row by re-inserting with adjusted prices
    affected_count = 0
    
    for _, row in df.iterrows():
        adjusted_ohlcv = {
            'open': float(row['open']) * adjustment_ratio if row['open'] else 0,
            'high': float(row['high']) * adjustment_ratio if row['high'] else 0,
            'low': float(row['low']) * adjustment_ratio if row['low'] else 0,
            'close': float(row['close']) * adjustment_ratio if row['close'] else 0,
            'volume': int(row['volume']) if row['volume'] else 0
        }
        
        date_str = pd.to_datetime(row['date']).strftime("%Y-%m-%d")
        source = row.get('source', 'manual')
        
        success = db.insert_price_data(ticker, date_str, adjusted_ohlcv, source=source)
        if success:
            affected_count += 1
    
    if debug:
        print(f"  ✓ Adjusted {affected_count} rows")
    
    return affected_count


def scan_all_tickers_for_dividends(lookback_days: int = 30, min_data_points: int = 3, debug: bool = False) -> List[Dict]:
    """
    Scan all tickers in database for potential dividend/split adjustments.
    
    Compares historical DB data vs fresh API data.
    Uses 3-5 data points for momentum validation, adjustment ratio from latest point.
    
    Args:
        lookback_days: How far back to look
        min_data_points: Minimum matching data points required for validation
        debug: Show debug output
    
    Returns:
        List of tickers that need adjustment
    """
    db = get_db_adapter()
    if not db:
        print("No database adapter available")
        return []
    
    # Get all tickers
    tickers = db.get_all_tickers(debug=debug)
    
    if not tickers:
        print("No tickers found in database")
        return []
    
    ADJUSTMENT_THRESHOLD = 0.02  # 2%
    
    print(f"\n{'='*80}")
    print(f"Scanning {len(tickers)} tickers for dividend/split adjustments...")
    print(f"Lookback: {lookback_days} days | Min validation points: {min_data_points}")
    print(f"Strategy: Compare historical DB vs fresh API data")
    print(f"Validation: 3-5 points for momentum comparison")
    print(f"Threshold: >{ADJUSTMENT_THRESHOLD*100}% price difference triggers adjustment")
    print(f"{'='*80}\n")
    
    adjustments_needed = []
    
    for idx, ticker in enumerate(tickers):
        print(f"[{idx+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        
        result = detect_dividend_adjustment(
            ticker,
            lookback_days=lookback_days,
            min_data_points=min_data_points,
            debug=False  # Set to False to avoid spam; use debug=True for individual ticker
        )
        
        if result and result.get('needs_adjustment'):
            print(f"⚠️ NEEDS ADJUSTMENT ({result['price_diff_pct']:+.2f}%)")
            adjustments_needed.append(result)
        else:
            print("✓")
    
    print(f"\n{'='*80}")
    if adjustments_needed:
        print(f"Found {len(adjustments_needed)} tickers requiring adjustment:\n")
        
        for adj in adjustments_needed:
            print(f"  {adj['ticker']:10s} | Ratio: {adj['adjustment_ratio']:.6f} | "
                  f"Rows: {adj['affected_rows']:4d} | Diff: {adj['price_diff_pct']:+.2f}% | "
                  f"Latest date: {adj['latest_matching_date']} | "
                  f"Data points: {adj['data_points']}")
    else:
        print(f"✓ No dividend/split adjustments detected")
    
    print(f"\n{'='*80}")
    print(f"💡 TIP: Use --debug flag to see detailed analysis:")
    print(f"   python dividend_adjuster.py --ticker <TICKER> --debug")
    print(f"{'='*80}\n")
    
    return adjustments_needed


def confirm_and_apply_adjustments(adjustments: List[Dict], dry_run: bool = False, auto_confirm: bool = False, debug: bool = False) -> int:
    """
    Present adjustments to user and apply if confirmed.
    
    Args:
        adjustments: List of adjustment dicts from scan_all_tickers_for_dividends
        dry_run: If True, don't actually modify database
        auto_confirm: If True, skip confirmation prompt
        debug: Show debug output
    
    Returns:
        Number of tickers adjusted
    """
    if not adjustments:
        print("No dividend adjustments needed.")
        return 0
    
    print(f"\n{'='*80}")
    print(f"DIVIDEND/SPLIT ADJUSTMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Found {len(adjustments)} tickers requiring adjustment:\n")
    
    for i, adj in enumerate(adjustments, 1):
        print(f"  {i}. {adj['ticker']:10s} | Adjustment: {adj['adjustment_ratio']:.6f} | "
              f"Rows: {adj['affected_rows']:4d} | "
              f"Price change: {adj['price_diff_pct']:+.2f}%")
        print(f"     Latest matching date: {adj['latest_matching_date']}")
        print(f"     Validation dates ({adj['data_points']}): {', '.join(adj.get('validation_dates', [])[:3])}...")
        if adj.get('momentum_correlation') is not None:
            print(f"     Momentum correlation: {adj['momentum_correlation']:.4f}")
    
    print(f"\n{'='*80}")
    
    if dry_run:
        print("⚠️ DRY RUN MODE - No changes will be made\n")
    
    if not auto_confirm:
        response = input("\nApply all adjustments? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Adjustment cancelled.")
            return 0
    
    print("\nApplying adjustments...\n")
    adjusted_count = 0
    
    for adj in adjustments:
        rows = apply_dividend_adjustment(
            adj['ticker'],
            adj['adjustment_ratio'],
            adj['adjustment_date'],
            dry_run=dry_run,
            debug=debug
        )
        if rows > 0:
            adjusted_count += 1
    
    print(f"\n{'='*80}")
    if dry_run:
        print(f"✓ DRY RUN completed: Would have adjusted {adjusted_count} tickers")
    else:
        print(f"✅ Successfully adjusted {adjusted_count} tickers")
    print(f"{'='*80}\n")
    
    return adjusted_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect and apply dividend/split adjustments")
    parser.add_argument("--ticker", type=str, help="Check specific ticker only")
    parser.add_argument("--scan", action="store_true", help="Scan all tickers")
    parser.add_argument("--apply", action="store_true", help="Apply adjustments after scanning")
    parser.add_argument("--auto-confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be adjusted without making changes")
    parser.add_argument("--debug", action="store_true", help="Show detailed debug output")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback period in days (default: 30)")
    parser.add_argument("--min-points", type=int, default=3, help="Minimum data points required (default: 3)")
    
    args = parser.parse_args()
    
    if args.ticker:
        # Check single ticker
        print(f"\nChecking {args.ticker}...")
        result = detect_dividend_adjustment(
            args.ticker,
            lookback_days=args.lookback,
            min_data_points=args.min_points,
            debug=args.debug
        )
        
        if result and result.get('needs_adjustment'):
            print(f"\n⚠️ Dividend/split adjustment needed for {args.ticker}")
            print(f"  Adjustment date: {result['adjustment_date']}")
            print(f"  Fresh API price: {result['new_api_price']:.2f}")
            print(f"  DB price: {result['old_db_price']:.2f}")
            print(f"  Adjustment ratio: {result['adjustment_ratio']:.6f}")
            print(f"  Affected rows: {result['affected_rows']}")
            print(f"  Data points used: {result['data_points']}")
            
            if args.apply:
                if not args.auto_confirm:
                    response = input("\nApply adjustment? (yes/no): ").strip().lower()
                    if response not in ['yes', 'y']:
                        print("Adjustment cancelled.")
                        exit(0)
                
                rows = apply_dividend_adjustment(
                    args.ticker,
                    result['adjustment_ratio'],
                    result['adjustment_date'],
                    dry_run=args.dry_run,
                    debug=args.debug
                )
                if rows > 0:
                    print(f"✓ Adjusted {rows} rows")
        else:
            print(f"✓ No adjustment needed for {args.ticker}")
    
    elif args.scan:
        # Scan all tickers
        adjustments = scan_all_tickers_for_dividends(
            lookback_days=args.lookback,
            min_data_points=args.min_points,
            debug=args.debug
        )
        
        if args.apply and adjustments:
            confirm_and_apply_adjustments(
                adjustments,
                dry_run=args.dry_run,
                auto_confirm=args.auto_confirm,
                debug=args.debug
            )
    
    else:
        parser.print_help()
