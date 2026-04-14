"""
Dividend/Split Adjuster — detects tickers needing re-import from AmiBroker.

Primary method: Scrape recent corporate events (dividends, stock splits, bonus shares)
from Vietstock and cross-reference with tickers in the database. Tickers whose ex-date
falls after their last DB update need a full re-import from AmiBroker.
"""
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Vietstock pages
VIETSTOCK_PAGE_URL = "https://finance.vietstock.vn/lich-su-kien.htm?page=1&tab=1"
VIETSTOCK_API_URL = "/data/eventstypedata"  # relative, called via browser fetch()

# Event types that affect price (require re-import)
PRICE_AFFECTING_EVENTS = {
    "Trả cổ tức bằng tiền mặt",           # Cash dividend
    "Trả cổ tức bằng cổ phiếu",           # Stock dividend
    "Thưởng cổ phiếu",                     # Bonus shares
    "Phát hành thêm",                      # Rights issue
    "Phát hành trái phiếu chuyển đổi",     # Convertible bond issuance
    "Hoán đổi cổ phiếu",                   # Stock swap/exchange
}


def get_db_adapter():
    """Get database adapter (MongoDB or SQLite)."""
    try:
        from db_adapter import get_db_adapter as get_adapter
        return get_adapter()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Vietstock scraper (primary method)
# ---------------------------------------------------------------------------

def fetch_vietstock_events(from_date: str = None, to_date: str = None,
                           lookback_days: int = 60, debug: bool = False) -> List[Dict]:
    """
    Fetch corporate events from Vietstock via its JSON API.

    Uses Playwright to load the page once (for CSRF token and cookies),
    then calls the internal API with a date range and pageSize=50.

    Args:
        from_date: Start date YYYY-MM-DD (default: today - lookback_days)
        to_date:   End date YYYY-MM-DD (default: today + 30 days)
        lookback_days: Days before today if from_date not given
        debug: Show debug output

    Returns:
        List of event dicts with keys: ticker, exchange, ex_date,
        ex_date_parsed, ex_date_dt, content, event_type
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium")
        return []

    now = datetime.now()
    if not from_date:
        from_date = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = (now + timedelta(days=30)).strftime("%Y-%m-%d")

    if debug:
        print(f"  Date range: {from_date} -> {to_date}")

    all_events = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load page once to get CSRF token and session cookies
        try:
            page.goto(VIETSTOCK_PAGE_URL, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_selector(
                'input[name="__RequestVerificationToken"]',
                state="attached", timeout=15000,
            )
        except Exception as e:
            if debug:
                print(f"  Page load failed: {e}")
            browser.close()
            return []

        # Fetch events via internal JSON API (pageSize=50)
        pg = 1
        while True:
            if debug:
                print(f"  API page {pg}...")

            result = page.evaluate("""
                async ({fDate, tDate, pg}) => {
                    const token = document.querySelector(
                        'input[name="__RequestVerificationToken"]'
                    )?.value;
                    if (!token) return {error: 'no_token'};

                    const params = new URLSearchParams({
                        eventTypeID: '1', channelID: '0', code: '',
                        catID: '-1',
                        fDate: fDate, tDate: tDate,
                        page: String(pg), pageSize: '50',
                        orderBy: 'Date1', orderDir: 'DESC',
                        __RequestVerificationToken: token,
                    });

                    const resp = await fetch('/data/eventstypedata', {
                        method: 'POST',
                        headers: {
                            'Content-Type':
                                'application/x-www-form-urlencoded; charset=UTF-8',
                            'X-Requested-With': 'XMLHttpRequest',
                        },
                        body: params.toString(),
                    });
                    if (!resp.ok) return {error: resp.status};
                    return await resp.json();
                }
            """, {"fDate": from_date, "tDate": to_date, "pg": pg})

            if isinstance(result, dict) and result.get('error'):
                if debug:
                    print(f"  API error: {result['error']}")
                break

            # Response: [[events...], [totalCount]]
            rows = result[0] if isinstance(result, list) and len(result) > 0 else []
            total = result[1][0] if isinstance(result, list) and len(result) > 1 else 0

            if not rows:
                break

            for r in rows:
                # GDKHQDate is "/Date(ms)/" format
                ms = 0
                gdkhq = r.get('GDKHQDate', '')
                if gdkhq:
                    import re as _re
                    m = _re.search(r'(\d+)', gdkhq)
                    if m:
                        ms = int(m.group(1))
                ex_dt = datetime.fromtimestamp(ms / 1000) if ms else None

                ex_name = (r.get('Exchange') or '').upper()
                if 'UPCOM' in ex_name or 'UPC' in ex_name:
                    exchange = 'UPCOM'
                elif 'HNX' in ex_name:
                    exchange = 'HNX'
                else:
                    exchange = 'HOSE'

                all_events.append({
                    'ticker': r.get('Code', ''),
                    'exchange': exchange,
                    'ex_date': ex_dt.strftime("%d/%m/%Y") if ex_dt else '',
                    'ex_date_parsed': ex_dt.strftime("%Y-%m-%d") if ex_dt else None,
                    'ex_date_dt': ex_dt,
                    'content': r.get('Note', ''),
                    'event_type': r.get('Name', ''),
                })

            if debug:
                print(f"  API page {pg}: {len(rows)} events (total {total})")

            if len(all_events) >= total:
                break
            pg += 1

        browser.close()

    return all_events


def find_tickers_needing_reimport(events: List[Dict],
                                   last_adjusted: str = None,
                                   debug: bool = False) -> List[Dict]:
    """
    Cross-reference Vietstock events with DB tickers to find those needing re-import.

    A ticker needs re-import if:
    1. It exists in our DB
    2. It had a price-affecting event (dividend, stock split, bonus shares)
    3. Either:
       a) last_adjusted is given and ex_date >= last_adjusted
          (flag everything since the last known good adjusted import)
       b) ex_date >= last_db_date - 5 days
          (DB may not yet reflect the adjustment)

    Args:
        events: Vietstock events from fetch_vietstock_events()
        last_adjusted: YYYY-MM-DD — the date of the last known good adjusted
                       import. All events on or after this date are flagged.
        debug: Show debug output
    """
    db = get_db_adapter()
    if not db:
        print("No database adapter available")
        return []

    db_tickers = set(db.get_all_tickers())
    if not db_tickers:
        print("No tickers found in database")
        return []

    last_adj_dt = None
    if last_adjusted:
        last_adj_dt = pd.Timestamp(last_adjusted)
        if debug:
            print(f"  Last adjusted date: {last_adjusted} (flag all events on/after)")

    results = []
    seen_tickers = set()

    for evt in events:
        ticker = evt.get('ticker', '')
        event_type = evt.get('event_type', '')
        ex_dt = evt.get('ex_date_dt')
        ex_date = evt.get('ex_date_parsed')

        if event_type not in PRICE_AFFECTING_EVENTS:
            continue
        if not ex_dt:
            continue
        if ticker not in db_tickers:
            if debug:
                print(f"  {ticker}: not in DB, skipping")
            continue
        if ticker in seen_tickers:
            continue

        df = db.load_price_range(ticker, "2020-01-01", datetime.now().strftime("%Y-%m-%d"))
        if df is None or df.empty:
            continue

        df['date'] = pd.to_datetime(df['date'])
        last_db_date = df['date'].max()

        ex_dt_ts = pd.Timestamp(ex_dt)

        if last_adj_dt is not None:
            # Flag everything since the last adjusted import date
            needs_reimport = ex_dt_ts >= last_adj_dt
        else:
            # Default: flag if ex-date is near/after DB last date
            needs_reimport = ex_dt_ts >= last_db_date - timedelta(days=5)

        if needs_reimport:
            seen_tickers.add(ticker)
            results.append({
                'ticker': ticker,
                'exchange': evt.get('exchange', ''),
                'ex_date': ex_date,
                'event_type': event_type,
                'content': evt.get('content', ''),
                'last_db_date': last_db_date.strftime("%Y-%m-%d"),
                'recommendation': 'full_reimport',
            })
            if debug:
                print(f"  {ticker}: {event_type} on {ex_date} | "
                      f"last DB: {last_db_date.strftime('%Y-%m-%d')} -> NEEDS REIMPORT")
        elif debug:
            print(f"  {ticker}: {event_type} on {ex_date} | "
                  f"last DB: {last_db_date.strftime('%Y-%m-%d')} -> OK (DB updated after ex-date)")

    return results


def scan_vietstock(from_date: str = None, to_date: str = None,
                   lookback_days: int = 60, last_adjusted: str = None,
                   debug: bool = False) -> List[Dict]:
    """
    Full pipeline: fetch Vietstock events for date range and find DB tickers
    needing re-import.

    Args:
        from_date: Start date YYYY-MM-DD (default: today - lookback_days)
        to_date:   End date YYYY-MM-DD (default: today + 30 days)
        lookback_days: Days before today if from_date not given
        last_adjusted: YYYY-MM-DD — last known adjusted import date.
                       If set, all events on/after this date are flagged.
        debug: Show debug output
    """
    now = datetime.now()
    if not from_date:
        from_date = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = (now + timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"\n{'=' * 80}")
    print(f"Scanning Vietstock corporate events for dividend/split adjustments")
    print(f"Date range: {from_date} -> {to_date}")
    if last_adjusted:
        print(f"Last adjusted import: {last_adjusted} (flag all events on/after)")
    print(f"{'=' * 80}\n")

    print("Fetching events from Vietstock...")
    events = fetch_vietstock_events(
        from_date=from_date, to_date=to_date,
        lookback_days=lookback_days, debug=debug,
    )
    if not events:
        print("No events fetched. Check internet connection or Vietstock availability.")
        return []

    price_events = [e for e in events if e.get('event_type') in PRICE_AFFECTING_EVENTS]
    print(f"  Total events: {len(events)} | Price-affecting: {len(price_events)}")

    print("\nCross-referencing with database...")
    results = find_tickers_needing_reimport(events, last_adjusted=last_adjusted, debug=debug)

    print(f"\n{'=' * 80}")
    if results:
        print(f"Found {len(results)} tickers needing re-import:\n")
        for r in results:
            print(f"  {r['ticker']:8s} | {r['exchange']:6s} | Ex-date: {r['ex_date']} | "
                  f"DB last: {r['last_db_date']} | {r['event_type']}")
            print(f"           {r['content']}")
        tickers_str = ','.join(r['ticker'] for r in results)
        print(f"\n  Recommendation: re-import from AmiBroker")
        print(f"   python ami_project_to_db.py --tickers {tickers_str} --from-date 2018-01-01")
    else:
        print("All DB tickers are up to date. No re-import needed.")
    print(f"{'=' * 80}\n")

    return results


# ---------------------------------------------------------------------------
# Legacy/compatibility wrappers
# ---------------------------------------------------------------------------

def detect_dividend_adjustment(ticker: str, lookback_days: int = 30, min_data_points: int = 3,
                                debug: bool = False) -> Optional[Dict]:
    """Legacy wrapper — checks if ticker has recent Vietstock events."""
    try:
        events = fetch_vietstock_events(lookback_days=lookback_days, debug=False)
        results = find_tickers_needing_reimport(events, debug=debug)
        for r in results:
            if r['ticker'] == ticker:
                return {
                    'needs_adjustment': True,
                    'ticker': ticker,
                    'adjustment_date': r['ex_date'],
                    'latest_matching_date': r['last_db_date'],
                    'new_api_price': 0,
                    'old_db_price': 0,
                    'adjustment_ratio': 1.0,
                    'affected_rows': 0,
                    'price_diff_pct': 0,
                    'source': 'vietstock',
                    'recommendation': 'full_reimport',
                    'event_type': r['event_type'],
                    'content': r['content'],
                    'data_points': 0,
                }
    except Exception:
        pass
    return {'needs_adjustment': False, 'ticker': ticker}


def scan_all_tickers_for_dividends(lookback_days: int = 30, min_data_points: int = 3,
                                    debug: bool = False, **kwargs) -> List[Dict]:
    """Legacy wrapper — scans via Vietstock."""
    results = scan_vietstock(lookback_days=lookback_days, debug=debug)
    legacy = []
    for r in results:
        legacy.append({
            'needs_adjustment': True,
            'ticker': r['ticker'],
            'adjustment_date': r['ex_date'],
            'latest_matching_date': r['last_db_date'],
            'adjustment_ratio': 1.0,
            'affected_rows': 0,
            'price_diff_pct': 0,
            'source': 'vietstock',
            'recommendation': 'full_reimport',
            'event_type': r['event_type'],
            'content': r['content'],
            'data_points': 0,
        })
    return legacy


# ---------------------------------------------------------------------------
# Apply adjustment & reimport helpers
# ---------------------------------------------------------------------------

def apply_dividend_adjustment(ticker: str, adjustment_ratio: float, adjustment_date: str,
                               dry_run: bool = False, debug: bool = False) -> int:
    """
    Apply dividend/split adjustment to historical prices before adjustment_date.
    Multiplies OHLC by adjustment_ratio for all rows where date < adjustment_date.
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

    start_date = "2000-01-01"
    adjust_dt = datetime.strptime(adjustment_date, "%Y-%m-%d")

    df = db.load_price_range(ticker, start_date, (adjust_dt - timedelta(days=1)).strftime("%Y-%m-%d"))

    if df is None or df.empty:
        if debug:
            print(f"  No data found before {adjustment_date}")
        return 0

    if debug:
        print(f"  Found {len(df)} rows before {adjustment_date}")

    if dry_run:
        if debug:
            print(f"  [DRY RUN] Would adjust {len(df)} rows by ratio {adjustment_ratio:.6f}")
        return len(df)

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
        print(f"  Adjusted {affected_count} rows")

    return affected_count


def reimport_from_amibroker(tickers: List[str], from_date: str = "2018-01-01",
                             auto_confirm: bool = False, debug: bool = False) -> int:
    """Re-import tickers from AmiBroker project to MongoDB."""
    if not tickers:
        print("No tickers to reimport.")
        return 0

    print(f"\n{'=' * 80}")
    print(f"REIMPORT FROM AMIBROKER")
    print(f"{'=' * 80}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"From date: {from_date}")
    print(f"{'=' * 80}\n")

    if not auto_confirm:
        response = input("Proceed with reimport? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Reimport cancelled.")
            return 0

    try:
        from ami_project_to_db import AmiProjectToMongo
    except ImportError:
        print("ERROR: ami_project_to_db not available. Run manually:")
        print(f"  python ami_project_to_db.py --tickers {','.join(tickers)} --from-date {from_date}")
        return 0

    loader = AmiProjectToMongo()
    success_count = 0

    for ticker in tickers:
        print(f"  Reimporting {ticker}...", end=" ", flush=True)
        try:
            loader.run_analysis_and_export(ticker_list=[ticker])
            loader.load_export_to_mongo()
            print("OK")
            success_count += 1
        except Exception as e:
            print(f"FAILED: {e}")
            if debug:
                import traceback
                traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"Reimported {success_count}/{len(tickers)} tickers from AmiBroker")
    print(f"{'=' * 80}\n")

    return success_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect dividend/split events from Vietstock and reimport from AmiBroker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dividend_adjuster.py --scan                               # Scan last 60 days
  python dividend_adjuster.py --scan --lookback 90                 # Look further back
  python dividend_adjuster.py --scan --last-adjusted 2026-02-01    # Flag all since Feb 1
  python dividend_adjuster.py --scan --from-date 2026-01-01       # Custom start date
  python dividend_adjuster.py --scan --to-date 2026-06-30         # Custom end date
  python dividend_adjuster.py --scan --reimport                    # Scan + auto reimport
  python dividend_adjuster.py --ticker VIC --debug                 # Check single ticker
        """
    )
    parser.add_argument("--ticker", type=str, help="Check specific ticker only")
    parser.add_argument("--scan", action="store_true", help="Scan Vietstock events and find tickers to reimport")
    parser.add_argument("--reimport", action="store_true", help="Reimport flagged tickers from AmiBroker")
    parser.add_argument("--auto-confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--debug", action="store_true", help="Show detailed debug output")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback period in days (default: 60)")
    parser.add_argument("--from-date", type=str, default=None,
                        help="Start date for event search YYYY-MM-DD (default: today - lookback)")
    parser.add_argument("--to-date", type=str, default=None,
                        help="End date for event search YYYY-MM-DD (default: today + 30)")
    parser.add_argument("--reimport-from", type=str, default="2018-01-01",
                        help="Start date for AmiBroker reimport (default: 2018-01-01)")
    parser.add_argument("--last-adjusted", type=str, default=None,
                        help="Date of last known good adjusted import YYYY-MM-DD. "
                             "All events on/after this date are flagged for reimport.")

    args = parser.parse_args()

    if args.ticker:
        print(f"\nChecking {args.ticker} against Vietstock events...")
        events = fetch_vietstock_events(
            from_date=args.from_date, to_date=args.to_date,
            lookback_days=args.lookback, debug=args.debug,
        )
        ticker_events = [e for e in events if e.get('ticker') == args.ticker
                         and e.get('event_type') in PRICE_AFFECTING_EVENTS]

        if ticker_events:
            print(f"\nFound {len(ticker_events)} price-affecting event(s) for {args.ticker}:")
            for e in ticker_events:
                print(f"  {e['ex_date']} | {e['exchange']} | {e['event_type']}")
                print(f"    {e['content']}")

            results = find_tickers_needing_reimport(events,
                                                     last_adjusted=args.last_adjusted,
                                                     debug=args.debug)
            needs = [r for r in results if r['ticker'] == args.ticker]

            if needs:
                print(f"\n  {args.ticker} NEEDS RE-IMPORT (ex-date after last DB update)")
                print(f"   python ami_project_to_db.py --tickers {args.ticker} --from-date {args.reimport_from}")

                if args.reimport:
                    reimport_from_amibroker(
                        [args.ticker], from_date=args.reimport_from,
                        auto_confirm=args.auto_confirm, debug=args.debug
                    )
            else:
                print(f"\n  {args.ticker} is up to date (DB already covers ex-date)")
        else:
            print(f"\n  No recent price-affecting events found for {args.ticker}")

    elif args.scan:
        results = scan_vietstock(
            from_date=args.from_date, to_date=args.to_date,
            lookback_days=args.lookback,
            last_adjusted=args.last_adjusted,
            debug=args.debug,
        )

        if results and args.reimport:
            flagged_tickers = [r['ticker'] for r in results]
            reimport_from_amibroker(
                flagged_tickers, from_date=args.reimport_from,
                auto_confirm=args.auto_confirm, debug=args.debug
            )

    else:
        parser.print_help()
