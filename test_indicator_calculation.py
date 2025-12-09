"""Test indicator calculation with proper date range."""
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd

# Add parent to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

load_dotenv()

# Import required modules
from utils.db_async import get_sync_db_adapter
from utils.indicators import calculate_all_indicators
from utils.macd_stage import detect_macd_stage

print("Testing Indicator Calculation Logic")
print("="*60)

db = get_sync_db_adapter()

ticker = "AAA"
target_date = datetime(2025, 12, 8)

# The key issue: we need ENOUGH historical data for indicators
# RSI needs 14+ days, MACD needs 26+ days, EMA200 needs 200+ days

print(f"\nTesting: {ticker} for date {target_date.date()}")

# Try different lookback periods
lookback_periods = [30, 60, 90, 200, 365]

for days_back in lookback_periods:
    start_date = target_date - timedelta(days=days_back)
    
    print(f"\n{'='*60}")
    print(f"Lookback: {days_back} days (from {start_date.date()} to {target_date.date()})")
    print(f"{'='*60}")
    
    df = db.get_price_data(ticker, start_date, target_date)
    
    if df.empty:
        print(f"  ❌ No price data")
        continue
    
    print(f"  ✓ Loaded {len(df)} bars")
    print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Calculate indicators
    df_with_indicators = calculate_all_indicators(df)
    
    # Check last row (target date)
    last_row = df_with_indicators.iloc[-1]
    
    print(f"\n  Indicator Results (latest bar):")
    print(f"    Date: {last_row['date'].date() if pd.notna(last_row['date']) else 'N/A'}")
    print(f"    Close: {last_row['close']:.2f}")
    print(f"    EMA10: {last_row['ema10']:.2f}" if pd.notna(last_row['ema10']) else "    EMA10: NaN")
    print(f"    EMA20: {last_row['ema20']:.2f}" if pd.notna(last_row['ema20']) else "    EMA20: NaN")
    print(f"    EMA50: {last_row['ema50']:.2f}" if pd.notna(last_row['ema50']) else "    EMA50: NaN")
    print(f"    EMA200: {last_row['ema200']:.2f}" if pd.notna(last_row['ema200']) else "    EMA200: NaN")
    print(f"    RSI: {last_row['rsi']:.2f}" if pd.notna(last_row['rsi']) else "    RSI: NaN")
    print(f"    MACD Hist: {last_row['macd_hist']:.6f}" if pd.notna(last_row['macd_hist']) else "    MACD Hist: NaN")
    
    # Calculate MACD stage
    if 'macd_hist' in df_with_indicators.columns:
        hist = df_with_indicators['macd_hist']
        stage = detect_macd_stage(hist, lookback=20)
        print(f"    MACD Stage: {stage}")
    
    # Summary
    valid_indicators = []
    if pd.notna(last_row['ema10']): valid_indicators.append("EMA10")
    if pd.notna(last_row['ema20']): valid_indicators.append("EMA20")
    if pd.notna(last_row['ema50']): valid_indicators.append("EMA50")
    if pd.notna(last_row['ema200']): valid_indicators.append("EMA200")
    if pd.notna(last_row['rsi']): valid_indicators.append("RSI")
    if pd.notna(last_row['macd_hist']): valid_indicators.append("MACD")
    
    print(f"\n  ✓ Valid indicators ({len(valid_indicators)}/6): {', '.join(valid_indicators) if valid_indicators else 'NONE'}")

print(f"\n{'='*60}")
print("Conclusion:")
print("="*60)
print("To get valid indicators, you need:")
print("  • EMA10: 10+ trading days")
print("  • EMA20: 20+ trading days")
print("  • RSI(14): 14+ trading days")
print("  • MACD(12,26,9): 26+ trading days")
print("  • EMA50: 50+ trading days")
print("  • EMA200: 200+ trading days")
print()
print("Recommended: Use at least 365 days (1 year) for calculations")
