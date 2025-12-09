"""
Debug script for testing RSI, MA, and MACD calculations on a specific ticker.
Supports both SQLite and MongoDB backends via db_adapter.
Usage: python debug_breadth_calculations.py <TICKER> [--days 365] [--lookback 20]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Try to import TA-Lib for RSI calculation
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("⚠️  TA-Lib not installed. Install with: pip install TA-Lib")
    print("   RSI will use manual Wilder's smoothing calculation")

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Load environment variables
try:
    from dotenv import load_dotenv
    ENV_PATH = os.path.join(SCRIPT_DIR, '.env')
    load_dotenv(dotenv_path=ENV_PATH, verbose=False)
except ImportError:
    pass

# Import db_adapter
try:
    from db_adapter import get_db_adapter
except ImportError as e:
    print(f"❌ Failed to import db_adapter: {e}")
    print("Make sure db_adapter.py is in the same directory.")
    sys.exit(1)

def load_price_range(ticker, start_date, end_date):
    """Load price data using db_adapter (supports both SQLite and MongoDB)."""
    try:
        db = get_db_adapter()
        
        # Convert dates to strings if needed
        start_str = start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d")
        end_str = end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
        
        df = db.load_price_range(ticker.upper(), start_str, end_str)
        
        if not df.empty:
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    except Exception as e:
        print(f"❌ Database error: {e}")
        return pd.DataFrame()

def macd_hist(close, fast=12, slow=26, signal=9):
    """
    Calculate MACD histogram matching AmiBroker's implementation.
    AmiBroker uses EMA for both MACD line and signal line.
    
    MACD Line = EMA(close, fast) - EMA(close, slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line
    """
    close = pd.Series(close).astype(float)
    
    # Calculate EMAs using adjust=False to match AmiBroker
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    
    # MACD Line
    macd_line = ema_fast - ema_slow
    
    # Signal Line - EMA of MACD line (not SMA!)
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    
    # Histogram
    hist = macd_line - macd_signal
    
    return macd_line, macd_signal, hist

def detect_stage(hist: pd.Series, lookback=20):
    """Return one of six stages for the latest bar with numeric prefix for sorting."""
    s = hist.dropna().reset_index(drop=True)
    if s.empty or len(s) < 3:
        return "N/A"
    last = float(s.iat[-1])
    prev = float(s.iat[-2])
    
    cross_up = (prev < 0 and last >= 0)
    cross_down = (prev > 0 and last <= 0)
    
    if cross_up:
        return "2. Confirmed Trough"
    if cross_down:
        return "5. Confirmed Peak"
    
    last_cross_idx = len(s) - 1
    for i in range(len(s)-2, max(0, len(s)-lookback-1), -1):
        if (s[i] < 0 and s[i+1] >= 0) or (s[i] > 0 and s[i+1] <= 0):
            last_cross_idx = i + 1
            break
    
    window_start = max(0, last_cross_idx)
    window = s.iloc[window_start:]
    
    if last < 0:
        if len(window) >= 3:
            min_idx_in_window = int(window.idxmin())
            min_val = float(window.min())
            if min_idx_in_window < len(s) - 1:
                recent_vals = s.iloc[min_idx_in_window:]
                if len(recent_vals) >= 2:
                    slope = np.polyfit(range(len(recent_vals)), recent_vals.values, 1)[0] if len(recent_vals) > 1 else (last - min_val)
                    if slope > 0:
                        return "1. Troughing"
        return "6. Falling below Zero"
    else:
        if len(window) >= 3:
            max_idx_in_window = int(window.idxmax())
            max_val = float(window.max())
            if max_idx_in_window < len(s) - 1:
                recent_vals = s.iloc[max_idx_in_window:]
                if len(recent_vals) >= 2:
                    slope = np.polyfit(range(len(recent_vals)), recent_vals.values, 1)[0] if len(recent_vals) > 1 else (last - max_val)
                    if slope < 0:
                        return "4. Peaking"
        return "3. Rising above Zero"

def calculate_ma(df, window):
    """Calculate moving average."""
    return df['close'].rolling(window).mean()

def calculate_rsi(df, period=14):
    """
    Calculate RSI using TA-Lib if available, otherwise use Wilder's smoothing.
    TA-Lib provides the most accurate RSI matching AmiBroker and other platforms.
    """
    if HAS_TALIB:
        try:
            # TA-Lib RSI uses Wilder's smoothing by default
            rsi = talib.RSI(df['close'].values, timeperiod=period)
            return pd.Series(rsi, index=df.index)
        except Exception as e:
            print(f"⚠️  TA-Lib RSI calculation failed: {e}")
            print("   Falling back to manual Wilder's smoothing...")
            return calculate_rsi_manual(df, period)
    else:
        return calculate_rsi_manual(df, period)

def calculate_rsi_manual(df, period=14):
    """
    Calculate RSI using Wilder's smoothing (fallback if TA-Lib not available).
    Matches AmiBroker's RSI calculation.
    """
    delta = df['close'].diff()
    
    # Calculate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Initialize arrays for Wilder's smoothing
    rsi = np.zeros(len(df))
    rsi[:] = np.nan
    
    # Calculate initial average gain/loss (simple average of first period)
    if len(df) >= period:
        avg_gain = gain.iloc[1:period+1].sum() / period
        avg_loss = loss.iloc[1:period+1].sum() / period
        
        # Set RSI starting from period+1
        for i in range(period, len(df)):
            if i == period:
                # First RSI value after period bars
                rs = avg_gain / avg_loss if avg_loss > 0 else 0
                rsi[i] = 100 - (100 / (1 + rs)) if rs > 0 else 0
            else:
                # Wilder's smoothing: weighted average
                avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
                avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period
                rs = avg_gain / avg_loss if avg_loss > 0 else 0
                rsi[i] = 100 - (100 / (1 + rs)) if rs > 0 else 0
    
    return pd.Series(rsi, index=df.index)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def debug_ticker(ticker, days_back=365, lookback=20):
    """Debug calculations for a specific ticker."""
    
    print_section(f"Debugging Ticker: {ticker.upper()}")
    
    # Load data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"📅 Date Range: {start_date} to {end_date}")
    print(f"📊 Days Requested: {days_back}")
    print(f"📍 MACD Lookback: {lookback}")
    print(f"🗄️  Using db_adapter (auto-detects SQLite or MongoDB)")
    if HAS_TALIB:
        print(f"📊 RSI Calculation: TA-Lib (most accurate)")
    else:
        print(f"📊 RSI Calculation: Manual Wilder's smoothing")
    
    df = load_price_range(ticker, start_date, end_date)
    
    if df.empty:
        print(f"❌ No data found for {ticker}")
        return
    
    # Ensure date is datetime and close is numeric
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    print(f"\n✅ Loaded {len(df)} bars")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Close range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Check for trading days only
    df['weekday'] = df['date'].dt.day_name()
    weekends = df[df['weekday'].isin(['Saturday', 'Sunday'])]
    if not weekends.empty:
        print(f"⚠️  Warning: Found {len(weekends)} weekend dates in data (should not happen)")
    
    # --- Moving Averages ---
    print_section("Moving Averages")
    
    df['ma20'] = calculate_ma(df, 20)
    df['ma50'] = calculate_ma(df, 50)
    df['ma200'] = calculate_ma(df, 200)
    
    latest = df.iloc[-1]
    
    print(f"\n📈 Latest Values (Bar {len(df)}):")
    print(f"   Date: {latest['date']}")
    print(f"   Close: {latest['close']:.2f}")
    print(f"   MA20:  {latest['ma20']:.2f}" if not pd.isna(latest['ma20']) else "   MA20:  N/A (not enough data)")
    print(f"   MA50:  {latest['ma50']:.2f}" if not pd.isna(latest['ma50']) else "   MA50:  N/A (not enough data)")
    print(f"   MA200: {latest['ma200']:.2f}" if not pd.isna(latest['ma200']) else "   MA200: N/A (not enough data)")
    
    # MA status
    print(f"\n🎯 MA Status (Breadth Signal):")
    if not pd.isna(latest['ma20']):
        above_ma20 = latest['close'] > latest['ma20']
        print(f"   Close {'>' if above_ma20 else '≤'} MA20: {above_ma20} {'✅' if above_ma20 else '❌'}")
    if not pd.isna(latest['ma50']):
        above_ma50 = latest['close'] > latest['ma50']
        print(f"   Close {'>' if above_ma50 else '≤'} MA50: {above_ma50} {'✅' if above_ma50 else '❌'}")
    if not pd.isna(latest['ma200']):
        above_ma200 = latest['close'] > latest['ma200']
        print(f"   Close {'>' if above_ma200 else '≤'} MA200: {above_ma200} {'✅' if above_ma200 else '❌'}")
    
    # --- RSI ---
    print_section("RSI (14-period)")
    
    df['rsi'] = calculate_rsi(df, period=14)
    
    latest_rsi = df['rsi'].iloc[-1]
    
    print(f"\n📊 Latest RSI Value (Bar {len(df)}):")
    print(f"   Date: {df['date'].iloc[-1].date()}")
    print(f"   RSI: {latest_rsi:.2f}" if not pd.isna(latest_rsi) else "   RSI: N/A (not enough data)")
    
    # Show RSI calculation details
    if len(df) >= 14:
        print(f"\n🔍 RSI Calculation Details (Wilder's Smoothing via {'TA-Lib' if HAS_TALIB else 'Manual'}):")
        recent_14 = df[['date', 'close']].tail(14).copy()
        recent_14['change'] = recent_14['close'].diff()
        recent_14['gain'] = recent_14['change'].where(recent_14['change'] > 0, 0)
        recent_14['loss'] = -recent_14['change'].where(recent_14['change'] < 0, 0)
        recent_14['bar'] = range(len(df) - 14 + 1, len(df) + 1)
        
        print(f"   Last 14 bars (for initial average):")
        total_gain = 0
        total_loss = 0
        for _, row in recent_14.iterrows():
            if not pd.isna(row['close']):
                total_gain += row['gain']
                total_loss += row['loss']
                print(f"   Bar {row['bar']:3d}: Date={row['date'].date()}  Close={row['close']:8.2f}  Change={row['change']:7.2f}  Gain={row['gain']:6.2f}  Loss={row['loss']:6.2f}")
        
        # Calculate initial average gain/loss
        initial_avg_gain = total_gain / 14
        initial_avg_loss = total_loss / 14
        
        print(f"\n   Initial Averages (first 14 bars):")
        print(f"   Total Gains: {total_gain:.6f}")
        print(f"   Total Losses: {total_loss:.6f}")
        print(f"   Initial Avg Gain: {initial_avg_gain:.6f}")
        print(f"   Initial Avg Loss: {initial_avg_loss:.6f}")
        
        print(f"\n   Wilder's Smoothing Formula:")
        print(f"   Next Avg Gain = (Previous Avg Gain × 13 + Current Gain) / 14")
        print(f"   Next Avg Loss = (Previous Avg Loss × 13 + Current Loss) / 14")
        print(f"   RS = Avg Gain / Avg Loss")
        print(f"   RSI = 100 - (100 / (1 + RS))")
        
        # Show final calculation
        if len(df) >= 15:
            # Get smoothed values from calculations
            smoothed_gains = []
            smoothed_losses = []
            
            current_avg_gain = initial_avg_gain
            current_avg_loss = initial_avg_loss
            
            smoothed_gains.append(current_avg_gain)
            smoothed_losses.append(current_avg_loss)
            
            # Simulate forward bars
            for i in range(14, min(14 + 5, len(df))):
                current_gain = df['close'].iloc[i] - df['close'].iloc[i-1] if df['close'].iloc[i] > df['close'].iloc[i-1] else 0
                current_loss = df['close'].iloc[i-1] - df['close'].iloc[i] if df['close'].iloc[i] < df['close'].iloc[i-1] else 0
                
                current_avg_gain = (current_avg_gain * 13 + current_gain) / 14
                current_avg_loss = (current_avg_loss * 13 + current_loss) / 14
                
                smoothed_gains.append(current_avg_gain)
                smoothed_losses.append(current_avg_loss)
            
            print(f"\n   Current Smoothed Values (at bar {len(df)}):")
            print(f"   Smoothed Avg Gain: {smoothed_gains[-1]:.6f}")
            print(f"   Smoothed Avg Loss: {smoothed_losses[-1]:.6f}")
            rs_current = smoothed_gains[-1] / smoothed_losses[-1] if smoothed_losses[-1] > 0 else 0
            rsi_calculated = 100 - (100 / (1 + rs_current)) if rs_current > 0 else 0
            print(f"   RS: {rs_current:.6f}")
            print(f"   RSI = 100 - (100/(1+RS)) = {rsi_calculated:.2f}")
    
    if not pd.isna(latest_rsi):
        print(f"\n🎯 RSI Status (Breadth Signal):")
        if latest_rsi > 70:
            print(f"   ⚠️  OVERBOUGHT (>70): {latest_rsi:.2f}")
        elif latest_rsi > 50:
            print(f"   ✅ BULLISH (>50): {latest_rsi:.2f}")
        elif latest_rsi < 30:
            print(f"   ✅ OVERSOLD (<30): {latest_rsi:.2f}")
        else:
            print(f"   ❌ BEARISH (≤50): {latest_rsi:.2f}")
    
    # RSI last 14 bars for debugging
    print(f"\n📈 RSI Last 14 Bars:")
    rsi_window = df[['date', 'close', 'rsi', 'weekday']].tail(14).copy()
    rsi_window['bar'] = range(len(df) - 14 + 1, len(df) + 1)
    for _, row in rsi_window.iterrows():
        rsi_status = ""
        if not pd.isna(row['rsi']):
            if row['rsi'] > 70:
                rsi_status = "⚠️  OB"
            elif row['rsi'] > 50:
                rsi_status = "✅ Bull"
            elif row['rsi'] < 30:
                rsi_status = "✅ OS"
            else:
                rsi_status = "❌ Bear"
        print(f"   Bar {row['bar']:3d}: {row['date'].date()} ({row['weekday'][:3]})  Close={row['close']:8.2f}  RSI={row['rsi']:6.2f}  {rsi_status}")
    
    # --- MACD ---
    print_section("MACD Histogram")
    
    close_series = df['close'].astype(float)
    macd_line, macd_signal, hist = macd_hist(close_series)
    
    df['macd_line'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_hist'] = hist
    
    latest_macd = df[['macd_line', 'macd_signal', 'macd_hist']].iloc[-1]
    
    print(f"\n📊 Latest MACD Values:")
    print(f"   MACD Line:   {latest_macd['macd_line']:.6f}" if not pd.isna(latest_macd['macd_line']) else "   MACD Line:   N/A")
    print(f"   Signal Line: {latest_macd['macd_signal']:.6f}" if not pd.isna(latest_macd['macd_signal']) else "   Signal Line: N/A")
    print(f"   Histogram:   {latest_macd['macd_hist']:.6f}" if not pd.isna(latest_macd['macd_hist']) else "   Histogram:   N/A")
    
    # Detect stage
    stage = detect_stage(hist, lookback=lookback)
    print(f"\n🎯 MACD Stage (Lookback={lookback}):")
    print(f"   {stage}")
    
    # MACD last 14 bars for debugging
    print(f"\n📈 MACD Histogram Last 14 Bars:")
    macd_window = df[['date', 'close', 'macd_hist']].tail(14).copy()
    macd_window['bar'] = range(len(df) - 14 + 1, len(df) + 1)
    for _, row in macd_window.iterrows():
        hist_val = row['macd_hist']
        direction = "📈" if not pd.isna(hist_val) and hist_val > 0 else "📉"
        print(f"   Bar {row['bar']:3d}: Date={row['date'].date()}  MACD_Hist={hist_val:10.6f}  {direction}")
    
    # --- Summary ---
    print_section("Breadth Summary")
    
    summary = f"""
Ticker: {ticker.upper()}
Date: {latest['date'].date()}
Close: {latest['close']:.2f}

MA Breadth Signals:
  • Above MA20:  {latest['close'] > latest['ma20'] if not pd.isna(latest['ma20']) else 'N/A'} {'✅' if not pd.isna(latest['ma20']) and latest['close'] > latest['ma20'] else '❌'}
  • Above MA50:  {latest['close'] > latest['ma50'] if not pd.isna(latest['ma50']) else 'N/A'} {'✅' if not pd.isna(latest['ma50']) and latest['close'] > latest['ma50'] else '❌'}
  • Above MA200: {latest['close'] > latest['ma200'] if not pd.isna(latest['ma200']) else 'N/A'} {'✅' if not pd.isna(latest['ma200']) and latest['close'] > latest['ma200'] else '❌'}

RSI Signal (14-period):
  • RSI Value: {latest_rsi:.2f} {'✅' if 30 < latest_rsi < 70 else '⚠️'}
  • Status: {'OVERBOUGHT (>70)' if latest_rsi > 70 else 'OVERSOLD (<30)' if latest_rsi < 30 else 'NEUTRAL (30-70)'}

MACD Signal:
  • Histogram: {latest_macd['macd_hist']:.6f} {'📈' if latest_macd['macd_hist'] > 0 else '📉'}
  • Stage: {stage}
  • Direction: {'Bullish' if 'Trough' in stage or 'Rising' in stage else 'Bearish' if 'Peak' in stage or 'Falling' in stage else 'Neutral'}
"""
    print(summary)
    
    # --- Data Validation ---
    print_section("Data Quality Check")
    
    print(f"\n✓ Total bars (trading days): {len(df)}")
    print(f"✓ Bars with close: {df['close'].notna().sum()}")
    print(f"✓ Bars with MA20: {df['ma20'].notna().sum()}")
    print(f"✓ Bars with MA50: {df['ma50'].notna().sum()}")
    print(f"✓ Bars with MA200: {df['ma200'].notna().sum()}")
    print(f"✓ Bars with RSI: {df['rsi'].notna().sum()}")
    print(f"✓ Bars with MACD: {df['macd_hist'].notna().sum()}")
    
    # Check trading days
    print(f"\n✓ Trading days verification:")
    weekday_counts = df['weekday'].value_counts()
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        count = weekday_counts.get(day, 0)
        print(f"   {day}: {count}")
    
    # Check for gaps (should be only 1-4 days for weekends/holidays)
    df['date_diff'] = df['date'].diff().dt.days
    gaps = df[df['date_diff'] > 4]
    if not gaps.empty:
        print(f"\n⚠️  Potential holidays detected (gaps > 4 days):")
        for _, row in gaps.iterrows():
            print(f"   Gap of {row['date_diff']} days at {row['date'].date()}")
    else:
        print(f"\n✓ No large gaps detected (only weekends)")
    
    # --- Save Debug Output ---
    output_file = f"debug_{ticker.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    debug_df = df[['date', 'close', 'ma20', 'ma50', 'ma200', 'rsi', 'macd_line', 'macd_signal', 'macd_hist']].copy()
    debug_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Debug data saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Debug RSI, MA, and MACD calculations for a specific ticker (with TA-Lib support)"
    )
    parser.add_argument("ticker", help="Ticker symbol (e.g., FPT, VCB, VNINDEX)")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data (default: 365)")
    parser.add_argument("--lookback", type=int, default=20, help="MACD lookback period (default: 20)")
    
    args = parser.parse_args()
    
    try:
        debug_ticker(args.ticker, days_back=args.days, lookback=args.lookback)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
