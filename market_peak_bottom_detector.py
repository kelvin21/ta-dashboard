"""
Market Peak/Bottom Detector using Breadth Indicators
Forecasts market turning points with 3-day and 1-week horizons.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add script directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


class MarketPeakBottomDetector:
    """Detect market peaks and bottoms using breadth indicators."""
    
    def __init__(self, db_path):
        """
        Initialize detector.
        
        Args:
            db_path: Path to SQLite database containing market_data table
        """
        self.db_path = db_path
        self.signals = []
        
        # Ensure signal history table exists
        self._create_signals_table()
    
    def _create_signals_table(self):
        """Create market_signals table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence INTEGER,
                    forecast_3d TEXT,
                    forecast_1w TEXT,
                    peak_score INTEGER,
                    bottom_score INTEGER,
                    mcclellan_oscillator REAL,
                    mcclellan_summation REAL,
                    ad_ratio REAL,
                    net_advances REAL,
                    adl_roc_5 REAL,
                    adl_roc_10 REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
            
            conn.commit()
            conn.close()
        except Exception:
            pass  # Silently fail if table creation fails
    
    def save_signal(self, date_str, signal_data):
        """
        Save a signal to the database.
        
        Args:
            date_str: Date string (YYYY-MM-DD)
            signal_data: Signal dictionary from detect_peaks_bottoms
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO market_signals 
                (date, signal, confidence, forecast_3d, forecast_1w, peak_score, bottom_score,
                 mcclellan_oscillator, mcclellan_summation, ad_ratio, net_advances, adl_roc_5, adl_roc_10)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str,
                signal_data['current_signal'],
                signal_data['confidence'],
                signal_data['forecast_3d'],
                signal_data['forecast_1w'],
                signal_data['peak_score'],
                signal_data['bottom_score'],
                signal_data['indicators']['mcclellan_oscillator'],
                signal_data['indicators']['mcclellan_summation'],
                signal_data['indicators']['ad_ratio'],
                signal_data['indicators']['net_advances'],
                signal_data['indicators']['adl_roc_5'],
                signal_data['indicators']['adl_roc_10']
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
    def get_signal_history(self, days=30):
        """
        Get historical signals from database.
        
        Args:
            days: Number of days of history to retrieve
        
        Returns:
            DataFrame with signal history
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            query = """
                SELECT * FROM market_signals
                WHERE date >= ? AND date <= ?
                ORDER BY date DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(start_date.strftime('%Y-%m-%d'),
                                                         end_date.strftime('%Y-%m-%d')))
            conn.close()
            
            return df
        except Exception:
            return pd.DataFrame()

    def calculate_breadth_indicators_from_tickers(self, debug=False):
        """
        Calculate breadth indicators from individual ticker MACD data.
        This mirrors the logic from Market Breadth page's calculate_breadth_metrics.
        
        Returns:
            Tuple of (success: bool, message: str, rows_calculated: int)
        """
        if debug:
            print(f"\n{'='*60}")
            print("Calculating Breadth Indicators from Ticker Data")
            print(f"{'='*60}\n")
        
        try:
            import sys
            import os
            
            # Import from main dashboard
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            
            from ta_dashboard import get_all_tickers, load_price_range, macd_hist, detect_stage
            
            # Get all tickers
            all_tickers = get_all_tickers(db_path=self.db_path, debug=False)
            
            if not all_tickers:
                return (False, "No tickers found in database", 0)
            
            if debug:
                print(f"Found {len(all_tickers)} tickers to analyze")
            
            # Calculate breadth for historical period
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365)  # Need enough data for calculations
            
            # Group tickers by date and calculate breadth metrics
            date_breadth = {}
            
            for ticker in all_tickers:
                if debug and len(date_breadth) % 50 == 0:
                    print(f"  Processing ticker {len(date_breadth)}/{len(all_tickers)}...")
                
                try:
                    df = load_price_range(ticker, start_date, end_date, db_path=self.db_path)
                    
                    if df.empty or len(df) < 50:
                        continue
                    
                    # Calculate MACD for this ticker
                    _, _, hist = macd_hist(df['close'].astype(float))
                    
                    # Get daily stages
                    for idx in range(len(df)):
                        date_str = df.iloc[idx]['date'].strftime('%Y-%m-%d')
                        
                        if date_str not in date_breadth:
                            date_breadth[date_str] = {
                                'advancing': 0,
                                'declining': 0,
                                'macd_positive': 0,
                                'macd_negative': 0
                            }
                        
                        # Check if MACD histogram is positive or negative
                        if idx < len(hist) and not pd.isna(hist.iloc[idx]):
                            if hist.iloc[idx] > 0:
                                date_breadth[date_str]['advancing'] += 1
                                date_breadth[date_str]['macd_positive'] += 1
                            else:
                                date_breadth[date_str]['declining'] += 1
                                date_breadth[date_str]['macd_negative'] += 1
                
                except Exception as e:
                    if debug:
                        print(f"    Error processing {ticker}: {e}")
                    continue
            
            if not date_breadth:
                return (False, "No breadth data could be calculated from tickers", 0)
            
            if debug:
                print(f"\nCalculated breadth for {len(date_breadth)} dates")
            
            # Now calculate indicators for each date
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure columns exist
            cursor.execute("PRAGMA table_info(market_breadth_history)")
            existing_columns = {col[1] for col in cursor.fetchall()}
            
            required_cols = ['net_advances', 'ad_ratio', 'ad_line', 'mcclellan_oscillator', 'mcclellan_summation']
            for col in required_cols:
                if col not in existing_columns:
                    cursor.execute(f"ALTER TABLE market_breadth_history ADD COLUMN {col} REAL")
            
            conn.commit()
            
            # Sort dates and calculate cumulative indicators
            sorted_dates = sorted(date_breadth.keys())
            cumulative_net_advances = 0
            net_advances_history = []
            
            rows_updated = 0
            
            for date_str in sorted_dates:
                breadth = date_breadth[date_str]
                
                # Calculate net advances
                net_advances = breadth['advancing'] - breadth['declining']
                net_advances_history.append(net_advances)
                
                # Calculate A/D ratio
                ad_ratio = breadth['advancing'] / max(breadth['declining'], 1)  # Avoid division by zero
                
                # Calculate A/D line (cumulative net advances)
                cumulative_net_advances += net_advances
                ad_line = cumulative_net_advances
                
                # Calculate McClellan Oscillator (need at least 39 days of history)
                if len(net_advances_history) >= 39:
                    # Convert to pandas Series for EMA calculation
                    net_adv_series = pd.Series(net_advances_history)
                    ema19 = net_adv_series.ewm(span=19, adjust=False, min_periods=1).mean().iloc[-1]
                    ema39 = net_adv_series.ewm(span=39, adjust=False, min_periods=1).mean().iloc[-1]
                    mcclellan_osc = ema19 - ema39
                else:
                    mcclellan_osc = 0
                
                # Calculate McClellan Summation (cumulative oscillator)
                # For simplicity, calculate from all available history
                if len(net_advances_history) >= 39:
                    net_adv_series = pd.Series(net_advances_history)
                    ema19_series = net_adv_series.ewm(span=19, adjust=False, min_periods=1).mean()
                    ema39_series = net_adv_series.ewm(span=39, adjust=False, min_periods=1).mean()
                    mcclellan_osc_series = ema19_series - ema39_series
                    mcclellan_sum = mcclellan_osc_series.sum()
                else:
                    mcclellan_sum = 0
                
                # Update database
                cursor.execute("""
                    UPDATE market_breadth_history
                    SET net_advances = ?,
                        ad_ratio = ?,
                        ad_line = ?,
                        mcclellan_oscillator = ?,
                        mcclellan_summation = ?
                    WHERE date = ?
                """, (
                    float(net_advances),
                    float(ad_ratio),
                    float(ad_line),
                    float(mcclellan_osc),
                    float(mcclellan_sum),
                    date_str
                ))
                
                if cursor.rowcount > 0:
                    rows_updated += 1
            
            conn.commit()
            conn.close()
            
            if debug:
                print(f"\n✅ Updated {rows_updated} rows with calculated breadth indicators")
            
            return (True, f"Successfully calculated breadth indicators for {rows_updated} dates from ticker data", rows_updated)
            
        except Exception as e:
            if debug:
                print(f"❌ Error calculating from ticker data: {e}")
                import traceback
                traceback.print_exc()
            return (False, f"Error: {e}", 0)
    
    def calculate_missing_indicators(self, debug=False):
        """
        Calculate missing breadth indicators and update the database.
        Now tries two approaches:
        1. Calculate from existing net_advances if available
        2. Calculate from individual ticker MACD data if net_advances is empty
        
        Returns:
            Tuple of (success: bool, message: str, rows_calculated: int)
        """
        if debug:
            print(f"\n{'='*60}")
            print("Calculating Missing Breadth Indicators")
            print(f"{'='*60}\n")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_breadth_history'")
            if not cursor.fetchone():
                conn.close()
                return (False, "Table 'market_breadth_history' does not exist", 0)
            
            # Get existing columns
            cursor.execute("PRAGMA table_info(market_breadth_history)")
            columns_info = cursor.fetchall()
            existing_columns = {col[1] for col in columns_info}
            
            if debug:
                print(f"Existing columns: {sorted(existing_columns)}")
            
            # Define required indicator columns
            required_indicators = {
                'net_advances', 'ad_ratio', 'ad_line', 
                'mcclellan_oscillator', 'mcclellan_summation'
            }
            
            # Check which indicators already exist as columns
            existing_indicators = required_indicators & existing_columns
            missing_indicators = required_indicators - existing_columns
            
            if debug:
                print(f"Existing indicators: {existing_indicators}")
                print(f"Missing indicators: {missing_indicators}")
            
            # Add missing columns first
            for col_name in missing_indicators:
                if debug:
                    print(f"Adding column: {col_name}")
                cursor.execute(f"ALTER TABLE market_breadth_history ADD COLUMN {col_name} REAL")
            
            conn.commit()
            
            # Now check if existing indicator columns have any non-NULL values
            indicators_need_calculation = set()
            
            for col in required_indicators:
                if col in existing_columns or col in missing_indicators:
                    # Check if column has any non-NULL values
                    cursor.execute(f"SELECT COUNT(*) FROM market_breadth_history WHERE {col} IS NOT NULL")
                    non_null_count = cursor.fetchone()[0]
                    
                    if debug:
                        print(f"  {col}: {non_null_count} non-NULL values")
                    
                    if non_null_count == 0:
                        indicators_need_calculation.add(col)
            
            # If all indicators have data, no need to calculate
            if not indicators_need_calculation and not missing_indicators:
                if debug:
                    print("✓ All indicators already have data in database")
                conn.close()
                return (True, "All required indicators already have data. No calculation needed.", 0)
            
            if debug:
                print(f"\nIndicators needing calculation: {indicators_need_calculation}")
            
            # Try loading existing data first
            query = """
                SELECT date, 
                       COALESCE(net_advances, 0) as net_advances,
                       COALESCE(ad_ratio, 0) as ad_ratio,
                       COALESCE(ad_line, 0) as ad_line
                FROM market_breadth_history
                ORDER BY date
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if debug:
                print(f"\nLoaded {len(df)} rows for indicator calculation")
            
            # Check if we have source data (net_advances)
            if df.empty or df['net_advances'].sum() == 0:
                if debug:
                    print("\n⚠️ net_advances is empty - need to calculate from ticker data")
                    print("  This requires analyzing individual ticker MACD values")
                
                # Try to calculate from ticker data
                return self.calculate_breadth_indicators_from_tickers(debug=debug)
            
            # If we have net_advances, calculate McClellan indicators
            df['net_advances'] = pd.to_numeric(df['net_advances'], errors='coerce').fillna(0)
            
            # 4. McClellan Oscillator (EMA19 - EMA39 of net advances)
            ema19 = df['net_advances'].ewm(span=19, adjust=False, min_periods=1).mean()
            ema39 = df['net_advances'].ewm(span=39, adjust=False, min_periods=1).mean()
            df['mcclellan_oscillator'] = ema19 - ema39
            
            # 5. McClellan Summation (cumulative sum of oscillator)
            df['mcclellan_summation'] = df['mcclellan_oscillator'].cumsum()
            
            if debug:
                print(f"\nCalculated McClellan indicators:")
                print(f"  McClellan Osc range: {df['mcclellan_oscillator'].min():.2f} to {df['mcclellan_oscillator'].max():.2f}")
                print(f"  McClellan Sum range: {df['mcclellan_summation'].min():.0f} to {df['mcclellan_summation'].max():.0f}")
            
            # Update database with calculated values
            rows_updated = 0
            for _, row in df.iterrows():
                # Only update the indicators that needed calculation
                update_cols = []
                update_vals = []
                
                if 'mcclellan_oscillator' in indicators_need_calculation:
                    update_cols.append('mcclellan_oscillator = ?')
                    update_vals.append(float(row['mcclellan_oscillator']))
                
                if 'mcclellan_summation' in indicators_need_calculation:
                    update_cols.append('mcclellan_summation = ?')
                    update_vals.append(float(row['mcclellan_summation']))
                
                if update_cols:
                    update_vals.append(row['date'])
                    sql = f"UPDATE market_breadth_history SET {', '.join(update_cols)} WHERE date = ?"
                    cursor.execute(sql, tuple(update_vals))
                    rows_updated += 1
            
            conn.commit()
            conn.close()
            
            if debug:
                print(f"\n✅ Updated {rows_updated} rows with calculated indicators")
            
            return (True, f"Successfully calculated indicators for {rows_updated} rows", rows_updated)
            
        except Exception as e:
            if debug:
                print(f"❌ Error calculating indicators: {e}")
                import traceback
                traceback.print_exc()
            return (False, f"Error: {e}", 0)
    
    def load_breadth_data(self, lookback_days=90, debug=False):
        """
        Load breadth indicators from database.
        
        Returns:
            DataFrame with breadth indicators
        """
        if not os.path.exists(self.db_path):
            if debug:
                print(f"❌ Database not found: {self.db_path}")
            return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if market_breadth_history table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_breadth_history'")
            if not cursor.fetchone():
                if debug:
                    print("❌ Table 'market_breadth_history' does not exist")
                    print("💡 Run Market Breadth page to create and populate this table")
                conn.close()
                return pd.DataFrame()
            
            # Check if table has data
            cursor.execute("SELECT COUNT(*) FROM market_breadth_history")
            row_count = cursor.fetchone()[0]
            
            if row_count == 0:
                if debug:
                    print("❌ Table 'market_breadth_history' exists but is empty")
                    print("💡 Run Market Breadth page to populate breadth indicators")
                conn.close()
                return pd.DataFrame()
            
            if debug:
                print(f"✓ Found {row_count} rows in market_breadth_history table")
            
            # First, get the actual column names from the table
            cursor.execute("PRAGMA table_info(market_breadth_history)")
            columns_info = cursor.fetchall()
            actual_columns = [col[1] for col in columns_info]
            
            if debug:
                print(f"  Actual columns: {actual_columns}")
            
            # Required core columns for peak/bottom detection
            required_columns = [
                'mcclellan_oscillator',
                'mcclellan_summation',
                'net_advances',
                'ad_line',
                'ad_ratio'
            ]
            
            # Alternative names
            alternative_names = {
                'advance_decline_line': 'ad_line',
                'advances_declines_ratio': 'ad_ratio',
                'advancing': 'advances',
                'declining': 'declines'
            }
            
            # Check which required columns are missing
            missing_columns = []
            for col in required_columns:
                # Check both the expected name and alternatives
                found = col in actual_columns
                if not found:
                    # Check if any alternative exists
                    for alt_key, alt_val in alternative_names.items():
                        if alt_val == col and alt_key in actual_columns:
                            found = True
                            break
                
                if not found:
                    missing_columns.append(col)
            
            if missing_columns:
                if debug:
                    print(f"❌ Missing required columns: {missing_columns}")
                    print(f"💡 These columns need to be calculated.")
                    print(f"💡 Go to Market Breadth page and click 'Update Breadth Data' to calculate indicators.")
                conn.close()
                return pd.DataFrame()
            
            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Build query using actual column names
            # Map expected columns to actual columns
            column_mapping = {
                'date': 'date',
                'advancing': 'advances' if 'advances' in actual_columns else 'advancing' if 'advancing' in actual_columns else None,
                'declining': 'declines' if 'declines' in actual_columns else 'declining' if 'declining' in actual_columns else None,
                'unchanged': 'unchanged' if 'unchanged' in actual_columns else None,
                'advances_declines_ratio': 'ad_ratio' if 'ad_ratio' in actual_columns else 'advances_declines_ratio' if 'advances_declines_ratio' in actual_columns else None,
                'net_advances': 'net_advances' if 'net_advances' in actual_columns else None,
                'mcclellan_oscillator': 'mcclellan_oscillator' if 'mcclellan_oscillator' in actual_columns else None,
                'mcclellan_summation': 'mcclellan_summation' if 'mcclellan_summation' in actual_columns else None,
                'advance_decline_line': 'ad_line' if 'ad_line' in actual_columns else 'advance_decline_line' if 'advance_decline_line' in actual_columns else None
            }
            
            # Build SELECT clause with actual column names
            select_parts = []
            for expected, actual in column_mapping.items():
                if actual and actual in actual_columns:
                    if expected != actual:
                        select_parts.append(f"{actual} as {expected}")
                    else:
                        select_parts.append(actual)
                elif expected == 'date':
                    # Date is required
                    if debug:
                        print(f"❌ Date column not found!")
                    conn.close()
                    return pd.DataFrame()
            
            if len(select_parts) < 6:  # date + 5 core indicators
                if debug:
                    print(f"❌ Insufficient columns found ({len(select_parts)-1} indicators). Need at least 5 core indicators.")
                    print(f"💡 Go to Market Breadth page and click 'Update Breadth Data' to calculate missing indicators.")
                conn.close()
                return pd.DataFrame()
            
            query = f"""
                SELECT 
                    {', '.join(select_parts)}
                FROM market_breadth_history
                WHERE date >= ? AND date <= ?
                ORDER BY date
            """
            
            if debug:
                print(f"  SQL Query:\n{query}")
            
            df = pd.read_sql_query(query, conn, params=(start_date.strftime('%Y-%m-%d'), 
                                                         end_date.strftime('%Y-%m-%d')))
            conn.close()
            
            if df.empty:
                if debug:
                    print(f"❌ No data found for date range {start_date} to {end_date}")
                    print(f"💡 Available data might be outside this range")
                    # Try to get available date range
                    try:
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute("SELECT MIN(date), MAX(date) FROM market_breadth_history")
                        min_date, max_date = cursor.fetchone()
                        conn.close()
                        if min_date and max_date:
                            print(f"   Available data: {min_date} to {max_date}")
                    except:
                        pass
                return df
            
            df['date'] = pd.to_datetime(df['date'])
            
            if debug:
                print(f"✓ Loaded {len(df)} rows of breadth data")
                print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
                print(f"  Columns in DataFrame: {list(df.columns)}")
            
            # Check if indicator columns exist but are all NaN
            indicator_cols = ['mcclellan_oscillator', 'mcclellan_summation', 'ad_line', 'ad_ratio', 'net_advances']
            has_valid_data = False
            
            for col in indicator_cols:
                if col in df.columns:
                    non_null_count = df[col].notna().sum()
                    if debug:
                        print(f"  {col}: {non_null_count} non-null values out of {len(df)}")
                    if non_null_count > 0:
                        has_valid_data = True
            
            if not has_valid_data:
                if debug:
                    print("⚠️ Indicator columns exist but all values are NULL/NaN")
                    print("  This indicates indicators need to be calculated")
                return pd.DataFrame()  # Return empty to trigger calculation
            
            return df
            
        except Exception as e:
            if debug:
                print(f"❌ Error loading breadth data: {e}")
                import traceback
                traceback.print_exc()
            return pd.DataFrame()
    
    def calculate_indicators(self, df):
        """
        Calculate additional technical indicators on breadth data.
        
        Args:
            df: DataFrame with breadth indicators
            
        Returns:
            DataFrame with additional indicators
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure all numeric columns are properly typed and handle None/NaN
        numeric_cols = [
            'mcclellan_oscillator', 'mcclellan_summation', 'advance_decline_line',
            'net_advances', 'advances_declines_ratio', 'advancing', 'declining'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check required columns exist
        required_cols = ['mcclellan_oscillator', 'mcclellan_summation', 'advance_decline_line', 
                        'net_advances', 'advances_declines_ratio']
        
        missing_cols = [col for col in required_cols if col not in df.columns or df[col].isna().all()]
        
        if missing_cols:
            print(f"Warning: Missing or empty required columns: {missing_cols}")
            return df
        
        # 1. McClellan Oscillator signals
        df['mcco_ma5'] = df['mcclellan_oscillator'].rolling(5, min_periods=1).mean()
        df['mcco_ma10'] = df['mcclellan_oscillator'].rolling(10, min_periods=1).mean()
        df['mcco_crossover'] = (df['mcco_ma5'] > df['mcco_ma10']).astype(int)
        df['mcco_crossover_signal'] = df['mcco_crossover'].diff().fillna(0)
        
        # 2. McClellan Summation extremes
        df['mcsum_ma20'] = df['mcclellan_summation'].rolling(20, min_periods=1).mean()
        df['mcsum_overbought'] = (df['mcclellan_summation'] > 1000).astype(int)
        df['mcsum_oversold'] = (df['mcclellan_summation'] < -1000).astype(int)
        
        # 3. Advance-Decline Line momentum
        df['adl_change'] = df['advance_decline_line'].diff().fillna(0)
        df['adl_roc_5'] = df['advance_decline_line'].pct_change(5).fillna(0) * 100
        df['adl_roc_10'] = df['advance_decline_line'].pct_change(10).fillna(0) * 100
        
        # 4. Net Advances momentum
        df['net_adv_ma5'] = df['net_advances'].rolling(5, min_periods=1).mean()
        df['net_adv_ma20'] = df['net_advances'].rolling(20, min_periods=1).mean()
        
        # 5. Advances/Declines ratio extremes
        df['ad_ratio_ma5'] = df['advances_declines_ratio'].rolling(5, min_periods=1).mean()
        df['ad_ratio_extreme_bullish'] = (df['advances_declines_ratio'] > 2.0).astype(int)
        df['ad_ratio_extreme_bearish'] = (df['advances_declines_ratio'] < 0.5).astype(int)
        
        # 6. Divergence detection (ADL vs McClellan)
        df['adl_trend'] = np.where(df['adl_roc_10'] > 0, 1, -1)
        df['mcco_trend'] = np.where(df['mcclellan_oscillator'] > 0, 1, -1)
        df['breadth_divergence'] = df['adl_trend'] - df['mcco_trend']
        
        return df
    
    def detect_peaks_bottoms(self, df, debug=False):
        """
        Detect potential market peaks and bottoms.
        
        Args:
            df: DataFrame with breadth indicators
            
        Returns:
            Dictionary with peak/bottom signals
        """
        if df.empty or len(df) < 20:
            return {
                'current_signal': 'NEUTRAL',
                'confidence': 0,
                'forecast_3d': 'NEUTRAL',
                'forecast_1w': 'NEUTRAL',
                'peak_score': 0,
                'bottom_score': 0,
                'indicators': {
                    'mcclellan_oscillator': 0.0,
                    'mcclellan_summation': 0.0,
                    'ad_ratio': 1.0,
                    'net_advances': 0.0,
                    'adl_roc_5': 0.0,
                    'adl_roc_10': 0.0,
                }
            }
        
        # Check if required indicator columns exist
        required_indicator_cols = [
            'mcco_crossover_signal', 'mcclellan_summation', 'ad_ratio_extreme_bearish',
            'ad_ratio_extreme_bullish', 'net_adv_ma5', 'adl_roc_5'
        ]
        
        missing_cols = [col for col in required_indicator_cols if col not in df.columns]
        
        if missing_cols:
            if debug:
                print(f"Warning: Missing indicator columns: {missing_cols}")
                print("Using basic analysis only")
            
            # Return basic analysis using only core columns
            latest = df.iloc[-1]
            
            return {
                'current_signal': 'NEUTRAL',
                'confidence': 0,
                'forecast_3d': 'NEUTRAL',
                'forecast_1w': 'NEUTRAL',
                'peak_score': 0,
                'bottom_score': 0,
                'indicators': {
                    'mcclellan_oscillator': float(latest.get('mcclellan_oscillator', 0)),
                    'mcclellan_summation': float(latest.get('mcclellan_summation', 0)),
                    'ad_ratio': float(latest.get('advances_declines_ratio', 1)),
                    'net_advances': float(latest.get('net_advances', 0)),
                    'adl_roc_5': 0.0,
                    'adl_roc_10': 0.0,
                }
            }
        
        latest = df.iloc[-1]
        recent_5 = df.tail(5)
        recent_10 = df.tail(10)
        
        # Initialize scores
        peak_score = 0
        bottom_score = 0
        
        # === BOTTOM SIGNALS (Bullish) ===
        
        # 1. McClellan Oscillator deeply oversold and turning up
        if latest['mcclellan_oscillator'] < -100 and latest.get('mcco_crossover_signal', 0) == 1:
            bottom_score += 3
            if debug:
                print("  ✓ McClellan Oscillator oversold + crossover (+3)")
        
        # 2. McClellan Summation deeply oversold
        if latest['mcclellan_summation'] < -1000:
            bottom_score += 2
            if debug:
                print("  ✓ McClellan Summation oversold (+2)")
        
        # 3. Extreme bearish A/D ratio followed by improvement
        if 'ad_ratio_extreme_bearish' in recent_5.columns and 'advances_declines_ratio' in recent_5.columns:
            if recent_5['ad_ratio_extreme_bearish'].sum() >= 2 and latest['advances_declines_ratio'] > recent_5['advances_declines_ratio'].iloc[-2]:
                bottom_score += 2
                if debug:
                    print("  ✓ A/D ratio improving from extreme bearish (+2)")
        
        # 4. Net Advances turning positive
        if 'net_adv_ma5' in df.columns:
            if latest['net_adv_ma5'] > 0 and recent_5['net_adv_ma5'].iloc[-2] < 0:
                bottom_score += 1
                if debug:
                    print("  ✓ Net Advances turning positive (+1)")
        
        # 5. ADL starting to rise
        if 'adl_roc_5' in df.columns:
            if latest['adl_roc_5'] > 0 and recent_5['adl_roc_5'].iloc[-2] < 0:
                bottom_score += 1
                if debug:
                    print("  ✓ ADL momentum turning positive (+1)")
        
        # === PEAK SIGNALS (Bearish) ===
        
        # 1. McClellan Oscillator deeply overbought and turning down
        if latest['mcclellan_oscillator'] > 100 and latest.get('mcco_crossover_signal', 0) == -1:
            peak_score += 3
            if debug:
                print("  ✓ McClellan Oscillator overbought + crossunder (+3)")
        
        # 2. McClellan Summation deeply overbought
        if latest['mcclellan_summation'] > 1000:
            peak_score += 2
            if debug:
                print("  ✓ McClellan Summation overbought (+2)")
        
        # 3. Extreme bullish A/D ratio followed by deterioration
        if 'ad_ratio_extreme_bullish' in recent_5.columns and 'advances_declines_ratio' in recent_5.columns:
            if recent_5['ad_ratio_extreme_bullish'].sum() >= 2 and latest['advances_declines_ratio'] < recent_5['advances_declines_ratio'].iloc[-2]:
                peak_score += 2
                if debug:
                    print("  ✓ A/D ratio deteriorating from extreme bullish (+2)")
        
        # 4. Net Advances turning negative
        if 'net_adv_ma5' in df.columns:
            if latest['net_adv_ma5'] < 0 and recent_5['net_adv_ma5'].iloc[-2] > 0:
                peak_score += 1
                if debug:
                    print("  ✓ Net Advances turning negative (+1)")
        
        # 5. ADL starting to decline
        if 'adl_roc_5' in df.columns:
            if latest['adl_roc_5'] < 0 and recent_5['adl_roc_5'].iloc[-2] > 0:
                peak_score += 1
                if debug:
                    print("  ✓ ADL momentum turning negative (+1)")
        
        # 6. Bearish divergence
        if 'breadth_divergence' in df.columns:
            if latest['breadth_divergence'] < 0:
                peak_score += 1
                if debug:
                    print("  ✓ Bearish divergence detected (+1)")
        
        # === Determine signal ===
        if bottom_score > peak_score and bottom_score >= 4:
            signal = 'BOTTOM'
            confidence = min(100, bottom_score * 15)
            forecast_3d = 'BULLISH'
            forecast_1w = 'BULLISH'
        elif peak_score > bottom_score and peak_score >= 4:
            signal = 'PEAK'
            confidence = min(100, peak_score * 15)
            forecast_3d = 'BEARISH'
            forecast_1w = 'BEARISH'
        elif bottom_score > 0 and bottom_score > peak_score:
            signal = 'BOTTOMING'
            confidence = min(100, bottom_score * 15)
            forecast_3d = 'NEUTRAL'
            forecast_1w = 'BULLISH'
        elif peak_score > 0 and peak_score > bottom_score:
            signal = 'TOPPING'
            confidence = min(100, peak_score * 15)
            forecast_3d = 'NEUTRAL'
            forecast_1w = 'BEARISH'
        else:
            signal = 'NEUTRAL'
            confidence = 0
            forecast_3d = 'NEUTRAL'
            forecast_1w = 'NEUTRAL'
        
        # === Build indicators dict with safe value extraction ===
        def safe_float(value, default=0.0):
            """Safely convert value to float, handling NaN/None."""
            try:
                val = float(value)
                return val if not pd.isna(val) else default
            except (TypeError, ValueError):
                return default
        
        return {
            'current_signal': signal,
            'confidence': confidence,
            'forecast_3d': forecast_3d,
            'forecast_1w': forecast_1w,
            'peak_score': peak_score,
            'bottom_score': bottom_score,
            'indicators': {
                'mcclellan_oscillator': safe_float(latest.get('mcclellan_oscillator', 0)),
                'mcclellan_summation': safe_float(latest.get('mcclellan_summation', 0)),
                'ad_ratio': safe_float(latest.get('advances_declines_ratio', 1), default=1.0),
                'net_advances': safe_float(latest.get('net_advances', 0)),
                'adl_roc_5': safe_float(latest.get('adl_roc_5', 0)),
                'adl_roc_10': safe_float(latest.get('adl_roc_10', 0)),
            }
        }
    
    def run_detection(self, lookback_days=90, debug=False, auto_calculate=True):
        """
        Run full peak/bottom detection analysis.
        
        Args:
            lookback_days: Number of days to look back
            debug: Enable debug output
            auto_calculate: If True, automatically calculate missing indicators
        
        Returns:
            Dictionary with detection results
        """
        if debug:
            print(f"\n{'='*60}")
            print("Market Peak/Bottom Detection")
            print(f"{'='*60}\n")
        
        # Load data
        df = self.load_breadth_data(lookback_days=lookback_days, debug=debug)
        
        if df.empty:
            # Check if we should try to calculate indicators
            if auto_calculate:
                if debug:
                    print("\n⚙️ Attempting to calculate missing indicators...")
                
                success, message, rows = self.calculate_missing_indicators(debug=debug)
                
                if success:
                    if rows > 0:  # Only if we actually calculated something
                        if debug:
                            print(f"✓ Calculated indicators for {rows} rows")
                        # Try loading data again
                        df = self.load_breadth_data(lookback_days=lookback_days, debug=debug)
                        
                        if df.empty:
                            return {
                                'success': False,
                                'message': 'Indicators calculated but no data in date range. Try increasing lookback period.',
                                'signal': None,
                                'hint': 'increase_lookback'
                            }
                    else:
                        # Indicators already exist, but data might be in wrong range
                        return {
                            'success': False,
                            'message': 'Breadth indicators exist but no data found in the requested date range. Try increasing lookback period to 180+ days or check if Market Breadth data is up to date.',
                            'signal': None,
                            'hint': 'increase_lookback'
                        }
                else:
                    return {
                        'success': False,
                        'message': f'Failed to calculate indicators: {message}. Please go to Market Breadth page and click "Update Breadth Data".',
                        'signal': None,
                        'hint': 'calculation_failed'
                    }
            else:
                return {
                    'success': False,
                    'message': 'No breadth indicators found. Enable "Auto-calculate missing indicators" or go to Market Breadth page.',
                    'signal': None,
                    'hint': 'calculate_indicators'
                }
        
        # Check if we have enough data points
        if len(df) < 20:
            return {
                'success': False,
                'message': f'Insufficient data: only {len(df)} days available. Need at least 20 days for reliable detection. Increase lookback period or update Market Breadth data.',
                'signal': None,
                'hint': 'insufficient_data'
            }
        
        # Calculate additional indicators
        df = self.calculate_indicators(df)
        
        # Detect peaks/bottoms
        signal = self.detect_peaks_bottoms(df, debug=debug)
        
        # Save signal to database
        latest_date = df.iloc[-1]['date'].strftime('%Y-%m-%d')
        self.save_signal(latest_date, signal)
        
        return {
            'success': True,
            'message': 'Detection completed',
            'signal': signal,
            'latest_date': latest_date,
            'data_points': len(df)
        }
    
    def format_report(self, result):
        """Format detection result as a readable report."""
        if not result['success']:
            return f"❌ {result['message']}"
        
        signal = result['signal']
        
        # Signal emoji
        signal_emoji = {
            'BOTTOM': '🟢',
            'PEAK': '🔴',
            'BOTTOMING': '🟡',
            'TOPPING': '🟠',
            'NEUTRAL': '⚪'
        }
        
        # Forecast emoji
        forecast_emoji = {
            'BULLISH': '📈',
            'BEARISH': '📉',
            'NEUTRAL': '➡️'
        }
        
        report = f"""
╔════════════════════════════════════════════════════════════╗
║          MARKET PEAK/BOTTOM DETECTION REPORT               ║
╚════════════════════════════════════════════════════════════╝

📅 Analysis Date: {result['latest_date']}
📊 Data Points: {result['data_points']} days

{signal_emoji[signal['current_signal']]} CURRENT SIGNAL: {signal['current_signal']}
   Confidence: {signal['confidence']}%

📅 FORECASTS:
   {forecast_emoji[signal['forecast_3d']]} 3-Day: {signal['forecast_3d']}
   {forecast_emoji[signal['forecast_1w']]} 1-Week: {signal['forecast_1w']}

📊 INDICATOR SCORES:
   Bottom Score: {signal['bottom_score']} {'✓' if signal['bottom_score'] >= 4 else ''}
   Peak Score: {signal['peak_score']} {'✓' if signal['peak_score'] >= 4 else ''}

📈 KEY BREADTH INDICATORS:
   McClellan Oscillator: {signal['indicators']['mcclellan_oscillator']:.2f}
   McClellan Summation: {signal['indicators']['mcclellan_summation']:.2f}
   A/D Ratio: {signal['indicators']['ad_ratio']:.2f}
   Net Advances: {signal['indicators']['net_advances']:.0f}
   ADL ROC (5d): {signal['indicators']['adl_roc_5']:.2f}%
   ADL ROC (10d): {signal['indicators']['adl_roc_10']:.2f}%

💡 INTERPRETATION:
"""
        
        # Add interpretation
        if signal['current_signal'] == 'BOTTOM':
            report += """   🟢 STRONG BOTTOM SIGNAL detected!
   → Market breadth shows extreme oversold conditions
   → Multiple indicators suggest bullish reversal
   → Consider increasing long positions
   → Expected upward movement in 3 days to 1 week
"""
        elif signal['current_signal'] == 'PEAK':
            report += """   🔴 STRONG PEAK SIGNAL detected!
   → Market breadth shows extreme overbought conditions
   → Multiple indicators suggest bearish reversal
   → Consider reducing long exposure or hedging
   → Expected downward movement in 3 days to 1 week
"""
        elif signal['current_signal'] == 'BOTTOMING':
            report += """   🟡 BOTTOMING PROCESS detected
   → Market showing signs of stabilization
   → Not yet confirmed bottom
   → Watch for confirmation in next 3-5 days
   → Cautiously accumulate on weakness
"""
        elif signal['current_signal'] == 'TOPPING':
            report += """   🟠 TOPPING PROCESS detected
   → Market showing signs of exhaustion
   → Not yet confirmed peak
   → Watch for confirmation in next 3-5 days
   → Consider taking partial profits
"""
        else:
            report += """   ⚪ NEUTRAL - No clear signal
   → Market breadth indicators are mixed
   → No extreme conditions detected
   → Monitor for developing patterns
"""
        
        report += "\n" + "="*62
        
        return report


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Peak/Bottom Detector')
    parser.add_argument('--db', type=str, default=os.path.join(SCRIPT_DIR, 'price_data.db'),
                       help='Path to database file')
    parser.add_argument('--lookback', type=int, default=90,
                       help='Lookback period in days (default: 90)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Detect command (default)
    detect_parser = subparsers.add_parser('detect', help='Detect market peaks and bottoms')
    
    # Calculate command
    calc_parser = subparsers.add_parser('calculate', help='Calculate missing breadth indicators')
    calc_parser.add_argument('--from-tickers', action='store_true',
                            help='Calculate from individual ticker MACD data')
    
    # Calculate-all command
    calc_all_parser = subparsers.add_parser('calculate-all', help='Calculate all historical breadth data')
    calc_all_parser.add_argument('--days', type=int, default=365,
                                help='Number of days to calculate (default: 365)')
    calc_all_parser.add_argument('--force', action='store_true',
                                help='Recalculate even if data exists')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update today\'s breadth data')
    
    args = parser.parse_args()
    
    # Create detector instance
    detector = MarketPeakBottomDetector(args.db)
    
    # Execute command
    if args.command == 'calculate':
        print(f"\n{'='*60}")
        print("Calculate Missing Breadth Indicators")
        print(f"{'='*60}\n")
        
        if args.from_tickers:
            success, message, rows = detector.calculate_breadth_indicators_from_tickers(debug=args.debug)
        else:
            success, message, rows = detector.calculate_missing_indicators(debug=args.debug)
        
        print(f"\n{'✅ Success' if success else '❌ Failed'}: {message}")
        print(f"Rows affected: {rows}")
        
        return 0 if success else 1
    
    elif args.command == 'calculate-all':
        print(f"\n{'='*60}")
        print(f"Calculate {args.days} Days of Historical Breadth Data")
        print(f"{'='*60}\n")
        
        success, message, rows = calculate_all_historical_breadth(
            detector, 
            days=args.days, 
            force=args.force, 
            debug=args.debug
        )
        
        print(f"\n{'✅ Success' if success else '❌ Failed'}: {message}")
        print(f"Days calculated: {rows}")
        
        return 0 if success else 1
    
    elif args.command == 'update':
        print(f"\n{'='*60}")
        print("Update Today's Breadth Data")
        print(f"{'='*60}\n")
        
        success, message, rows = update_todays_breadth(detector, debug=args.debug)
        
        print(f"\n{'✅ Success' if success else '❌ Failed'}: {message}")
        
        return 0 if success else 1
    
    else:
        # Default: run detection
        result = detector.run_detection(lookback_days=args.lookback, debug=args.debug)
        
        # Print report
        print(detector.format_report(result))
        
        # Return exit code based on signal
        if result['success']:
            signal = result['signal']['current_signal']
            if signal in ['BOTTOM', 'BOTTOMING']:
                return 0  # Bullish
            elif signal in ['PEAK', 'TOPPING']:
                return 2  # Bearish
            else:
                return 1  # Neutral
        else:
            return 3  # Error


def calculate_all_historical_breadth(detector, days=365, force=False, debug=False):
    """
    Calculate breadth indicators for all historical dates.
    
    Args:
        detector: MarketPeakBottomDetector instance
        days: Number of days back to calculate
        force: If True, recalculate even if data exists
        debug: Enable debug output
    
    Returns:
        Tuple of (success: bool, message: str, days_calculated: int)
    """
    try:
        # Import required modules
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        from ta_dashboard import get_all_tickers, load_price_range, macd_hist
        
        # Get all tickers
        all_tickers = get_all_tickers(db_path=detector.db_path, debug=False)
        
        if not all_tickers:
            return (False, "No tickers found in database", 0)
        
        if debug:
            print(f"Found {len(all_tickers)} tickers")
            print(f"Calculating {days} days of historical breadth data...")
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Build date list (skip weekends)
        calc_dates = []
        current_date = end_date
        while current_date >= start_date:
            # Skip weekends (rough check - doesn't account for holidays)
            if current_date.weekday() < 5:  # Monday=0 to Friday=4
                calc_dates.append(current_date)
            current_date -= timedelta(days=1)
        
        calc_dates.reverse()  # Process oldest to newest
        
        if debug:
            print(f"Will calculate {len(calc_dates)} trading days")
        
        # Connect to database
        conn = sqlite3.connect(detector.db_path)
        cursor = conn.cursor()
        
        # Ensure columns exist
        cursor.execute("PRAGMA table_info(market_breadth_history)")
        existing_columns = {col[1] for col in cursor.fetchall()}
        
        required_cols = ['net_advances', 'ad_ratio', 'ad_line', 'mcclellan_oscillator', 'mcclellan_summation']
        for col in required_cols:
            if col not in existing_columns:
                cursor.execute(f"ALTER TABLE market_breadth_history ADD COLUMN {col} REAL")
        
        conn.commit()
        
        # Calculate breadth for each date
        days_calculated = 0
        days_skipped = 0
        cumulative_net_advances = 0
        net_advances_history = []
        
        for idx, calc_date in enumerate(calc_dates):
            date_str = calc_date.strftime('%Y-%m-%d')
            
            if debug and (idx + 1) % 10 == 0:
                print(f"  Processing {idx + 1}/{len(calc_dates)}: {date_str}")
            
            # Check if already calculated (unless force=True)
            if not force:
                cursor.execute("""
                    SELECT net_advances FROM market_breadth_history
                    WHERE date = ? AND net_advances IS NOT NULL
                """, (date_str,))
                
                if cursor.fetchone():
                    days_skipped += 1
                    # Need to load existing values for cumulative calculations
                    cursor.execute("""
                        SELECT net_advances FROM market_breadth_history WHERE date = ?
                    """, (date_str,))
                    result = cursor.fetchone()
                    if result and result[0] is not None:
                        net_advances_history.append(float(result[0]))
                        cumulative_net_advances += float(result[0])
                    continue
            
            # Calculate breadth for this date
            advancing = 0
            declining = 0
            
            # Need data from before this date for MACD calculation
            calc_start = calc_date - timedelta(days=300)
            
            for ticker in all_tickers:
                try:
                    df = load_price_range(ticker, calc_start.strftime('%Y-%m-%d'), date_str, db_path=detector.db_path)
                    
                    if df.empty or len(df) < 50:
                        continue
                    
                    # Find this specific date in the dataframe
                    date_mask = df['date'].dt.date == calc_date
                    if not date_mask.any():
                        continue
                    
                    date_idx = df[date_mask].index[0]
                    
                    # Calculate MACD up to this date
                    df_up_to_date = df.iloc[:date_idx + 1]
                    _, _, hist = macd_hist(df_up_to_date['close'].astype(float))
                    
                    # Check latest MACD histogram value
                    if len(hist) > 0 and not pd.isna(hist.iloc[-1]):
                        if hist.iloc[-1] > 0:
                            advancing += 1
                        else:
                            declining += 1
                
                except Exception as e:
                    if debug:
                        print(f"    Error processing {ticker} for {date_str}: {e}")
                    continue
            
            # Skip if no data
            if advancing == 0 and declining == 0:
                days_skipped += 1
                continue
            
            # Calculate indicators
            net_advances = advancing - declining
            net_advances_history.append(net_advances)
            
            ad_ratio = advancing / max(declining, 1)
            
            cumulative_net_advances += net_advances
            ad_line = cumulative_net_advances
            
            # Calculate McClellan Oscillator (need at least 39 days)
            if len(net_advances_history) >= 39:
                net_adv_series = pd.Series(net_advances_history)
                ema19 = net_adv_series.ewm(span=19, adjust=False, min_periods=1).mean().iloc[-1]
                ema39 = net_adv_series.ewm(span=39, adjust=False, min_periods=1).mean().iloc[-1]
                mcclellan_osc = ema19 - ema39
                
                # McClellan Summation (cumulative oscillator)
                ema19_series = net_adv_series.ewm(span=19, adjust=False, min_periods=1).mean()
                ema39_series = net_adv_series.ewm(span=39, adjust=False, min_periods=1).mean()
                mcclellan_osc_series = ema19_series - ema39_series
                mcclellan_sum = mcclellan_osc_series.sum()
            else:
                mcclellan_osc = 0
                mcclellan_sum = 0
            
            # Update database
            cursor.execute("""
                UPDATE market_breadth_history
                SET net_advances = ?,
                    ad_ratio = ?,
                    ad_line = ?,
                    mcclellan_oscillator = ?,
                    mcclellan_summation = ?
                WHERE date = ?
            """, (
                float(net_advances),
                float(ad_ratio),
                float(ad_line),
                float(mcclellan_osc),
                float(mcclellan_sum),
                date_str
            ))
            
            if cursor.rowcount > 0:
                days_calculated += 1
        
        conn.commit()
        conn.close()
        
        message = f"Calculated {days_calculated} days, skipped {days_skipped} days (already exist or no data)"
        
        if debug:
            print(f"\n✅ {message}")
        
        return (True, message, days_calculated)
        
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        return (False, f"Error: {e}", 0)


def update_todays_breadth(detector, debug=False):
    """
    Update today's breadth indicators.
    
    Args:
        detector: MarketPeakBottomDetector instance
        debug: Enable debug output
    
    Returns:
        Tuple of (success: bool, message: str, rows: int)
    """
    try:
        # Import required modules
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        from ta_dashboard import get_all_tickers, load_price_range, macd_hist
        
        # Get all tickers
        all_tickers = get_all_tickers(db_path=detector.db_path, debug=False)
        
        if not all_tickers:
            return (False, "No tickers found in database", 0)
        
        # Calculate for today
        today = datetime.now().date()
        date_str = today.strftime('%Y-%m-%d')
        
        if debug:
            print(f"Calculating breadth for {date_str}...")
        
        # Need historical data for MACD
        start_date = today - timedelta(days=300)
        
        advancing = 0
        declining = 0
        
        for ticker in all_tickers:
            try:
                df = load_price_range(ticker, start_date.strftime('%Y-%m-%d'), date_str, db_path=detector.db_path)
                
                if df.empty or len(df) < 50:
                    continue
                
                # Calculate MACD
                _, _, hist = macd_hist(df['close'].astype(float))
                
                # Check latest value
                if len(hist) > 0 and not pd.isna(hist.iloc[-1]):
                    if hist.iloc[-1] > 0:
                        advancing += 1
                    else:
                        declining += 1
            
            except Exception as e:
                if debug:
                    print(f"  Error processing {ticker}: {e}")
                continue
        
        if advancing == 0 and declining == 0:
            return (False, "No ticker data available for today", 0)
        
        # Load historical net advances for cumulative calculations
        conn = sqlite3.connect(detector.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT net_advances FROM market_breadth_history
            WHERE date < ?
            ORDER BY date
        """, (date_str,))
        
        historical = cursor.fetchall()
        net_advances_history = [float(row[0]) for row in historical if row[0] is not None]
        
        # Calculate today's values
        net_advances = advancing - declining
        net_advances_history.append(net_advances)
        
        ad_ratio = advancing / max(declining, 1)
        
        cumulative = sum(net_advances_history)
        ad_line = cumulative
        
        # McClellan indicators
        if len(net_advances_history) >= 39:
            net_adv_series = pd.Series(net_advances_history)
            ema19 = net_adv_series.ewm(span=19, adjust=False, min_periods=1).mean().iloc[-1]
            ema39 = net_adv_series.ewm(span=39, adjust=False, min_periods=1).mean().iloc[-1]
            mcclellan_osc = ema19 - ema39
            
            ema19_series = net_adv_series.ewm(span=19, adjust=False, min_periods=1).mean()
            ema39_series = net_adv_series.ewm(span=39, adjust=False, min_periods=1).mean()
            mcclellan_sum = (ema19_series - ema39_series).sum()
        else:
            mcclellan_osc = 0
            mcclellan_sum = 0
        
        # Update database
        cursor.execute("""
            UPDATE market_breadth_history
            SET net_advances = ?,
                ad_ratio = ?,
                ad_line = ?,
                mcclellan_oscillator = ?,
                mcclellan_summation = ?
            WHERE date = ?
        """, (
            float(net_advances),
            float(ad_ratio),
            float(ad_line),
            float(mcclellan_osc),
            float(mcclellan_sum),
            date_str
        ))
        
        conn.commit()
        conn.close()
        
        if debug:
            print(f"\n✅ Updated {date_str}")
            print(f"  Net Advances: {net_advances}")
            print(f"  A/D Ratio: {ad_ratio:.2f}")
            print(f"  McClellan Osc: {mcclellan_osc:.2f}")
        
        return (True, f"Updated breadth data for {date_str}", 1)
        
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        return (False, f"Error: {e}", 0)


if __name__ == "__main__":
    sys.exit(main())
