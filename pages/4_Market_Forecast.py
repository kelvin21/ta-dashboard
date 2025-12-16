"""
Market Peak/Bottom Forecast Page
Displays market turning point predictions using breadth indicators.
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from market_peak_bottom_detector import MarketPeakBottomDetector

st.set_page_config(page_title="Market Forecast", layout="wide", page_icon="🎯")

# Get DB path from main dashboard
try:
    import ta_dashboard
    DB_PATH = ta_dashboard.DB_PATH
except:
    DB_PATH = os.path.join(PARENT_DIR, "price_data.db")

st.markdown("## 🎯 Market Peak/Bottom Forecast")
st.markdown("*Using breadth indicators to predict market turning points*")

# Sidebar controls
st.sidebar.markdown("### Forecast Controls")

# Check if there's a lookback override in session state
if 'lookback_override' in st.session_state:
    default_lookback = st.session_state['lookback_override']
    del st.session_state['lookback_override']  # Clear after use
else:
    default_lookback = 365  # Changed from 90 to 365

lookback_days = st.sidebar.slider("Lookback period (days)", 30, 730, default_lookback, step=10,
                                  help="Number of days to look back for analysis")
auto_calculate = st.sidebar.checkbox("Auto-calculate missing indicators", value=True,
                                     help="Automatically calculate indicators if they're missing")
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 5 min)", value=False)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

st.sidebar.markdown("---")

# Manual calculation button in sidebar
st.sidebar.markdown("### 🛠️ Manual Tools")
if st.sidebar.button("⚙️ Calculate Indicators", use_container_width=True):
    with st.spinner("Calculating breadth indicators..."):
        detector = MarketPeakBottomDetector(DB_PATH)
        success, message, rows = detector.calculate_missing_indicators(debug=show_debug)
        
        if success:
            st.sidebar.success(f"✅ {message}")
        else:
            st.sidebar.error(f"❌ {message}")

if st.sidebar.button("🔄 Refresh Analysis", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 Current Market Signal")
    
    with st.spinner("Analyzing market breadth indicators..."):
        detector = MarketPeakBottomDetector(DB_PATH)
        result = detector.run_detection(lookback_days=lookback_days, debug=show_debug, auto_calculate=auto_calculate)
    
    if not result['success']:
        st.error(f"❌ {result['message']}")
        
        # Provide clear instructions
        st.markdown("---")
        st.markdown("### 📋 Setup Instructions")
        
        # Check if it's a date range issue (most common)
        if result.get('hint') == 'increase_lookback' or "date range" in result.get('message', '').lower():
            st.error(f"❌ {result['message']}")
            
            st.warning("""
            **⚠️ Date Range Issue Detected**
            
            The breadth indicators exist in your database, but no data was found in the requested time period.
            """)
            
            # Show current date range being searched
            from datetime import datetime, timedelta
            search_start = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            search_end = datetime.now().strftime('%Y-%m-%d')
            
            st.info(f"""
            **Current Search Parameters:**
            - Lookback period: **{lookback_days} days**
            - Searching from: **{search_start}** to **{search_end}**
            
            **This usually means:**
            - Your Market Breadth data is from an older date range
            - You need to extend the lookback period
            - Or update Market Breadth data to include recent dates
            """)
            
            # Prominent quick-fix buttons
            st.markdown("### 🚀 Quick Fix Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Try Longer Period**")
                if st.button("📆 180 Days", use_container_width=True, type="primary"):
                    st.session_state['lookback_override'] = 180
                    st.rerun()
                if st.button("📆 365 Days", use_container_width=True):
                    st.session_state['lookback_override'] = 365
                    st.rerun()
                if st.button("📆 730 Days (2 years)", use_container_width=True):
                    st.session_state['lookback_override'] = 730
                    st.rerun()
            
            with col2:
                st.markdown("**Update Data**")
                if st.button("🔄 Update Breadth Data", use_container_width=True, type="primary"):
                    st.switch_page("pages/1_📊_Market_Breadth.py")
                st.caption("Go to Market Breadth page to update indicators with latest data")
            
            with col3:
                st.markdown("**Check Database**")
                if st.button("🔍 Show Available Dates", use_container_width=True):
                    # Query database to show actual date range
                    try:
                        import sqlite3
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM market_breadth_history")
                        min_date, max_date, count = cursor.fetchone()
                        conn.close()
                        
                        if min_date and max_date:
                            st.success(f"""
                            **Database Information:**
                            - Available from: **{min_date}**
                            - Available to: **{max_date}**
                            - Total rows: **{count}**
                            
                            Try setting lookback period to cover this range.
                            """)
                        else:
                            st.warning("No date information found in market_breadth_history table")
                    except Exception as e:
                        st.error(f"Error querying database: {e}")
            
            st.stop()
        
        if result.get('hint') == 'calculate_indicators':
            st.warning("""
            **⚠️ Breadth Indicators Not Found**
            
            The forecast requires these indicators:
            - McClellan Oscillator
            - McClellan Summation  
            - Net Advances
            - Advance/Decline Line
            - A/D Ratio
            """)
            
            # Check if indicators might already exist but data range is wrong
            detector = MarketPeakBottomDetector(DB_PATH)
            success, calc_msg, rows = detector.calculate_missing_indicators(debug=show_debug)
            
            if "already exist" in calc_msg or "already calculated" in calc_msg:
                st.success("""
                ✅ **Good news!** The breadth indicators already exist in your database.
                
                The issue is likely with the date range or data availability.
                """)
                
                st.info("""
                **Try these solutions:**
                
                1. **Increase Lookback Period**: Use the slider in sidebar to increase from 90 to 180+ days
                2. **Check Data Date Range**: Your breadth data might be from an older date range
                3. **Update Breadth Data**: Go to Market Breadth page and click "Update Breadth Data" to refresh
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Retry with Full Range", use_container_width=True):
                        st.session_state['lookback_override'] = 365
                        st.rerun()
                with col2:
                    if st.button("➡️ Update Breadth Data", use_container_width=True):
                        st.switch_page("pages/1_📊_Market_Breadth.py")
            else:
                st.info("""
                **📝 How to Calculate Indicators:**
                
                1. **Option 1: Auto-Calculate**
                   - Check "Auto-calculate missing indicators" in sidebar
                   - Click "Refresh Analysis"
                
                2. **Option 2: Market Breadth Page** (Recommended)
                   - Go to Market Breadth page (📊 button)
                   - Click "Update Breadth Data"
                   - Return here for forecast
                """)
                
                # Add manual calculation button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("⚙️ Calculate Indicators Now", type="primary", use_container_width=True, key="calc_main"):
                        with st.spinner("Calculating breadth indicators..."):
                            success, message, rows = detector.calculate_missing_indicators(debug=show_debug)
                            
                            if success:
                                st.success(f"✅ {message}")
                                st.balloons()
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                                st.info("The breadth data needs to be calculated from Market Breadth page first.")
        
        elif result.get('hint') == 'calculation_failed':
            st.error("""
            **❌ Failed to Calculate Indicators**
            
            There was an error calculating the breadth indicators automatically.
            """)
            
            st.info("""
            **Possible Causes:**
            - Missing source data (advances, declines columns)
            - Database permission issues
            - Data format problems
            
            **Solution:**
            Go to Market Breadth page and use the "Update Breadth Data" button to calculate indicators properly.
            """)
            
            # Show calculate button option
            if st.button("⚙️ Try Manual Calculation", use_container_width=True):
                with st.spinner("Attempting manual calculation..."):
                    success, message, rows = detector.calculate_missing_indicators(debug=True)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                        with st.expander("Show Error Details"):
                            st.code(message)
        
        elif result.get('hint') == 'insufficient_data':
            st.warning("""
            **⚠️ Insufficient Data Points**
            
            Found breadth data but not enough for reliable peak/bottom detection.
            """)
            
            st.info("""
            **Requirement:** At least 20 days of data needed
            
            **Solutions:**
            1. Increase "Lookback period" slider to 180+ days
            2. Go to Market Breadth page and ensure data covers recent dates
            3. Wait for more trading days to accumulate data
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📆 Try 365 Days", use_container_width=True, type="primary"):
                    st.session_state['lookback_override'] = 365
                    st.rerun()
            with col2:
                if st.button("➡️ Update Breadth Data", use_container_width=True):
                    st.switch_page("pages/1_📊_Market_Breadth.py")
            
            st.stop()
        
        else:
            st.info("""
            **Market Forecast requires breadth data to work.**
            
            To populate the required data:
            
            1. **Go to Market Breadth page** (📊 Breadth button in sidebar)
            2. **Load at least 90 days of data** (365+ days recommended)
            3. **Click "Update Breadth Data"** to calculate indicators
            4. **Return here** to see market peak/bottom forecasts
            
            The breadth indicators (McClellan Oscillator, A/D Line, etc.) are stored in the `market_breadth_history` table.
            """)
        
        # Add navigation buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➡️ Go to Market Breadth Page", use_container_width=True):
                st.switch_page("pages/1_📊_Market_Breadth.py")
        with col2:
            if st.button("🔄 Retry Detection", use_container_width=True):
                st.rerun()
        
        st.stop()  # Don't show rest of page
    
    else:
        signal = result['signal']
        
        # Display main signal
        signal_colors = {
            'BOTTOM': 'success',
            'PEAK': 'error',
            'BOTTOMING': 'warning',
            'TOPPING': 'warning',
            'NEUTRAL': 'info'
        }
        
        signal_type = signal_colors.get(signal['current_signal'], 'info')
        
        if signal_type == 'success':
            st.success(f"## 🟢 {signal['current_signal']}")
        elif signal_type == 'error':
            st.error(f"## 🔴 {signal['current_signal']}")
        elif signal_type == 'warning':
            st.warning(f"## 🟡 {signal['current_signal']}")
        else:
            st.info(f"## 🔵 {signal['current_signal']}")
        
        # Confidence meter
        st.markdown(f"**Confidence:** {signal['confidence']}%")
        st.progress(signal['confidence'] / 100)
        
        st.markdown("---")
        
        # Forecasts
        st.markdown("### 📅 Forecasts")
        
        fcol1, fcol2 = st.columns(2)
        
        with fcol1:
            forecast_3d = signal.get('forecast_3d', 'NEUTRAL')
            delta_3d = "Bullish" if forecast_3d == 'BULLISH' else "Bearish" if forecast_3d == 'BEARISH' else None
            st.metric("3-Day Forecast", forecast_3d,
                     delta=delta_3d,
                     delta_color="normal")
        
        with fcol2:
            forecast_1w = signal.get('forecast_1w', 'NEUTRAL')
            delta_1w = "Bullish" if forecast_1w == 'BULLISH' else "Bearish" if forecast_1w == 'BEARISH' else None
            st.metric("1-Week Forecast", forecast_1w,
                     delta=delta_1w,
                     delta_color="normal")

with col2:
    st.markdown("### 📈 Signal Scores")
    
    if result['success']:
        signal = result['signal']
        
        # Bottom score
        bottom_score = signal.get('bottom_score', 0)
        st.metric("Bottom Score", bottom_score, 
                 delta="Strong" if bottom_score >= 4 else None)
        st.progress(min(1.0, bottom_score / 10))
        
        # Peak score
        peak_score = signal.get('peak_score', 0)
        st.metric("Peak Score", peak_score,
                 delta="Strong" if peak_score >= 4 else None)
        st.progress(min(1.0, peak_score / 10))
        
        st.caption("Scores ≥4 indicate significant signals")

# Breadth indicators
if result['success']:
    st.markdown("---")
    st.markdown("### 📊 Key Breadth Indicators")
    
    indicators = signal.get('indicators', {})
    
    # Helper function to safely format indicators
    def format_indicator(value, decimal_places=2, default="N/A"):
        """Safely format indicator value, handling NaN/None."""
        try:
            if value is None:
                return default
            # Check for NaN using math.isnan or numpy
            if isinstance(value, float):
                if np.isnan(value) or value == float('nan') or value == float('inf') or value == float('-inf'):
                    return default
            return f"{float(value):.{decimal_places}f}"
        except (TypeError, ValueError):
            return default
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mcco = indicators.get('mcclellan_oscillator', 0)
        mcco_str = format_indicator(mcco)
        if mcco_str != "N/A":
            mcco_val = float(mcco)
            mcco_delta = "Overbought" if mcco_val > 100 else "Oversold" if mcco_val < -100 else None
        else:
            mcco_delta = None
        st.metric("McClellan Oscillator", mcco_str, delta=mcco_delta)
        
        mcsum = indicators.get('mcclellan_summation', 0)
        mcsum_str = format_indicator(mcsum, 0)
        if mcsum_str != "N/A":
            mcsum_val = float(mcsum)
            mcsum_delta = "Overbought" if mcsum_val > 1000 else "Oversold" if mcsum_val < -1000 else None
        else:
            mcsum_delta = None
        st.metric("McClellan Summation", mcsum_str, delta=mcsum_delta)
    
    with col2:
        ad_ratio = indicators.get('ad_ratio', 1)
        ad_ratio_str = format_indicator(ad_ratio)
        if ad_ratio_str != "N/A":
            ad_ratio_val = float(ad_ratio)
            ad_delta = "Extreme Bullish" if ad_ratio_val > 2.0 else "Extreme Bearish" if ad_ratio_val < 0.5 else None
        else:
            ad_delta = None
        st.metric("A/D Ratio", ad_ratio_str, delta=ad_delta)
        
        net_adv = indicators.get('net_advances', 0)
        net_adv_str = format_indicator(net_adv, 0)
        if net_adv_str != "N/A":
            net_adv_val = float(net_adv)
            net_delta = "Positive" if net_adv_val > 0 else "Negative" if net_adv_val < 0 else None
        else:
            net_delta = None
        st.metric("Net Advances", net_adv_str, delta=net_delta)
    
    with col3:
        adl_roc5 = indicators.get('adl_roc_5', 0)
        adl_roc5_str = format_indicator(adl_roc5)
        if adl_roc5_str != "N/A":
            adl_roc5_val = float(adl_roc5)
            # Cap extreme percentages for display
            if abs(adl_roc5_val) > 100:
                adl_roc5_str = ">100" if adl_roc5_val > 0 else "<-100"
            else:
                adl_roc5_str = f"{adl_roc5_val:.2f}"
            adl5_delta = "Rising" if adl_roc5_val > 0 else "Falling" if adl_roc5_val < 0 else None
        else:
            adl5_delta = None
        st.metric("ADL ROC (5d)", f"{adl_roc5_str}%", delta=adl5_delta)
        
        adl_roc10 = indicators.get('adl_roc_10', 0)
        adl_roc10_str = format_indicator(adl_roc10)
        if adl_roc10_str != "N/A":
            adl_roc10_val = float(adl_roc10)
            # Cap extreme percentages for display
            if abs(adl_roc10_val) > 100:
                adl_roc10_str = ">100" if adl_roc10_val > 0 else "<-100"
            else:
                adl_roc10_str = f"{adl_roc10_val:.2f}"
            adl10_delta = "Rising" if adl_roc10_val > 0 else "Falling" if adl_roc10_val < 0 else None
        else:
            adl10_delta = None
        st.metric("ADL ROC (10d)", f"{adl_roc10_str}%", delta=adl10_delta)
    
    # Add warning if key indicators are missing
    if (mcco_str == "N/A" or mcsum_str == "N/A" or ad_ratio_str == "N/A" or net_adv_str == "N/A"):
        st.warning("""
        ⚠️ **Some indicators are missing data (showing as N/A)**
        
        This usually means the breadth calculations haven't been completed properly.
        
        **To fix:**
        1. Go to Market Breadth page (📊 button in sidebar)
        2. Click "🔄 Recalculate Historical Data" button
        3. Wait for calculation to complete
        4. Return here to see updated indicators
        """)

# Interpretation
st.markdown("---")
st.markdown("### 📜 Interpretation of Signals")

# Current signal
current_signal = signal.get('current_signal', 'NEUTRAL')
if current_signal == 'BOTTOM':
    st.markdown("**Market is in a Bottom phase**: Potential buying opportunity")
elif current_signal == 'PEAK':
    st.markdown("**Market is at a Peak phase**: Potential selling opportunity")
elif current_signal == 'BOTTOMING':
    st.markdown("**Market is Bottoming**: Strengthening bullish signals")
elif current_signal == 'TOPPING':
    st.markdown("**Market is Topping**: Strengthening bearish signals")
else:
    st.markdown("**Market is in a Neutral phase**: No clear action suggested")

# Signal history
st.markdown("---")
st.markdown("### 📈 Signal History")

# Footer with methodology
with st.expander("📚 Methodology"):
    st.markdown("""
    ### Detection Methodology
    
    This forecast uses multiple breadth indicators to detect market turning points:
    
    **Bottom Signals (Bullish):**
    - McClellan Oscillator < -100 and crossing up
    - McClellan Summation < -1000 (extreme oversold)
    - A/D Ratio < 0.5 with improvement
    - Net Advances turning positive
    - ADL momentum turning positive
    
    **Peak Signals (Bearish):**
    - McClellan Oscillator > +100 and crossing down
    - McClellan Summation > +1000 (extreme overbought)
    - A/D Ratio > 2.0 with deterioration
    - Net Advances turning negative
    - ADL momentum turning negative
    
    **Confidence Levels:**
    - Score ≥ 6: High confidence (90%+)
    - Score 4-5: Moderate confidence (60-75%)
    - Score < 4: Low confidence (< 60%)
    
    **Forecast Horizons:**
    - 3-Day: Near-term reaction (intraday to 3 days)
    - 1-Week: Short-term trend (3-7 days)
    
    *Historical accuracy varies by market conditions. Use as one input in your decision-making process.*
    """)

# Auto-refresh - remove any signal history queries
if auto_refresh:
    import time
    time.sleep(300)  # 5 minutes
    st.rerun()
