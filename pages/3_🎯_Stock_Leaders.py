"""
Stock Leaders Detection Page
Daily prediction list of stocks with high probability of outperforming VNINDEX.
Primarily based on Relative Strength (RS) vs VNINDEX.
"""
import os
import sys
from pathlib import Path
from typing import Dict
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import utilities
try:
    from utils.indicators import calculate_all_indicators, calculate_rsi
    from utils.relative_strength import (
        calculate_relative_strength, calculate_rs_percentile, is_rs_near_high,
        calculate_obv, analyze_obv_status, calculate_leader_score,
        classify_prediction_list, generate_expectation, check_entry_trigger,
        check_exit_signal, format_score_breakdown
    )
    from utils.db_async import get_sync_db_adapter
except ImportError as e:
    st.error(f"Failed to import utility modules: {e}")
    st.stop()

# Load Material CSS
css_path = SCRIPT_DIR / "styles" / "material.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.markdown("# 🎯 Outperforming Stock Detection")
st.markdown("Daily prediction list based on **Relative Strength vs VNINDEX**")

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

@st.cache_resource(ttl=3600)
def get_db():
    """Get database adapter (cached)."""
    return get_sync_db_adapter()

@st.cache_data(ttl=1800)
def get_latest_date_from_db():
    """Get latest date from database."""
    try:
        db = get_db()
        return db.get_latest_date()
    except Exception as e:
        st.error(f"Error getting latest date: {e}")
        return None

@st.cache_data(ttl=1800)
def get_all_tickers_cached():
    """Get all tickers from database (cached)."""
    try:
        db = get_db()
        return db.get_all_tickers()
    except Exception as e:
        st.error(f"Error getting tickers: {e}")
        return []

def load_price_data_for_ticker(ticker: str, start_date: datetime, end_date: datetime):
    """Load price data for a single ticker."""
    try:
        db = get_db()
        return db.get_price_data(ticker, start_date, end_date)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def analyze_stock_leadership(ticker: str, analysis_date: datetime, vnindex_data: pd.DataFrame) -> Dict:
    """
    Analyze single stock for leadership potential based primarily on RS.
    
    Args:
        ticker: Stock ticker
        analysis_date: Analysis date
        vnindex_data: VNINDEX price data for RS calculation
    
    Returns:
        Dictionary with analysis results
    """
    # Load stock data (6 months for weekly analysis)
    start_date = analysis_date - timedelta(days=180)
    df = load_price_data_for_ticker(ticker, start_date, analysis_date)
    
    if df.empty or len(df) < 20:
        return None
    
    # Calculate Relative Strength (PRIMARY METRIC)
    rs = calculate_relative_strength(df['close'], vnindex_data['close'])
    if rs.empty or rs.isna().all():
        return None
    
    df['rs'] = rs
    
    # Calculate RS MA20 for crossover detection
    df['rs_ma20'] = df['rs'].rolling(window=20, min_periods=20).mean()
    
    # Calculate RSI on RS values (to detect overbought RS)
    rs_rsi = calculate_rsi(df['rs'], period=14)
    df['rs_rsi'] = rs_rsi
    
    # Get current values
    current = df.iloc[-1]
    rs_current = current['rs']
    rs_ma20 = current.get('rs_ma20', np.nan)
    rs_rsi_value = current.get('rs_rsi', np.nan)
    
    # Calculate distance from RS MA20 (percentage)
    rs_distance_pct = np.nan
    if not np.isnan(rs_ma20) and rs_ma20 != 0:
        rs_distance_pct = ((rs_current - rs_ma20) / rs_ma20) * 100
    
    # Flag overextended RS (potential distribution zone)
    distribution_warning = False
    distribution_reason = []
    
    if not np.isnan(rs_rsi_value) and rs_rsi_value > 70:
        distribution_warning = True
        distribution_reason.append(f"RS RSI Overbought ({rs_rsi_value:.1f})")
    
    if not np.isnan(rs_distance_pct) and rs_distance_pct > 10:
        distribution_warning = True
        distribution_reason.append(f"RS {rs_distance_pct:.1f}% Above MA20")
    
    # Detect RS crossover with MA20
    rs_crossover_signal = None
    rs_crossover_status = 'None'
    
    if len(df) >= 21 and not np.isnan(rs_ma20):
        prev = df.iloc[-2]
        rs_prev = prev['rs']
        rs_ma20_prev = prev.get('rs_ma20', np.nan)
        
        if not np.isnan(rs_ma20_prev):
            # Just crossed above
            if rs_prev <= rs_ma20_prev and rs_current > rs_ma20:
                rs_crossover_signal = 'Bullish Cross'
                rs_crossover_status = '🚀 Just Crossed Above'
            # Just crossed below
            elif rs_prev >= rs_ma20_prev and rs_current < rs_ma20:
                rs_crossover_signal = 'Bearish Cross'
                rs_crossover_status = '⚠️ Just Crossed Below'
            # Close to crossing above (within 2% of MA20 and below it)
            elif rs_current < rs_ma20 and (rs_ma20 - rs_current) / rs_ma20 <= 0.02:
                rs_crossover_signal = 'Near Bullish'
                rs_crossover_status = '📈 Near Cross Above'
            # Above MA20
            elif rs_current > rs_ma20:
                rs_crossover_status = '✓ Above MA20'
            else:
                rs_crossover_status = 'Below MA20'
    
    # Optional: Calculate indicators if available
    try:
        df = calculate_all_indicators(df)
        rsi_daily = current.get('rsi', np.nan)
    except:
        rsi_daily = np.nan
    
    # Optional: Calculate OBV if available
    try:
        if 'volume' in df.columns:
            df['obv'] = calculate_obv(df['close'], df['volume'])
            obv_analysis = analyze_obv_status(df['obv'])
            obv_status = obv_analysis['status']
        else:
            obv_status = 'N/A'
    except:
        obv_status = 'N/A'
    
    # Simple RS-based score (will be overwritten with percentile later)
    score = 50  # Placeholder
    
    # Add bonus for RS crossover signals
    if rs_crossover_signal == 'Bullish Cross':
        score += 15  # Just crossed - highest priority
    elif rs_crossover_signal == 'Near Bullish':
        score += 10  # About to cross
    
    # Reduce score for overextended RS (distribution warning)
    if distribution_warning:
        score -= 10  # Penalize overextended stocks
    
    return {
        'ticker': ticker,
        'close': current['close'],
        'rs_current': rs_current,
        'rs_ma20': rs_ma20,
        'rs_rsi': rs_rsi_value,
        'rs_distance_pct': rs_distance_pct,
        'rs_crossover_signal': rs_crossover_signal,
        'rs_crossover_status': rs_crossover_status,
        'distribution_warning': distribution_warning,
        'distribution_reason': ', '.join(distribution_reason) if distribution_reason else 'None',
        'rs_percentile': 0,  # Will update later
        'rsi_daily': rsi_daily,
        'obv_status': obv_status,
        'score': score,
        'list_class': 'Monitor',
        'badge_color': 'gray',
        'description': 'Pending classification',
    }

# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================

with st.sidebar:
    st.header("📅 Analysis Settings")
    
    # Get latest date
    latest_date = get_latest_date_from_db()
    
    if latest_date is None:
        st.error("No data found in database")
        st.stop()
    
    st.info(f"Latest data: {latest_date.strftime('%Y-%m-%d')}")
    
    # Date selector
    selected_date = st.date_input(
        "Analysis Date",
        value=latest_date.date(),
        max_value=latest_date.date()
    )
    analysis_datetime = datetime.combine(selected_date, datetime.min.time())
    
    st.markdown("---")
    
    # Universe selection
    st.subheader("🌍 Universe Selection")
    universe = st.radio(
        "Select Universe",
        options=["VN30", "HOSE", "All"],
        index=0,
        help="VN30 recommended for focused leaders"
    )
    
    # RS Percentile filter
    min_percentile = st.slider(
        "Minimum RS Percentile",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        help="Higher RS = outperforming VNINDEX"
    )
    
    st.markdown("---")
    
    # Crossover filter
    st.subheader("🎯 RS Crossover Filter")
    filter_crossover = st.selectbox(
        "Show Only",
        options=["All Stocks", "RS Crossovers Only", "Near Crossover", "Above MA20"],
        index=0,
        help="Filter by RS vs MA20 status"
    )
    
    # Distribution filter
    hide_overextended = st.checkbox(
        "Hide Overextended (Distribution Zone)",
        value=False,
        help="Hide stocks with RS RSI >70 or RS >10% above MA20"
    )
    
    st.markdown("---")
    
    # Display options
    st.subheader("📊 Display Options")
    
    show_rsi = st.checkbox("Show RSI (Optional)", value=True)
    show_obv = st.checkbox("Show OBV (Optional)", value=False)
    show_rs_ma20 = st.checkbox("Show RS MA20", value=True)
    show_rs_indicators = st.checkbox("Show RS RSI & Distance", value=True)
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# =====================================================================
# MAIN CONTENT
# =====================================================================

# Load VNINDEX data
st.markdown("## 📈 Loading VNINDEX Reference Data...")
vnindex_start = analysis_datetime - timedelta(days=180)
vnindex_data = load_price_data_for_ticker('VNINDEX', vnindex_start, analysis_datetime)

if vnindex_data.empty:
    st.error("❌ Could not load VNINDEX data. Please check database.")
    st.stop()

vnindex_data = calculate_all_indicators(vnindex_data)
vnindex_current = vnindex_data.iloc[-1]

# Display VNINDEX overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="material-card elevation-2" style="text-align: center; padding: 1.5rem;">
        <div style="color: #666; font-size: 0.875rem; margin-bottom: 0.5rem;">VNINDEX</div>
        <div style="font-size: 2rem; font-weight: bold; color: #1976D2;">{:.2f}</div>
        <div style="color: #4CAF50; font-size: 0.875rem; margin-top: 0.5rem;">Reference Index</div>
    </div>
    """.format(vnindex_current['close']), unsafe_allow_html=True)

with col2:
    rsi_vn = vnindex_current.get('rsi', 50)
    rsi_color = '#4CAF50' if rsi_vn >= 50 else '#F44336'
    st.markdown(f"""
    <div class="material-card elevation-2" style="text-align: center; padding: 1.5rem;">
        <div style="color: #666; font-size: 0.875rem; margin-bottom: 0.5rem;">VN RSI</div>
        <div style="font-size: 2rem; font-weight: bold; color: {rsi_color};">{rsi_vn:.1f}</div>
        <div style="color: #666; font-size: 0.875rem; margin-top: 0.5rem;">Market Momentum</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    ema_status = "✓ Above" if vnindex_current['close'] > vnindex_current.get('ema50', 0) else "✗ Below"
    ema_color = '#4CAF50' if "Above" in ema_status else '#F44336'
    st.markdown(f"""
    <div class="material-card elevation-2" style="text-align: center; padding: 1.5rem;">
        <div style="color: #666; font-size: 0.875rem; margin-bottom: 0.5rem;">EMA50 Status</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {ema_color};">{ema_status}</div>
        <div style="color: #666; font-size: 0.875rem; margin-top: 0.5rem;">Trend Filter</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="material-card elevation-2" style="text-align: center; padding: 1.5rem;">
        <div style="color: #666; font-size: 0.875rem; margin-bottom: 0.5rem;">Analysis Date</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #1976D2;">{selected_date.strftime('%d %b')}</div>
        <div style="color: #666; font-size: 0.875rem; margin-top: 0.5rem;">{selected_date.year}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Analyze all stocks
st.markdown("## 🔍 Analyzing Stock Universe...")

all_tickers = get_all_tickers_cached()

# Filter universe
if universe == "VN30":
    vn30_tickers = ['ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
                    'KBC', 'KDH', 'MBB', 'MSN', 'MWG', 'NLG', 'NVL', 'PDR', 'PLX', 'PNJ',
                    'POW', 'SAB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIC',
                    'VJC', 'VNM', 'VPB', 'VRE']
    filtered_tickers = [t for t in all_tickers if t in vn30_tickers]
elif universe == "HOSE":
    filtered_tickers = all_tickers  # Simplified - would need exchange filter
else:
    filtered_tickers = all_tickers

progress_bar = st.progress(0)
status_text = st.empty()

results = []
rs_values = []

for idx, ticker in enumerate(filtered_tickers):
    if ticker == 'VNINDEX':
        continue
    
    status_text.text(f"Analyzing {ticker} ({idx + 1}/{len(filtered_tickers)})...")
    
    try:
        analysis = analyze_stock_leadership(ticker, analysis_datetime, vnindex_data)
        if analysis:
            results.append(analysis)
            rs_values.append(analysis['rs_current'])
    except Exception as e:
        pass
    
    progress_bar.progress((idx + 1) / len(filtered_tickers))

progress_bar.empty()
status_text.empty()

if not results:
    st.warning("⚠️ No stocks analyzed. Please check data availability.")
    st.stop()

# Calculate RS percentiles
for result in results:
    result['rs_percentile'] = calculate_rs_percentile(result['rs_current'], rs_values)
    # Score is primarily based on RS percentile
    result['score'] = result['rs_percentile']
    
    # Reclassify based on RS percentile
    if result['rs_percentile'] >= 80:
        result['list_class'] = 'List A'
        result['badge_color'] = 'green'
        result['description'] = 'Top 20% RS - Strong Leader'
    elif result['rs_percentile'] >= 70:
        result['list_class'] = 'List B'
        result['badge_color'] = 'blue'
        result['description'] = 'Top 30% RS - Watchlist'
    else:
        result['list_class'] = 'List C'
        result['badge_color'] = 'gray'
        result['description'] = 'Below 70% RS - Monitor'

# Filter by minimum RS percentile
filtered_results = [r for r in results if r['rs_percentile'] >= min_percentile]

# Apply crossover filter
if filter_crossover == "RS Crossovers Only":
    filtered_results = [r for r in filtered_results if r.get('rs_crossover_signal') == 'Bullish Cross']
elif filter_crossover == "Near Crossover":
    filtered_results = [r for r in filtered_results if r.get('rs_crossover_signal') in ['Bullish Cross', 'Near Bullish']]
elif filter_crossover == "Above MA20":
    filtered_results = [r for r in filtered_results if 'Above MA20' in r.get('rs_crossover_status', '')]

# Apply distribution filter
if hide_overextended:
    filtered_results = [r for r in filtered_results if not r.get('distribution_warning', False)]
elif filter_crossover == "Above MA20":
    filtered_results = [r for r in filtered_results if 'Above MA20' in r.get('rs_crossover_status', '')]

# Sort by score (prioritizes recent crossovers due to bonus)
filtered_results.sort(key=lambda x: x['score'], reverse=True)

st.markdown("---")

# Summary Statistics
st.markdown("## 📊 Prediction List Summary")

col1, col2, col3, col4, col5, col6 = st.columns(6)

list_a_count = sum(1 for r in filtered_results if r['list_class'] == 'List A')
list_b_count = sum(1 for r in filtered_results if r['list_class'] == 'List B')
crossover_count = sum(1 for r in filtered_results if r.get('rs_crossover_signal') == 'Bullish Cross')
near_crossover_count = sum(1 for r in filtered_results if r.get('rs_crossover_signal') == 'Near Bullish')
distribution_count = sum(1 for r in results if r.get('distribution_warning', False))
total_analyzed = len(results)

with col1:
    st.metric("List A (High Conviction)", list_a_count, delta=None)

with col2:
    st.metric("List B (Watchlist)", list_b_count, delta=None)

with col3:
    st.metric("🚀 RS Crossovers", crossover_count, delta=None, help="Just crossed above MA20")

with col4:
    st.metric("📈 Near Cross", near_crossover_count, delta=None, help="About to cross MA20")

with col5:
    st.metric("⚠️ Overextended", distribution_count, delta=None, help="RS RSI >70 or >10% above MA20")

with col6:
    st.metric("Total Analyzed", total_analyzed, delta=None)

st.markdown("---")

# Display prediction lists
if not filtered_results:
    st.info(f"No stocks meet the minimum RS percentile threshold of {min_percentile}%.")
else:
    st.markdown("## 🎯 Daily Prediction List")
    st.caption(f"Showing {len(filtered_results)} stocks ranked by RS percentile")
    
    # Create results table
    for idx, result in enumerate(filtered_results[:50], start=1):  # Limit to top 50
        badge_class = f"chip-{result['badge_color']}"
        
        # Add crossover indicator to title
        crossover_indicator = ""
        if result.get('rs_crossover_signal') == 'Bullish Cross':
            crossover_indicator = " 🚀"
        elif result.get('rs_crossover_signal') == 'Near Bullish':
            crossover_indicator = " 📈"
        elif result.get('distribution_warning', False):
            crossover_indicator = " ⚠️"
        
        # Create expandable card
        with st.expander(f"#{idx} | {result['ticker']}{crossover_indicator} | RS: {result['rs_percentile']:.1f}% | {result['list_class']}", expanded=(idx <= 3)):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Price:** {result['close']:.2f}")
                st.markdown(f"**RS Percentile:** {result['rs_percentile']:.1f}%")
                st.markdown(f"**RS Value:** {result['rs_current']:.2f}")
                if show_rs_ma20 and not np.isnan(result.get('rs_ma20', np.nan)):
                    st.markdown(f"**RS MA20:** {result['rs_ma20']:.2f}")
                if show_rs_indicators:
                    if not np.isnan(result.get('rs_rsi', np.nan)):
                        rs_rsi_color = '🔴' if result['rs_rsi'] > 70 else '🟢' if result['rs_rsi'] < 30 else '🟡'
                        st.markdown(f"**RS RSI:** {rs_rsi_color} {result['rs_rsi']:.1f}")
                    if not np.isnan(result.get('rs_distance_pct', np.nan)):
                        st.markdown(f"**RS vs MA20:** {result['rs_distance_pct']:+.1f}%")
                st.markdown(f"**Classification:** `{result['list_class']}`")
                
            with col2:
                st.markdown(f"**Description:** {result['description']}")
                
                # Distribution warning
                if result.get('distribution_warning', False):
                    st.error(f"⚠️ **Overextended:** {result['distribution_reason']}")
                
                # Highlight crossover status
                crossover_status = result.get('rs_crossover_status', 'None')
                if '🚀' in crossover_status or '📈' in crossover_status:
                    st.success(f"**RS Status:** {crossover_status}")
                elif '⚠️' in crossover_status:
                    st.warning(f"**RS Status:** {crossover_status}")
                else:
                    st.markdown(f"**RS Status:** {crossover_status}")
                    
                if show_rsi and not np.isnan(result['rsi_daily']):
                    st.markdown(f"**RSI (Daily):** {result['rsi_daily']:.1f}")
                if show_obv and result['obv_status'] != 'N/A':
                    st.markdown(f"**OBV Status:** {result['obv_status']}")

st.markdown("---")

# Export functionality
st.markdown("## 💾 Export Results")

if filtered_results:
    # Create DataFrame for export
    export_df = pd.DataFrame([
        {
            'Rank': idx + 1,
            'Ticker': r['ticker'],
            'RS Percentile': f"{r['rs_percentile']:.1f}",
            'RS Value': f"{r['rs_current']:.2f}",
            'RS MA20': f"{r.get('rs_ma20', 0):.2f}" if not np.isnan(r.get('rs_ma20', np.nan)) else 'N/A',
            'RS RSI': f"{r.get('rs_rsi', 0):.1f}" if not np.isnan(r.get('rs_rsi', np.nan)) else 'N/A',
            'RS Distance %': f"{r.get('rs_distance_pct', 0):+.1f}" if not np.isnan(r.get('rs_distance_pct', np.nan)) else 'N/A',
            'RS Status': r.get('rs_crossover_status', 'None'),
            'Crossover Signal': r.get('rs_crossover_signal', 'None'),
            'Distribution Warning': 'Yes' if r.get('distribution_warning', False) else 'No',
            'Distribution Reason': r.get('distribution_reason', 'None'),
            'List': r['list_class'],
            'Price': f"{r['close']:.2f}",
            'RSI': f"{r['rsi_daily']:.1f}" if not np.isnan(r['rsi_daily']) else 'N/A',
            'OBV Status': r['obv_status']
        }
        for idx, r in enumerate(filtered_results[:50])
    ])
    
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction List (CSV)",
        data=csv,
        file_name=f"stock_leaders_{selected_date.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

st.markdown("---")

# Strategy Notes
st.markdown("## 📚 Trading Strategy Notes")

st.markdown("""
<div class="material-card elevation-2" style="padding: 1.5rem;">
<h4>⚠️ Important Reminders</h4>

**DO NOT Buy Immediately**
- Prediction list identifies stocks BEFORE acceleration
- Wait for valid entry trigger

**Valid Entry Triggers:**
- RSI crosses above 55
- Price reclaims EMA20
- Break of consolidation range
- Volume expansion confirms OBV

**Exit Rules:**
- RS breaks below weekly EMA10
- OBV makes lower low
- Daily RSI <45 for 3 consecutive sessions
- Underperforms VNINDEX for 5 sessions

**Stop-Loss:**
- Below EMA50 or last swing low
- Leaders should not lose EMA50 decisively

</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("_Stock Leaders Detection • Relative Strength Focus • Daily EOD Analysis_")

