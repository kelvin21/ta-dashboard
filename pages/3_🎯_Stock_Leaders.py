"""
Stock Leaders Detection Page
Daily prediction list of stocks with high probability of outperforming VNINDEX.
Based on Relative Strength, RSI, and OBV analysis.
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

# Page config
st.set_page_config(page_title="Stock Leaders Detection", layout="wide", page_icon="🎯")

# Load Material CSS
css_path = SCRIPT_DIR / "styles" / "material.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.markdown("# 🎯 Outperforming Stock Detection")
st.markdown("Daily prediction list using Relative Strength, RSI, and OBV")

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
    Analyze single stock for leadership potential.
    
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
    
    # Calculate indicators
    df = calculate_all_indicators(df)
    
    # Calculate OBV
    if 'volume' in df.columns:
        df['obv'] = calculate_obv(df['close'], df['volume'])
    else:
        return None
    
    # Calculate Relative Strength
    rs = calculate_relative_strength(df['close'], vnindex_data['close'])
    if rs.empty or rs.isna().all():
        return None
    
    df['rs'] = rs
    
    # Get current values
    current = df.iloc[-1]
    
    # RSI values
    rsi_daily = current.get('rsi', np.nan)
    rsi_weekly = rsi_daily  # Simplified - would need weekly data for accurate weekly RSI
    
    # OBV analysis
    obv_analysis = analyze_obv_status(df['obv'])
    
    # RS percentile (will be calculated across universe later)
    rs_current = current['rs']
    
    # Calculate score (without percentile for now)
    score, breakdown = calculate_leader_score(
        rs=df['rs'],
        rs_percentile=0,  # Will update later
        rsi_daily=rsi_daily,
        rsi_weekly=rsi_weekly,
        obv_analysis=obv_analysis
    )
    
    # Classification
    list_class, badge_color, description = classify_prediction_list(score)
    
    # Expectation
    expectation = generate_expectation(score, obv_analysis['status'], rsi_daily)
    
    # Entry triggers
    ema20 = current.get('ema20', np.nan)
    volume_ratio = 1.0  # Simplified
    has_trigger, trigger_reasons = check_entry_trigger(rsi_daily, current['close'], ema20, volume_ratio)
    
    # Exit signals
    rsi_last_3 = df['rsi'].iloc[-3:].tolist() if len(df) >= 3 else []
    should_exit, exit_reasons = check_exit_signal(df['rs'], df['obv'], rsi_last_3)
    
    return {
        'ticker': ticker,
        'close': current['close'],
        'rs_current': rs_current,
        'rs_percentile': 0,  # Will update
        'rsi_daily': rsi_daily,
        'rsi_weekly': rsi_weekly,
        'obv_status': obv_analysis['status'],
        'score': score,
        'score_breakdown': breakdown,
        'list_class': list_class,
        'badge_color': badge_color,
        'description': description,
        'expectation': expectation,
        'has_entry_trigger': has_trigger,
        'entry_triggers': trigger_reasons,
        'should_exit': should_exit,
        'exit_reasons': exit_reasons,
        'ema20': ema20,
        'ema50': current.get('ema50', np.nan)
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
    
    # Score filter
    min_score = st.slider(
        "Minimum Leader Score",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        help="60+ = List B, 75+ = List A"
    )
    
    st.markdown("---")
    
    # Display options
    st.subheader("📊 Display Options")
    
    show_score_breakdown = st.checkbox("Show Score Breakdown", value=True)
    show_entry_triggers = st.checkbox("Show Entry Triggers", value=True)
    show_exit_signals = st.checkbox("Show Exit Signals", value=False)
    
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
    # Recalculate score with percentile
    score, breakdown = calculate_leader_score(
        rs=pd.Series([result['rs_current']]),  # Simplified
        rs_percentile=result['rs_percentile'],
        rsi_daily=result['rsi_daily'],
        rsi_weekly=result['rsi_weekly'],
        obv_analysis={'above_ema20': 'Above' in result['obv_status'], 'score': 15 if 'Accumulating' in result['obv_status'] else 0}
    )
    result['score'] = score
    result['score_breakdown'] = breakdown
    result['list_class'], result['badge_color'], result['description'] = classify_prediction_list(score)

# Filter by minimum score
filtered_results = [r for r in results if r['score'] >= min_score]

# Sort by score
filtered_results.sort(key=lambda x: x['score'], reverse=True)

st.markdown("---")

# Summary Statistics
st.markdown("## 📊 Prediction List Summary")

col1, col2, col3, col4 = st.columns(4)

list_a_count = sum(1 for r in filtered_results if r['list_class'] == 'List A')
list_b_count = sum(1 for r in filtered_results if r['list_class'] == 'List B')
total_analyzed = len(results)
avg_score = np.mean([r['score'] for r in filtered_results]) if filtered_results else 0

with col1:
    st.metric("List A (High Conviction)", list_a_count, delta=None)

with col2:
    st.metric("List B (Watchlist)", list_b_count, delta=None)

with col3:
    st.metric("Total Analyzed", total_analyzed, delta=None)

with col4:
    st.metric("Avg Score", f"{avg_score:.1f}", delta=None)

st.markdown("---")

# Display prediction lists
if not filtered_results:
    st.info(f"No stocks meet the minimum score threshold of {min_score}.")
else:
    st.markdown("## 🎯 Daily Prediction List")
    st.caption(f"Showing {len(filtered_results)} stocks ranked by leader score")
    
    # Create results table
    for idx, result in enumerate(filtered_results[:50], start=1):  # Limit to top 50
        badge_class = f"chip-{result['badge_color']}"
        
        # Create expandable card
        with st.expander(f"#{idx} | {result['ticker']} | Score: {result['score']} | {result['list_class']}", expanded=(idx <= 3)):
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                st.markdown(f"**Price:** {result['close']:.2f}")
                st.markdown(f"**RS Percentile:** {result['rs_percentile']:.1f}%")
                st.markdown(f"**RSI:** {result['rsi_daily']:.1f}")
                
            with col2:
                st.markdown(f"**OBV Status:** {result['obv_status']}")
                st.markdown(f"**Classification:** `{result['list_class']}`")
                st.markdown(f"**Expectation:** {result['expectation']}")
            
            with col3:
                if show_score_breakdown:
                    st.markdown("**Score Breakdown:**")
                    breakdown_text = format_score_breakdown(result['score_breakdown'])
                    st.code(breakdown_text, language=None)
            
            # Entry triggers
            if show_entry_triggers and result['has_entry_trigger']:
                st.success(f"✅ Entry Triggers: {', '.join(result['entry_triggers'])}")
            
            # Exit signals
            if show_exit_signals and result['should_exit']:
                st.error(f"⚠️ Exit Signals: {', '.join(result['exit_reasons'])}")

st.markdown("---")

# Export functionality
st.markdown("## 💾 Export Results")

if filtered_results:
    # Create DataFrame for export
    export_df = pd.DataFrame([
        {
            'Rank': idx + 1,
            'Ticker': r['ticker'],
            'Score': r['score'],
            'List': r['list_class'],
            'RS %ile': f"{r['rs_percentile']:.1f}",
            'RSI': f"{r['rsi_daily']:.1f}",
            'OBV Status': r['obv_status'],
            'Expectation': r['expectation'],
            'Price': f"{r['close']:.2f}",
            'Entry Trigger': 'Yes' if r['has_entry_trigger'] else 'No'
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
st.markdown("_Stock Leaders Detection • RS + RSI + OBV • Daily EOD Analysis_")
