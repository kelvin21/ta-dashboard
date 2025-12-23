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

# Load Material Design CSS
css_path = SCRIPT_DIR / "styles" / "material.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load FontAwesome
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">', unsafe_allow_html=True)

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

# Title
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 32px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 10px 20px rgba(0,0,0,0.19);">
    <h1 style="color: white; margin: 0; font-size: 42px;"><i class="fas fa-chart-line"></i> Stock Leaders Detection</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 18px;">
        Identify stocks outperforming or underperforming VNINDEX using Relative Strength analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Summary/Introduction Section
st.markdown("### 📊 What is Relative Strength (RS)?")
st.write("""
**Relative Strength** measures how a stock performs compared to the market index (VNINDEX). 
A rising RS line means the stock is **outperforming** the market, while a falling RS indicates **underperformance**.
""")

col_intro1, col_intro2 = st.columns(2)

with col_intro1:
    st.success("**🚀 Outperforming Leaders**")
    st.write("""
    Stocks with high RS percentile (top 20-30%) are **market leaders**. 
    Look for RS crossing above MA20 as entry signals, but avoid overextended stocks.
    """)

with col_intro2:
    st.error("**📉 Underperforming Laggards**")
    st.write("""
    Stocks with low RS percentile (bottom 30%) are **lagging** the market. 
    Consider reducing exposure or looking for mean reversion opportunities.
    """)

st.warning("**💡 Key Insight:** Strong stocks become stronger, weak stocks become weaker. Focus on relative strength, not just price.")

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
    st.markdown("### <i class='far fa-calendar'></i> Analysis Settings", unsafe_allow_html=True)
    
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
    
    st.markdown("### <i class='fas fa-globe'></i> Universe Selection", unsafe_allow_html=True)
    universe = st.radio(
        "Select Universe",
        options=["All", "VN30", "HOSE"],
        index=0,
        help="Analyze all stocks by default"
    )
    
    st.markdown("---")
    
    st.markdown("### <i class='fas fa-filter'></i> Filters", unsafe_allow_html=True)
    
    # RS Percentile filter
    min_percentile = st.slider(
        "Minimum RS Percentile",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="0 = show all stocks, higher values filter for stronger RS"
    )
    
    # Crossover filter
    filter_crossover = st.selectbox(
        "RS Crossover Filter",
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
    
    st.markdown("### <i class='fas fa-cog'></i> Display Options", unsafe_allow_html=True)
    
    show_rsi = st.checkbox("Show RSI (Optional)", value=True)
    show_obv = st.checkbox("Show OBV (Optional)", value=False)
    show_rs_ma20 = st.checkbox("Show RS MA20", value=True)
    show_rs_indicators = st.checkbox("Show RS RSI & Distance", value=True)
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Data", type="primary"):
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
        result['performance_type'] = 'outperforming'
    elif result['rs_percentile'] >= 70:
        result['list_class'] = 'List B'
        result['badge_color'] = 'blue'
        result['description'] = 'Top 30% RS - Watchlist'
        result['performance_type'] = 'outperforming'
    elif result['rs_percentile'] <= 30:
        result['list_class'] = 'Laggard'
        result['badge_color'] = 'red'
        result['description'] = 'Bottom 30% RS - Underperforming'
        result['performance_type'] = 'underperforming'
    else:
        result['list_class'] = 'List C'
        result['badge_color'] = 'gray'
        result['description'] = 'Middle Range RS'
        result['performance_type'] = 'neutral'

# Split into outperforming and underperforming
outperforming_results = [r for r in results if r['performance_type'] == 'outperforming']
underperforming_results = [r for r in results if r['performance_type'] == 'underperforming']

# Filter by minimum RS percentile
outperforming_filtered = [r for r in outperforming_results if r['rs_percentile'] >= min_percentile]
underperforming_filtered = [r for r in underperforming_results if r['rs_percentile'] <= (100 - min_percentile)]

# Apply crossover filter to outperforming
if filter_crossover == "RS Crossovers Only":
    outperforming_filtered = [r for r in outperforming_filtered if r.get('rs_crossover_signal') == 'Bullish Cross']
elif filter_crossover == "Near Crossover":
    outperforming_filtered = [r for r in outperforming_filtered if r.get('rs_crossover_signal') in ['Bullish Cross', 'Near Bullish']]
elif filter_crossover == "Above MA20":
    outperforming_filtered = [r for r in outperforming_filtered if 'Above MA20' in r.get('rs_crossover_status', '')]

# Apply distribution filter
if hide_overextended:
    outperforming_filtered = [r for r in outperforming_filtered if not r.get('distribution_warning', False)]

# Sort by score (prioritizes recent crossovers due to bonus)
outperforming_filtered.sort(key=lambda x: x['score'], reverse=True)
underperforming_filtered.sort(key=lambda x: x['score'], reverse=False)  # Lowest RS first

st.markdown("---")

# Summary Statistics
st.markdown("## 📊 Market RS Distribution")

col1, col2, col3, col4, col5, col6 = st.columns(6)

list_a_count = sum(1 for r in outperforming_results if r['list_class'] == 'List A')
list_b_count = sum(1 for r in outperforming_results if r['list_class'] == 'List B')
laggard_count = len(underperforming_results)
crossover_count = sum(1 for r in outperforming_filtered if r.get('rs_crossover_signal') == 'Bullish Cross')
near_crossover_count = sum(1 for r in outperforming_filtered if r.get('rs_crossover_signal') == 'Near Bullish')
distribution_count = sum(1 for r in results if r.get('distribution_warning', False))
total_analyzed = len(results)

with col1:
    st.metric("🚀 List A (Top 20%)", list_a_count, delta=None)

with col2:
    st.metric("📈 List B (Top 30%)", list_b_count, delta=None)

with col3:
    st.metric("⚡ RS Crossovers", crossover_count, delta=None, help="Just crossed above MA20")

with col4:
    st.metric("📉 Laggards (Bottom 30%)", laggard_count, delta=None)

with col5:
    st.metric("⚠️ Overextended", distribution_count, delta=None, help="RS RSI >70 or >10% above MA20")

with col6:
    st.metric("📊 Total Analyzed", total_analyzed, delta=None)

# Crossover Summary with Ticker Names
bullish_cross_tickers = [r['ticker'] for r in results if r.get('rs_crossover_signal') == 'Bullish Cross']
near_bullish_tickers = [r['ticker'] for r in results if r.get('rs_crossover_signal') == 'Near Bullish']
bearish_cross_tickers = [r['ticker'] for r in results if r.get('rs_crossover_signal') == 'Bearish Cross']
near_bearish_tickers = [r['ticker'] for r in results if r.get('rs_crossover_signal') == 'Near Bearish']

summary_parts = []
if bullish_cross_tickers:
    tickers_str = ", ".join(bullish_cross_tickers)
    summary_parts.append(f"**{len(bullish_cross_tickers)}** stocks just crossed above RS MA20 🚀: {tickers_str}")
if near_bullish_tickers:
    tickers_str = ", ".join(near_bullish_tickers)
    summary_parts.append(f"**{len(near_bullish_tickers)}** stocks are near bullish crossing 📈: {tickers_str}")
if bearish_cross_tickers:
    tickers_str = ", ".join(bearish_cross_tickers)
    summary_parts.append(f"**{len(bearish_cross_tickers)}** stocks just crossed below RS MA20 📉: {tickers_str}")
if near_bearish_tickers:
    tickers_str = ", ".join(near_bearish_tickers)
    summary_parts.append(f"**{len(near_bearish_tickers)}** stocks are near bearish crossing ⚠️: {tickers_str}")

if summary_parts:
    summary_text = "<br/>".join(summary_parts)
    st.markdown(f"""
    <div class="material-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 padding: 16px; border-radius: 8px; margin: 16px 0;">
        <div style="color: white; font-size: 15px; line-height: 1.8;">
            <div style="font-weight: bold; margin-bottom: 8px;">
                <i class="fas fa-chart-line"></i> RS Crossover Activity:
            </div>
            {summary_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Create tabs for Outperforming vs Underperforming
tab1, tab2 = st.tabs([
    f"🚀 Outperforming Leaders ({len(outperforming_filtered)})",
    f"📉 Underperforming Laggards ({len(underperforming_filtered)})"
])

# TAB 1: OUTPERFORMING
with tab1:
    if not outperforming_filtered:
        st.info(f"No outperforming stocks meet the criteria.")
    else:
        st.markdown(f"### Top {len(outperforming_filtered[:50])} Market Leaders")
        st.caption("Sorted by RS percentile - stocks with strongest relative performance")
        st.markdown("")
        
        # Display outperforming stocks in card format
        # Create 3 columns for card layout
        cols_per_row = 3
        for idx, result in enumerate(outperforming_filtered[:50], start=1):  # Limit to top 50
            if (idx - 1) % cols_per_row == 0:
                cols = st.columns(cols_per_row)
            
            col = cols[(idx - 1) % cols_per_row]
            
            with col:
                # Determine momentum badge
                momentum_badge = ""
                momentum_color = "#4CAF50"
                if result.get('rs_crossover_signal') == 'Bullish Cross':
                    momentum_badge = "Strong Momentum"
                    momentum_color = "#4CAF50"
                elif result.get('rs_crossover_signal') == 'Near Bullish':
                    momentum_badge = "Building Momentum"
                    momentum_color = "#2196F3"
                elif result['rs_percentile'] >= 80:
                    momentum_badge = "Strong Momentum"
                    momentum_color = "#4CAF50"
                elif result['rs_percentile'] >= 70:
                    momentum_badge = "Good Momentum"
                    momentum_color = "#2196F3"
                else:
                    momentum_badge = "Moderate"
                    momentum_color = "#9E9E9E"
                
                # Calculate strength rating (1-5 stars)
                strength_rating = min(5, max(1, int((result['rs_percentile'] / 100) * 5) + 1))
                stars = "⭐" * strength_rating
                
                # Determine arrow direction
                arrow = "↑" if result['rs_current'] > result.get('rs_ma20', result['rs_current']) else "↓"
                arrow_color = "#4CAF50" if arrow == "↑" else "#F44336"
                
                # Distribution warning
                warning_badge = ""
                if result.get('distribution_warning', False):
                    dist_pct = result.get('rs_distance_pct', 0)
                    if not np.isnan(dist_pct) and abs(dist_pct) > 0:
                        warning_badge = f'<div style="display: flex; align-items: center; gap: 4px; color: #FF9800; font-size: 12px; margin-top: 4px;"><i class="fas fa-exclamation-triangle"></i><span>{abs(dist_pct):.1f}% Tight</span></div>'
                
                # RS status indicator
                rs_status_text = ""
                rs_ma20_value = result.get('rs_ma20', 0)
                if not np.isnan(rs_ma20_value):
                    rs_diff_pct = ((result['rs_current'] - rs_ma20_value) / rs_ma20_value * 100) if rs_ma20_value != 0 else 0
                    rs_status_text = f"S: RS MA20 ({rs_diff_pct:+.1f}%)"
                else:
                    rs_status_text = "S: RS MA20 (N/A)"
                
                # Violations check
                violations_text = "No recent violations"
                if result.get('distribution_warning', False):
                    violations_text = result.get('distribution_reason', 'Overextended zone')
                
                # Build description text from analysis
                description_parts = []
                if result.get('rs_crossover_signal') == 'Bullish Cross':
                    description_parts.append("Golden cross")
                if result['rs_current'] > result.get('rs_ma20', 0):
                    description_parts.append("bullish RS alignment")
                if result['rs_percentile'] >= 80:
                    description_parts.append("in BUY zone")
                if result.get('rs_rsi', 50) < 70:
                    description_parts.append("RSs converging")
                if result.get('rs_crossover_signal') == 'Near Bullish':
                    description_parts.append("breakout imminent")
                
                description_text = " - ".join(description_parts) if description_parts else result['description']
                
                # Button text based on list class
                button_text = "BUY NOW" if result['list_class'] == 'List A' else "WATCH"
                button_color = "#4CAF50" if result['list_class'] == 'List A' else "#2196F3"
                
                # Escape HTML special characters in text
                import html
                description_escaped = html.escape(description_text)
                violations_escaped = html.escape(violations_text)
                ticker_escaped = html.escape(result['ticker'])
                
                # Create card HTML
                card_html = f'''<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 16px; height: 100%%; position: relative;"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;"><div style="display: flex; align-items: center; gap: 8px;"><span style="font-size: 20px; font-weight: bold; color: #1976D2;">{ticker_escaped}</span><span style="font-size: 18px; color: {arrow_color};">{arrow}</span></div><div style="text-align: right;">{stars}</div></div><div style="margin-bottom: 12px;"><span style="background: {momentum_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 500;">✓ {momentum_badge}</span>{warning_badge}</div><div style="font-size: 32px; font-weight: bold; color: #1976D2; margin-bottom: 8px;">{result['close']:.2f}</div><div style="font-size: 13px; color: #666; margin-bottom: 8px;">{rs_status_text}</div><div style="font-size: 13px; color: #666; margin-bottom: 4px;"><strong>Strength:</strong> {strength_rating}/5</div><div style="font-size: 13px; color: #666; margin-bottom: 12px;">{violations_escaped}</div><div style="margin-bottom: 12px;"><div style="background: {button_color}; color: white; padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 14px; cursor: pointer;">{button_text}</div></div><div style="margin-bottom: 12px;"><div style="background: #2196F3; color: white; padding: 8px; border-radius: 8px; text-align: center; font-size: 12px; cursor: pointer;">READ RS MA20</div></div><div style="font-size: 12px; color: #666; line-height: 1.5; border-top: 1px solid #e0e0e0; padding-top: 12px;">{description_escaped}</div></div>'''
                
                st.markdown(card_html, unsafe_allow_html=True)
        
        # Add spacing after cards
        st.markdown("")

# TAB 2: UNDERPERFORMING
with tab2:
    if not underperforming_filtered:
        st.info(f"No underperforming stocks found.")
    else:
        st.markdown(f"### Bottom {len(underperforming_filtered[:50])} Market Laggards")
        st.caption("Sorted by RS percentile (lowest first) - stocks with weakest relative performance")
        st.markdown("")
        
        # Display underperforming stocks
        for idx, result in enumerate(underperforming_filtered[:50], start=1):
            with st.expander(f"#{idx} | {result['ticker']} | RS: {result['rs_percentile']:.1f}% | {result['list_class']}", expanded=(idx <= 3)):
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
                    st.warning("⚠️ **Weak RS:** Underperforming the market")
                    
                    # RS status
                    crossover_status = result.get('rs_crossover_status', 'None')
                    st.markdown(f"**RS Status:** {crossover_status}")
                        
                    if show_rsi and not np.isnan(result['rsi_daily']):
                        st.markdown(f"**RSI (Daily):** {result['rsi_daily']:.1f}")
                    if show_obv and result['obv_status'] != 'N/A':
                        st.markdown(f"**OBV Status:** {result['obv_status']}")

st.markdown("---")

# Export functionality
st.markdown("## 💾 Export Results")

col1, col2 = st.columns(2)

with col1:
    if outperforming_filtered:
        # Create DataFrame for export
        export_df_out = pd.DataFrame([
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
            for idx, r in enumerate(outperforming_filtered[:50])
        ])
        
        csv = export_df_out.to_csv(index=False)
        st.download_button(
            label="📥 Download Outperforming Leaders (CSV)",
            data=csv,
            file_name=f"outperforming_leaders_{selected_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if underperforming_filtered:
        # Create DataFrame for export
        export_df_under = pd.DataFrame([
            {
                'Rank': idx + 1,
                'Ticker': r['ticker'],
                'RS Percentile': f"{r['rs_percentile']:.1f}",
                'RS Value': f"{r['rs_current']:.2f}",
                'RS MA20': f"{r.get('rs_ma20', 0):.2f}" if not np.isnan(r.get('rs_ma20', np.nan)) else 'N/A',
                'RS RSI': f"{r.get('rs_rsi', 0):.1f}" if not np.isnan(r.get('rs_rsi', np.nan)) else 'N/A',
                'List': r['list_class'],
                'Price': f"{r['close']:.2f}",
                'RSI': f"{r['rsi_daily']:.1f}" if not np.isnan(r['rsi_daily']) else 'N/A',
            }
            for idx, r in enumerate(underperforming_filtered[:50])
        ])
        
        csv = export_df_under.to_csv(index=False)
        st.download_button(
            label="📥 Download Underperforming Laggards (CSV)",
            data=csv,
            file_name=f"underperforming_laggards_{selected_date.strftime('%Y%m%d')}.csv",
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

