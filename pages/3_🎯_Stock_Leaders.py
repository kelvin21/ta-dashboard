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

# Add custom CSS for container width
st.markdown("""
<style>
    .stMainBlockContainer.block-container.st-emotion-cache-1w723zb.e4man114,
    .main .block-container {
        max-width: 1200px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

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
        calculate_relative_strength, calculate_multi_period_rs, calculate_rs_percentile, is_rs_near_high,
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
st.markdown("### 📊 What is Multi-Period Relative Strength?")
st.write("""
**Multi-Period RS** measures stock momentum across **three timeframes** (1M, 2M, 3M) for comprehensive analysis:

**Individual Periods:**
- **1M RS** (21 days): Short-term momentum - Most recent strength
- **2M RS** (42 days): Medium-term momentum - Swing trading horizon  
- **3M RS** (63 days): Longer-term momentum - Position building trend

**Composite RS** = Weighted average (1M: 50%, 2M: 30%, 3M: 20%)
- Emphasizes recent momentum while considering longer trends
- **RS > 1.0**: Outperforming VNINDEX
- **RS < 1.0**: Underperforming VNINDEX

**Trend Analysis:**
- **Accelerating**: 1M > 2M > 3M (Getting stronger - Best setup)
- **Decelerating**: 1M < 2M < 3M (Getting weaker - Exit signal)

This multi-timeframe approach captures momentum **consistency** and **direction**, 
providing better ticker comparison than single-period analysis.
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

def create_rs_chart(ticker: str, analysis_date: datetime, vnindex_data: pd.DataFrame):
    """
    Create RS chart with supplemental indicators.
    
    Args:
        ticker: Stock ticker
        analysis_date: Analysis date
        vnindex_data: VNINDEX data for RS calculation
    
    Returns:
        Plotly figure with RS chart and indicators
    """
    # Load stock data (6 months)
    start_date = analysis_date - timedelta(days=180)
    df = load_price_data_for_ticker(ticker, start_date, analysis_date)
    
    if df.empty or len(df) < 20:
        return None
    
    # Calculate RS using momentum-based method (2M/42-day for charts)
    # This matches the baseline used for MA calculations in analysis
    rs = calculate_relative_strength(df['close'], vnindex_data['close'], method='momentum', lookback=42)
    if rs.empty or rs.isna().all():
        return None
    
    df['rs'] = rs
    df['rs_ma20'] = df['rs'].rolling(window=20, min_periods=20).mean()
    df['rs_ma50'] = df['rs'].rolling(window=50, min_periods=50).mean()
    
    # Calculate RS RSI
    rs_rsi = calculate_rsi(df['rs'], period=14)
    df['rs_rsi'] = rs_rsi
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05,
        subplot_titles=(
            f'{ticker} - Relative Strength vs VNINDEX',
            'RS RSI (14)',
            'Price'
        )
    )
    
    # Plot 1: RS with MA20 and MA50
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['rs'],
            name='RS',
            line=dict(color='#1976D2', width=2),
            hovertemplate='RS: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['rs_ma20'],
            name='RS MA20',
            line=dict(color='#FF9800', width=1.5, dash='dash'),
            hovertemplate='MA20: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['rs_ma50'],
            name='RS MA50',
            line=dict(color='#9E9E9E', width=1, dash='dot'),
            hovertemplate='MA50: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot 2: RS RSI
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['rs_rsi'],
            name='RS RSI',
            line=dict(color='#9C27B0', width=2),
            fill='tozeroy',
            fillcolor='rgba(156, 39, 176, 0.1)',
            hovertemplate='RSI: %{y:.1f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add RSI reference lines
    fig.add_hline(y=70, line=dict(color='red', width=1, dash='dash'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='green', width=1, dash='dash'), row=2, col=1)
    fig.add_hline(y=50, line=dict(color='gray', width=0.5, dash='dot'), row=2, col=1)
    
    # Plot 3: Price
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['close'],
            name='Price',
            line=dict(color='#4CAF50', width=2),
            hovertemplate='Price: %{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Update layout with fullscreen support
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.7)'
        )
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor='#e0e0e0')
    fig.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
    
    # Y-axis labels
    fig.update_yaxes(title_text="RS Value", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Price", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

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
    
    # Calculate Multi-Period RS (1M, 2M, 3M) for comprehensive momentum analysis
    rs_analysis = calculate_multi_period_rs(df['close'], vnindex_data['close'])
    
    # Use composite RS as primary metric (weighted: 1M=50%, 2M=30%, 3M=20%)
    rs_current = rs_analysis['rs_composite']
    if np.isnan(rs_current):
        return None
    
    # Store all RS periods in dataframe (use 2M for MA calculations)
    df['rs'] = rs_analysis['rs_2m_series']  # 2-month for MA baseline
    df['rs_1m'] = rs_analysis['rs_1m_series']
    df['rs_2m'] = rs_analysis['rs_2m_series']
    df['rs_3m'] = rs_analysis['rs_3m_series']
    
    # Calculate RS MA20 for crossover detection (using 2M RS)
    df['rs_ma20'] = df['rs'].rolling(window=20, min_periods=20).mean()
    
    # Calculate RSI on RS values (to detect overbought RS) - using 2M RS
    rs_rsi = calculate_rsi(df['rs'], period=14)
    df['rs_rsi'] = rs_rsi
    
    # Get current values
    current = df.iloc[-1]
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
    
    # Calculate 5-day RS momentum change
    rs_5d_change = np.nan
    if len(df) >= 6:
        rs_5d_ago = df.iloc[-6]['rs'] if 'rs' in df.columns else np.nan
        if not np.isnan(rs_5d_ago) and rs_5d_ago != 0:
            rs_5d_change = ((rs_current - rs_5d_ago) / rs_5d_ago) * 100
    
    return {
        'ticker': ticker,
        'close': current['close'],
        'rs_current': rs_current,  # Composite RS (weighted 1M, 2M, 3M)
        'rs_1m': rs_analysis['rs_1m'],
        'rs_2m': rs_analysis['rs_2m'],
        'rs_3m': rs_analysis['rs_3m'],
        'rs_trend': rs_analysis['rs_trend'],  # Accelerating/Decelerating/Stable
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
        'rs_5d_change': rs_5d_change,  # Add 5-day momentum
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

# Split into outperforming, middle ground, and underperforming
outperforming_results = [r for r in results if r['performance_type'] == 'outperforming']
middle_ground_results = [r for r in results if r['performance_type'] == 'neutral']
underperforming_results = [r for r in results if r['performance_type'] == 'underperforming']

# Filter by minimum RS percentile
outperforming_filtered = [r for r in outperforming_results if r['rs_percentile'] >= min_percentile]
middle_ground_filtered = middle_ground_results  # No percentile filter for middle ground
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

# Create tabs for Outperforming, Middle Ground, and Underperforming
tab1, tab2, tab3 = st.tabs([
    f"🚀 Outperforming Leaders ({len(outperforming_filtered)})",
    f"⚖️ Middle Ground ({len(middle_ground_filtered)})",
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
        # Create responsive columns for card layout (4 cards per row)
        cols_per_row = 4
        for idx, result in enumerate(outperforming_filtered[:100], start=1):  # Limit to top 100
            if (idx - 1) % cols_per_row == 0:
                cols = st.columns(cols_per_row, gap="medium")
            
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
                
                # Calculate strength rating (1-5 stars) based on RS percentile and momentum
                base_rating = int((result['rs_percentile'] / 100) * 5)
                
                # Adjust rating based on RS momentum (RS vs MA20)
                rs_ma20_value = result.get('rs_ma20', 0)
                if not np.isnan(rs_ma20_value) and rs_ma20_value != 0:
                    rs_momentum = ((result['rs_current'] - rs_ma20_value) / rs_ma20_value * 100)
                    # Boost rating if RS is rising above MA20
                    if rs_momentum > 5:  # Strong upward momentum
                        base_rating += 1
                    elif rs_momentum > 2:  # Moderate upward momentum
                        base_rating += 0.5
                    elif rs_momentum < -5:  # Negative momentum
                        base_rating -= 1
                    elif rs_momentum < -2:  # Weak momentum
                        base_rating -= 0.5
                
                # Additional boost for bullish crossover
                if result.get('rs_crossover_signal') == 'Bullish Cross':
                    base_rating += 0.5
                
                strength_rating = min(5, max(1, int(base_rating) + 1))
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
                
                # Escape HTML special characters in text
                import html
                description_escaped = html.escape(description_text)
                ticker_escaped = html.escape(result['ticker'])
                
                # Create visual signal indicators
                # RS Percentile indicator
                rs_perc = result['rs_percentile']
                if rs_perc >= 90:
                    rs_perc_badge = f'<span style="background: #1B5E20; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS: {rs_perc:.0f}% 🔥</span>'
                elif rs_perc >= 80:
                    rs_perc_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS: {rs_perc:.0f}% ⬆️</span>'
                elif rs_perc >= 70:
                    rs_perc_badge = f'<span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS: {rs_perc:.0f}% 📈</span>'
                else:
                    rs_perc_badge = f'<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS: {rs_perc:.0f}%</span>'
                
                # Multi-period RS badges (1M, 2M, 3M)
                rs_1m = result.get('rs_1m', np.nan)
                rs_2m = result.get('rs_2m', np.nan)
                rs_3m = result.get('rs_3m', np.nan)
                rs_trend = result.get('rs_trend', 'N/A')
                
                # Color code based on value relative to 1.0
                def get_rs_color(rs_val):
                    if np.isnan(rs_val):
                        return '#9E9E9E'
                    elif rs_val >= 1.05:
                        return '#4CAF50'  # Strong green
                    elif rs_val >= 1.02:
                        return '#8BC34A'  # Light green
                    elif rs_val >= 0.98:
                        return '#FFC107'  # Yellow (neutral)
                    elif rs_val >= 0.95:
                        return '#FF9800'  # Orange
                    else:
                        return '#F44336'  # Red
                
                rs_1m_color = get_rs_color(rs_1m)
                rs_2m_color = get_rs_color(rs_2m)
                rs_3m_color = get_rs_color(rs_3m)
                
                if not np.isnan(rs_1m):
                    rs_1m_badge = f'<span style="background: {rs_1m_color}; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px; font-weight: bold;">1M: {rs_1m:.2f}</span>'
                else:
                    rs_1m_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px;">1M: N/A</span>'
                
                if not np.isnan(rs_2m):
                    rs_2m_badge = f'<span style="background: {rs_2m_color}; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px; font-weight: bold;">2M: {rs_2m:.2f}</span>'
                else:
                    rs_2m_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px;">2M: N/A</span>'
                
                if not np.isnan(rs_3m):
                    rs_3m_badge = f'<span style="background: {rs_3m_color}; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px; font-weight: bold;">3M: {rs_3m:.2f}</span>'
                else:
                    rs_3m_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px;">3M: N/A</span>'
                
                # RS Trend badge
                if rs_trend == 'Accelerating':
                    rs_trend_badge = '<span style="background: #1B5E20; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px; font-weight: bold;">🚀 Accelerating</span>'
                elif rs_trend == 'Strengthening':
                    rs_trend_badge = '<span style="background: #4CAF50; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px; font-weight: bold;">📈 Strengthening</span>'
                elif rs_trend == 'Decelerating':
                    rs_trend_badge = '<span style="background: #F44336; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px; font-weight: bold;">⬇️ Decelerating</span>'
                elif rs_trend == 'Weakening':
                    rs_trend_badge = '<span style="background: #FF9800; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px; font-weight: bold;">📉 Weakening</span>'
                elif rs_trend == 'Stable':
                    rs_trend_badge = '<span style="background: #2196F3; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px; font-weight: bold;">↔️ Stable</span>'
                else:
                    rs_trend_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 6px; border-radius: 6px; font-size: 9px;">Trend: N/A</span>'
                
                # 5-day momentum indicator
                rs_5d = result.get('rs_5d_change', np.nan)
                if not np.isnan(rs_5d):
                    if rs_5d >= 5:
                        momentum_5d_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: +{rs_5d:.1f}% 🚀</span>'
                    elif rs_5d >= 2:
                        momentum_5d_badge = f'<span style="background: #8BC34A; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: +{rs_5d:.1f}% ↗️</span>'
                    elif rs_5d >= 0:
                        momentum_5d_badge = f'<span style="background: #FFC107; color: #333; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: +{rs_5d:.1f}%</span>'
                    elif rs_5d >= -2:
                        momentum_5d_badge = f'<span style="background: #FF9800; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: {rs_5d:.1f}%</span>'
                    else:
                        momentum_5d_badge = f'<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: {rs_5d:.1f}% 🔻</span>'
                else:
                    momentum_5d_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: N/A</span>'
                
                # RS RSI indicator
                rs_rsi_value = result.get('rs_rsi', np.nan)
                if not np.isnan(rs_rsi_value):
                    if rs_rsi_value > 70:
                        rs_rsi_badge = f'<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RSI: {rs_rsi_value:.0f} 🔴</span>'
                    elif rs_rsi_value > 60:
                        rs_rsi_badge = f'<span style="background: #FF9800; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RSI: {rs_rsi_value:.0f} 🟠</span>'
                    elif rs_rsi_value >= 40:
                        rs_rsi_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RSI: {rs_rsi_value:.0f} 🟢</span>'
                    else:
                        rs_rsi_badge = f'<span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RSI: {rs_rsi_value:.0f} 🔵</span>'
                else:
                    rs_rsi_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RSI: N/A</span>'
                
                # Crossover status badge
                crossover = result.get('rs_crossover_status', 'None')
                if '🚀' in crossover:
                    crossover_badge = '<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">🚀 Crossed</span>'
                elif '📈' in crossover:
                    crossover_badge = '<span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">📈 Near Cross</span>'
                elif 'Above MA20' in crossover:
                    crossover_badge = '<span style="background: #8BC34A; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">✓ Above MA20</span>'
                elif 'Below MA20' in crossover:
                    crossover_badge = '<span style="background: #FF9800; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">✗ Below MA20</span>'
                else:
                    crossover_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">N/A</span>'
                
                # OBV status badge
                obv = result['obv_status']
                if 'Strong' in obv or 'Accumulation' in obv:
                    obv_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">OBV: ⬆️</span>'
                elif 'Weak' in obv or 'Distribution' in obv:
                    obv_badge = f'<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">OBV: ⬇️</span>'
                elif 'Neutral' in obv:
                    obv_badge = f'<span style="background: #FFC107; color: #333; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">OBV: ↔️</span>'
                else:
                    obv_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">OBV: N/A</span>'
                
                # Daily RSI badge
                rsi_daily = result['rsi_daily']
                if not np.isnan(rsi_daily):
                    if rsi_daily > 70:
                        daily_rsi_badge = f'<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: {rsi_daily:.0f} 🔴</span>'
                    elif rsi_daily > 55:
                        daily_rsi_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: {rsi_daily:.0f} 🟢</span>'
                    elif rsi_daily >= 45:
                        daily_rsi_badge = f'<span style="background: #FFC107; color: #333; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: {rsi_daily:.0f}</span>'
                    else:
                        daily_rsi_badge = f'<span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: {rsi_daily:.0f} 🔵</span>'
                else:
                    daily_rsi_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: N/A</span>'
                
                # Create card HTML with visual signals
                card_html = f'''<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 14px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 16px; min-height: 360px; display: flex; flex-direction: column;"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;"><div style="display: flex; align-items: center; gap: 8px;"><span style="font-size: 18px; font-weight: bold; color: #1976D2;">{ticker_escaped}</span><span style="font-size: 16px; color: {arrow_color};">{arrow}</span></div><div style="text-align: right; font-size: 14px;">{stars}</div></div><div style="margin-bottom: 10px;"><span style="background: {momentum_color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;">✓ {momentum_badge}</span>{warning_badge}</div><div style="font-size: 28px; font-weight: bold; color: #1976D2; margin-bottom: 8px;">{result['close']:.2f}</div><div style="font-size: 12px; color: #666; margin-bottom: 8px;">{rs_status_text}</div><div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px;">{rs_perc_badge}{momentum_5d_badge}</div><div style="display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 6px; padding: 4px; background: #F5F5F5; border-radius: 6px;">{rs_1m_badge}{rs_2m_badge}{rs_3m_badge}{rs_trend_badge}</div><div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px;">{rs_rsi_badge}{crossover_badge}</div><div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px;">{obv_badge}{daily_rsi_badge}</div><div style="margin-top: auto;"><div style="font-size: 11px; color: #666; line-height: 1.4; border-top: 1px solid #e0e0e0; padding-top: 8px;">{description_escaped}</div></div></div>'''
                
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Add chart expander below the card
                with st.expander(f"📊 View RS Chart for {result['ticker']}", expanded=False):
                    chart = create_rs_chart(result['ticker'], analysis_datetime, vnindex_data)
                    if chart:
                        st.plotly_chart(
                            chart, 
                            use_container_width=True,
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToAdd': ['toggleSpikelines'],
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': f'{result["ticker"]}_RS_chart',
                                    'height': 1200,
                                    'width': 1600,
                                    'scale': 2
                                }
                            }
                        )
                    else:
                        st.warning("Unable to generate chart - insufficient data")
        
        # Add spacing after cards
        st.markdown("")

# TAB 2: MIDDLE GROUND
with tab2:
    if not middle_ground_filtered:
        st.info(f"No middle ground stocks found.")
    else:
        st.markdown(f"### {len(middle_ground_filtered[:100])} Neutral Performers")
        st.caption("Sorted by RS percentile - stocks performing in line with market (30-70% RS)")
        st.markdown("")
        
        # Display middle ground stocks in card format
        cols_per_row = 4
        for idx, result in enumerate(middle_ground_filtered[:100], start=1):  # Limit to 100
            if (idx - 1) % cols_per_row == 0:
                cols = st.columns(cols_per_row, gap="medium")
            
            col = cols[(idx - 1) % cols_per_row]
            
            with col:
                # Determine momentum badge for neutral
                momentum_badge = "Neutral"
                momentum_color = "#9E9E9E"  # Gray for neutral
                if result['rs_percentile'] >= 60:
                    momentum_badge = "Above Average"
                    momentum_color = "#2196F3"
                elif result['rs_percentile'] >= 50:
                    momentum_badge = "Average"
                    momentum_color = "#00BCD4"
                elif result['rs_percentile'] >= 40:
                    momentum_badge = "Slightly Below"
                    momentum_color = "#FF9800"
                else:
                    momentum_badge = "Below Average"
                    momentum_color = "#FFC107"
                
                # Calculate strength rating with momentum adjustment
                base_rating = int((result['rs_percentile'] / 100) * 5)
                
                # Adjust rating based on RS momentum
                rs_ma20_value = result.get('rs_ma20', 0)
                if not np.isnan(rs_ma20_value) and rs_ma20_value != 0:
                    rs_momentum = ((result['rs_current'] - rs_ma20_value) / rs_ma20_value * 100)
                    if rs_momentum > 5:
                        base_rating += 1
                    elif rs_momentum > 2:
                        base_rating += 0.5
                    elif rs_momentum < -5:
                        base_rating -= 1
                    elif rs_momentum < -2:
                        base_rating -= 0.5
                
                strength_rating = max(1, min(5, int(base_rating) + 1))
                stars = "⭐" * strength_rating
                
                # Determine arrow direction
                arrow = "↑" if result['rs_current'] > result.get('rs_ma20', result['rs_current']) else "↓"
                arrow_color = "#4CAF50" if arrow == "↑" else "#FF9800"
                
                # Warning badge for borderline cases
                warning_badge = ""
                dist_pct = result.get('rs_distance_pct', 0)
                if not np.isnan(dist_pct) and abs(dist_pct) > 8:
                    if dist_pct > 0:
                        warning_badge = '<div style="display: flex; align-items: center; gap: 4px; color: #2196F3; font-size: 12px; margin-top: 4px;"><i class="fas fa-info-circle"></i><span>Breaking Higher</span></div>'
                    else:
                        warning_badge = '<div style="display: flex; align-items: center; gap: 4px; color: #FF9800; font-size: 12px; margin-top: 4px;"><i class="fas fa-exclamation-triangle"></i><span>Weakening</span></div>'
                
                # RS status indicator
                rs_ma20_value = result.get('rs_ma20', 0)
                if not np.isnan(rs_ma20_value):
                    rs_diff_pct = ((result['rs_current'] - rs_ma20_value) / rs_ma20_value * 100) if rs_ma20_value != 0 else 0
                    rs_status_text = f"S: RS MA20 ({rs_diff_pct:+.1f}%)"
                else:
                    rs_status_text = "S: RS MA20 (N/A)"
                
                # Build description
                description_parts = []
                if result['rs_percentile'] >= 60:
                    description_parts.append("Above average performer")
                elif result['rs_percentile'] >= 50:
                    description_parts.append("In line with market")
                elif result['rs_percentile'] >= 40:
                    description_parts.append("Slightly lagging market")
                else:
                    description_parts.append("Below market average")
                
                description = result.get('description', 'Middle Range RS')
                if description_parts:
                    description += " • " + " • ".join(description_parts)
                
                ticker_escaped = html.escape(result['ticker'])
                description_escaped = html.escape(description)
                
                # Visual signal badges
                rs_perc_badge = f'<span style="background: {momentum_color}; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS: {result["rs_percentile"]:.0f}%</span>'
                
                # 5-day RS momentum badge
                rs_5d_change = result.get('rs_5d_change', np.nan)
                if not np.isnan(rs_5d_change):
                    if rs_5d_change > 0:
                        momentum_5d_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: +{rs_5d_change:.1f}% ▲</span>'
                    else:
                        momentum_5d_badge = f'<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: {rs_5d_change:.1f}% ▼</span>'
                else:
                    momentum_5d_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: N/A</span>'
                
                # RS RSI badge
                rs_rsi_value = result.get('rs_rsi', np.nan)
                if not np.isnan(rs_rsi_value):
                    if rs_rsi_value >= 70:
                        rs_rsi_badge = f'<span style="background: #FF9800; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS RSI: {rs_rsi_value:.0f} 🔶</span>'
                    elif rs_rsi_value <= 30:
                        rs_rsi_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS RSI: {rs_rsi_value:.0f} 🟢</span>'
                    else:
                        rs_rsi_badge = f'<span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS RSI: {rs_rsi_value:.0f} 🔵</span>'
                else:
                    rs_rsi_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS RSI: N/A</span>'
                
                # Crossover status badge
                crossover_status = result.get('rs_crossover_status', 'N/A')
                if 'Bullish' in crossover_status:
                    crossover_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">{crossover_status} ✓</span>'
                elif 'Near' in crossover_status:
                    crossover_badge = f'<span style="background: #FF9800; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">{crossover_status} ⚠</span>'
                elif 'Above' in crossover_status:
                    crossover_badge = f'<span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">{crossover_status} ↑</span>'
                else:
                    crossover_badge = f'<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">{crossover_status} ↓</span>'
                
                # OBV badge
                obv_status = result.get('obv_status', 'N/A')
                if obv_status == 'Above EMA':
                    obv_badge = '<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">OBV: Strong 💪</span>'
                elif obv_status == 'Below EMA':
                    obv_badge = '<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">OBV: Weak ⚠</span>'
                else:
                    obv_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">OBV: N/A</span>'
                
                # Daily RSI badge
                rsi_daily = result.get('rsi_daily', np.nan)
                if not np.isnan(rsi_daily):
                    if rsi_daily >= 70:
                        daily_rsi_badge = f'<span style="background: #FF9800; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: {rsi_daily:.0f} 🔶</span>'
                    elif rsi_daily <= 30:
                        daily_rsi_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: {rsi_daily:.0f} 🟢</span>'
                    else:
                        daily_rsi_badge = f'<span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: {rsi_daily:.0f} 🔵</span>'
                else:
                    daily_rsi_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: N/A</span>'
                
                # Create card HTML with neutral styling
                card_html = f'''<div style="border: 1px solid #BDBDBD; border-radius: 12px; padding: 14px; background: white; box-shadow: 0 2px 8px rgba(158,158,158,0.2); margin-bottom: 16px; min-height: 320px; display: flex; flex-direction: column;"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;"><div style="display: flex; align-items: center; gap: 8px;"><span style="font-size: 18px; font-weight: bold; color: #616161;">{ticker_escaped}</span><span style="font-size: 16px; color: {arrow_color};">{arrow}</span></div><div style="text-align: right; font-size: 14px;">{stars}</div></div><div style="margin-bottom: 10px;"><span style="background: {momentum_color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;">⚖ {momentum_badge}</span>{warning_badge}</div><div style="font-size: 28px; font-weight: bold; color: #616161; margin-bottom: 8px;">{result['close']:.2f}</div><div style="font-size: 12px; color: #666; margin-bottom: 8px;">{rs_status_text}</div><div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px;">{rs_perc_badge}{momentum_5d_badge}</div><div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px;">{rs_rsi_badge}{crossover_badge}</div><div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px;">{obv_badge}{daily_rsi_badge}</div><div style="margin-top: auto;"><div style="font-size: 11px; color: #666; line-height: 1.4; border-top: 1px solid #BDBDBD; padding-top: 8px;">{description_escaped}</div></div></div>'''
                
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Add chart expander below the card
                with st.expander(f"📊 View RS Chart for {result['ticker']}", expanded=False):
                    chart = create_rs_chart(result['ticker'], analysis_datetime, vnindex_data)
                    if chart:
                        st.plotly_chart(
                            chart, 
                            use_container_width=True,
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToAdd': ['toggleSpikelines'],
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': f'{result["ticker"]}_RS_chart',
                                    'height': 1200,
                                    'width': 1600,
                                    'scale': 2
                                }
                            }
                        )
                    else:
                        st.warning("Unable to generate chart - insufficient data")
        
        # Add spacing after cards
        st.markdown("")

# TAB 3: UNDERPERFORMING
with tab3:
    if not underperforming_filtered:
        st.info(f"No underperforming stocks found.")
    else:
        st.markdown(f"### Bottom {len(underperforming_filtered[:100])} Market Laggards")
        st.caption("Sorted by RS percentile (lowest first) - stocks with weakest relative performance")
        st.markdown("")
        
        # Display underperforming stocks in card format
        cols_per_row = 4
        for idx, result in enumerate(underperforming_filtered[:100], start=1):  # Limit to bottom 100
            if (idx - 1) % cols_per_row == 0:
                cols = st.columns(cols_per_row, gap="medium")
            
            col = cols[(idx - 1) % cols_per_row]
            
            with col:
                # Determine momentum badge for underperforming (inverted logic)
                momentum_badge = "Weak Momentum"
                momentum_color = "#F44336"  # Red for underperforming
                if result['rs_percentile'] <= 10:
                    momentum_badge = "Very Weak"
                    momentum_color = "#D32F2F"
                elif result['rs_percentile'] <= 20:
                    momentum_badge = "Weak Momentum"
                    momentum_color = "#F44336"
                else:
                    momentum_badge = "Lagging"
                    momentum_color = "#FF9800"
                
                # Calculate strength rating (inverted - lower is worse) with momentum adjustment
                base_rating = int((result['rs_percentile'] / 100) * 5)
                
                # Adjust rating based on RS momentum
                rs_ma20_value = result.get('rs_ma20', 0)
                if not np.isnan(rs_ma20_value) and rs_ma20_value != 0:
                    rs_momentum = ((result['rs_current'] - rs_ma20_value) / rs_ma20_value * 100)
                    # Penalize if RS is falling below MA20
                    if rs_momentum < -5:  # Strong downward momentum
                        base_rating -= 1
                    elif rs_momentum < -2:  # Moderate downward momentum
                        base_rating -= 0.5
                    elif rs_momentum > 2:  # Showing recovery
                        base_rating += 0.5
                
                strength_rating = max(1, min(5, int(base_rating) + 1))
                stars = "⭐" * strength_rating
                
                # Determine arrow direction
                arrow = "↑" if result['rs_current'] > result.get('rs_ma20', result['rs_current']) else "↓"
                arrow_color = "#4CAF50" if arrow == "↑" else "#F44336"
                
                # Warning badge for extremely weak
                warning_badge = ""
                if result['rs_percentile'] <= 10:
                    warning_badge = '<div style="display: flex; align-items: center; gap: 4px; color: #F44336; font-size: 12px; margin-top: 4px;"><i class="fas fa-exclamation-triangle"></i><span>Extreme Weakness</span></div>'
                
                # RS status indicator
                rs_ma20_value = result.get('rs_ma20', 0)
                if not np.isnan(rs_ma20_value):
                    rs_diff_pct = ((result['rs_current'] - rs_ma20_value) / rs_ma20_value * 100) if rs_ma20_value != 0 else 0
                    rs_status_text = f"S: RS MA20 ({rs_diff_pct:+.1f}%)"
                else:
                    rs_status_text = "S: RS MA20 (N/A)"
                
                # Build description
                description_parts = []
                if result['rs_percentile'] <= 10:
                    description_parts.append("Extreme underperformance")
                elif result['rs_percentile'] <= 20:
                    description_parts.append("Significant weakness")
                else:
                    description_parts.append("Lagging market")
                
                if result['rs_current'] < result.get('rs_ma20', 0):
                    description_parts.append("below RS MA20")
                
                if result.get('rs_crossover_signal') == 'Near Bullish':
                    description_parts.append("potential reversal")
                    
                description_text = " - ".join(description_parts) if description_parts else result['description']
                
                # Escape HTML
                import html
                description_escaped = html.escape(description_text)
                ticker_escaped = html.escape(result['ticker'])
                
                # Create visual signal indicators for underperforming
                # RS Percentile indicator (red theme)
                rs_perc = result['rs_percentile']
                if rs_perc <= 10:
                    rs_perc_badge = f'<span style="background: #B71C1C; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS: {rs_perc:.0f}% 🔻</span>'
                elif rs_perc <= 20:
                    rs_perc_badge = f'<span style="background: #D32F2F; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS: {rs_perc:.0f}% ⬇️</span>'
                elif rs_perc <= 30:
                    rs_perc_badge = f'<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS: {rs_perc:.0f}% 📉</span>'
                else:
                    rs_perc_badge = f'<span style="background: #FF9800; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RS: {rs_perc:.0f}%</span>'
                
                # 5-day momentum indicator
                rs_5d = result.get('rs_5d_change', np.nan)
                if not np.isnan(rs_5d):
                    if rs_5d >= 2:
                        momentum_5d_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: +{rs_5d:.1f}% 👍</span>'
                    elif rs_5d >= 0:
                        momentum_5d_badge = f'<span style="background: #FFC107; color: #333; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: +{rs_5d:.1f}%</span>'
                    elif rs_5d >= -2:
                        momentum_5d_badge = f'<span style="background: #FF9800; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: {rs_5d:.1f}%</span>'
                    elif rs_5d >= -5:
                        momentum_5d_badge = f'<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: {rs_5d:.1f}% ↘️</span>'
                    else:
                        momentum_5d_badge = f'<span style="background: #B71C1C; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: {rs_5d:.1f}% 🔻</span>'
                else:
                    momentum_5d_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">5D: N/A</span>'
                
                # RS RSI indicator
                rs_rsi_value = result.get('rs_rsi', np.nan)
                if not np.isnan(rs_rsi_value):
                    if rs_rsi_value > 70:
                        rs_rsi_badge = f'<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RSI: {rs_rsi_value:.0f} 🔴</span>'
                    elif rs_rsi_value > 40:
                        rs_rsi_badge = f'<span style="background: #FFC107; color: #333; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RSI: {rs_rsi_value:.0f} 🟡</span>'
                    else:
                        rs_rsi_badge = f'<span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RSI: {rs_rsi_value:.0f} 🔵</span>'
                else:
                    rs_rsi_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">RSI: N/A</span>'
                
                # Crossover status badge
                crossover = result.get('rs_crossover_status', 'None')
                if '📈' in crossover:
                    crossover_badge = '<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">📈 Recovery?</span>'
                elif 'Above MA20' in crossover:
                    crossover_badge = '<span style="background: #FFC107; color: #333; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">✓ Above MA20</span>'
                elif 'Below MA20' in crossover:
                    crossover_badge = '<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">✗ Below MA20</span>'
                else:
                    crossover_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">N/A</span>'
                
                # OBV status badge
                obv = result['obv_status']
                if 'Strong' in obv or 'Accumulation' in obv:
                    obv_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">OBV: ⬆️</span>'
                elif 'Weak' in obv or 'Distribution' in obv:
                    obv_badge = f'<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">OBV: ⬇️</span>'
                elif 'Neutral' in obv:
                    obv_badge = f'<span style="background: #FFC107; color: #333; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">OBV: ↔️</span>'
                else:
                    obv_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">OBV: N/A</span>'
                
                # Daily RSI badge
                rsi_daily = result['rsi_daily']
                if not np.isnan(rsi_daily):
                    if rsi_daily > 70:
                        daily_rsi_badge = f'<span style="background: #F44336; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: {rsi_daily:.0f} 🔴</span>'
                    elif rsi_daily > 45:
                        daily_rsi_badge = f'<span style="background: #FFC107; color: #333; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: {rsi_daily:.0f}</span>'
                    elif rsi_daily >= 30:
                        daily_rsi_badge = f'<span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: {rsi_daily:.0f} 🔵</span>'
                    else:
                        daily_rsi_badge = f'<span style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: {rsi_daily:.0f} 🟢</span>'
                else:
                    daily_rsi_badge = '<span style="background: #9E9E9E; color: white; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: bold;">Daily RSI: N/A</span>'
                
                # Create card HTML
                card_html = f'''<div style="border: 1px solid #ffcdd2; border-radius: 12px; padding: 14px; background: white; box-shadow: 0 2px 8px rgba(244,67,54,0.1); margin-bottom: 16px; min-height: 320px; display: flex; flex-direction: column;"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;"><div style="display: flex; align-items: center; gap: 8px;"><span style="font-size: 18px; font-weight: bold; color: #F44336;">{ticker_escaped}</span><span style="font-size: 16px; color: {arrow_color};">{arrow}</span></div><div style="text-align: right; font-size: 14px;">{stars}</div></div><div style="margin-bottom: 10px;"><span style="background: {momentum_color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;">✗ {momentum_badge}</span>{warning_badge}</div><div style="font-size: 28px; font-weight: bold; color: #F44336; margin-bottom: 8px;">{result['close']:.2f}</div><div style="font-size: 12px; color: #666; margin-bottom: 8px;">{rs_status_text}</div><div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px;">{rs_perc_badge}{momentum_5d_badge}</div><div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px;">{rs_rsi_badge}{crossover_badge}</div><div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px;">{obv_badge}{daily_rsi_badge}</div><div style="margin-top: auto;"><div style="font-size: 11px; color: #666; line-height: 1.4; border-top: 1px solid #ffcdd2; padding-top: 8px;">{description_escaped}</div></div></div>'''                
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Add chart expander below the card
                with st.expander(f"📊 View RS Chart for {result['ticker']}", expanded=False):
                    chart = create_rs_chart(result['ticker'], analysis_datetime, vnindex_data)
                    if chart:
                        st.plotly_chart(
                            chart, 
                            use_container_width=True,
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToAdd': ['toggleSpikelines'],
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': f'{result["ticker"]}_RS_chart',
                                    'height': 1200,
                                    'width': 1600,
                                    'scale': 2
                                }
                            }
                        )
                    else:
                        st.warning("Unable to generate chart - insufficient data")
        
        # Add spacing after cards
        st.markdown("")

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

# Methodology Section
st.markdown("## 📚 Methodology & Rating System")

st.markdown("""
<div class="material-card elevation-2" style="padding: 1.5rem; margin-bottom: 1rem;">
<h3>🎯 Multi-Period Relative Strength Analysis</h3>

**Core Concept - Triple Timeframe Momentum:**
- **1M RS**: (Stock's 21-day ROC) / (VNINDEX's 21-day ROC) - **Short-term**
- **2M RS**: (Stock's 42-day ROC) / (VNINDEX's 42-day ROC) - **Medium-term**
- **3M RS**: (Stock's 63-day ROC) / (VNINDEX's 63-day ROC) - **Longer-term**

**Composite RS Formula:**
- Weighted Average: (1M × 50%) + (2M × 30%) + (3M × 20%)
- Emphasizes recent momentum while validating with longer trends

**Interpretation:**
- RS > 1.0: Stock's momentum is **outperforming** the market
- RS < 1.0: Stock's momentum is **underperforming** the market
- RS = 1.0: Moving **in line** with the market

**Trend Signals:**
- **🚀 Accelerating**: 1M > 2M > 3M (Building momentum - Best setup)
- **📈 Strengthening**: 1M > 3M (Recent strength)
- **↔️ Stable**: Consistent across periods
- **📉 Weakening**: 1M < 3M (Losing momentum)
- **⬇️ Decelerating**: 1M < 2M < 3M (Breaking down - Exit)

**Why Multi-Period?**
- **Confirms Momentum**: All periods aligned = high confidence
- **Detects Changes**: Divergences show momentum shifts early
- **Reduces Noise**: Filters false signals from single timeframe
- **Better Ranking**: Compares tickers fairly across multiple horizons

**RS Percentile Ranking:**
- **Top 20% (RS ≥80%)**: List A - Strong Leaders, highest probability
- **Top 30% (RS 70-80%)**: List B - Emerging Leaders, watchlist
- **Bottom 30% (RS ≤30%)**: Laggards - Underperformers, avoid/reduce
- **Middle Range**: Neutral - no clear edge

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="material-card elevation-2" style="padding: 1.5rem; margin-bottom: 1rem;">
<h3>⭐ Strength Rating System (1-5 Stars)</h3>

**Base Rating:** Derived from RS Percentile
- 5 stars: Top 80-100% RS
- 4 stars: 60-80% RS
- 3 stars: 40-60% RS
- 2 stars: 20-40% RS
- 1 star: 0-20% RS

**Momentum Adjustments:**
- **+1 star**: RS >5% above MA20 (strong upward momentum)
- **+0.5 star**: RS 2-5% above MA20 (building momentum)
- **+0.5 star**: Bullish crossover (RS just crossed above MA20)
- **-0.5 star**: RS 2-5% below MA20 (weakening momentum)
- **-1 star**: RS >5% below MA20 (negative momentum)

**Key Insight:** A stock with 75% RS percentile but rising momentum (RS above MA20) may be rated higher than a stock with 85% RS but falling momentum.

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="material-card elevation-2" style="padding: 1.5rem; margin-bottom: 1rem;">
<h3>🚦 Entry & Exit Rules</h3>

**⚠️ DO NOT Buy Immediately**
- List identifies stocks BEFORE they accelerate
- High RS alone is not an entry signal
- Wait for proper setup and trigger

**Valid Entry Triggers:**
1. **RSI crosses above 55** (momentum confirmation)
2. **Price reclaims EMA20** (trend support)
3. **Breakout from consolidation** (volatility contraction)
4. **Volume expansion + OBV confirmation** (institutional buying)

**Exit Signals:**
1. RS breaks below weekly EMA10 (loss of relative strength)
2. OBV makes lower low (distribution)
3. Daily RSI <45 for 3 consecutive days (momentum break)
4. Underperforms VNINDEX for 5 consecutive sessions

**Stop-Loss:**
- Below EMA50 or last swing low
- Leaders should not lose EMA50 decisively
- Adjust stops as stock progresses

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="material-card elevation-2" style="padding: 1.5rem; margin-bottom: 1rem;">
<h3>📊 Technical Indicators Explained</h3>

**RS MA20:** 20-day moving average of Relative Strength
- Smooths RS line to identify trend
- Crossovers signal momentum shifts
- Distance from MA20 shows momentum strength

**RS RSI:** RSI calculated on RS values (not price)
- >70: RS overbought (distribution risk)
- <30: RS oversold (potential reversal)
- 40-60: Healthy RS momentum

**RS Status:**
- **🚀 Just Crossed Above**: Bullish momentum trigger
- **📈 Near Cross Above**: Setup forming, watch closely
- **✓ Above MA20**: Positive momentum trend
- **⚠️ Just Crossed Below**: Losing momentum

**OBV (On-Balance Volume):**
- Tracks cumulative volume flow
- Confirms price moves with volume
- Divergences warn of trend changes

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="material-card elevation-2" style="padding: 1.5rem;">
<h3>⚠️ Important Disclaimers</h3>

**Risk Warning:**
- Past relative strength does not guarantee future performance
- High RS stocks can still decline in bear markets
- Always use proper position sizing and risk management
- This is analysis, not trading advice

**Best Practices:**
- Focus on top RS stocks in confirmed uptrends
- Diversify across multiple high-RS stocks
- Don't chase overextended stocks (RS >10% above MA20)
- Monitor market breadth and VNINDEX direction
- Combine RS with your own trading system

**Data Updates:**
- Analysis based on end-of-day (EOD) data
- RS calculations use daily closing prices
- Update frequency: Daily after market close

</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("_Stock Leaders Detection • Relative Strength Focus • Daily EOD Analysis • Not Financial Advice_")

