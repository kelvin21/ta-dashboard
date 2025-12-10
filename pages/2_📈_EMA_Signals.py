"""
EMA Signals Analysis Page
Comprehensive EMA signal analysis for all tickers with Material Design UI.
"""
import os
import sys
from pathlib import Path
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

# Import project utilities
try:
    from utils.indicators import calculate_all_indicators
    from utils.ema_utils import (
        calculate_ema_alignment, calculate_ema_strength_score,
        determine_ema_zone, calculate_ema_distances, calculate_ema_convergence,
        create_mini_sparkline, rank_ema_urgency,
        get_zone_color, get_alignment_color, format_distance
    )
    from utils.conclusion_builder import (
        generate_immediate_buy_signals, generate_immediate_sell_signals,
        calculate_market_breadth_summary, generate_market_strategy,
        format_signal_card
    )
    from utils.db_async import get_sync_db_adapter
    USE_UTILS = True
except ImportError as e:
    st.error(f"Failed to import utility modules: {e}")
    st.info("Please ensure utils/ directory exists with required modules")
    st.stop()

# Page config
st.set_page_config(
    page_title="EMA Signals Analysis",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CACHE & DATABASE FUNCTIONS
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
        latest = db.get_latest_date()
        return latest
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

@st.cache_data(ttl=900)
def load_price_data_for_ticker(ticker: str, start_date: datetime, end_date: datetime):
    """Load price data for a single ticker."""
    try:
        db = get_db()
        return db.get_price_data(ticker, start_date, end_date)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def calculate_indicators_for_all_tickers(tickers: list, date: datetime, lookback_days: int = 200):
    """Calculate indicators for all tickers."""
    warmup_days = 365
    start_date = date - timedelta(days=lookback_days + warmup_days)
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"Processing {ticker} ({idx + 1}/{len(tickers)})...")
        
        try:
            df = load_price_data_for_ticker(ticker, start_date, date)
            if df.empty or 'close' not in df.columns:
                continue
            
            # Calculate indicators
            df = calculate_all_indicators(df)
            
            # Get latest row
            df_recent = df[df['date'] <= date].tail(1)
            if df_recent.empty:
                continue
            
            latest = df_recent.iloc[0].copy()
            latest['ticker'] = ticker
            
            # Calculate EMA metrics
            latest['ema_alignment'] = calculate_ema_alignment(latest)
            latest['ema_strength'] = calculate_ema_strength_score(latest)
            latest['ema_zone'] = determine_ema_zone(latest)
            latest['ema_convergence'] = calculate_ema_convergence(latest)
            
            # Calculate distances
            distances = calculate_ema_distances(latest)
            for key, val in distances.items():
                latest[f'{key}_dist'] = val
            
            results.append(latest)
            
        except Exception as e:
            st.warning(f"Error processing {ticker}: {e}")
            continue
        
        progress_bar.progress((idx + 1) / len(tickers))
    
    progress_bar.empty()
    status_text.empty()
    
    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()

# =====================================================================
# UI HELPER FUNCTIONS
# =====================================================================

def render_vnindex_card(vnindex_data: pd.Series):
    """Render VNINDEX overview card."""
    if vnindex_data is None or vnindex_data.empty:
        st.warning("VNINDEX data not available")
        return
    
    close = vnindex_data.get('close', 0)
    ema20 = vnindex_data.get('ema20', 0)
    ema50 = vnindex_data.get('ema50', 0)
    ema200 = vnindex_data.get('ema200', 0)
    strength = vnindex_data.get('ema_strength', 3)
    zone = vnindex_data.get('ema_zone', 'neutral')
    alignment = vnindex_data.get('ema_alignment', 'neutral')
    
    # Calculate change
    ema20_dist = vnindex_data.get('ema20_dist', 0)
    
    # Determine trend
    if close > ema50 > ema200:
        trend = '<i class="fas fa-chart-line"></i> UPTREND'
        trend_color = "#4CAF50"
    elif close < ema50 < ema200:
        trend = '<i class="fas fa-chart-line" style="transform: scaleY(-1);"></i> DOWNTREND'
        trend_color = "#F44336"
    else:
        trend = '<i class="fas fa-arrows-alt-h"></i> SIDEWAYS'
        trend_color = "#FF9800"
    
    st.markdown(f"""
    <div class="material-card elevation-2">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; color: {trend_color};">VNINDEX</h2>
                <p style="font-size: 32px; font-weight: bold; margin: 8px 0; color: #212121;">
                    {close:.2f}
                </p>
                <p style="font-size: 14px; color: #757575; margin: 4px 0;">
                    EMA20: {ema20:.2f} ({ema20_dist:+.2f}%)
                </p>
            </div>
            <div style="text-align: right;">
                <div style="background: {trend_color}; color: white; padding: 8px 16px; border-radius: 16px; font-weight: bold; margin-bottom: 8px;">
                    {trend}
                </div>
                <div style="background: {get_zone_color(zone)}; color: white; padding: 6px 12px; border-radius: 12px; font-size: 12px;">
                    Zone: {zone.upper()}
                </div>
            </div>
        </div>
        <div style="margin-top: 16px; display: flex; gap: 12px;">
            <div class="chip chip-info">Strength: {strength}/5</div>
            <div class="chip" style="background: {get_alignment_color(alignment)}; color: white;">
                {alignment.upper()}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_breadth_summary(breadth: dict):
    """Render market breadth summary."""
    above_50 = breadth.get('above_ema50_pct', 0)
    above_200 = breadth.get('above_ema200_pct', 0)
    bullish = breadth.get('bullish_alignment_pct', 0)
    
    st.markdown(f"""
    <div class="material-card elevation-1">
        <h3 style="margin: 0 0 16px 0; color: #212121;"><i class="fas fa-chart-bar"></i> Market Breadth</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
            <div style="text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #2196F3;">
                    {above_50:.0f}%
                </div>
                <div style="font-size: 14px; color: #757575;">Above EMA50</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #4CAF50;">
                    {above_200:.0f}%
                </div>
                <div style="font-size: 14px; color: #757575;">Above EMA200</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #FF9800;">
                    {bullish:.0f}%
                </div>
                <div style="font-size: 14px; color: #757575;">Bullish Alignment</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_action_conclusion(buy_signals: list, sell_signals: list, strategy: str):
    """Render immediate action conclusion card."""
    st.markdown("""
    <div class="material-card elevation-4" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
        <h2 style="margin: 0 0 16px 0; color: white;"><i class="fas fa-bullseye"></i> Immediate Action Plan</h2>
    """, unsafe_allow_html=True)
    
    # Buy signals
    if buy_signals:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 16px; border-radius: 8px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 12px 0; color: white;"><i class="fas fa-circle" style="color: #4CAF50;"></i> Top Buy Opportunities</h3>
        """, unsafe_allow_html=True)
        
        for signal in buy_signals:
            card = format_signal_card(signal, 'buy')
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 18px; font-weight: bold;">{card['ticker']}</span>
                    <span style="background: {card['priority_color']}; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                        {card['priority']}
                    </span>
                </div>
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">
                    Price: {card['close']:.2f} | Score: {card['score']}
                </div>
                <div style="font-size: 13px; opacity: 0.8;">
                    {card['reason_text']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Sell signals
    if sell_signals:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 16px; border-radius: 8px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 12px 0; color: white;"><i class="fas fa-circle" style="color: #F44336;"></i> Top Sell/Avoid Positions</h3>
        """, unsafe_allow_html=True)
        
        for signal in sell_signals:
            card = format_signal_card(signal, 'sell')
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 18px; font-weight: bold;">{card['ticker']}</span>
                    <span style="background: {card['priority_color']}; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                        {card['priority']}
                    </span>
                </div>
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">
                    Price: {card['close']:.2f} | Score: {card['score']}
                </div>
                <div style="font-size: 13px; opacity: 0.8;">
                    {card['reason_text']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Strategy
    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.2); padding: 16px; border-radius: 8px; border-left: 4px solid #FFD700;">
            <h4 style="margin: 0 0 8px 0; color: white;"><i class="fas fa-lightbulb"></i> Market Strategy</h4>
            <p style="margin: 0; font-size: 16px; line-height: 1.6; color: white;">
                {strategy}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def get_ema_position_summary(row: pd.Series) -> str:
    """Get compact EMA position summary."""
    close = row.get('close', np.nan)
    if pd.isna(close):
        return 'N/A'
    
    ema_periods = [10, 20, 50, 100, 150, 200]
    above_emas = []
    below_emas = []
    
    for period in ema_periods:
        col = f'ema{period}'
        ema_val = row.get(col, np.nan)
        if pd.notna(ema_val):
            if close > ema_val:
                above_emas.append(period)
            else:
                below_emas.append(period)
    
    # If above all EMAs, show strongest position
    if len(above_emas) == len(ema_periods):
        return '<i class="fas fa-check-circle" style="color: #4CAF50;"></i> >EMA10 (All EMAs)'
    
    # If below all EMAs
    if len(below_emas) == len(ema_periods):
        return '<i class="fas fa-times-circle" style="color: #F44336;"></i> <EMA200 (All EMAs)'
    
    # Show highest EMA above and lowest EMA below
    if above_emas:
        highest_above = max(above_emas)
        return f'<i class="fas fa-check-circle" style="color: #4CAF50;"></i> >EMA{highest_above}'
    elif below_emas:
        lowest_below = min(below_emas)
        return f'<i class="fas fa-times-circle" style="color: #F44336;"></i> <EMA{lowest_below}'
    
    return 'N/A'

def render_ticker_card(row: pd.Series, show_sparkline: bool = False):
    """Render individual ticker card."""
    ticker = row.get('ticker', 'N/A')
    close = row.get('close', 0)
    strength = row.get('ema_strength', 3)
    zone = row.get('ema_zone', 'neutral')
    alignment = row.get('ema_alignment', 'neutral')
    convergence = row.get('ema_convergence', 0)
    
    # Get compact EMA summary
    ema_summary = get_ema_position_summary(row)
    
    # Get distances
    ema20_dist = row.get('ema20_dist', 0)
    ema50_dist = row.get('ema50_dist', 0)
    
    zone_color = get_zone_color(zone)
    align_color = get_alignment_color(alignment)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div class="material-card elevation-1" style="padding: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h3 style="margin: 0; color: #212121; font-size: 18px;">{ticker}</h3>
                <div style="background: {zone_color}; color: white; padding: 3px 10px; border-radius: 10px; font-size: 11px; font-weight: bold;">
                    {zone.upper()}
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                <span style="font-size: 22px; font-weight: bold; color: #212121;">{close:.2f}</span>
                <span style="font-size: 13px; color: #757575;">{ema_summary}</span>
            </div>
            <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                <span class="chip chip-info" style="height: 24px; line-height: 24px; font-size: 11px; padding: 0 10px;"><i class="fas fa-star"></i> {strength}/5</span>
                <span class="chip" style="background: {align_color}; color: white; height: 24px; line-height: 24px; font-size: 11px; padding: 0 10px;">{alignment}</span>
                <span class="chip" style="height: 24px; line-height: 24px; font-size: 11px; padding: 0 10px;"><i class="fas fa-chart-bar"></i> {convergence:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if show_sparkline:
            # Load recent data for sparkline
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            df_recent = load_price_data_for_ticker(ticker, start_date, end_date)
            if not df_recent.empty:
                df_recent = calculate_all_indicators(df_recent)
                fig = create_mini_sparkline(df_recent, ticker)
                st.plotly_chart(fig, use_container_width=True, key=f"spark_{ticker}")

# =====================================================================
# MAIN PAGE
# =====================================================================

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 32px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 10px 20px rgba(0,0,0,0.19);">
    <h1 style="color: white; margin: 0; font-size: 42px;"><i class="fas fa-chart-line"></i> EMA Signals Analysis</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 18px;">
        Comprehensive EMA signal analysis for all tickers
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### <i class='far fa-calendar'></i> Analysis Settings", unsafe_allow_html=True)
    
    latest_date = get_latest_date_from_db()
    if latest_date:
        selected_date = st.date_input(
            "Analysis Date",
            value=latest_date.date(),
            max_value=latest_date.date()
        )
        selected_datetime = datetime.combine(selected_date, datetime.min.time())
    else:
        st.error("Database connection failed")
        st.stop()
    
    st.markdown("---")
    
    st.markdown("### <i class='fas fa-cog'></i> Display Options", unsafe_allow_html=True)
    show_sparklines = st.checkbox("Show price sparklines", value=False)
    show_details = st.checkbox("Show detailed metrics", value=True)
    
    st.markdown("---")
    
    st.markdown("### <i class='fas fa-filter'></i> Filters", unsafe_allow_html=True)
    
    # Zone filter
    zone_filter = st.multiselect(
        "Trading Zone",
        options=['buy', 'accumulate', 'distribute', 'sell', 'risk', 'neutral'],
        default=['buy', 'accumulate', 'distribute', 'sell', 'risk', 'neutral']
    )
    
    # Strength filter
    min_strength = st.slider("Minimum Strength", 1, 5, 3)
    
    # Alignment filter
    alignment_filter = st.multiselect(
        "EMA Alignment",
        options=['bullish', 'bearish', 'mixed', 'neutral'],
        default=['bullish']
    )
    
    st.markdown("---")
    
    # Recalculate button
    if st.button("Recalculate", type="primary"):
        st.cache_data.clear()
        st.rerun()

# Load data
with st.spinner("Loading market data..."):
    all_tickers = get_all_tickers_cached()
    if not all_tickers:
        st.error("No tickers found in database")
        st.stop()
    
    # Calculate indicators for all tickers
    df_indicators = calculate_indicators_for_all_tickers(all_tickers, selected_datetime)

if df_indicators.empty:
    st.warning("No indicator data available. Please check database.")
    st.stop()

# Extract VNINDEX data
vnindex_data = df_indicators[df_indicators['ticker'] == 'VNINDEX']
if not vnindex_data.empty:
    vnindex_row = vnindex_data.iloc[0]
else:
    vnindex_row = None

# Generate signals
buy_signals = generate_immediate_buy_signals(df_indicators, top_n=3)
sell_signals = generate_immediate_sell_signals(df_indicators, top_n=3)
breadth_summary = calculate_market_breadth_summary(df_indicators)
strategy = generate_market_strategy(vnindex_row, breadth_summary, buy_signals, sell_signals)

# Display sections
col1, col2 = st.columns([2, 1])

with col1:
    render_vnindex_card(vnindex_row)

with col2:
    render_breadth_summary(breadth_summary)

st.markdown("---")

# Action conclusion
render_action_conclusion(buy_signals, sell_signals, strategy)

st.markdown("---")

# Universe analysis
st.markdown("## <i class='fas fa-globe'></i> Market Universe Analysis", unsafe_allow_html=True)

# Filter tickers
df_filtered = df_indicators[df_indicators['ticker'] != 'VNINDEX'].copy()

if zone_filter:
    df_filtered = df_filtered[df_filtered['ema_zone'].isin(zone_filter)]

if alignment_filter:
    df_filtered = df_filtered[df_filtered['ema_alignment'].isin(alignment_filter)]

df_filtered = df_filtered[df_filtered['ema_strength'] >= min_strength]

# Sort by urgency
df_ranked = rank_ema_urgency(df_filtered, top_n=50)

# Display stats
st.markdown(f"""
<div class="material-card elevation-1">
    <div style="display: flex; justify-content: space-around;">
        <div style="text-align: center;">
            <div style="font-size: 28px; font-weight: bold; color: #2196F3;">{len(df_filtered)}</div>
            <div style="font-size: 14px; color: #757575;">Filtered Tickers</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 28px; font-weight: bold; color: #4CAF50;">{len(df_filtered[df_filtered['ema_zone'] == 'buy'])}</div>
            <div style="font-size: 14px; color: #757575;">Buy Zone</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 28px; font-weight: bold; color: #FF9800;">{len(df_filtered[df_filtered['ema_alignment'] == 'bullish'])}</div>
            <div style="font-size: 14px; color: #757575;">Bullish Alignment</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### <i class='fas fa-trophy'></i> Top Ranked Opportunities", unsafe_allow_html=True)

if not df_ranked.empty:
    # Display in grid
    num_cols = 2 if not show_sparklines else 1
    rows = []
    for i in range(0, len(df_ranked), num_cols):
        rows.append(df_ranked.iloc[i:i+num_cols])
    
    for row_df in rows:
        cols = st.columns(num_cols)
        for idx, (_, ticker_row) in enumerate(row_df.iterrows()):
            with cols[idx]:
                render_ticker_card(
                    df_indicators[df_indicators['ticker'] == ticker_row['ticker']].iloc[0],
                    show_sparkline=show_sparklines
                )
else:
    st.info("No tickers match the selected filters")

# Detailed table
if show_details:
    st.markdown("---")
    st.markdown("### <i class='fas fa-list'></i> Detailed Metrics Table", unsafe_allow_html=True)
    
    display_cols = ['ticker', 'close', 'ema_strength', 'ema_zone', 'ema_alignment', 
                    'ema_convergence', 'ema20_dist', 'ema50_dist']
    
    df_display = df_filtered[display_cols].copy()
    df_display.columns = ['Ticker', 'Close', 'Strength', 'Zone', 'Alignment', 
                          'Convergence', 'EMA20%', 'EMA50%']
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Export button
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"ema_signals_{selected_date}.csv",
        mime="text/csv"
    )

# Methodology section
st.markdown("---")
with st.expander("EMA Alignment & Methodology", expanded=False):
    st.markdown("""
    ### EMA Calculation Method
    
    **Exponential Moving Average (EMA)** gives more weight to recent prices, making it more responsive to price changes than Simple Moving Average (SMA).
    
    **Periods Used**: 10, 20, 50, 100, 150, 200 days
    
    ---
    
    ### EMA Alignment Categories
    
    **🟢 Bullish Alignment**
    - Price > EMA10 > EMA20 > EMA50 > EMA100 > EMA200
    - All EMAs in proper ascending order
    - Strong uptrend with momentum
    - **Action**: Accumulate on dips to EMA10/20
    
    **🔴 Bearish Alignment**
    - Price < EMA10 < EMA20 < EMA50 < EMA100 < EMA200
    - All EMAs in descending order
    - Strong downtrend with selling pressure
    - **Action**: Avoid or wait for EMA50 reclaim
    
    **🟡 Mixed Alignment**
    - EMAs are intertwined
    - Price crossing between short and medium EMAs
    - Consolidation or transition phase
    - **Action**: Wait for clear direction
    
    **⚪ Neutral**
    - Insufficient data or unclear pattern
    - **Action**: No action until clarity emerges
    
    ---
    
    ### Trading Zones
    
    **🟢 BUY ZONE**
    - Price > EMA20 > EMA50 > EMA200
    - Perfect bullish alignment
    - Strong momentum confirming
    - **Risk**: Low (riding the trend)
    
    **🟢 ACCUMULATE ZONE**
    - Price > EMA200 but mixed shorter EMAs
    - Long-term uptrend intact
    - Potential consolidation before next leg
    - **Risk**: Medium (selective buying)
    
    **🟡 DISTRIBUTE ZONE**
    - Price < EMA50 but > EMA200
    - Warning of weakness
    - Consider reducing exposure
    - **Risk**: Medium-High (caution)
    
    **🔴 SELL ZONE**
    - Price < EMA50 and approaching EMA200
    - Breakdown imminent
    - Protect capital
    - **Risk**: High (exit positions)
    
    **⚫ RISK ZONE**
    - Price < EMA200
    - All major supports lost
    - Avoid new positions
    - **Risk**: Very High (cash preferred)
    
    ---
    
    ### Strength Score (1-5)
    
    **Calculation Components**:
    1. Price position vs all EMAs (above = stronger)
    2. EMA order (proper sequence = stronger)
    3. Distance from EMAs (further above = stronger)
    4. Momentum indicators (RSI, MACD if available)
    
    **Score Interpretation**:
    - **5/5**: 🔥 Very Strong - Maximum bullish alignment
    - **4/5**: ✅ Strong - Clear uptrend
    - **3/5**: 😐 Neutral - Mixed signals
    - **2/5**: ⚠️ Weak - Downtrend developing
    - **1/5**: 🚫 Very Weak - Strong bearish
    
    ---
    
    ### Convergence Metric
    
    **Definition**: Standard deviation of all EMAs relative to their mean
    
    **Interpretation**:
    - **< 2%**: Very tight - Breakout imminent (either direction)
    - **2-5%**: Moderate - Normal market
    - **> 5%**: Wide spread - Trending market
    
    **Trading Implication**:
    - Low convergence + bullish setup = Strong breakout potential
    - Low convergence + bearish setup = Breakdown risk
    
    ---
    
    ### Signal Scoring Logic
    
    **Buy Signals (Score >= 5)**:
    - ✅ FOMO breakout above EMA50 (+3 points)
    - ✅ Golden cross (EMA10 > EMA20 > EMA50) (+3 points)
    - ✅ Above all major EMAs (+2 points)
    - ✅ Strong momentum (Price > EMA10 > EMA20) (+2 points)
    - ✅ EMA convergence < 3% (+2 points)
    - ✅ Near EMA20 support (+1 point)
    - ✅ RSI oversold < 40 (+2 points)
    - ✅ MACD bullish (+1 point)
    
    **Sell Signals (Score >= 5)**:
    - ❌ Breakdown below EMA50 (+3 points)
    - ❌ Critical breakdown below EMA100 (+2 points)
    - ❌ Death cross (EMA10 < EMA20 < EMA50) (+3 points)
    - ❌ Below all major EMAs (+2 points)
    - ❌ Declining momentum (Price < EMA10 < EMA20) (+2 points)
    - ❌ Failed EMA20 resistance (+1 point)
    - ❌ Overextended below EMA50 > 10% (+2 points)
    - ❌ RSI overbought > 70 (+2 points)
    - ❌ MACD bearish (+1 point)
    
    ---
    
    ### Data Source & Updates
    
    - **Database**: MongoDB (macd_reversal collection)
    - **Update Frequency**: Daily after market close
    - **Calculation**: Uses latest available data with 365-day warmup period
    - **Indicators**: Reuses `utils/indicators.py` with TA-Lib/pandas-ta
    
    ---
    
    ### Risk Disclaimer
    
    ⚠️ **This tool is for educational and informational purposes only.**
    
    - EMA signals are lagging indicators (react to past prices)
    - No guarantee of future performance
    - Always use proper risk management
    - Combine with other analysis (fundamentals, volume, market conditions)
    - Past EMA patterns may not repeat
    - Recommended: 2-5% position sizing, stop-loss at EMA50/100
    """, unsafe_allow_html=False)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #757575; font-size: 14px;">
    <i class="fas fa-chart-bar"></i> EMA Signals Analysis • Material Design UI • Data from MongoDB
</div>
""", unsafe_allow_html=True)
