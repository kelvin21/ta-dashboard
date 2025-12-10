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
        trend = "📈 UPTREND"
        trend_color = "#4CAF50"
    elif close < ema50 < ema200:
        trend = "📉 DOWNTREND"
        trend_color = "#F44336"
    else:
        trend = "➡️ SIDEWAYS"
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
        <h3 style="margin: 0 0 16px 0; color: #212121;">📊 Market Breadth</h3>
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
        <h2 style="margin: 0 0 16px 0; color: white;">🎯 Immediate Action Plan</h2>
    """, unsafe_allow_html=True)
    
    # Buy signals
    if buy_signals:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 16px; border-radius: 8px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 12px 0; color: white;">🟢 Top Buy Opportunities</h3>
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
            <h3 style="margin: 0 0 12px 0; color: white;">🔴 Top Sell/Avoid Positions</h3>
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
            <h4 style="margin: 0 0 8px 0; color: white;">💡 Market Strategy</h4>
            <p style="margin: 0; font-size: 16px; line-height: 1.6; color: white;">
                {strategy}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_ticker_card(row: pd.Series, show_sparkline: bool = False):
    """Render individual ticker card."""
    ticker = row.get('ticker', 'N/A')
    close = row.get('close', 0)
    strength = row.get('ema_strength', 3)
    zone = row.get('ema_zone', 'neutral')
    alignment = row.get('ema_alignment', 'neutral')
    convergence = row.get('ema_convergence', 0)
    
    # Get distances
    ema20_dist = row.get('ema20_dist', 0)
    ema50_dist = row.get('ema50_dist', 0)
    
    zone_color = get_zone_color(zone)
    align_color = get_alignment_color(alignment)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div class="material-card elevation-1" style="padding: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="margin: 0; color: #212121;">{ticker}</h3>
                <div style="background: {zone_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                    {zone.upper()}
                </div>
            </div>
            <div style="font-size: 24px; font-weight: bold; color: #212121; margin-bottom: 8px;">
                {close:.2f}
            </div>
            <div style="display: flex; gap: 8px; margin-bottom: 12px;">
                <span class="chip chip-info">Strength: {strength}/5</span>
                <span class="chip" style="background: {align_color}; color: white;">{alignment}</span>
            </div>
            <div style="font-size: 13px; color: #757575;">
                EMA20: {format_distance(ema20_dist)} | EMA50: {format_distance(ema50_dist)}<br/>
                Convergence: {convergence:.2f}%
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
    <h1 style="color: white; margin: 0; font-size: 42px;">📈 EMA Signals Analysis</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 18px;">
        Comprehensive EMA signal analysis for all tickers
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📅 Analysis Settings")
    
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
    
    st.markdown("### 🎯 Display Options")
    show_sparklines = st.checkbox("Show price sparklines", value=False)
    show_details = st.checkbox("Show detailed metrics", value=True)
    
    st.markdown("---")
    
    st.markdown("### 🔍 Filters")
    
    # Zone filter
    zone_filter = st.multiselect(
        "Trading Zone",
        options=['buy', 'accumulate', 'distribute', 'sell', 'risk', 'neutral'],
        default=['buy', 'accumulate']
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
    if st.button("🔄 Recalculate", type="primary"):
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
st.markdown("## 🌐 Market Universe Analysis")

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

st.markdown("### 🎯 Top Ranked Opportunities")

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
    st.markdown("### 📋 Detailed Metrics Table")
    
    display_cols = ['ticker', 'close', 'ema_strength', 'ema_zone', 'ema_alignment', 
                    'ema_convergence', 'ema20_dist', 'ema50_dist']
    
    df_display = df_filtered[display_cols].copy()
    df_display.columns = ['Ticker', 'Close', 'Strength', 'Zone', 'Alignment', 
                          'Convergence', 'EMA20%', 'EMA50%']
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Export button
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"ema_signals_{selected_date}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #757575; font-size: 14px;">
    EMA Signals Analysis • Material Design UI • Data from MongoDB
</div>
""", unsafe_allow_html=True)
