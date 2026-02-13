"""
Pattern Scanner Page
Identifies classical trading patterns that can form over 6-18 months.
Provides buy/sell signals with target prices and stop loss levels.
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
from scipy.signal import argrelextrema
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

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

# Import utilities
try:
    from utils.pattern_detection import PatternDetector, rank_patterns_by_quality
    from utils.db_async import get_sync_db_adapter
    from utils.indicators import calculate_all_indicators
    USE_UTILS = True
except ImportError as e:
    st.error(f"Failed to import utility modules: {e}")
    st.info("Please ensure utils/ directory exists with pattern_detection.py and db_async.py")
    st.stop()

# Page config
st.set_page_config(
    page_title="Pattern Scanner",
    layout="wide",
    page_icon="📈"
)

# Title
st.markdown("# 📈 Trading Pattern Scanner")
st.markdown("Identify classical chart patterns with buy/sell signals and price targets")

# Initialize session state
if 'patterns_analyzed' not in st.session_state:
    st.session_state.patterns_analyzed = False
if 'all_patterns' not in st.session_state:
    st.session_state.all_patterns = []
if 'analysis_progress' not in st.session_state:
    st.session_state.analysis_progress = 0
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

@st.cache_resource(ttl=3600)
def get_db():
    """Get database adapter (cached)."""
    return get_sync_db_adapter()

@st.cache_data(ttl=1800)
def get_all_tickers_cached():
    """Get all tickers from database (cached)."""
    try:
        db = get_db()
        tickers = db.get_all_tickers()
        return tickers if tickers else []
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")
        return []

def analyze_ticker_patterns(ticker: str, lookback_months: int = 18, include_forming: bool = False) -> list:
    """
    Analyze patterns for a single ticker.
    
    Args:
        ticker: Ticker symbol
        lookback_months: Number of months to look back
        include_forming: If True, include patterns still forming
    
    Returns:
        List of detected patterns
    """
    try:
        db = get_db()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months * 30)
        
        # Get price data
        df = db.get_price_data(ticker, start_date, end_date)
        
        if df.empty or len(df) < 30:
            return []
        
        # Initialize pattern detector
        detector = PatternDetector(min_pattern_days=30, max_pattern_days=lookback_months * 30)
        
        # Detect all patterns
        patterns = detector.detect_all_patterns(df, ticker, include_forming=include_forming)
        
        # Add timeframe targets
        for pattern in patterns:
            targets = detector.calculate_timeframe_targets(pattern, df)
            pattern.update(targets)
        
        return patterns
    
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return []

def analyze_all_tickers(tickers: list, lookback_months: int = 18, include_forming: bool = False, progress_callback=None):
    """
    Analyze patterns for all tickers using parallel processing.
    
    Args:
        tickers: List of ticker symbols
        lookback_months: Number of months to look back
        include_forming: If True, include patterns still forming
        progress_callback: Function to call with progress updates
    
    Returns:
        List of all detected patterns
    """
    all_patterns = []
    total = len(tickers)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_ticker_patterns, ticker, lookback_months, include_forming): ticker 
                   for ticker in tickers}
        
        for i, future in enumerate(futures):
            try:
                patterns = future.result(timeout=30)
                all_patterns.extend(patterns)
                
                if progress_callback:
                    progress_callback(i + 1, total)
            except Exception as e:
                print(f"Error processing ticker: {e}")
    
    # Rank by quality
    ranked_patterns = rank_patterns_by_quality(all_patterns)
    
    return ranked_patterns

def create_pattern_chart(ticker: str, pattern: dict, lookback_days: int = None):
    """Create interactive chart showing the pattern with shape overlay."""
    try:
        db = get_db()
        
        # Use pattern formation days or default lookback
        if lookback_days is None:
            lookback_days = max(pattern.get('formation_days', 180) + 30, 180)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        df = db.get_price_data(ticker, start_date, end_date)
        
        if df.empty:
            return None
        
        # Create candlestick chart
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Add pattern shape overlay
        pattern_type = pattern['pattern'].lower()
        formation_days = pattern.get('formation_days', 180)
        pattern_start_idx = max(0, len(df) - formation_days)
        pattern_df = df.iloc[pattern_start_idx:]
        
        # Add pattern region highlight
        if len(pattern_df) > 0:
            fig.add_vrect(
                x0=pattern_df.iloc[0]['date'],
                x1=pattern_df.iloc[-1]['date'],
                fillcolor="yellow",
                opacity=0.1,
                line_width=0,
                annotation_text="Pattern Formation",
                annotation_position="top left"
            )
            
            # Draw Head & Shoulders pattern shape
            if 'shoulder' in pattern_type or 'head' in pattern_type:
                key_points = pattern.get('key_points', {})
                
                if 'left_shoulder' in key_points and 'head' in key_points and 'right_shoulder' in key_points:
                    ls = key_points['left_shoulder']
                    head = key_points['head']
                    rs = key_points['right_shoulder']
                    
                    # Draw connecting lines for peaks
                    fig.add_trace(go.Scatter(
                        x=[ls['date'], head['date'], rs['date']],
                        y=[ls['price'], head['price'], rs['price']],
                        mode='lines+markers',
                        line=dict(color='orange', width=3, dash='solid'),
                        marker=dict(size=12, color='orange', symbol='diamond'),
                        name='Pattern Shape',
                        showlegend=False
                    ))
                    
                    # Add labels
                    yshift = 25 if 'inverse' not in pattern_type.lower() else -25
                    fig.add_annotation(x=ls['date'], y=ls['price'], text="LS", showarrow=False, yshift=yshift, font=dict(size=12, color='orange', family='Arial Black'))
                    fig.add_annotation(x=head['date'], y=head['price'], text="HEAD", showarrow=False, yshift=yshift, font=dict(size=12, color='orange', family='Arial Black'))
                    fig.add_annotation(x=rs['date'], y=rs['price'], text="RS", showarrow=False, yshift=yshift, font=dict(size=12, color='orange', family='Arial Black'))
            
            # Draw Triangle pattern trendlines
            elif 'triangle' in pattern_type:
                key_points = pattern.get('key_points', {})
                trendlines = pattern.get('trendlines', {})
                
                if key_points and trendlines and 'peaks' in key_points and 'troughs' in key_points:
                    peaks = key_points['peaks']
                    troughs = key_points['troughs']
                    
                    if len(peaks) >= 2 and len(troughs) >= 2:
                        # Ascending Triangle: Flat resistance (top), Rising support (bottom)
                        if 'ascending' in pattern_type.lower():
                            # Flat resistance at peak level
                            resistance_level = trendlines['resistance']['level']
                            fig.add_trace(go.Scatter(
                                x=[peaks[0]['date'], peaks[-1]['date']],
                                y=[resistance_level, resistance_level],
                                mode='lines',
                                line=dict(color='red', width=3, dash='solid'),
                                name='Flat Resistance',
                                showlegend=False
                            ))
                            # Mark peaks
                            for p in peaks[-3:]:
                                fig.add_trace(go.Scatter(
                                    x=[p['date']], y=[p['price']],
                                    mode='markers',
                                    marker=dict(size=10, color='red', symbol='triangle-down'),
                                    showlegend=False, hoverinfo='skip'
                                ))
                            
                            # Rising support line
                            fig.add_trace(go.Scatter(
                                x=[troughs[0]['date'], troughs[-1]['date']],
                                y=[troughs[0]['price'], troughs[-1]['price']],
                                mode='lines+markers',
                                line=dict(color='green', width=3, dash='solid'),
                                marker=dict(size=10, color='green', symbol='triangle-up'),
                                name='Rising Support',
                                showlegend=False
                            ))
                        
                        # Descending Triangle: Descending resistance (top), Flat support (bottom)
                        elif 'descending' in pattern_type.lower():
                            # Descending resistance line
                            fig.add_trace(go.Scatter(
                                x=[peaks[0]['date'], peaks[-1]['date']],
                                y=[peaks[0]['price'], peaks[-1]['price']],
                                mode='lines+markers',
                                line=dict(color='red', width=3, dash='solid'),
                                marker=dict(size=10, color='red', symbol='triangle-down'),
                                name='Descending Resistance',
                                showlegend=False
                            ))
                            
                            # Flat support at trough level
                            support_level = trendlines['support']['level']
                            fig.add_trace(go.Scatter(
                                x=[troughs[0]['date'], troughs[-1]['date']],
                                y=[support_level, support_level],
                                mode='lines',
                                line=dict(color='green', width=3, dash='solid'),
                                name='Flat Support',
                                showlegend=False
                            ))
                            # Mark troughs
                            for t in troughs[-3:]:
                                fig.add_trace(go.Scatter(
                                    x=[t['date']], y=[t['price']],
                                    mode='markers',
                                    marker=dict(size=10, color='green', symbol='triangle-up'),
                                    showlegend=False, hoverinfo='skip'
                                ))
                        
                        # Symmetrical Triangle: Both converging
                        else:
                            # Descending resistance
                            fig.add_trace(go.Scatter(
                                x=[peaks[0]['date'], peaks[-1]['date']],
                                y=[peaks[0]['price'], peaks[-1]['price']],
                                mode='lines+markers',
                                line=dict(color='red', width=3, dash='solid'),
                                marker=dict(size=10, color='red', symbol='triangle-down'),
                                name='Resistance',
                                showlegend=False
                            ))
                            
                            # Rising support
                            fig.add_trace(go.Scatter(
                                x=[troughs[0]['date'], troughs[-1]['date']],
                                y=[troughs[0]['price'], troughs[-1]['price']],
                                mode='lines+markers',
                                line=dict(color='green', width=3, dash='solid'),
                                marker=dict(size=10, color='green', symbol='triangle-up'),
                                name='Support',
                                showlegend=False
                            ))
            
            # Draw Flag/Pennant pattern (pole + flag channel)
            elif 'flag' in pattern_type:
                key_points = pattern.get('key_points', {})
                
                if key_points and 'pole_start' in key_points and 'pole_end' in key_points:
                    pole_start = key_points['pole_start']
                    pole_end = key_points['pole_end']
                    flag_start_idx = key_points.get('flag_start', 0)
                    
                    # Draw pole (strong trend line)
                    fig.add_trace(go.Scatter(
                        x=[pole_start['date'], pole_end['date']],
                        y=[pole_start['price'], pole_end['price']],
                        mode='lines',
                        line=dict(color='purple', width=4, dash='solid'),
                        name='Pole',
                        showlegend=False
                    ))
                    
                    # Mark pole endpoints
                    fig.add_trace(go.Scatter(
                        x=[pole_start['date'], pole_end['date']],
                        y=[pole_start['price'], pole_end['price']],
                        mode='markers',
                        marker=dict(size=10, color='purple', symbol='circle'),
                        showlegend=False
                    ))
                    
                    # Draw flag channel lines
                    flag_df = pattern_df.iloc[flag_start_idx:]
                    if len(flag_df) >= 3:
                        flag_high = key_points.get('flag_high', flag_df['high'].max())
                        flag_low = key_points.get('flag_low', flag_df['low'].min())
                        
                        # Upper channel
                        fig.add_trace(go.Scatter(
                            x=[flag_df.iloc[0]['date'], flag_df.iloc[-1]['date']],
                            y=[flag_high, flag_high],
                            mode='lines',
                            line=dict(color='orange', width=2, dash='dash'),
                            showlegend=False
                        ))
                        
                        # Lower channel
                        fig.add_trace(go.Scatter(
                            x=[flag_df.iloc[0]['date'], flag_df.iloc[-1]['date']],
                            y=[flag_low, flag_low],
                            mode='lines',
                            line=dict(color='orange', width=2, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(255, 165, 0, 0.1)',
                            showlegend=False
                        ))
            
            # Draw Wedge pattern (converging trendlines)
            elif 'wedge' in pattern_type:
                key_points = pattern.get('key_points', {})
                
                if key_points and 'peaks' in key_points and 'troughs' in key_points:
                    peaks = key_points['peaks']
                    troughs = key_points['troughs']
                    
                    if len(peaks) >= 2 and len(troughs) >= 2:
                        # Draw upper trendline (resistance)
                        peaks_dates = [p['date'] for p in peaks]
                        peaks_prices = [p['price'] for p in peaks]
                        
                        fig.add_trace(go.Scatter(
                            x=peaks_dates,
                            y=peaks_prices,
                            mode='lines+markers',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=10, color='red', symbol='triangle-down'),
                            name='Resistance',
                            showlegend=False
                        ))
                        
                        # Draw lower trendline (support)
                        troughs_dates = [t['date'] for t in troughs]
                        troughs_prices = [t['price'] for t in troughs]
                        
                        fig.add_trace(go.Scatter(
                            x=troughs_dates,
                            y=troughs_prices,
                            mode='lines+markers',
                            line=dict(color='green', width=2, dash='dash'),
                            marker=dict(size=10, color='green', symbol='triangle-up'),
                            name='Support',
                            showlegend=False
                        ))
                        
                        # Fill between trendlines to show wedge
                        # Extend lines to show convergence
                        all_dates = peaks_dates + troughs_dates
                        fig.add_trace(go.Scatter(
                            x=[peaks_dates[0], peaks_dates[-1]],
                            y=[peaks_prices[0], peaks_prices[-1]],
                            mode='lines',
                            line=dict(color='red', width=1, dash='solid'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[troughs_dates[0], troughs_dates[-1]],
                            y=[troughs_prices[0], troughs_prices[-1]],
                            mode='lines',
                            line=dict(color='green', width=1, dash='solid'),
                            fill='tonexty',
                            fillcolor='rgba(128, 128, 128, 0.1)',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
            
            # Draw Cup and Handle pattern
            elif 'cup' in pattern_type and 'handle' in pattern_type:
                key_points = pattern.get('key_points', {})
                
                if key_points and 'cup_left' in key_points:
                    cup_left = key_points['cup_left']
                    cup_bottom = key_points['cup_bottom']
                    cup_right = key_points['cup_right']
                    cup_length = key_points.get('cup_length', int(len(pattern_df) * 0.67))
                    
                    # Get cup portion
                    cup_df = pattern_df.iloc[:cup_length]
                    
                    if len(cup_df) > 5:
                        # Draw cup curve (smooth spline through prices)
                        fig.add_trace(go.Scatter(
                            x=cup_df['date'],
                            y=cup_df['close'],
                            mode='lines',
                            line=dict(color='blue', width=3, shape='spline'),
                            name='Cup',
                            showlegend=False
                        ))
                        
                        # Mark cup edges and bottom
                        fig.add_trace(go.Scatter(
                            x=[cup_left['date'], cup_bottom['date'], cup_right['date']],
                            y=[cup_left['price'], cup_bottom['price'], cup_right['price']],
                            mode='markers',
                            marker=dict(size=12, color='blue', symbol=['circle', 'star', 'circle']),
                            showlegend=False
                        ))
                        
                        # Draw rim line
                        rim_price = (cup_left['price'] + cup_right['price']) / 2
                        fig.add_trace(go.Scatter(
                            x=[cup_left['date'], cup_right['date']],
                            y=[rim_price, rim_price],
                            mode='lines',
                            line=dict(color='blue', width=2, dash='solid'),
                            name='Rim',
                            showlegend=False
                        ))
                        
                        # Draw handle
                        handle_df = pattern_df.iloc[cup_length:]
                        if len(handle_df) >= 3:
                            fig.add_trace(go.Scatter(
                                x=handle_df['date'],
                                y=handle_df['high'],
                                mode='lines',
                                line=dict(color='lightblue', width=2, dash='dot'),
                                showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=handle_df['date'],
                                y=handle_df['low'],
                                mode='lines',
                                line=dict(color='lightblue', width=2, dash='dot'),
                                fill='tonexty',
                                fillcolor='rgba(173, 216, 230, 0.2)',
                                showlegend=False
                            ))
            
            # Draw Double Top/Bottom pattern
            elif 'double' in pattern_type:
                key_points = pattern.get('key_points', {})
                
                if 'top' in pattern_type and 'peak1' in key_points and 'peak2' in key_points:
                    peak1 = key_points['peak1']
                    peak2 = key_points['peak2']
                    trough = key_points.get('trough', {})
                    
                    # Draw M shape (peak1 -> trough -> peak2)
                    if trough:
                        fig.add_trace(go.Scatter(
                            x=[peak1['date'], trough['date'], peak2['date']],
                            y=[peak1['price'], trough['price'], peak2['price']],
                            mode='lines+markers',
                            line=dict(color='red', width=3, dash='solid'),
                            marker=dict(size=14, color='red', symbol=['triangle-down', 'circle', 'triangle-down']),
                            name='Double Top',
                            showlegend=False
                        ))
                        
                        # Add labels
                        fig.add_annotation(x=peak1['date'], y=peak1['price'], text="P1", showarrow=False, yshift=20, font=dict(size=12, color='red', family='Arial Black'))
                        fig.add_annotation(x=peak2['date'], y=peak2['price'], text="P2", showarrow=False, yshift=20, font=dict(size=12, color='red', family='Arial Black'))
                
                elif 'bottom' in pattern_type and 'trough1' in key_points and 'trough2' in key_points:
                    trough1 = key_points['trough1']
                    trough2 = key_points['trough2']
                    peak = key_points.get('peak', {})
                    
                    # Draw W shape (trough1 -> peak -> trough2)
                    if peak:
                        fig.add_trace(go.Scatter(
                            x=[trough1['date'], peak['date'], trough2['date']],
                            y=[trough1['price'], peak['price'], trough2['price']],
                            mode='lines+markers',
                            line=dict(color='green', width=3, dash='solid'),
                            marker=dict(size=14, color='green', symbol=['triangle-up', 'circle', 'triangle-up']),
                            name='Double Bottom',
                            showlegend=False
                        ))
                        
                        # Add labels
                        fig.add_annotation(x=trough1['date'], y=trough1['price'], text="T1", showarrow=False, yshift=-20, font=dict(size=12, color='green', family='Arial Black'))
                        fig.add_annotation(x=trough2['date'], y=trough2['price'], text="T2", showarrow=False, yshift=-20, font=dict(size=12, color='green', family='Arial Black'))
        
        # Add pattern markers
        current_price = pattern['current_price']
        target_price = pattern['target_price']
        stop_loss = pattern['stop_loss']
        
        # Current price line
        fig.add_hline(y=current_price, line_dash="solid", line_color="blue", 
                      annotation_text=f"Current: {current_price:,.0f}",
                      annotation_position="right")
        
        # Target price line
        fig.add_hline(y=target_price, line_dash="dash", line_color="green" if pattern['signal'] == 'BUY' else "red",
                      annotation_text=f"Target: {target_price:,.0f}",
                      annotation_position="right")
        
        # Stop loss line
        fig.add_hline(y=stop_loss, line_dash="dot", line_color="red" if pattern['signal'] == 'BUY' else "green",
                      annotation_text=f"Stop: {stop_loss:,.0f}",
                      annotation_position="right")
        
        # Neckline if available
        if pattern.get('neckline'):
            fig.add_hline(y=pattern['neckline'], line_dash="dash", line_color="orange",
                         annotation_text="Neckline",
                         annotation_position="left")
        
        fig.update_layout(
            title=f"{ticker} - {pattern['pattern']}",
            yaxis_title="Price",
            xaxis_title="Date",
            template="plotly_white",
            height=400,
            showlegend=False,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def display_pattern_card(pattern: dict):
    """Display a compact pattern card (1/3 original size)."""
    signal_icon = "📈" if pattern['signal'] == 'BUY' else "📉"
    status_badge = "🔮" if pattern.get('status') == 'forming' else ""
    
    current = pattern['current_price']
    target_full = pattern.get('target_full', pattern['target_price'])
    stop = pattern['stop_loss']
    potential_full = ((target_full - current) / current * 100)
    confidence_pct = pattern.get('confidence', 0.5) * 100
    
    # Compact card with header and key metrics only
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**{signal_icon} {pattern['ticker']}** - {pattern['pattern']} {status_badge}")
        st.caption(f"{pattern.get('formation_days', 0)}d | Conf: {confidence_pct:.0f}%")
    with col2:
        st.metric("Current", f"{current:,.0f}")
    with col3:
        st.metric("Target", f"{target_full:,.0f}", f"{potential_full:+.1f}%")

# =====================================================================
# MAIN UI
# =====================================================================

# Info section
with st.expander("ℹ️ About Pattern Scanner", expanded=False):
    st.markdown("""
    ### What are Chart Patterns?
    Chart patterns are formations created by the price movements of stocks over time. These patterns can take 
    **6-18 months** to fully develop and provide insights into potential future price movements.
    
    ### Patterns Detected:
    
    **Reversal Patterns (Trend Change):**
    - 📊 **Head and Shoulders**: Strong reversal pattern after uptrend
    - 📊 **Inverse Head and Shoulders**: Bullish reversal after downtrend
    - 📊 **Double/Triple Top/Bottom**: Multiple tests of support/resistance
    - 📊 **Rising/Falling Wedge**: Price converging with directional bias
    
    **Continuation Patterns (Trend Continues):**
    - 📈 **Bull/Bear Flags**: Brief consolidation before trend continues
    - 📈 **Triangles**: Ascending, descending, or symmetrical consolidation
    - 📈 **Cup and Handle**: U-shaped recovery with small pullback
    
    ### How to Use:
    1. Click **"Scan All Patterns"** to analyze entire database (takes 2-5 minutes)
    2. Filter by signal type, pattern, or timeframe
    3. Review recommendations with target prices and stop loss
    4. View charts to confirm pattern visually
    
    ### Target Timeframes:
    - **1-3 Days**: Conservative short-term target
    - **1 Month**: Medium-term target (~50% of full pattern move)
    - **Full Target**: Complete pattern target (may take several months)
    
    ### Risk Management:
    Always use stop loss levels provided. Risk/Reward ratio shows potential gain vs. potential loss.
    """)

# Control panel
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    lookback_months = st.slider(
        "Analysis Period (Months)",
        min_value=3,
        max_value=24,
        value=18,
        help="How far back to look for pattern formation"
    )

with col2:
    min_quality = st.slider(
        "Minimum Quality Score",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Filter patterns by quality score"
    )

with col3:
    min_risk_reward = st.slider(
        "Min Risk/Reward Ratio",
        min_value=1.0,
        max_value=5.0,
        value=1.5,
        step=0.5,
        help="Minimum acceptable risk/reward ratio"
    )

# Additional options
col_opt1, col_opt2, col_opt3 = st.columns([2, 2, 2])

with col_opt1:
    include_forming = st.checkbox(
        "🔮 Show Forming Patterns",
        value=False,
        help="Include patterns that are still forming (not yet complete). These have lower confidence but may signal future opportunities."
    )

if include_forming:
    st.info("💡 **Forming patterns** are shown with reduced confidence. They may provide early signals but carry higher risk.")

# Scan button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    if st.button("🔍 Scan All Patterns", type="primary", use_container_width=True, disabled=st.session_state.is_analyzing):
        st.session_state.is_analyzing = True
        st.session_state.all_patterns = []
        st.session_state.analysis_progress = 0
        
        # Get all tickers
        all_tickers = get_all_tickers_cached()
        
        if not all_tickers:
            st.error("No tickers found in database!")
            st.session_state.is_analyzing = False
        else:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total):
                progress = current / total
                st.session_state.analysis_progress = progress
                progress_bar.progress(progress)
                status_text.text(f"Analyzing patterns... {current}/{total} tickers ({progress*100:.1f}%)")
            
            status_text.text(f"Starting analysis of {len(all_tickers)} tickers...")
            
            # Run analysis
            start_time = time.time()
            patterns = analyze_all_tickers(all_tickers, lookback_months, include_forming, update_progress)
            elapsed = time.time() - start_time
            
            st.session_state.all_patterns = patterns
            st.session_state.patterns_analyzed = True
            st.session_state.is_analyzing = False
            
            progress_bar.empty()
            status_text.success(f"✅ Analysis complete! Found {len(patterns)} patterns in {elapsed:.1f} seconds")
            time.sleep(2)
            status_text.empty()
            st.rerun()

# Display results
if st.session_state.patterns_analyzed and st.session_state.all_patterns:
    st.markdown("---")
    st.markdown("## 📊 Detected Patterns")
    
    patterns = st.session_state.all_patterns
    
    # Apply filters
    filtered_patterns = [
        p for p in patterns
        if p.get('quality_score', 0) >= min_quality
        and p.get('risk_reward', 0) >= min_risk_reward
    ]
    
    # Filter controls
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        signal_filter = st.multiselect(
            "Signal Type",
            options=['BUY', 'SELL'],
            default=['BUY', 'SELL']
        )
    
    with col_f2:
        pattern_types = list(set([p['pattern'] for p in filtered_patterns]))
        pattern_filter = st.multiselect(
            "Pattern Type",
            options=sorted(pattern_types),
            default=sorted(pattern_types)
        )
    
    with col_f3:
        timeframe_filter = st.selectbox(
            "Focus Timeframe",
            options=['All', '1-3 Days', '1 Month', 'Full Target']
        )
    
    with col_f4:
        sort_by = st.selectbox(
            "Sort By",
            options=['Quality Score', 'Risk/Reward', 'Potential Gain', 'Confidence']
        )
    
    # Apply additional filters
    if signal_filter:
        filtered_patterns = [p for p in filtered_patterns if p['signal'] in signal_filter]
    if pattern_filter:
        filtered_patterns = [p for p in filtered_patterns if p['pattern'] in pattern_filter]
    
    # Sort
    if sort_by == 'Quality Score':
        filtered_patterns.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    elif sort_by == 'Risk/Reward':
        filtered_patterns.sort(key=lambda x: x.get('risk_reward', 0), reverse=True)
    elif sort_by == 'Potential Gain':
        filtered_patterns.sort(key=lambda x: abs((x['target_price'] - x['current_price']) / x['current_price']), reverse=True)
    elif sort_by == 'Confidence':
        filtered_patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Summary metrics
    st.markdown("### 📈 Summary Statistics")
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    
    buy_patterns = [p for p in filtered_patterns if p['signal'] == 'BUY']
    sell_patterns = [p for p in filtered_patterns if p['signal'] == 'SELL']
    avg_quality = np.mean([p.get('quality_score', 0) for p in filtered_patterns]) if filtered_patterns else 0
    avg_rr = np.mean([p.get('risk_reward', 0) for p in filtered_patterns]) if filtered_patterns else 0
    
    col_s1.metric("Total Patterns", len(filtered_patterns))
    col_s2.metric("Buy Signals", len(buy_patterns), delta="Bullish", delta_color="normal")
    col_s3.metric("Sell Signals", len(sell_patterns), delta="Bearish", delta_color="inverse")
    col_s4.metric("Avg Quality", f"{avg_quality:.2f}")
    col_s5.metric("Avg R/R", f"1:{avg_rr:.2f}")
    
    st.markdown("---")
    
    # Display patterns
    if not filtered_patterns:
        st.info("No patterns match the current filters. Try adjusting the criteria.")
    else:
        # Tabs for BUY and SELL
        tab_buy, tab_sell, tab_all = st.tabs([
            f"🟢 BUY Signals ({len(buy_patterns)})",
            f"🔴 SELL Signals ({len(sell_patterns)})",
            f"📊 All Patterns ({len(filtered_patterns)})"
        ])
        
        with tab_buy:
            if buy_patterns:
                st.markdown(f"### Top Buy Opportunities")
                for i, pattern in enumerate(buy_patterns[:20]):  # Limit to top 20
                    display_pattern_card(pattern)
                    
                    with st.expander(f"View Chart - {pattern['ticker']}"):
                        chart = create_pattern_chart(pattern['ticker'], pattern)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True, key=f"buy_chart_{i}_{pattern['ticker']}")
            else:
                st.info("No buy signals found with current filters.")
        
        with tab_sell:
            if sell_patterns:
                st.markdown(f"### Top Sell Opportunities")
                for i, pattern in enumerate(sell_patterns[:20]):  # Limit to top 20
                    display_pattern_card(pattern)
                    
                    with st.expander(f"View Chart - {pattern['ticker']}"):
                        chart = create_pattern_chart(pattern['ticker'], pattern)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True, key=f"sell_chart_{i}_{pattern['ticker']}")
            else:
                st.info("No sell signals found with current filters.")
        
        with tab_all:
            st.markdown(f"### All Detected Patterns")
            for i, pattern in enumerate(filtered_patterns[:30]):  # Limit to top 30
                display_pattern_card(pattern)
                
                with st.expander(f"View Chart - {pattern['ticker']}"):
                    chart = create_pattern_chart(pattern['ticker'], pattern)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True, key=f"all_chart_{i}_{pattern['ticker']}")

elif st.session_state.is_analyzing:
    st.info("Analysis in progress... Please wait.")
else:
    st.info("👆 Click 'Scan All Patterns' to begin analysis of historical data")
    
    # Show example pattern explanation
    st.markdown("---")
    st.markdown("### 📚 Pattern Examples")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.markdown("""
        #### 🟢 Bullish Patterns (Buy Signals)
        
        **Inverse Head and Shoulders**
        - Three troughs: left shoulder, head (lowest), right shoulder
        - Breaks above neckline = BUY signal
        - Target: Neckline + pattern height
        
        **Cup and Handle**
        - U-shaped cup formation over months
        - Small handle pullback near rim
        - Breaks above rim = BUY signal
        
        **Ascending Triangle**
        - Flat resistance, rising support
        - Compression leads to upward breakout
        - Target: Triangle height added to breakout
        """)
    
    with col_e2:
        st.markdown("""
        #### 🔴 Bearish Patterns (Sell Signals)
        
        **Head and Shoulders Top**
        - Three peaks: left shoulder, head (highest), right shoulder
        - Breaks below neckline = SELL signal
        - Target: Neckline - pattern height
        
        **Double Top**
        - Two peaks at similar levels
        - Breaks below valley = SELL signal
        - Target: Pattern height subtracted
        
        **Descending Triangle**
        - Flat support, declining resistance
        - Compression leads to downward break
        - Target: Triangle height subtracted
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>⚠️ <strong>Disclaimer:</strong> Pattern analysis is not financial advice. Always do your own research and use proper risk management.</p>
    <p>Patterns can fail. Use stop losses and never risk more than you can afford to lose.</p>
</div>
""", unsafe_allow_html=True)
