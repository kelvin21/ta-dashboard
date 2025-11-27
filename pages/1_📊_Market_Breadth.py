import streamlit as st
import os

# Check if page should be visible
if os.getenv("SHOW_MARKET_BREADTH_PAGE", "true").lower() == "false":
    st.error("This page is not available in the current deployment.")
    st.stop()

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add parent directory to path to import shared modules
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import from main dashboard
try:
    from ta_dashboard import (
        DB_PATH, DEFAULT_LOCAL_DB, HAS_BDB,
        load_price_range, get_all_tickers, macd_hist, detect_stage
    )
except ImportError:
    st.error("Could not import from ta_dashboard. Make sure ta_dashboard.py is in the parent directory.")
    st.stop()

st.set_page_config(page_title="Market Breadth", layout="wide", page_icon="📊")
st.title("📊 Market Breadth Analysis")

# Sidebar controls
st.sidebar.header("Analysis Settings")
lookback = st.sidebar.slider("MACD Lookback (bars)", 5, 60, 20)
days_back = st.sidebar.number_input("Days of history", 200, 730, 365)
debug = st.sidebar.checkbox("Show debug info", value=False)

st.sidebar.info("💡 **Tip:** Market Breadth requires 200+ bars (≈10 months) for MA200 calculation.")

# Historical data settings
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Historical Charts")
show_historical = st.sidebar.checkbox("Show historical breadth charts", value=True)
historical_days = st.sidebar.slider("Historical period (days)", 365, 730, 548) if show_historical else 365  # ~1.5 years default

# Recalculate button
if st.sidebar.button("🔄 Recalculate Historical Data", help="Recalculate and save historical market breadth data"):
    st.session_state['recalculate_breadth'] = True

# Load all tickers
all_tickers = get_all_tickers(debug=False)

if not all_tickers:
    st.warning("No tickers found in database")
    st.stop()

st.info(f"Analyzing {len(all_tickers)} tickers...")

# Calculate breadth metrics
end_date = datetime.now().date()
start_date = end_date - timedelta(days=days_back)

@st.cache_data(ttl=300)
def calculate_breadth_metrics(tickers, start_date, end_date, lookback, debug=False):
    """Calculate market breadth metrics for all tickers."""
    results = []
    errors = []
    skipped_reasons = {"empty": 0, "too_short": 0, "error": 0}
    
    for ticker in tickers:
        try:
            df = load_price_range(ticker, start_date, end_date)
            
            if df.empty:
                skipped_reasons["empty"] += 1
                if debug:
                    errors.append(f"{ticker}: Empty dataframe")
                continue
            
            if len(df) < 200:  # Need at least 200 bars for MA200
                skipped_reasons["too_short"] += 1
                if debug:
                    errors.append(f"{ticker}: Only {len(df)} bars (need 200+)")
                continue
            
            # Get latest data
            latest = df.iloc[-1]
            close = float(latest['close'])
            
            # Calculate moving averages
            df['ma20'] = df['close'].rolling(20).mean()
            df['ma50'] = df['close'].rolling(50).mean()
            df['ma200'] = df['close'].rolling(200).mean()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            _, _, hist = macd_hist(df['close'].astype(float))
            stage = detect_stage(hist, lookback=lookback)
            
            # Get latest values
            latest_ma20 = float(df['ma20'].iloc[-1]) if not pd.isna(df['ma20'].iloc[-1]) else None
            latest_ma50 = float(df['ma50'].iloc[-1]) if not pd.isna(df['ma50'].iloc[-1]) else None
            latest_ma200 = float(df['ma200'].iloc[-1]) if not pd.isna(df['ma200'].iloc[-1]) else None
            latest_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
            
            results.append({
                'ticker': ticker,
                'close': close,
                'above_ma20': close > latest_ma20 if latest_ma20 else None,
                'above_ma50': close > latest_ma50 if latest_ma50 else None,
                'above_ma200': close > latest_ma200 if latest_ma200 else None,
                'rsi': latest_rsi,
                'macd_stage': stage
            })
        except Exception as e:
            skipped_reasons["error"] += 1
            if debug:
                errors.append(f"{ticker}: {str(e)[:100]}")
            continue
    
    return pd.DataFrame(results), errors, skipped_reasons

with st.spinner("Calculating market breadth..."):
    breadth_df, errors, skipped_reasons = calculate_breadth_metrics(all_tickers, start_date, end_date, lookback, debug=debug)

if breadth_df.empty:
    st.error("⚠️ Not enough data to calculate breadth metrics")
    
    # Show debug information
    st.markdown("### 🔍 Debug Information")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tickers", len(all_tickers))
    with col2:
        st.metric("Date Range", f"{days_back} days")
    with col3:
        st.metric("Required Bars", "200+")
    with col4:
        st.metric("Start Date", start_date.strftime("%Y-%m-%d"))
    
    st.markdown("#### Skipped Tickers Breakdown")
    skip_df = pd.DataFrame([
        {"Reason": "Empty data", "Count": skipped_reasons["empty"], "Impact": f"{skipped_reasons['empty']/len(all_tickers)*100:.1f}%"},
        {"Reason": "Too short (<200 bars)", "Count": skipped_reasons["too_short"], "Impact": f"{skipped_reasons['too_short']/len(all_tickers)*100:.1f}%"},
        {"Reason": "Errors", "Count": skipped_reasons["error"], "Impact": f"{skipped_reasons['error']/len(all_tickers)*100:.1f}%"},
    ])
    st.dataframe(skip_df, use_container_width=True)
    
    if debug and errors:
        st.markdown("#### Detailed Errors")
        with st.expander(f"Show {len(errors)} error messages"):
            for err in errors[:50]:  # Show first 50 errors
                st.text(err)
            if len(errors) > 50:
                st.caption(f"... and {len(errors) - 50} more errors")
    
    st.markdown("#### 💡 Recommended Actions")
    
    # Determine the most likely issue
    if skipped_reasons["too_short"] > len(all_tickers) * 0.5:
        st.warning("""
        **Primary Issue: Insufficient historical data**
        
        Most tickers have less than 200 bars of data. This usually means:
        - You need to increase the date range to 365+ days
        - OR your database lacks historical data
        """)
        
        st.markdown("**Solution Steps:**")
        st.markdown("""
        1. **Increase date range** in the sidebar to **365 days** or more
        2. If still no data, use **TCBS refresh** from main dashboard to fetch historical data
        3. Ensure you're requesting at least 12 months when refreshing
        """)
    
    elif skipped_reasons["empty"] > len(all_tickers) * 0.5:
        st.error("""
        **Primary Issue: Empty database**
        
        Most tickers have no data at all.
        """)
        
        st.markdown("**Solution Steps:**")
        st.markdown("""
        1. Go back to **MACD Overview** page
        2. Use **"Force refresh ALL tickers"** button
        3. Confirm and wait for data to download
        4. Return here after refresh completes
        """)
    
    else:
        st.info("""
        **Mixed Issues Detected**
        
        Try these solutions in order:
        """)
        st.markdown("""
        1. **Increase date range:** Set to 365+ days in sidebar
        2. **Refresh data:** Use TCBS refresh on main dashboard
        3. **Check date calculations:** Verify start_date = end_date - days_back
        """)
    
    # Show sample of tickers and their data availability
    st.markdown("#### Sample Ticker Data Check")
    sample_tickers = all_tickers[:10]
    check_data = []
    for ticker in sample_tickers:
        df = load_price_range(ticker, start_date, end_date)
        check_data.append({
            "Ticker": ticker,
            "Bars": len(df),
            "Date Range": f"{df['date'].min().date()} to {df['date'].max().date()}" if not df.empty else "No data",
            "Status": "✓ OK" if len(df) >= 200 else "✗ Insufficient"
        })
    st.dataframe(pd.DataFrame(check_data), use_container_width=True)
    
    st.stop()

st.success(f"Successfully analyzed {len(breadth_df)} tickers")

# Display current date
current_date_str = datetime.now().strftime("%Y-%m-%d")
st.markdown(f"### Market Breadth Snapshot — {current_date_str}")

# --- Moving Average Breadth ---
st.markdown("#### 📈 Moving Average Breadth")
col1, col2, col3 = st.columns(3)

ma20_above = breadth_df['above_ma20'].sum()
ma20_pct = (ma20_above / len(breadth_df) * 100) if len(breadth_df) > 0 else 0

ma50_above = breadth_df['above_ma50'].sum()
ma50_pct = (ma50_above / len(breadth_df) * 100) if len(breadth_df) > 0 else 0

ma200_above = breadth_df['above_ma200'].sum()
ma200_pct = (ma200_above / len(breadth_df) * 100) if len(breadth_df) > 0 else 0

with col1:
    st.metric("Above MA20", f"{ma20_pct:.1f}%", f"{ma20_above}/{len(breadth_df)}")
with col2:
    st.metric("Above MA50", f"{ma50_pct:.1f}%", f"{ma50_above}/{len(breadth_df)}")
with col3:
    st.metric("Above MA200", f"{ma200_pct:.1f}%", f"{ma200_above}/{len(breadth_df)}")

# MA Breadth chart
fig_ma = go.Figure()
fig_ma.add_trace(go.Bar(
    x=['MA20', 'MA50', 'MA200'],
    y=[ma20_pct, ma50_pct, ma200_pct],
    marker_color=['#66bb6a' if x >= 50 else '#ff5252' for x in [ma20_pct, ma50_pct, ma200_pct]],
    text=[f"{x:.1f}%" for x in [ma20_pct, ma50_pct, ma200_pct]],
    textposition='auto'
))
fig_ma.update_layout(
    title="% of Stocks Above Moving Averages",
    yaxis_title="Percentage (%)",
    height=400,
    showlegend=False
)
fig_ma.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
st.plotly_chart(fig_ma, use_container_width=True)

# --- RSI Breadth ---
st.markdown("#### 🔄 RSI Breadth")

rsi_df = breadth_df[breadth_df['rsi'].notna()].copy()
rsi_above_50 = (rsi_df['rsi'] > 50).sum()
rsi_below_50 = (rsi_df['rsi'] <= 50).sum()
rsi_oversold = (rsi_df['rsi'] < 30).sum()
rsi_overbought = (rsi_df['rsi'] > 70).sum()

total_rsi = len(rsi_df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("RSI > 50", f"{(rsi_above_50/total_rsi*100):.1f}%", f"{rsi_above_50}/{total_rsi}")
with col2:
    st.metric("RSI ≤ 50", f"{(rsi_below_50/total_rsi*100):.1f}%", f"{rsi_below_50}/{total_rsi}")
with col3:
    st.metric("Oversold (<30)", f"{(rsi_oversold/total_rsi*100):.1f}%", f"{rsi_oversold}/{total_rsi}")
with col4:
    st.metric("Overbought (>70)", f"{(rsi_overbought/total_rsi*100):.1f}%", f"{rsi_overbought}/{total_rsi}")

# RSI distribution histogram
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Histogram(
    x=rsi_df['rsi'],
    nbinsx=20,
    marker_color='#1f77b4'
))
fig_rsi.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
fig_rsi.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="Neutral (50)")
fig_rsi.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
fig_rsi.update_layout(
    title="RSI Distribution Across All Stocks",
    xaxis_title="RSI Value",
    yaxis_title="Number of Stocks",
    height=400
)
st.plotly_chart(fig_rsi, use_container_width=True)

# --- MACD Stage Breadth ---
st.markdown("#### 📊 MACD Histogram Stage Distribution")

stage_counts = breadth_df['macd_stage'].value_counts()
stage_order = [
    "1. Troughing",
    "2. Confirmed Trough",
    "3. Rising above Zero",
    "4. Peaking",
    "5. Confirmed Peak",
    "6. Falling below Zero"
]

# Prepare data for all stages (including zero counts)
stage_data = []
for stage in stage_order:
    count = stage_counts.get(stage, 0)
    pct = (count / len(breadth_df) * 100) if len(breadth_df) > 0 else 0
    stage_data.append({
        'stage': stage,
        'count': count,
        'percentage': pct
    })

stage_summary_df = pd.DataFrame(stage_data)

# Display metrics
cols = st.columns(6)
for idx, row in stage_summary_df.iterrows():
    with cols[idx]:
        st.metric(row['stage'].split('. ')[1], f"{row['percentage']:.1f}%", f"{row['count']}/{len(breadth_df)}")

# MACD stage chart
stage_colors = {
    "1. Troughing": "#c8e6c9",
    "2. Confirmed Trough": "#39ff14",
    "3. Rising above Zero": "#2e7d32",
    "4. Peaking": "#ffccbc",
    "5. Confirmed Peak": "#ff5252",
    "6. Falling below Zero": "#c62828"
}

fig_macd = go.Figure()
fig_macd.add_trace(go.Bar(
    x=stage_summary_df['stage'],
    y=stage_summary_df['percentage'],
    marker_color=[stage_colors.get(s, '#1f77b4') for s in stage_summary_df['stage']],
    text=[f"{p:.1f}%" for p in stage_summary_df['percentage']],
    textposition='auto'
))
fig_macd.update_layout(
    title="% of Stocks in Each MACD Stage",
    xaxis_title="MACD Stage",
    yaxis_title="Percentage (%)",
    height=400,
    showlegend=False
)
st.plotly_chart(fig_macd, use_container_width=True)

# --- Market Sentiment Summary ---
st.markdown("#### 💡 Market Sentiment Summary")

bullish_signals = 0
bearish_signals = 0

# MA breadth signals
if ma20_pct > 60:
    bullish_signals += 1
elif ma20_pct < 40:
    bearish_signals += 1

if ma50_pct > 60:
    bullish_signals += 1
elif ma50_pct < 40:
    bearish_signals += 1

# RSI signals
if rsi_oversold > rsi_overbought:
    bullish_signals += 1
elif rsi_overbought > rsi_oversold:
    bearish_signals += 1

# MACD signals
bullish_macd = stage_counts.get("2. Confirmed Trough", 0) + stage_counts.get("1. Troughing", 0)
bearish_macd = stage_counts.get("5. Confirmed Peak", 0) + stage_counts.get("4. Peaking", 0)

if bullish_macd > bearish_macd:
    bullish_signals += 1
elif bearish_macd > bullish_macd:
    bearish_signals += 1

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Bullish Signals", bullish_signals, help="Number of bullish market breadth indicators")
with col2:
    st.metric("Bearish Signals", bearish_signals, help="Number of bearish market breadth indicators")
with col3:
    if bullish_signals > bearish_signals:
        sentiment = "🟢 Bullish"
        delta = f"+{bullish_signals - bearish_signals}"
    elif bearish_signals > bullish_signals:
        sentiment = "🔴 Bearish"
        delta = f"-{bearish_signals - bullish_signals}"
    else:
        sentiment = "⚪ Neutral"
        delta = "0"
    st.metric("Overall Sentiment", sentiment, delta)

# --- Download Data ---
st.markdown("---")
st.markdown("### 📥 Download Data")

# Prepare summary CSV
summary_data = {
    'Metric': [
        'Above MA20 (%)', 'Above MA50 (%)', 'Above MA200 (%)',
        'RSI > 50 (%)', 'RSI <= 50 (%)', 'RSI Oversold (%)', 'RSI Overbought (%)'
    ],
    'Value': [
        f"{ma20_pct:.1f}",
        f"{ma50_pct:.1f}",
        f"{ma200_pct:.1f}",
        f"{(rsi_above_50/total_rsi*100):.1f}",
        f"{(rsi_below_50/total_rsi*100):.1f}",
        f"{(rsi_oversold/total_rsi*100):.1f}",
        f"{(rsi_overbought/total_rsi*100):.1f}"
    ],
    'Count': [
        f"{ma20_above}/{len(breadth_df)}",
        f"{ma50_above}/{len(breadth_df)}",
        f"{ma200_above}/{len(breadth_df)}",
        f"{rsi_above_50}/{total_rsi}",
        f"{rsi_below_50}/{total_rsi}",
        f"{rsi_oversold}/{total_rsi}",
        f"{rsi_overbought}/{total_rsi}"
    ]
}

for _, row in stage_summary_df.iterrows():
    summary_data['Metric'].append(f"MACD: {row['stage']}")
    summary_data['Value'].append(f"{row['percentage']:.1f}")
    summary_data['Count'].append(f"{row['count']}/{len(breadth_df)}")

summary_df = pd.DataFrame(summary_data)

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Download Summary CSV",
        summary_df.to_csv(index=False).encode('utf-8'),
        f"market_breadth_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )

with col2:
    st.download_button(
        "Download Detailed Data CSV",
        breadth_df.to_csv(index=False).encode('utf-8'),
        f"market_breadth_detailed_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )

# Database functions for historical breadth data
def create_breadth_history_table(db_path=DB_PATH):
    """Create table for storing historical market breadth data."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_breadth_history (
            date TEXT PRIMARY KEY,
            ma20_above INTEGER,
            ma20_pct REAL,
            ma50_above INTEGER,
            ma50_pct REAL,
            ma200_above INTEGER,
            ma200_pct REAL,
            rsi_above_50 INTEGER,
            rsi_below_50 INTEGER,
            rsi_oversold INTEGER,
            rsi_overbought INTEGER,
            macd_troughing INTEGER,
            macd_confirmed_trough INTEGER,
            macd_rising INTEGER,
            macd_peaking INTEGER,
            macd_confirmed_peak INTEGER,
            macd_falling INTEGER,
            total_tickers INTEGER,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_breadth_snapshot(date_str, breadth_data, db_path=DB_PATH):
    """Save a daily breadth snapshot to database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
        INSERT OR REPLACE INTO market_breadth_history 
        (date, ma20_above, ma20_pct, ma50_above, ma50_pct, ma200_above, ma200_pct,
         rsi_above_50, rsi_below_50, rsi_oversold, rsi_overbought,
         macd_troughing, macd_confirmed_trough, macd_rising, macd_peaking, 
         macd_confirmed_peak, macd_falling, total_tickers)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        date_str,
        breadth_data['ma20_above'], breadth_data['ma20_pct'],
        breadth_data['ma50_above'], breadth_data['ma50_pct'],
        breadth_data['ma200_above'], breadth_data['ma200_pct'],
        breadth_data['rsi_above_50'], breadth_data['rsi_below_50'],
        breadth_data['rsi_oversold'], breadth_data['rsi_overbought'],
        breadth_data['macd_troughing'], breadth_data['macd_confirmed_trough'],
        breadth_data['macd_rising'], breadth_data['macd_peaking'],
        breadth_data['macd_confirmed_peak'], breadth_data['macd_falling'],
        breadth_data['total_tickers']
    ))
    
    conn.commit()
    conn.close()

def load_breadth_history(days=548, db_path=DB_PATH):
    """Load historical breadth data from database."""
    conn = sqlite3.connect(db_path)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    query = """
        SELECT * FROM market_breadth_history 
        WHERE date >= ? 
        ORDER BY date
    """
    
    df = pd.read_sql_query(query, conn, params=(start_date.strftime("%Y-%m-%d"),))
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert numeric columns to proper types
        numeric_cols = [
            'ma20_above', 'ma20_pct', 'ma50_above', 'ma50_pct', 
            'ma200_above', 'ma200_pct', 'rsi_above_50', 'rsi_below_50',
            'rsi_oversold', 'rsi_overbought', 'macd_troughing',
            'macd_confirmed_trough', 'macd_rising', 'macd_peaking',
            'macd_confirmed_peak', 'macd_falling', 'total_tickers'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                # Convert integers
                if col.endswith('_above') or col in ['rsi_above_50', 'rsi_below_50', 'rsi_oversold', 'rsi_overbought', 'total_tickers'] or col.startswith('macd_'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                # Convert floats
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
    
    return df

def calculate_daily_breadth(date_str, all_tickers, lookback=20, db_path=DB_PATH):
    """Calculate breadth metrics for a specific date."""
    # Set date range: need 250+ days before target date for MA200 calculation
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    calc_start = target_date - timedelta(days=300)  # Extra buffer for MA200
    calc_end = target_date
    
    breadth_df, _, _ = calculate_breadth_metrics(
        all_tickers, 
        calc_start.strftime("%Y-%m-%d"), 
        calc_end.strftime("%Y-%m-%d"), 
        lookback, 
        debug=False
    )
    
    if breadth_df.empty:
        return None
    
    # Calculate all metrics
    total = len(breadth_df)
    
    ma20_above = int(breadth_df['above_ma20'].sum())
    ma50_above = int(breadth_df['above_ma50'].sum())
    ma200_above = int(breadth_df['above_ma200'].sum())
    
    rsi_df = breadth_df[breadth_df['rsi'].notna()]
    total_rsi = len(rsi_df) if len(rsi_df) > 0 else 1  # Avoid division by zero
    rsi_above_50 = int((rsi_df['rsi'] > 50).sum())
    rsi_below_50 = int((rsi_df['rsi'] <= 50).sum())
    rsi_oversold = int((rsi_df['rsi'] < 30).sum())
    rsi_overbought = int((rsi_df['rsi'] > 70).sum())
    
    stage_counts = breadth_df['macd_stage'].value_counts()
    
    return {
        'ma20_above': ma20_above,
        'ma20_pct': (ma20_above / total * 100) if total > 0 else 0,
        'ma50_above': ma50_above,
        'ma50_pct': (ma50_above / total * 100) if total > 0 else 0,
        'ma200_above': ma200_above,
        'ma200_pct': (ma200_above / total * 100) if total > 0 else 0,
        'rsi_above_50': rsi_above_50,
        'rsi_below_50': rsi_below_50,
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,
        'macd_troughing': int(stage_counts.get("1. Troughing", 0)),
        'macd_confirmed_trough': int(stage_counts.get("2. Confirmed Trough", 0)),
        'macd_rising': int(stage_counts.get("3. Rising above Zero", 0)),
        'macd_peaking': int(stage_counts.get("4. Peaking", 0)),
        'macd_confirmed_peak': int(stage_counts.get("5. Confirmed Peak", 0)),
        'macd_falling': int(stage_counts.get("6. Falling below Zero", 0)),
        'total_tickers': total
    }

# Initialize database table
create_breadth_history_table()

# Save today's snapshot
if not breadth_df.empty:
    today_data = {
        'ma20_above': ma20_above,
        'ma20_pct': ma20_pct,
        'ma50_above': ma50_above,
        'ma50_pct': ma50_pct,
        'ma200_above': ma200_above,
        'ma200_pct': ma200_pct,
        'rsi_above_50': rsi_above_50,
        'rsi_below_50': rsi_below_50,
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,
        'macd_troughing': int(stage_counts.get("1. Troughing", 0)),
        'macd_confirmed_trough': int(stage_counts.get("2. Confirmed Trough", 0)),
        'macd_rising': int(stage_counts.get("3. Rising above Zero", 0)),
        'macd_peaking': int(stage_counts.get("4. Peaking", 0)),
        'macd_confirmed_peak': int(stage_counts.get("5. Confirmed Peak", 0)),
        'macd_falling': int(stage_counts.get("6. Falling below Zero", 0)),
        'total_tickers': len(breadth_df)
    }
    save_breadth_snapshot(current_date_str, today_data)

# --- Historical Breadth Charts -----------------------------------------------
if show_historical:
    st.markdown("---")
    st.markdown("### 📈 Historical Market Breadth")
    
    # Check if recalculation requested
    if st.session_state.get('recalculate_breadth', False):
        with st.spinner(f"Recalculating {historical_days} days of historical breadth data..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calculate for each trading day going backwards from today
            end_date_calc = datetime.now().date()
            successful = 0
            failed = 0
            
            for days_ago in range(historical_days):
                target_date = end_date_calc - timedelta(days=days_ago)
                date_str = target_date.strftime("%Y-%m-%d")
                
                # Skip weekends (rough check - doesn't account for holidays)
                if target_date.weekday() >= 5:  # Saturday=5, Sunday=6
                    continue
                
                status_text.text(f"Processing {date_str} ({days_ago + 1}/{historical_days})...")
                
                try:
                    breadth_data = calculate_daily_breadth(date_str, all_tickers, lookback, DB_PATH)
                    if breadth_data:
                        save_breadth_snapshot(date_str, breadth_data, DB_PATH)
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    if debug:
                        st.error(f"Error calculating {date_str}: {e}")
                
                progress_bar.progress((days_ago + 1) / historical_days)
            
            status_text.empty()
            progress_bar.empty()
            st.success(f"✓ Recalculated {historical_days} days: {successful} successful, {failed} failed")
            
            if failed > 0:
                st.warning(f"⚠️ {failed} days could not be calculated (likely weekends or insufficient data)")
        
        st.session_state['recalculate_breadth'] = False
        st.rerun()
    
    # Load historical data
    hist_df = load_breadth_history(days=historical_days, db_path=DB_PATH)
    
    if hist_df.empty:
        st.warning("📭 No historical data available. Click '🔄 Recalculate Historical Data' button in the sidebar to generate it.")
        st.info(f"""
        **Note:** This will calculate breadth metrics for the past {historical_days} days.
        - Each day requires data from 300 days prior for MA200 calculation
        - Estimated time: {historical_days * 0.5:.0f} seconds (~{historical_days * 0.5 / 60:.1f} minutes)
        - Weekend days will be skipped automatically
        """)
    else:
        st.success(f"✅ Loaded {len(hist_df)} days of historical data (from {hist_df['date'].min().date()} to {hist_df['date'].max().date()})")
        
        # Show data quality metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", len(hist_df))
        with col2:
            avg_tickers = hist_df['total_tickers'].mean()
            st.metric("Avg Tickers/Day", f"{avg_tickers:.0f}")
        with col3:
            date_range_days = (hist_df['date'].max() - hist_df['date'].min()).days
            st.metric("Date Range", f"{date_range_days} days")
        
        # --- VNINDEX Technical Analysis with Market Context ---
        st.markdown("---")
        st.markdown("### 📊 VNINDEX Technical Analysis with Market Breadth Context")
        
        st.info("💡 **Tip:** Hover over any date on the charts below to see a vertical line across all subplots with synchronized data.")
        
        # Calculate bullish/bearish percentages FIRST (before merging)
        hist_df['macd_bullish_pct'] = ((hist_df['macd_troughing'].astype(int) + hist_df['macd_confirmed_trough'].astype(int) + hist_df['macd_rising'].astype(int)) / hist_df['total_tickers'].astype(int) * 100).fillna(0)
        hist_df['macd_bearish_pct'] = ((hist_df['macd_peaking'].astype(int) + hist_df['macd_confirmed_peak'].astype(int) + hist_df['macd_falling'].astype(int)) / hist_df['total_tickers'].astype(int) * 100).fillna(0)
        
        # Load VNINDEX data - USE SAME DATE RANGE AS HISTORICAL BREADTH DATA
        vnindex_start = hist_df['date'].min().date() - timedelta(days=100)  # Extra days for EMA50 calculation
        vnindex_end = hist_df['date'].max().date()
        vnindex_df = load_price_range('VNINDEX', vnindex_start, vnindex_end)
        
        if vnindex_df.empty:
            st.warning("⚠️ VNINDEX data not available. Please ensure VNINDEX is in your database.")
        else:
            # Calculate technical indicators
            vnindex_df = vnindex_df.sort_values('date').copy()
            vnindex_df['close'] = pd.to_numeric(vnindex_df['close'], errors='coerce')
            
            # EMAs
            vnindex_df['ema10'] = vnindex_df['close'].ewm(span=10, adjust=False).mean()
            vnindex_df['ema20'] = vnindex_df['close'].ewm(span=20, adjust=False).mean()
            vnindex_df['ema50'] = vnindex_df['close'].ewm(span=50, adjust=False).mean()
            
            # Bollinger Bands
            vnindex_df['bb_middle'] = vnindex_df['close'].rolling(20).mean()
            bb_std = vnindex_df['close'].rolling(20).std()
            vnindex_df['bb_upper'] = vnindex_df['bb_middle'] + (bb_std * 2)
            vnindex_df['bb_lower'] = vnindex_df['bb_middle'] - (bb_std * 2)
            
            # RSI
            delta = vnindex_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            vnindex_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            macd_line, macd_signal, macd_hist_vals = macd_hist(vnindex_df['close'])
            vnindex_df['macd_line'] = macd_line
            vnindex_df['macd_signal'] = macd_signal
            vnindex_df['macd_hist'] = macd_hist_vals
            
            # Merge with historical breadth data to identify peak/bottom regions
            vnindex_merged = vnindex_df.merge(
                hist_df[['date', 'ma20_pct', 'rsi_oversold', 'rsi_overbought', 'total_tickers', 'macd_bullish_pct', 'macd_bearish_pct']],
                on='date',
                how='left'
            )
            
            # Filter merged data to match breadth history date range
            vnindex_merged = vnindex_merged[
                (vnindex_merged['date'] >= hist_df['date'].min()) &
                (vnindex_merged['date'] <= hist_df['date'].max())
            ].reset_index(drop=True)  # IMPORTANT: Reset index to make it continuous
            
            # Define peak/bottom regions based on breadth indicators
            # Bottom region: MA20 < 30% OR RSI oversold > 20% OR MACD bullish > 60%
            vnindex_merged['is_bottom'] = (
                (vnindex_merged['ma20_pct'] < 30) |
                ((vnindex_merged['rsi_oversold'] / vnindex_merged['total_tickers'] * 100) > 20) |
                (vnindex_merged['macd_bullish_pct'] > 60)
            )
            
            # Peak region: MA20 > 70% OR RSI overbought > 20% OR MACD bearish > 60%
            vnindex_merged['is_peak'] = (
                (vnindex_merged['ma20_pct'] > 70) |
                ((vnindex_merged['rsi_overbought'] / vnindex_merged['total_tickers'] * 100) > 20) |
                (vnindex_merged['macd_bearish_pct'] > 60)
            )
            
            # Create figure with subplots
            fig_vnindex = make_subplots(
                rows=4, cols=1,
                row_heights=[0.4, 0.2, 0.2, 0.2],
                subplot_titles=('VNINDEX Price with EMAs & Bollinger Bands', 'Volume', 'RSI (14)', 'MACD Histogram'),
                vertical_spacing=0.05,
                specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            # Add shaded regions for peaks and bottoms to all subplots
            for row in range(1, 5):
                # Add bottom regions (green shade) - find contiguous regions
                bottom_dates = vnindex_merged[vnindex_merged['is_bottom']]['date']
                if len(bottom_dates) > 0:
                    # Group contiguous dates
                    regions = []
                    start_date = bottom_dates.iloc[0]
                    prev_idx = vnindex_merged[vnindex_merged['date'] == start_date].index[0]
                    
                    for i, date in enumerate(bottom_dates):
                        curr_idx = vnindex_merged[vnindex_merged['date'] == date].index[0]
                        # Check if this is a new region (gap in indices)
                        if i > 0 and curr_idx - prev_idx > 1:
                            # End previous region
                            regions.append((start_date, bottom_dates.iloc[i-1]))
                            start_date = date
                        prev_idx = curr_idx
                    
                    # Add last region
                    regions.append((start_date, bottom_dates.iloc[-1]))
                    
                    # Draw rectangles for each region
                    for start, end in regions:
                        fig_vnindex.add_vrect(
                            x0=start, x1=end,
                            fillcolor="green", opacity=0.1, layer="below", line_width=0,
                            row=row, col=1
                        )
                
                # Add peak regions (red shade) - find contiguous regions
                peak_dates = vnindex_merged[vnindex_merged['is_peak']]['date']
                if len(peak_dates) > 0:
                    # Group contiguous dates
                    regions = []
                    start_date = peak_dates.iloc[0]
                    prev_idx = vnindex_merged[vnindex_merged['date'] == start_date].index[0]
                    
                    for i, date in enumerate(peak_dates):
                        curr_idx = vnindex_merged[vnindex_merged['date'] == date].index[0]
                        # Check if this is a new region (gap in indices)
                        if i > 0 and curr_idx - prev_idx > 1:
                            # End previous region
                            regions.append((start_date, peak_dates.iloc[i-1]))
                            start_date = date
                        prev_idx = curr_idx
                    
                    # Add last region
                    regions.append((start_date, peak_dates.iloc[-1]))
                    
                    # Draw rectangles for each region
                    for start, end in regions:
                        fig_vnindex.add_vrect(
                            x0=start, x1=end,
                            fillcolor="red", opacity=0.1, layer="below", line_width=0,
                            row=row, col=1
                        )
            
            # Row 1: Price with EMAs and Bollinger Bands
            fig_vnindex.add_trace(go.Scatter(
                x=vnindex_merged['date'], y=vnindex_merged['close'],
                name='VNINDEX', line=dict(color='#1f77b4', width=2)
            ), row=1, col=1)
            
            fig_vnindex.add_trace(go.Scatter(
                x=vnindex_merged['date'], y=vnindex_merged['ema10'],
                name='EMA10', line=dict(color='#ff7f0e', width=1, dash='dot')
            ), row=1, col=1)
            
            fig_vnindex.add_trace(go.Scatter(
                x=vnindex_merged['date'], y=vnindex_merged['ema20'],
                name='EMA20', line=dict(color='#2ca02c', width=1, dash='dash')
            ), row=1, col=1)
            
            fig_vnindex.add_trace(go.Scatter(
                x=vnindex_merged['date'], y=vnindex_merged['ema50'],
                name='EMA50', line=dict(color='#d62728', width=1.5)
            ), row=1, col=1)
            
            # Bollinger Bands
            fig_vnindex.add_trace(go.Scatter(
                x=vnindex_merged['date'], y=vnindex_merged['bb_upper'],
                name='BB Upper', line=dict(color='gray', width=1, dash='dot'),
                showlegend=False
            ), row=1, col=1)
            
            fig_vnindex.add_trace(go.Scatter(
                x=vnindex_merged['date'], y=vnindex_merged['bb_lower'],
                name='BB Lower', line=dict(color='gray', width=1, dash='dot'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ), row=1, col=1)
            
            # Row 2: Volume
            fig_vnindex.add_trace(go.Bar(
                x=vnindex_merged['date'], y=vnindex_merged['volume'],
                name='Volume', marker_color='#9467bd', showlegend=False
            ), row=2, col=1)
            
            # Row 3: RSI
            fig_vnindex.add_trace(go.Scatter(
                x=vnindex_merged['date'], y=vnindex_merged['rsi'],
                name='RSI', line=dict(color='#8c564b', width=2),
                showlegend=False
            ), row=3, col=1)
            
            fig_vnindex.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Overbought (70)")
            fig_vnindex.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Oversold (30)")
            fig_vnindex.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
            
            # Row 4: MACD Histogram
            colors = ['#2ca02c' if v >= 0 else '#d62728' for v in vnindex_merged['macd_hist']]
            fig_vnindex.add_trace(go.Bar(
                x=vnindex_merged['date'], y=vnindex_merged['macd_hist'],
                name='MACD Hist', marker_color=colors,
                showlegend=False
            ), row=4, col=1)
            
            fig_vnindex.add_trace(go.Scatter(
                x=vnindex_merged['date'], y=vnindex_merged['macd_line'],
                name='MACD Line', line=dict(color='blue', width=1.5),
                showlegend=False
            ), row=4, col=1)
            
            fig_vnindex.add_trace(go.Scatter(
                x=vnindex_merged['date'], y=vnindex_merged['macd_signal'],
                name='Signal', line=dict(color='orange', width=1.5),
                showlegend=False
            ), row=4, col=1)
            
            fig_vnindex.add_hline(y=0, line_dash="solid", line_color="black", row=4, col=1)
            
            # Update layout with synchronized hover
            fig_vnindex.update_layout(
                height=1000,
                hovermode='x unified',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                title_text="VNINDEX Technical Analysis with Market Breadth Context",
                xaxis=dict(matches='x'),
                xaxis2=dict(matches='x'),
                xaxis3=dict(matches='x'),
                xaxis4=dict(matches='x')
            )
            
            # Add prominent vertical line on hover for all x-axes
            for i in range(1, 5):
                fig_vnindex.update_xaxes(
                    showspikes=True,
                    spikemode='across',
                    spikesnap='cursor',
                    spikecolor='rgba(255,0,0,0.5)',  # More visible red color
                    spikethickness=2,  # Thicker line
                    spikedash='solid',
                    row=i, col=1
                )
            
            st.plotly_chart(fig_vnindex, use_container_width=True)
            
            # Add interpretation guide
            with st.expander("📖 How to Read This Chart"):
                st.markdown("""
                **Shaded Regions:**
                - 🟢 **Green shading**: Market bottom conditions detected
                  - MA20 breadth < 30% OR
                  - High RSI oversold (>20% of stocks) OR
                  - High MACD bullish (>60% bullish stages)
                
                - 🔴 **Red shading**: Market peak conditions detected
                  - MA20 breadth > 70% OR
                  - High RSI overbought (>20% of stocks) OR
                  - High MACD bearish (>60% bearish stages)
                
                **Technical Indicators:**
                - **EMAs (10/20/50)**: Trend direction and support/resistance levels
                - **Bollinger Bands**: Volatility and potential reversal zones
                - **RSI**: Overbought (>70) / Oversold (<30) momentum
                - **MACD Histogram**: Trend strength and momentum shifts
                
                **Trading Signals:**
                - Look for bullish setups in green (bottom) regions
                - Consider taking profits or caution in red (peak) regions
                - Confirm signals across multiple timeframes and breadth indicators
                """)
        
        # --- Moving Average Breadth History ---
        st.markdown("#### 📊 Moving Average Breadth (Historical)")
        
        fig_ma_hist = go.Figure()
        fig_ma_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=hist_df['ma20_pct'],
            name='MA20', line=dict(color='#2196F3', width=2)
        ))
        fig_ma_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=hist_df['ma50_pct'],
            name='MA50', line=dict(color='#FF9800', width=2)
        ))
        fig_ma_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=hist_df['ma200_pct'],
            name='MA200', line=dict(color='#9C27B0', width=2)
        ))
        
        fig_ma_hist.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
        fig_ma_hist.add_hline(y=70, line_dash="dot", line_color="green", annotation_text="Bullish (70%)")
        fig_ma_hist.add_hline(y=30, line_dash="dot", line_color="red", annotation_text="Bearish (30%)")
        
        fig_ma_hist.update_layout(
            title="% of Stocks Above Moving Averages (Time Series)",
            xaxis_title="Date",
            yaxis_title="Percentage (%)",
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Add prominent vertical line on hover
        fig_ma_hist.update_xaxes(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='rgba(255,0,0,0.5)',  # More visible red
            spikethickness=2,  # Thicker
            spikedash='solid'
        )
        
        st.plotly_chart(fig_ma_hist, use_container_width=True)
        
        # --- RSI Breadth History ---
        st.markdown("#### 🔄 RSI Breadth (Historical)")
        
        fig_rsi_hist = make_subplots(
            rows=2, cols=1, 
            subplot_titles=("RSI Above/Below 50", "RSI Oversold/Overbought"),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        # RSI above/below 50 - ensure we're working with numeric Series
        total_rsi_series = hist_df['rsi_above_50'].astype(int) + hist_df['rsi_below_50'].astype(int)
        # Avoid division by zero
        total_rsi_series = total_rsi_series.replace(0, 1)
        
        rsi_above_50_pct = (hist_df['rsi_above_50'].astype(int) / total_rsi_series * 100).fillna(0)
        rsi_below_50_pct = (hist_df['rsi_below_50'].astype(int) / total_rsi_series * 100).fillna(0)
        
        fig_rsi_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=rsi_above_50_pct,
            name='RSI > 50', fill='tozeroy', line=dict(color='#4CAF50')
        ), row=1, col=1)
        fig_rsi_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=rsi_below_50_pct,
            name='RSI ≤ 50', fill='tozeroy', line=dict(color='#F44336')
        ), row=1, col=1)
        
        # RSI oversold/overbought
        rsi_oversold_pct = (hist_df['rsi_oversold'].astype(int) / total_rsi_series * 100).fillna(0)
        rsi_overbought_pct = (hist_df['rsi_overbought'].astype(int) / total_rsi_series * 100).fillna(0)
        
        fig_rsi_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=rsi_oversold_pct,
            name='Oversold (<30)', line=dict(color='#00BCD4', width=2)
        ), row=2, col=1)
        fig_rsi_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=rsi_overbought_pct,
            name='Overbought (>70)', line=dict(color='#FF5722', width=2)
        ), row=2, col=1)
        
        fig_rsi_hist.update_xaxes(title_text="Date", row=2, col=1)
        fig_rsi_hist.update_yaxes(title_text="Percentage (%)", row=1, col=1)
        fig_rsi_hist.update_yaxes(title_text="Percentage (%)", row=2, col=1)
        
        fig_rsi_hist.update_layout(
            height=700,
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(matches='x'),
            xaxis2=dict(matches='x')
        )
        
        # Add prominent vertical line on hover for both subplots
        for row_num in [1, 2]:
            fig_rsi_hist.update_xaxes(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor='rgba(255,0,0,0.5)',  # More visible red
                spikethickness=2,  # Thicker
                spikedash='solid',
                row=row_num, col=1
            )
        
        st.plotly_chart(fig_rsi_hist, use_container_width=True)
        
        # --- MACD Stage Distribution History ---
        st.markdown("#### 📊 MACD Stage Distribution (Historical)")
        
        fig_macd_hist = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Bullish vs Bearish MACD Stages", "Detailed MACD Stage Breakdown"),
            vertical_spacing=0.15,
            row_heights=[0.4, 0.6]
        )
        
        # Bullish vs Bearish
        fig_macd_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=hist_df['macd_bullish_pct'],
            name='Bullish Stages', fill='tozeroy', line=dict(color='#4CAF50', width=2)
        ), row=1, col=1)
        fig_macd_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=hist_df['macd_bearish_pct'],
            name='Bearish Stages', fill='tozeroy', line=dict(color='#F44336', width=2)
        ), row=1, col=1)
        
        # Detailed breakdown
        fig_macd_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=(hist_df['macd_troughing'] / hist_df['total_tickers'] * 100),
            name='Troughing', line=dict(color='#c8e6c9', width=1.5)
        ), row=2, col=1)
        fig_macd_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=(hist_df['macd_confirmed_trough'] / hist_df['total_tickers'] * 100),
            name='Confirmed Trough', line=dict(color='#39ff14', width=2)
        ), row=2, col=1)
        fig_macd_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=(hist_df['macd_rising'] / hist_df['total_tickers'] * 100),
            name='Rising', line=dict(color='#2e7d32', width=1.5)
        ), row=2, col=1)
        fig_macd_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=(hist_df['macd_peaking'] / hist_df['total_tickers'] * 100),
            name='Peaking', line=dict(color='#ffccbc', width=1.5)
        ), row=2, col=1)
        fig_macd_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=(hist_df['macd_confirmed_peak'] / hist_df['total_tickers'] * 100),
            name='Confirmed Peak', line=dict(color='#ff5252', width=2)
        ), row=2, col=1)
        fig_macd_hist.add_trace(go.Scatter(
            x=hist_df['date'], y=(hist_df['macd_falling'] / hist_df['total_tickers'] * 100),
            name='Falling', line=dict(color='#c62828', width=1.5)
        ), row=2, col=1)
        
        fig_macd_hist.update_xaxes(title_text="Date", row=2, col=1)
        fig_macd_hist.update_yaxes(title_text="Percentage (%)", row=1, col=1)
        fig_macd_hist.update_yaxes(title_text="Percentage (%)", row=2, col=1)
        
        fig_macd_hist.update_layout(
            height=800,
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(matches='x'),
            xaxis2=dict(matches='x')
        )
        
        # Add prominent vertical line on hover for both subplots
        for row_num in [1, 2]:
            fig_macd_hist.update_xaxes(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor='rgba(255,0,0,0.5)',  # More visible red
                spikethickness=2,  # Thicker
                spikedash='solid',
                row=row_num, col=1
            )
        
        st.plotly_chart(fig_macd_hist, use_container_width=True)
        
        # --- Market Sentiment Summary (Historical) ---
        st.markdown("#### 💡 Market Sentiment Summary (Historical)")
        
        # Calculate bullish and bearish signals
        hist_df['bullish_signals'] = 0
        hist_df['bearish_signals'] = 0
        
        # MA breadth signals
        hist_df.loc[hist_df['ma20_pct'] > 60, 'bullish_signals'] += 1
        hist_df.loc[hist_df['ma20_pct'] < 40, 'bearish_signals'] += 1
        
        hist_df.loc[hist_df['ma50_pct'] > 60, 'bullish_signals'] += 1
        hist_df.loc[hist_df['ma50_pct'] < 40, 'bearish_signals'] += 1
        
        # RSI signals
        hist_df.loc[hist_df['rsi_oversold'] > hist_df['rsi_overbought'], 'bullish_signals'] += 1
        hist_df.loc[hist_df['rsi_overbought'] > hist_df['rsi_oversold'], 'bearish_signals'] += 1
        
        # MACD signals
        bullish_macd_hist = hist_df['macd_confirmed_trough'] + hist_df['macd_troughing']
        bearish_macd_hist = hist_df['macd_confirmed_peak'] + hist_df['macd_peaking']
        
        hist_df.loc[bullish_macd_hist > bearish_macd_hist, 'bullish_signals'] += 1
        hist_df.loc[bearish_macd_hist > bullish_macd_hist, 'bearish_signals'] += 1
        
        # Calculate total signals
        total_signals = hist_df['bullish_signals'] + hist_df['bearish_signals']
        hist_df['bullish_signals_pct'] = (hist_df['bullish_signals'] / total_signals * 100).fillna(0)
        hist_df['bearish_signals_pct'] = (hist_df['bearish_signals'] / total_signals * 100).fillna(0)
        
        # Display latest sentiment summary
        latest_sentiment = hist_df.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bullish Signals", int(latest_sentiment['bullish_signals']), help="Number of bullish market breadth indicators")
        with col2:
            st.metric("Bearish Signals", int(latest_sentiment['bearish_signals']), help="Number of bearish market breadth indicators")
        with col3:
            if latest_sentiment['bullish_signals'] > latest_sentiment['bearish_signals']:
                sentiment = "🟢 Bullish"
                delta = f"+{latest_sentiment['bullish_signals'] - latest_sentiment['bearish_signals']}"
            elif latest_sentiment['bearish_signals'] > latest_sentiment['bullish_signals']:
                sentiment = "🔴 Bearish"
                delta = f"-{latest_sentiment['bearish_signals'] - latest_sentiment['bullish_signals']}"
            else:
                sentiment = "⚪ Neutral"
                delta = "0"
            st.metric("Overall Sentiment", sentiment, delta)
        
        # --- Download Historical Data ---
        st.markdown("---")
        st.markdown("### 📥 Download Historical Data")
        
        # Prepare historical summary CSV
        hist_summary_data = {
            'Date': hist_df['date'],
            'Bullish Signals': hist_df['bullish_signals'],
            'Bearish Signals': hist_df['bearish_signals'],
            'Overall Sentiment': ["Bullish" if bs > sbs else "Bearish" if bs < sbs else "Neutral" for bs, sbs in zip(hist_df['bullish_signals'], hist_df['bearish_signals'])]
        }
        
        hist_summary_df = pd.DataFrame(hist_summary_data)
        
        st.download_button(
            "Download Historical Sentiment Summary CSV",
            hist_summary_df.to_csv(index=False).encode('utf-8'),
            f"market_breadth_historical_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
