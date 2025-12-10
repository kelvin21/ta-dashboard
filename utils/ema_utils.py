"""
EMA Signal Utilities
Helper functions for EMA analysis, scoring, and zone detection.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go


def calculate_ema_alignment(row: pd.Series, ema_periods: List[int] = [10, 20, 50, 100, 150, 200]) -> str:
    """
    Calculate EMA alignment status.
    
    Args:
        row: DataFrame row with close and EMA columns
        ema_periods: List of EMA periods to check
    
    Returns:
        Alignment status: 'bullish', 'bearish', 'mixed', 'neutral'
    """
    close = row.get('close', np.nan)
    if pd.isna(close):
        return 'neutral'
    
    # Get available EMAs
    emas = []
    for period in ema_periods:
        col = f'ema{period}'
        if col in row.index and pd.notna(row[col]):
            emas.append((period, row[col]))
    
    if len(emas) < 2:
        return 'neutral'
    
    # Sort by period (shortest to longest)
    emas.sort(key=lambda x: x[0])
    
    # Check if EMAs are in proper order
    bullish_count = 0
    bearish_count = 0
    
    # Price vs EMAs
    above_emas = sum(1 for _, ema_val in emas if close > ema_val)
    below_emas = len(emas) - above_emas
    
    # EMA order (shorter > longer is bullish)
    for i in range(len(emas) - 1):
        if emas[i][1] > emas[i+1][1]:
            bullish_count += 1
        else:
            bearish_count += 1
    
    # Combine price position and EMA order
    if above_emas >= len(emas) * 0.7 and bullish_count >= bearish_count:
        return 'bullish'
    elif below_emas >= len(emas) * 0.7 and bearish_count >= bullish_count:
        return 'bearish'
    elif abs(above_emas - below_emas) <= 1:
        return 'mixed'
    else:
        return 'neutral'


def calculate_ema_strength_score(row: pd.Series, ema_periods: List[int] = [10, 20, 50, 100, 150, 200]) -> int:
    """
    Calculate EMA strength score (1-5).
    
    Args:
        row: DataFrame row with close and EMA columns
        ema_periods: List of EMA periods to check
    
    Returns:
        Strength score 1-5 (5 = strongest bullish, 1 = strongest bearish)
    """
    close = row.get('close', np.nan)
    if pd.isna(close):
        return 3  # Neutral
    
    # Get available EMAs
    emas = []
    for period in ema_periods:
        col = f'ema{period}'
        if col in row.index and pd.notna(row[col]):
            emas.append((period, row[col]))
    
    if len(emas) < 2:
        return 3  # Neutral
    
    # Calculate components
    score = 0
    max_score = 0
    
    # 1. Price position relative to EMAs
    above_emas = sum(1 for _, ema_val in emas if close > ema_val)
    max_score += len(emas)
    score += above_emas
    
    # 2. EMA order (shorter > longer)
    emas.sort(key=lambda x: x[0])
    bullish_order = 0
    for i in range(len(emas) - 1):
        if emas[i][1] > emas[i+1][1]:
            bullish_order += 1
        max_score += 1
    score += bullish_order
    
    # 3. Distance to EMAs (bonus for being far above or penalty for being far below)
    if len(emas) > 0:
        avg_ema = np.mean([val for _, val in emas])
        distance_pct = ((close - avg_ema) / avg_ema) * 100
        if distance_pct > 5:
            score += 2  # Strong bullish
        elif distance_pct > 2:
            score += 1
        elif distance_pct < -5:
            score -= 2  # Strong bearish
        elif distance_pct < -2:
            score -= 1
        max_score += 2
    
    # Normalize to 1-5 scale
    if max_score > 0:
        normalized = (score / max_score) * 4 + 1
        return int(np.clip(round(normalized), 1, 5))
    
    return 3  # Neutral


def determine_ema_zone(row: pd.Series, ema_periods: List[int] = [10, 20, 50, 100, 150, 200]) -> str:
    """
    Determine trading zone based on EMA position.
    
    Args:
        row: DataFrame row with close and EMA columns
        ema_periods: List of EMA periods to check
    
    Returns:
        Zone: 'buy', 'accumulate', 'distribute', 'sell', 'risk'
    """
    close = row.get('close', np.nan)
    if pd.isna(close):
        return 'neutral'
    
    # Get key EMAs
    ema20 = row.get('ema20', np.nan)
    ema50 = row.get('ema50', np.nan)
    ema100 = row.get('ema100', np.nan)
    ema200 = row.get('ema200', np.nan)
    
    # Buy Zone: Above all major EMAs with bullish alignment
    if all(pd.notna(ema) for ema in [ema20, ema50, ema200]):
        if close > ema20 > ema50 > ema200:
            return 'buy'
    
    # Accumulate Zone: Above EMA200 but mixed shorter EMAs
    if pd.notna(ema200) and close > ema200:
        if pd.notna(ema50):
            if close > ema50:
                return 'accumulate'
    
    # Distribute Zone: Below short EMAs but above long EMAs
    if pd.notna(ema50) and pd.notna(ema200):
        if close < ema50 and close > ema200:
            return 'distribute'
    
    # Sell Zone: Below EMA50 and approaching EMA200
    if pd.notna(ema50) and close < ema50:
        if pd.notna(ema200):
            if close > ema200 * 0.98:  # Within 2% of EMA200
                return 'sell'
    
    # Risk Zone: Below all major EMAs
    if pd.notna(ema200) and close < ema200:
        return 'risk'
    
    return 'neutral'


def calculate_ema_distances(row: pd.Series, ema_periods: List[int] = [10, 20, 50, 100, 150, 200]) -> Dict[str, float]:
    """
    Calculate percentage distances from price to each EMA.
    
    Args:
        row: DataFrame row with close and EMA columns
        ema_periods: List of EMA periods to check
    
    Returns:
        Dictionary of {ema_period: distance_percentage}
    """
    close = row.get('close', np.nan)
    distances = {}
    
    if pd.isna(close):
        return distances
    
    for period in ema_periods:
        col = f'ema{period}'
        if col in row.index and pd.notna(row[col]):
            ema_val = row[col]
            distance_pct = ((close - ema_val) / ema_val) * 100
            distances[f'ema{period}'] = round(distance_pct, 2)
    
    return distances


def calculate_ema_convergence(row: pd.Series, ema_periods: List[int] = [10, 20, 50, 100, 150, 200]) -> float:
    """
    Calculate EMA convergence metric (standard deviation of EMAs).
    Lower value = more converged = potential breakout.
    
    Args:
        row: DataFrame row with close and EMA columns
        ema_periods: List of EMA periods to check
    
    Returns:
        Convergence score (lower = more converged)
    """
    close = row.get('close', np.nan)
    if pd.isna(close):
        return 0.0
    
    emas = []
    for period in ema_periods:
        col = f'ema{period}'
        if col in row.index and pd.notna(row[col]):
            emas.append(row[col])
    
    if len(emas) < 2:
        return 0.0
    
    # Calculate coefficient of variation (normalized std dev)
    mean_ema = np.mean(emas)
    if mean_ema == 0:
        return 0.0
    
    std_ema = np.std(emas)
    convergence = (std_ema / mean_ema) * 100
    
    return round(convergence, 2)


def create_mini_sparkline(df: pd.DataFrame, ticker: str, show_emas: bool = True) -> go.Figure:
    """
    Create a mini sparkline chart for a ticker.
    
    Args:
        df: DataFrame with OHLCV and indicator data
        ticker: Ticker symbol
        show_emas: Whether to show EMA lines
    
    Returns:
        Plotly figure object
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=80,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False
        )
        return fig
    
    fig = go.Figure()
    
    # Take last 30 days
    df_recent = df.tail(30)
    
    # Add close price line
    fig.add_trace(go.Scatter(
        x=df_recent.index if 'date' not in df_recent.columns else df_recent['date'],
        y=df_recent['close'],
        mode='lines',
        name='Close',
        line=dict(color='#1976D2', width=2),
        showlegend=False
    ))
    
    # Add EMA20 and EMA50 if available
    if show_emas:
        if 'ema20' in df_recent.columns:
            fig.add_trace(go.Scatter(
                x=df_recent.index if 'date' not in df_recent.columns else df_recent['date'],
                y=df_recent['ema20'],
                mode='lines',
                name='EMA20',
                line=dict(color='#FF9800', width=1, dash='dot'),
                showlegend=False
            ))
        
        if 'ema50' in df_recent.columns:
            fig.add_trace(go.Scatter(
                x=df_recent.index if 'date' not in df_recent.columns else df_recent['date'],
                y=df_recent['ema50'],
                mode='lines',
                name='EMA50',
                line=dict(color='#4CAF50', width=1, dash='dot'),
                showlegend=False
            ))
    
    # Minimal layout for sparkline
    fig.update_layout(
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def rank_ema_urgency(df_indicators: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Rank tickers by EMA signal urgency for immediate action.
    
    Args:
        df_indicators: DataFrame with all ticker indicators
        top_n: Number of top results to return
    
    Returns:
        DataFrame with ranked tickers and urgency scores
    """
    if df_indicators.empty:
        return pd.DataFrame()
    
    results = []
    
    for _, row in df_indicators.iterrows():
        ticker = row.get('ticker', '')
        if not ticker:
            continue
        
        # Calculate urgency factors
        strength = calculate_ema_strength_score(row)
        zone = determine_ema_zone(row)
        alignment = calculate_ema_alignment(row)
        convergence = calculate_ema_convergence(row)
        
        # Urgency score calculation
        urgency_score = 0
        
        # Zone urgency
        zone_scores = {
            'buy': 5,
            'accumulate': 3,
            'distribute': -3,
            'sell': -4,
            'risk': -5,
            'neutral': 0
        }
        urgency_score += zone_scores.get(zone, 0)
        
        # Strength contribution
        urgency_score += (strength - 3)  # -2 to +2
        
        # Alignment bonus
        if alignment == 'bullish':
            urgency_score += 2
        elif alignment == 'bearish':
            urgency_score -= 2
        
        # Convergence bonus (potential breakout)
        if convergence < 2:  # Very converged
            urgency_score += 1
        
        results.append({
            'ticker': ticker,
            'urgency_score': urgency_score,
            'strength': strength,
            'zone': zone,
            'alignment': alignment,
            'convergence': convergence,
            'close': row.get('close', np.nan)
        })
    
    df_ranked = pd.DataFrame(results)
    if df_ranked.empty:
        return df_ranked
    
    # Sort by urgency score
    df_ranked = df_ranked.sort_values('urgency_score', ascending=False)
    
    return df_ranked.head(top_n)


def get_zone_color(zone: str) -> str:
    """Get color for trading zone."""
    colors = {
        'buy': '#4CAF50',
        'accumulate': '#8BC34A',
        'distribute': '#FF9800',
        'sell': '#FF5722',
        'risk': '#F44336',
        'neutral': '#9E9E9E'
    }
    return colors.get(zone, '#9E9E9E')


def get_alignment_color(alignment: str) -> str:
    """Get color for EMA alignment."""
    colors = {
        'bullish': '#4CAF50',
        'bearish': '#F44336',
        'mixed': '#FF9800',
        'neutral': '#9E9E9E'
    }
    return colors.get(alignment, '#9E9E9E')


def format_distance(distance: float) -> str:
    """Format distance percentage with color indicator."""
    if distance > 0:
        return f"+{distance:.2f}%"
    else:
        return f"{distance:.2f}%"
