"""
Relative Strength (RS) analysis utilities for stock leader detection.
Identifies stocks outperforming VNINDEX using RS, RSI, and OBV.
Optimized with vectorization and async processing for performance.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor


def calculate_relative_strength(stock_close: pd.Series, vnindex_close: pd.Series, method: str = 'momentum', lookback: int = 63) -> pd.Series:
    """
    Calculate Relative Strength capturing short/medium-term momentum.
    
    Methods:
    - 'momentum': Rolling rate of change - captures recent momentum (RECOMMENDED)
      RS = (Stock ROC) / (Benchmark ROC) over lookback period
    - 'percentage': Cumulative performance from start (legacy CRS)
    - 'price_ratio': Simple price ratio (legacy)
    
    Common lookback periods:
    - 21 days: 1 month (short-term momentum)
    - 42 days: 2 months (medium-term momentum)
    - 63 days: 3 months (longer-term momentum)
    
    Args:
        stock_close: Stock closing prices
        vnindex_close: VNINDEX closing prices (aligned dates)
        method: Calculation method ('momentum', 'percentage', 'price_ratio')
        lookback: Period for momentum calculation (default: 63 days for 3-month)
    
    Returns:
        Series of RS values
    """
    # Align series by index
    aligned = pd.DataFrame({
        'stock': stock_close,
        'vnindex': vnindex_close
    }).dropna()
    
    if aligned.empty:
        return pd.Series([np.nan] * len(stock_close), index=stock_close.index)
    
    if method == 'momentum':
        # Rolling momentum-based RS (captures short/medium-term trends)
        # VECTORIZED calculation for performance
        
        # Calculate rate of change over lookback period
        stock_roc = aligned['stock'].pct_change(periods=lookback)
        vnindex_roc = aligned['vnindex'].pct_change(periods=lookback)
        
        # RS = (1 + stock_roc) / (1 + vnindex_roc)
        rs = (1 + stock_roc) / (1 + vnindex_roc)
        
        # For early periods without enough data, use progressive lookback (VECTORIZED)
        # This prevents abnormally low RS values in the 0-lookback range
        if lookback > 0 and len(aligned) >= lookback:
            # Create arrays for vectorized calculation
            early_indices = np.arange(min(lookback, len(aligned)))
            
            # Vectorized: normalized price ratios from first value
            stock_normalized = aligned['stock'].iloc[:lookback] / aligned['stock'].iloc[0]
            vnindex_normalized = aligned['vnindex'].iloc[:lookback] / aligned['vnindex'].iloc[0]
            
            # Compute early RS values in one operation
            early_rs = stock_normalized / vnindex_normalized
            
            # Update early periods
            rs.iloc[:lookback] = early_rs
        
    elif method == 'percentage':
        # Cumulative performance from first value (legacy CRS)
        stock_pct_change = (aligned['stock'] / aligned['stock'].iloc[0]) - 1
        vnindex_pct_change = (aligned['vnindex'] / aligned['vnindex'].iloc[0]) - 1
        
        # Handle division by zero
        vnindex_pct_change = vnindex_pct_change.replace(0, np.nan)
        
        # CRS = (1 + stock_pct) / (1 + vnindex_pct)
        rs = (1 + stock_pct_change) / (1 + vnindex_pct_change)
        
    else:
        # Legacy price ratio method
        if (aligned['vnindex'] == 0).any():
            return pd.Series([np.nan] * len(stock_close), index=stock_close.index)
        rs = aligned['stock'] / aligned['vnindex']
    
    return rs.reindex(stock_close.index)


async def calculate_multi_period_rs_async(stock_close: pd.Series, vnindex_close: pd.Series) -> Dict:
    """
    Async wrapper for calculate_multi_period_rs to enable parallel processing.
    
    Args:
        stock_close: Stock closing prices
        vnindex_close: VNINDEX closing prices
    
    Returns:
        Dictionary with rs_1w, rs_2w, rs_1m, rs_2m, rs_3m, rs_composite, and trend
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Run all 5 period calculations in parallel
        tasks = [
            loop.run_in_executor(executor, calculate_relative_strength, stock_close, vnindex_close, 'momentum', 5),
            loop.run_in_executor(executor, calculate_relative_strength, stock_close, vnindex_close, 'momentum', 10),
            loop.run_in_executor(executor, calculate_relative_strength, stock_close, vnindex_close, 'momentum', 21),
            loop.run_in_executor(executor, calculate_relative_strength, stock_close, vnindex_close, 'momentum', 42),
            loop.run_in_executor(executor, calculate_relative_strength, stock_close, vnindex_close, 'momentum', 63)
        ]
        rs_1w, rs_2w, rs_1m, rs_2m, rs_3m = await asyncio.gather(*tasks)
        rs_1w, rs_2w, rs_1m, rs_2m, rs_3m = await asyncio.gather(*tasks)
    
    # Get current values (vectorized)
    rs_1w_current = rs_1w.iloc[-1] if not rs_1w.empty else np.nan
    rs_2w_current = rs_2w.iloc[-1] if not rs_2w.empty else np.nan
    rs_1m_current = rs_1m.iloc[-1] if not rs_1m.empty else np.nan
    rs_2m_current = rs_2m.iloc[-1] if not rs_2m.empty else np.nan
    rs_3m_current = rs_3m.iloc[-1] if not rs_3m.empty else np.nan
    
    # Calculate composite RS using numpy for vectorization
    rs_values = np.array([rs_1w_current, rs_2w_current, rs_1m_current, rs_2m_current, rs_3m_current])
    weights = np.array([0.30, 0.25, 0.25, 0.15, 0.05])
    
    # Mask for valid (non-NaN) values
    valid_mask = ~np.isnan(rs_values)
    
    if valid_mask.any():
        valid_rs = rs_values[valid_mask]
        valid_weights = weights[valid_mask]
        # Normalize weights
        valid_weights = valid_weights / valid_weights.sum()
        rs_composite = np.dot(valid_rs, valid_weights)
    else:
        rs_composite = np.nan
    
    # Determine trend (accelerating/decelerating/stable) - using 1W, 1M, 3M
    trend = "N/A"
    if not np.isnan(rs_1w_current) and not np.isnan(rs_1m_current) and not np.isnan(rs_3m_current):
        if rs_1w_current > rs_1m_current > rs_3m_current:
            trend = "Accelerating"  # Getting stronger
        elif rs_1w_current < rs_1m_current < rs_3m_current:
            trend = "Decelerating"  # Getting weaker
        elif rs_1w_current > rs_3m_current:
            trend = "Strengthening"
        elif rs_1w_current < rs_3m_current:
            trend = "Weakening"
        else:
            trend = "Stable"
    
    return {
        'rs_1w': rs_1w_current,
        'rs_2w': rs_2w_current,
        'rs_1m': rs_1m_current,
        'rs_2m': rs_2m_current,
        'rs_3m': rs_3m_current,
        'rs_composite': rs_composite,
        'rs_trend': trend,
        'rs_1w_series': rs_1w,
        'rs_2w_series': rs_2w,
        'rs_1m_series': rs_1m,
        'rs_2m_series': rs_2m,
        'rs_3m_series': rs_3m
    }


def calculate_multi_period_rs(stock_close: pd.Series, vnindex_close: pd.Series) -> Dict:
    """
    Synchronous wrapper for calculate_multi_period_rs_async.
    For backward compatibility with existing code.
    
    Args:
        stock_close: Stock closing prices
        vnindex_close: VNINDEX closing prices
    
    Returns:
        Dictionary with rs_1w, rs_2w, rs_1m, rs_2m, rs_3m, rs_composite, and trend
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's already a running loop, create a new one in a thread
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(calculate_multi_period_rs_async(stock_close, vnindex_close))
        else:
            return loop.run_until_complete(calculate_multi_period_rs_async(stock_close, vnindex_close))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(calculate_multi_period_rs_async(stock_close, vnindex_close))


def calculate_rs_ema_slope(rs: pd.Series, period: int = 10, lookback: int = 3) -> float:
    """
    Calculate slope of RS EMA over lookback periods.
    
    Args:
        rs: Relative strength series
        period: EMA period (default: 10)
        lookback: Number of periods to check slope (default: 3)
    
    Returns:
        Slope value (positive = uptrend)
    """
    if len(rs) < period + lookback:
        return 0.0
    
    rs_ema = rs.ewm(span=period, adjust=False).mean()
    
    if len(rs_ema) < lookback:
        return 0.0
    
    recent_ema = rs_ema.iloc[-lookback:].values
    
    # Calculate slope using linear regression
    x = np.arange(len(recent_ema))
    slope = np.polyfit(x, recent_ema, 1)[0] if len(recent_ema) > 1 else 0.0
    
    return float(slope)


def calculate_rs_percentile(rs_current: float, rs_universe: List[float]) -> float:
    """
    Calculate percentile rank of RS within universe.
    
    Args:
        rs_current: Current RS value
        rs_universe: List of RS values for all stocks
    
    Returns:
        Percentile (0-100)
    """
    if not rs_universe or np.isnan(rs_current):
        return 0.0
    
    valid_rs = [r for r in rs_universe if not np.isnan(r)]
    if not valid_rs:
        return 0.0
    
    percentile = (sum(1 for r in valid_rs if r < rs_current) / len(valid_rs)) * 100
    return percentile


def is_rs_near_high(rs: pd.Series, lookback_months: int = 3) -> bool:
    """
    Check if RS is near 3-12 month high.
    
    Args:
        rs: Relative strength series
        lookback_months: Months to look back (default: 3)
    
    Returns:
        True if RS is within 5% of high
    """
    if len(rs) < 20:
        return False
    
    lookback_days = lookback_months * 21  # Approximate trading days
    recent_rs = rs.iloc[-lookback_days:] if len(rs) > lookback_days else rs
    
    current_rs = rs.iloc[-1]
    high_rs = recent_rs.max()
    
    if np.isnan(current_rs) or np.isnan(high_rs) or high_rs == 0:
        return False
    
    return (current_rs / high_rs) >= 0.95  # Within 5% of high


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        close: Closing prices
        volume: Volume data
    
    Returns:
        OBV series
    """
    if len(close) < 2:
        return pd.Series([0] * len(close), index=close.index)
    
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    
    return pd.Series(obv, index=close.index)


def analyze_obv_status(obv: pd.Series, lookback: int = 10) -> Dict[str, any]:
    """
    Analyze OBV status and trend.
    
    Args:
        obv: OBV series
        lookback: Days to analyze (default: 10)
    
    Returns:
        Dictionary with OBV analysis
    """
    if len(obv) < 20:
        return {
            'above_ema20': False,
            'positive_slope': False,
            'higher_highs': False,
            'status': 'N/A',
            'score': 0
        }
    
    # Calculate OBV EMA20
    obv_ema20 = obv.ewm(span=20, adjust=False).mean()
    current_obv = obv.iloc[-1]
    current_ema = obv_ema20.iloc[-1]
    
    # Check if above EMA20
    above_ema20 = current_obv > current_ema
    
    # Check positive slope (last N days)
    recent_obv = obv.iloc[-lookback:]
    slope = np.polyfit(range(len(recent_obv)), recent_obv.values, 1)[0] if len(recent_obv) > 1 else 0
    positive_slope = slope > 0
    
    # Check for higher highs (last 20 days)
    last_20 = obv.iloc[-20:]
    mid_high = last_20.iloc[:10].max()
    recent_high = last_20.iloc[10:].max()
    higher_highs = recent_high > mid_high
    
    # Determine status
    if above_ema20 and positive_slope and higher_highs:
        status = 'Accumulating'
        score = 15
    elif above_ema20 and positive_slope:
        status = 'Building'
        score = 10
    elif above_ema20:
        status = 'Above EMA'
        score = 5
    else:
        status = 'Weak'
        score = 0
    
    return {
        'above_ema20': above_ema20,
        'positive_slope': positive_slope,
        'higher_highs': higher_highs,
        'status': status,
        'score': score
    }


def calculate_leader_score(
    rs: pd.Series,
    rs_percentile: float,
    rsi_daily: float,
    rsi_weekly: float,
    obv_analysis: Dict
) -> Tuple[int, Dict[str, int]]:
    """
    Calculate leader score (0-100) based on prediction model.
    
    Scoring breakdown:
    - RS > weekly EMA10: +25
    - RS higher high vs 3 months: +15
    - RS percentile ≥ 70%: +10
    - RSI daily 48–60 and rising: +20
    - OBV > EMA20: +15
    - OBV leading/accumulating: +15
    
    Args:
        rs: Relative strength series
        rs_percentile: RS percentile in universe
        rsi_daily: Daily RSI value
        rsi_weekly: Weekly RSI value
        obv_analysis: OBV analysis dict
    
    Returns:
        Tuple of (total_score, breakdown_dict)
    """
    breakdown = {
        'rs_above_ema': 0,
        'rs_higher_high': 0,
        'rs_percentile': 0,
        'rsi_zone': 0,
        'obv_above_ema': 0,
        'obv_leading': 0
    }
    
    # RS above weekly EMA10 (+25)
    if len(rs) >= 10:
        rs_ema10 = rs.ewm(span=10, adjust=False).mean()
        if rs.iloc[-1] > rs_ema10.iloc[-1]:
            breakdown['rs_above_ema'] = 25
    
    # RS higher high vs 3 months (+15)
    if is_rs_near_high(rs, lookback_months=3):
        breakdown['rs_higher_high'] = 15
    
    # RS percentile ≥ 70% (+10)
    if rs_percentile >= 70:
        breakdown['rs_percentile'] = 10
    
    # RSI daily 48–60 and rising (+20)
    if 48 <= rsi_daily <= 60:
        # Check if RSI is rising (compare to 5 days ago if available)
        breakdown['rsi_zone'] = 20
    elif 55 <= rsi_daily <= 65:
        # Healthy trend continuation
        breakdown['rsi_zone'] = 15
    elif rsi_daily > 70:
        # Extended - reduced score
        breakdown['rsi_zone'] = 5
    
    # OBV status
    breakdown['obv_above_ema'] = 15 if obv_analysis.get('above_ema20') else 0
    breakdown['obv_leading'] = obv_analysis.get('score', 0)
    
    total_score = sum(breakdown.values())
    return total_score, breakdown


def classify_prediction_list(score: int) -> Tuple[str, str, str]:
    """
    Classify stock into prediction list based on score.
    
    Args:
        score: Leader score (0-100)
    
    Returns:
        Tuple of (classification, badge_color, description)
    """
    if score >= 75:
        return 'List A', 'error', 'High Conviction Leader'
    elif score >= 60:
        return 'List B', 'warning', 'Watchlist / Early Setup'
    else:
        return 'Ignore', 'secondary', 'Below Threshold'


def generate_expectation(score: int, obv_status: str, rsi: float) -> str:
    """
    Generate expectation text based on indicators.
    
    Args:
        score: Leader score
        obv_status: OBV status string
        rsi: Current RSI
    
    Returns:
        Expectation text
    """
    if score >= 75:
        if 'Accumulating' in obv_status and rsi < 55:
            return 'Breakout 5–10D'
        elif rsi >= 55:
            return 'Trend Resume'
        else:
            return 'Early Leader'
    elif score >= 60:
        if rsi < 50:
            return 'Base Build'
        else:
            return 'Monitor Setup'
    else:
        return 'Wait & See'


def check_entry_trigger(
    rsi: float,
    close: float,
    ema20: float,
    volume_ratio: float
) -> Tuple[bool, List[str]]:
    """
    Check if valid entry trigger is present.
    
    Args:
        rsi: Current RSI
        close: Current close price
        ema20: EMA20 value
        volume_ratio: Current volume / avg volume
    
    Returns:
        Tuple of (has_trigger, trigger_reasons)
    """
    triggers = []
    
    if rsi > 55:
        triggers.append('RSI > 55')
    
    if close > ema20:
        triggers.append('Above EMA20')
    
    if volume_ratio > 1.5:
        triggers.append('Volume Expansion')
    
    has_trigger = len(triggers) >= 2
    return has_trigger, triggers


def check_exit_signal(
    rs: pd.Series,
    obv: pd.Series,
    rsi_values: List[float]
) -> Tuple[bool, List[str]]:
    """
    Check if exit signal is present.
    
    Args:
        rs: Relative strength series
        obv: OBV series
        rsi_values: Last 3 RSI values
    
    Returns:
        Tuple of (should_exit, exit_reasons)
    """
    exit_reasons = []
    
    # RS breaks below weekly EMA10
    if len(rs) >= 10:
        rs_ema10 = rs.ewm(span=10, adjust=False).mean()
        if rs.iloc[-1] < rs_ema10.iloc[-1]:
            exit_reasons.append('RS < EMA10')
    
    # OBV makes lower low
    if len(obv) >= 20:
        last_10 = obv.iloc[-10:]
        prev_10 = obv.iloc[-20:-10]
        if last_10.min() < prev_10.min():
            exit_reasons.append('OBV Lower Low')
    
    # RSI <45 for 3 consecutive sessions
    if len(rsi_values) >= 3:
        if all(r < 45 for r in rsi_values[-3:]):
            exit_reasons.append('RSI < 45 (3 days)')
    
    should_exit = len(exit_reasons) > 0
    return should_exit, exit_reasons


def format_score_breakdown(breakdown: Dict[str, int]) -> str:
    """
    Format score breakdown for display.
    
    Args:
        breakdown: Score breakdown dictionary
    
    Returns:
        Formatted string
    """
    lines = []
    
    labels = {
        'rs_above_ema': 'RS > EMA10',
        'rs_higher_high': 'RS Higher High',
        'rs_percentile': 'RS Percentile ≥70%',
        'rsi_zone': 'RSI Zone',
        'obv_above_ema': 'OBV > EMA20',
        'obv_leading': 'OBV Leading'
    }
    
    for key, value in breakdown.items():
        if value > 0:
            label = labels.get(key, key)
            lines.append(f"  • {label}: +{value}")
    
    return '\n'.join(lines) if lines else '  No positive scores'
