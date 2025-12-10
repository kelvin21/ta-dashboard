"""
MACD Analysis Utilities
Functions for MACD histogram stage detection and classification.
"""
import pandas as pd
import numpy as np


def detect_macd_stage(hist: pd.Series, lookback: int = 20) -> str:
    """
    Detect MACD histogram stage for the latest bar.
    
    Returns one of six stages with numeric prefix for sorting:
    - "1. Troughing" - Below zero, turning up from bottom
    - "2. Confirmed Trough" - Just crossed above zero (bullish)
    - "3. Rising above Zero" - Above zero, continuing to rise
    - "4. Peaking" - Above zero, turning down from peak
    - "5. Confirmed Peak" - Just crossed below zero (bearish)
    - "6. Falling below Zero" - Below zero, continuing to fall
    
    Args:
        hist: MACD histogram Series
        lookback: Number of bars to look back for trend detection
    
    Returns:
        String stage name with numeric prefix, or "N/A" if insufficient data
    """
    s = hist.dropna().reset_index(drop=True)
    if s.empty or len(s) < 3:
        return "N/A"
    
    last = float(s.iat[-1])
    prev = float(s.iat[-2])
    
    # Check for zero line crosses
    cross_up = (prev < 0 and last >= 0)
    cross_down = (prev > 0 and last <= 0)
    
    if cross_up:
        return "2. Confirmed Trough"
    if cross_down:
        return "5. Confirmed Peak"
    
    # Find last cross
    last_cross_idx = len(s) - 1
    for i in range(len(s) - 2, max(0, len(s) - lookback - 1), -1):
        if (s[i] < 0 and s[i + 1] >= 0) or (s[i] > 0 and s[i + 1] <= 0):
            last_cross_idx = i + 1
            break
    
    window_start = max(0, last_cross_idx)
    window = s.iloc[window_start:]
    
    # Below zero
    if last < 0:
        if len(window) >= 3:
            min_idx_in_window = int(window.idxmin())
            min_val = float(window.min())
            if min_idx_in_window < len(s) - 1:
                recent_vals = s.iloc[min_idx_in_window:]
                if len(recent_vals) >= 2:
                    slope = (np.polyfit(range(len(recent_vals)), recent_vals.values, 1)[0] 
                            if len(recent_vals) > 1 else (last - min_val))
                    if slope > 0:
                        return "1. Troughing"
        return "6. Falling below Zero"
    
    # Above zero
    else:
        if len(window) >= 3:
            max_idx_in_window = int(window.idxmax())
            max_val = float(window.max())
            if max_idx_in_window < len(s) - 1:
                recent_vals = s.iloc[max_idx_in_window:]
                if len(recent_vals) >= 2:
                    slope = (np.polyfit(range(len(recent_vals)), recent_vals.values, 1)[0] 
                            if len(recent_vals) > 1 else (last - max_val))
                    if slope < 0:
                        return "4. Peaking"
        return "3. Rising above Zero"


def get_simple_macd_stage(hist_current: float, hist_prev: float) -> str:
    """
    Get simplified MACD stage based on current and previous histogram values.
    
    Returns lowercase stage names compatible with momentum detection:
    - "rising" - Histogram increasing
    - "peaking" - Histogram decreasing from positive
    - "declining" - Histogram decreasing
    - "troughing" - Histogram increasing from negative
    - "neutral" - No clear direction
    
    Args:
        hist_current: Current MACD histogram value
        hist_prev: Previous MACD histogram value
    
    Returns:
        Simple stage name (lowercase)
    """
    if pd.isna(hist_current) or pd.isna(hist_prev):
        return "neutral"
    
    is_rising = hist_current > hist_prev
    is_falling = hist_current < hist_prev
    is_positive = hist_current > 0
    
    if is_rising:
        return "troughing" if not is_positive else "rising"
    elif is_falling:
        return "peaking" if is_positive else "declining"
    else:
        return "neutral"


def calculate_macd_stage_from_series(hist_series: pd.Series, lookback: int = 20) -> str:
    """
    Calculate MACD stage from a histogram series and return simplified stage name.
    
    Args:
        hist_series: MACD histogram Series
        lookback: Lookback period for stage detection
    
    Returns:
        Simplified stage name (lowercase) for momentum detection
    """
    full_stage = detect_macd_stage(hist_series, lookback)
    
    # Map full stage names to simplified names
    stage_map = {
        "1. Troughing": "troughing",
        "2. Confirmed Trough": "rising",
        "3. Rising above Zero": "rising",
        "4. Peaking": "peaking",
        "5. Confirmed Peak": "declining",
        "6. Falling below Zero": "declining",
        "N/A": "neutral"
    }
    
    return stage_map.get(full_stage, "neutral")


def stage_score(stage: str) -> int:
    """
    Convert stage name to numeric score for sorting/comparison.
    
    Scores:
    - Confirmed Trough: +3 (most bullish)
    - Troughing: +2
    - Rising above Zero: +1
    - Falling below Zero: -1
    - Peaking: -2
    - Confirmed Peak: -3 (most bearish)
    
    Args:
        stage: Stage name (with or without numeric prefix)
    
    Returns:
        Numeric score
    """
    if "Confirmed Trough" in stage:
        return 3
    if "Troughing" in stage:
        return 2
    if "Rising above Zero" in stage:
        return 1
    if "Peaking" in stage:
        return -2
    if "Confirmed Peak" in stage:
        return -3
    if "Falling below Zero" in stage:
        return -1
    return 0
