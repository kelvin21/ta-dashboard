"""
MACD stage detection and categorization.
Re-uses logic from ta_dashboard.py for consistency.
"""
import pandas as pd
import numpy as np


def detect_macd_stage(hist: pd.Series, lookback: int = 20) -> str:
    """
    Detect MACD histogram stage for the latest bar.
    
    Returns one of six stages:
    - "1. Troughing"
    - "2. Confirmed Trough"
    - "3. Rising above Zero"
    - "4. Peaking"
    - "5. Confirmed Peak"
    - "6. Falling below Zero"
    - "N/A" (if not enough data)
    
    Args:
        hist: MACD histogram series
        lookback: Number of bars to look back for peak/trough detection
    
    Returns:
        String representing the stage
    """
    s = hist.dropna().reset_index(drop=True)
    if s.empty or len(s) < 3:
        return "N/A"
    
    last = float(s.iat[-1])
    prev = float(s.iat[-2])
    
    # Check for zero crossings
    cross_up = (prev < 0 and last >= 0)
    cross_down = (prev > 0 and last <= 0)
    
    if cross_up:
        return "2. Confirmed Trough"
    if cross_down:
        return "5. Confirmed Peak"
    
    # Find last crossing point
    last_cross_idx = len(s) - 1
    for i in range(len(s)-2, max(0, len(s)-lookback-1), -1):
        if (s[i] < 0 and s[i+1] >= 0) or (s[i] > 0 and s[i+1] <= 0):
            last_cross_idx = i + 1
            break
    
    window_start = max(0, last_cross_idx)
    window = s.iloc[window_start:]
    
    if last < 0:
        # Below zero: check for troughing or falling
        if len(window) >= 3:
            min_idx_in_window = int(window.idxmin())
            min_pos = min_idx_in_window - window_start
            last_pos = len(window) - 1
            
            if min_pos < last_pos:
                return "1. Troughing"
        return "6. Falling below Zero"
    else:
        # Above zero: check for peaking or rising
        if len(window) >= 3:
            max_idx_in_window = int(window.idxmax())
            max_pos = max_idx_in_window - window_start
            last_pos = len(window) - 1
            
            if max_pos < last_pos:
                return "4. Peaking"
        return "3. Rising above Zero"


def categorize_macd_stage(stage: str) -> str:
    """
    Categorize MACD stage into simplified buckets.
    
    Args:
        stage: Stage string from detect_macd_stage
    
    Returns:
        Category: 'troughing', 'confirmed_trough', 'rising', 'peaking', 'confirmed_peak', 'declining', or 'N/A'
    """
    if "N/A" in stage:
        return "N/A"
    elif "Troughing" in stage:
        return "troughing"
    elif "Confirmed Trough" in stage:
        return "confirmed_trough"
    elif "Rising" in stage:
        return "rising"
    elif "Peaking" in stage:
        return "peaking"
    elif "Confirmed Peak" in stage:
        return "confirmed_peak"
    elif "Falling" in stage:
        return "declining"
    else:
        return "N/A"


def macd_stage_score(stage: str) -> int:
    """
    Convert MACD stage to numeric score for sorting/comparison.
    
    Args:
        stage: Stage string
    
    Returns:
        Integer score: +3 (most bullish) to -3 (most bearish)
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


def get_macd_stage_display_name(stage: str) -> str:
    """
    Get display name for MACD stage (removes numeric prefix if present).
    
    Args:
        stage: Stage string
    
    Returns:
        Display name without numeric prefix
    """
    # Remove numeric prefix (e.g., "1. " or "2. ")
    if stage and len(stage) > 3 and stage[0].isdigit() and stage[1] == '.' and stage[2] == ' ':
        return stage[3:]
    return stage


def get_all_macd_stages() -> list:
    """
    Get list of all possible MACD stages in order.
    
    Returns:
        List of stage strings
    """
    return [
        "1. Troughing",
        "2. Confirmed Trough",
        "3. Rising above Zero",
        "4. Peaking",
        "5. Confirmed Peak",
        "6. Falling below Zero"
    ]


def get_macd_stage_color(stage: str) -> str:
    """
    Get color code for MACD stage visualization.
    
    Args:
        stage: Stage string
    
    Returns:
        Hex color code
    """
    if "Confirmed Trough" in stage:
        return "#00ff00"  # Bright green
    elif "Troughing" in stage:
        return "#90EE90"  # Light green
    elif "Rising" in stage:
        return "#87CEEB"  # Sky blue
    elif "Peaking" in stage:
        return "#FFB6C1"  # Light pink
    elif "Confirmed Peak" in stage:
        return "#ff0000"  # Red
    elif "Falling" in stage:
        return "#FFA07A"  # Light salmon
    else:
        return "#808080"  # Gray
