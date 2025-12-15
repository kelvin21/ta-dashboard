"""
Technical indicator calculations using TA-Lib and pandas.
Provides EMA, RSI, MACD, and Bollinger Bands calculations.
"""
import pandas as pd
import numpy as np

# Try to import TA-Lib and pandas-ta
HAS_TALIB = False
HAS_PANDAS_TA = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    pass

# Try pandas-ta as fallback
if not HAS_TALIB:
    try:
        import pandas_ta as ta
        HAS_PANDAS_TA = True
        print("✓ Using pandas-ta for indicator calculations")
    except ImportError:
        print("⚠️ Neither TA-Lib nor pandas-ta installed. Using manual pandas calculations.")


def calculate_ema(close_series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        close_series: Series of closing prices
        period: EMA period
    
    Returns:
        Series with EMA values
    """
    if HAS_TALIB:
        return pd.Series(talib.EMA(close_series.values, timeperiod=period), index=close_series.index)
    elif HAS_PANDAS_TA:
        import pandas_ta as ta
        return ta.ema(close_series, length=period)
    else:
        return close_series.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_sma(close_series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        close_series: Series of closing prices
        period: SMA period
    
    Returns:
        Series with SMA values
    """
    if HAS_TALIB:
        return pd.Series(talib.SMA(close_series.values, timeperiod=period), index=close_series.index)
    elif HAS_PANDAS_TA:
        import pandas_ta as ta
        return ta.sma(close_series, length=period)
    else:
        return close_series.rolling(window=period).mean()


def calculate_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI using TA-Lib if available, otherwise pandas-ta or Wilder's smoothing.
    
    Args:
        close_series: Series of closing prices
        period: RSI period (default 14)
    
    Returns:
        Series with RSI values
    """
    if HAS_TALIB:
        return pd.Series(talib.RSI(close_series.values, timeperiod=period), index=close_series.index)
    elif HAS_PANDAS_TA:
        import pandas_ta as ta
        return ta.rsi(close_series, length=period)
    else:
        return _calculate_rsi_manual(close_series, period)


def _calculate_rsi_manual(close_series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI using Wilder's smoothing (fallback if TA-Lib not available).
    Matches AmiBroker's RSI calculation.
    """
    delta = close_series.diff()
    
    # Calculate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Initialize arrays for Wilder's smoothing
    rsi = np.zeros(len(close_series))
    rsi[:] = np.nan
    
    # Calculate initial average gain/loss (simple average of first period)
    if len(close_series) >= period:
        avg_gain = gain.iloc[1:period+1].sum() / period
        avg_loss = loss.iloc[1:period+1].sum() / period
        
        # Set RSI starting from period+1
        for i in range(period, len(close_series)):
            current_gain = gain.iloc[i]
            current_loss = loss.iloc[i]
            
            # Wilder's smoothing
            avg_gain = (avg_gain * (period - 1) + current_gain) / period
            avg_loss = (avg_loss * (period - 1) + current_loss) / period
            
            # Calculate RS and RSI
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
    
    return pd.Series(rsi, index=close_series.index)


def calculate_macd(close_series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Calculate MACD line, signal line, and histogram.
    
    Args:
        close_series: Series of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    if HAS_TALIB:
        macd, signal_line, hist = talib.MACD(
            close_series.values,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal
        )
        return (
            pd.Series(macd, index=close_series.index),
            pd.Series(signal_line, index=close_series.index),
            pd.Series(hist, index=close_series.index)
        )
    elif HAS_PANDAS_TA:
        import pandas_ta as ta
        df_temp = pd.DataFrame({'close': close_series})
        macd_result = ta.macd(df_temp['close'], fast=fast, slow=slow, signal=signal)
        return (
            macd_result[f'MACD_{fast}_{slow}_{signal}'],
            macd_result[f'MACDs_{fast}_{slow}_{signal}'],
            macd_result[f'MACDh_{fast}_{slow}_{signal}']
        )
    else:
        # Calculate EMAs using adjust=False to match AmiBroker
        ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean()
        
        # MACD Line
        macd_line = ema_fast - ema_slow
        
        # Signal Line - EMA of MACD line
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        
        # Histogram
        hist = macd_line - signal_line
        
        return macd_line, signal_line, hist


def calculate_bollinger_bands(close_series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands.
    
    Args:
        close_series: Series of closing prices
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if HAS_TALIB:
        upper, middle, lower = talib.BBANDS(
            close_series.values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0  # SMA
        )
        return (
            pd.Series(upper, index=close_series.index),
            pd.Series(middle, index=close_series.index),
            pd.Series(lower, index=close_series.index)
        )
    elif HAS_PANDAS_TA:
        import pandas_ta as ta
        df_temp = pd.DataFrame({'close': close_series})
        bbands = ta.bbands(df_temp['close'], length=period, std=std_dev)
        
        # pandas-ta uses different column naming (handles float formatting)
        # Try multiple possible column name formats
        if bbands is not None and not bbands.empty:
            # Get column names from the result
            cols = bbands.columns.tolist()
            # Find upper, middle, lower bands by prefix
            upper_col = next((c for c in cols if c.startswith('BBU_')), None)
            middle_col = next((c for c in cols if c.startswith('BBM_')), None)
            lower_col = next((c for c in cols if c.startswith('BBL_')), None)
            
            if upper_col and middle_col and lower_col:
                return (bbands[upper_col], bbands[middle_col], bbands[lower_col])
        
        # Fallback if pandas-ta fails
        middle = close_series.rolling(window=period).mean()
        std = close_series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    else:
        # Calculate SMA
        middle = close_series.rolling(window=period).mean()
        
        # Calculate standard deviation
        std = close_series.rolling(window=period).std()
        
        # Calculate bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower


def calculate_all_indicators(df: pd.DataFrame, ema_periods=None, rsi_period=14, 
                             macd_params=(12, 26, 9), bb_params=(20, 2.0)) -> pd.DataFrame:
    """
    Calculate all technical indicators for a DataFrame with OHLCV data.
    
    Args:
        df: DataFrame with at least 'close' column and 'date' index or column
        ema_periods: List of EMA periods (default [10, 20, 50, 100, 200])
        rsi_period: RSI period (default 14)
        macd_params: MACD parameters (fast, slow, signal) - default (12, 26, 9)
        bb_params: Bollinger Bands parameters (period, std_dev) - default (20, 2.0)
    
    Returns:
        DataFrame with all indicators added as new columns
    """
    if ema_periods is None:
        ema_periods = [10, 20, 50, 100, 200]
    
    df = df.copy()
    
    # Ensure close is numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Calculate EMAs
    for period in ema_periods:
        df[f'ema{period}'] = calculate_ema(df['close'], period)
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    
    # Calculate MACD
    macd_line, signal_line, hist = calculate_macd(df['close'], *macd_params)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = hist
    df['macd_histogram'] = hist  # Alias for compatibility
    
    # Calculate MACD stage (simplified for momentum detection)
    try:
        from .macd_utils import get_simple_macd_stage
        
        # Calculate stage for each row using current and previous histogram
        df['macd_stage'] = 'neutral'
        for i in range(1, len(df)):
            if pd.notna(df.iloc[i]['macd_hist']) and pd.notna(df.iloc[i-1]['macd_hist']):
                df.at[df.index[i], 'macd_stage'] = get_simple_macd_stage(
                    df.iloc[i]['macd_hist'],
                    df.iloc[i-1]['macd_hist']
                )
    except Exception as e:
        # If MACD stage calculation fails, continue without it
        df['macd_stage'] = 'neutral'
    
    # Calculate Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df['close'], *bb_params)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    
    return df


def categorize_rsi(rsi_value: float) -> str:
    """
    Categorize RSI value into buckets.
    
    Args:
        rsi_value: RSI value
    
    Returns:
        String category: 'oversold', '<50', '>50', 'overbought', or 'N/A'
    """
    if pd.isna(rsi_value):
        return 'N/A'
    
    if rsi_value < 30:
        return 'oversold'
    elif rsi_value < 50:
        return '<50'
    elif rsi_value <= 70:
        return '>50'
    else:
        return 'overbought'


def categorize_rsi_vectorized(rsi_series: pd.Series) -> pd.Series:
    """
    Vectorized version: Categorize RSI values into buckets.
    
    Args:
        rsi_series: Series of RSI values
    
    Returns:
        Series of string categories: 'oversold', '<50', '>50', 'overbought', or 'N/A'
    """
    conditions = [
        rsi_series.isna(),
        rsi_series < 30,
        rsi_series < 50,
        rsi_series <= 70
    ]
    choices = ['N/A', 'oversold', '<50', '>50']
    return pd.Series(
        np.select(conditions, choices, default='overbought'),
        index=rsi_series.index
    )


def check_price_above_ema(close: float, ema: float) -> bool:
    """
    Check if price is above EMA.
    
    Args:
        close: Closing price
        ema: EMA value
    
    Returns:
        True if close > ema, False otherwise
    """
    if pd.isna(close) or pd.isna(ema):
        return False
    return close > ema


def check_price_above_ema_vectorized(close_series: pd.Series, ema_series: pd.Series) -> pd.Series:
    """
    Vectorized version: Check if prices are above EMA.
    
    Args:
        close_series: Series of closing prices
        ema_series: Series of EMA values
    
    Returns:
        Boolean series indicating if close > ema (False for NaN values)
    """
    # Handle NaN values - return False where either is NaN
    valid_mask = close_series.notna() & ema_series.notna()
    return (close_series > ema_series) & valid_mask
