"""
Trading Pattern Detection
Identifies classical chart patterns including those that form over 6-18 months.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy.signal import argrelextrema
from scipy.stats import linregress


class PatternDetector:
    """Detect classical trading patterns in price data."""
    
    def __init__(self, min_pattern_days: int = 30, max_pattern_days: int = 540):
        """
        Initialize pattern detector.
        
        Args:
            min_pattern_days: Minimum days for pattern formation (default 30)
            max_pattern_days: Maximum days for pattern formation (default 540 = 18 months)
        """
        self.min_pattern_days = min_pattern_days
        self.max_pattern_days = max_pattern_days
    
    def detect_all_patterns(self, df: pd.DataFrame, ticker: str, include_forming: bool = False) -> List[Dict]:
        """
        Detect all patterns in the given price data.
        
        Args:
            df: DataFrame with OHLCV data and date column
            ticker: Ticker symbol
            include_forming: If True, include patterns still forming (not complete)
        
        Returns:
            List of detected patterns with signals
        """
        if len(df) < self.min_pattern_days:
            return []
        
        patterns = []
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Detect various patterns
        patterns.extend(self._detect_head_shoulders(df, ticker, include_forming))
        patterns.extend(self._detect_double_top_bottom(df, ticker, include_forming))
        patterns.extend(self._detect_triangles(df, ticker, include_forming))
        patterns.extend(self._detect_cup_handle(df, ticker, include_forming))
        patterns.extend(self._detect_flags_pennants(df, ticker, include_forming))
        patterns.extend(self._detect_wedges(df, ticker, include_forming))
        
        return patterns
    
    def _find_peaks_troughs(self, data: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find local peaks and troughs in price data."""
        peaks = argrelextrema(data.values, np.greater, order=order)[0]
        troughs = argrelextrema(data.values, np.less, order=order)[0]
        return peaks, troughs
    
    def _detect_head_shoulders(self, df: pd.DataFrame, ticker: str, include_forming: bool = False) -> List[Dict]:
        """Detect Head and Shoulders (top) and Inverse Head and Shoulders (bottom)."""
        patterns = []
        
        # Look at last 3-18 months of data
        for lookback_days in [90, 180, 365, 540]:
            if len(df) < lookback_days:
                continue
            
            window_df = df.tail(lookback_days).reset_index(drop=True)
            high = window_df['high'].values
            low = window_df['low'].values
            close = window_df['close'].values
            
            # Find peaks and troughs
            peaks, _ = self._find_peaks_troughs(pd.Series(high), order=10)
            _, troughs = self._find_peaks_troughs(pd.Series(low), order=10)
            
            # Head and Shoulders Top (bearish)
            if len(peaks) >= 3:
                for i in range(len(peaks) - 2):
                    left_shoulder = peaks[i]
                    head = peaks[i + 1]
                    right_shoulder = peaks[i + 2]
                    
                    # Check if head is higher than shoulders
                    if (high[head] > high[left_shoulder] and 
                        high[head] > high[right_shoulder] and
                        abs(high[left_shoulder] - high[right_shoulder]) / high[head] < 0.05):
                        
                        # Find neckline (lows between peaks)
                        neckline_lows = low[left_shoulder:right_shoulder+1]
                        neckline = np.min(neckline_lows)
                        current_price = close[-1]
                        
                        # Check if pattern is complete or near completion
                        is_complete = current_price <= neckline * 1.02  # Within 2% of neckline break
                        is_forming = current_price > neckline and current_price < high[right_shoulder] * 1.05
                        
                        if is_complete or (include_forming and is_forming):
                            height = high[head] - neckline
                            target = neckline - height
                            stop_loss = high[right_shoulder] * 1.02
                            
                            pattern_start_date = window_df.iloc[left_shoulder]['date']
                            pattern_days = (window_df.iloc[-1]['date'] - pattern_start_date).days
                            
                            patterns.append({
                                'ticker': ticker,
                                'pattern': 'Head and Shoulders Top',
                                'type': 'bearish',
                                'signal': 'SELL',
                                'status': 'complete' if is_complete else 'forming',
                                'current_price': current_price,
                                'target_price': target,
                                'stop_loss': stop_loss,
                                'confidence': self._calculate_confidence(current_price, neckline, 'bearish') * (1.0 if is_complete else 0.6),
                                'neckline': neckline,
                                'formation_days': pattern_days,
                                'detected_date': df.iloc[-1]['date'],
                                'risk_reward': abs(target - current_price) / abs(stop_loss - current_price),
                                'key_points': {
                                    'left_shoulder': {'index': left_shoulder, 'price': high[left_shoulder], 'date': window_df.iloc[left_shoulder]['date']},
                                    'head': {'index': head, 'price': high[head], 'date': window_df.iloc[head]['date']},
                                    'right_shoulder': {'index': right_shoulder, 'price': high[right_shoulder], 'date': window_df.iloc[right_shoulder]['date']}
                                }
                            })
            
            # Inverse Head and Shoulders (bullish)
            if len(troughs) >= 3:
                for i in range(len(troughs) - 2):
                    left_shoulder = troughs[i]
                    head = troughs[i + 1]
                    right_shoulder = troughs[i + 2]
                    
                    # Check if head is lower than shoulders
                    if (low[head] < low[left_shoulder] and 
                        low[head] < low[right_shoulder] and
                        abs(low[left_shoulder] - low[right_shoulder]) / low[head] < 0.05):
                        
                        # Find neckline (highs between troughs)
                        neckline_highs = high[left_shoulder:right_shoulder+1]
                        neckline = np.max(neckline_highs)
                        current_price = close[-1]
                        
                        # Check if pattern is complete or near completion
                        is_complete = current_price >= neckline * 0.98  # Within 2% of neckline break
                        is_forming = current_price < neckline and current_price > low[right_shoulder] * 0.95
                        
                        if is_complete or (include_forming and is_forming):
                            height = neckline - low[head]
                            target = neckline + height
                            stop_loss = low[right_shoulder] * 0.98
                            
                            pattern_start_date = window_df.iloc[left_shoulder]['date']
                            pattern_days = (window_df.iloc[-1]['date'] - pattern_start_date).days
                            
                            patterns.append({
                                'ticker': ticker,
                                'pattern': 'Inverse Head and Shoulders',
                                'type': 'bullish',
                                'signal': 'BUY',
                                'status': 'complete' if is_complete else 'forming',
                                'current_price': current_price,
                                'target_price': target,
                                'stop_loss': stop_loss,
                                'confidence': self._calculate_confidence(current_price, neckline, 'bullish') * (1.0 if is_complete else 0.6),
                                'neckline': neckline,
                                'formation_days': pattern_days,
                                'detected_date': df.iloc[-1]['date'],
                                'risk_reward': abs(target - current_price) / abs(current_price - stop_loss),
                                'key_points': {
                                    'left_shoulder': {'index': left_shoulder, 'price': low[left_shoulder], 'date': window_df.iloc[left_shoulder]['date']},
                                    'head': {'index': head, 'price': low[head], 'date': window_df.iloc[head]['date']},
                                    'right_shoulder': {'index': right_shoulder, 'price': low[right_shoulder], 'date': window_df.iloc[right_shoulder]['date']}
                                }
                            })
        
        return patterns
    
    def _detect_double_top_bottom(self, df: pd.DataFrame, ticker: str, include_forming: bool = False) -> List[Dict]:
        """Detect Double Top (bearish) and Double Bottom (bullish) patterns."""
        patterns = []
        
        for lookback_days in [60, 120, 250, 365]:
            if len(df) < lookback_days:
                continue
            
            window_df = df.tail(lookback_days).reset_index(drop=True)
            high = window_df['high'].values
            low = window_df['low'].values
            close = window_df['close'].values
            
            peaks, _ = self._find_peaks_troughs(pd.Series(high), order=8)
            _, troughs = self._find_peaks_troughs(pd.Series(low), order=8)
            
            # Double Top
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    peak1, peak2 = peaks[i], peaks[i + 1]
                    
                    # Check if peaks are similar height (within 2%)
                    if abs(high[peak1] - high[peak2]) / high[peak1] < 0.02:
                        # Find the trough between peaks
                        if peak2 - peak1 > 10:  # At least 10 days between peaks
                            trough_val = np.min(low[peak1:peak2+1])
                            current_price = close[-1]
                            
                            if current_price <= trough_val * 1.02:
                                height = (high[peak1] + high[peak2]) / 2 - trough_val
                                target = trough_val - height
                                stop_loss = max(high[peak1], high[peak2]) * 1.02
                                
                                pattern_start_date = window_df.iloc[peak1]['date']
                                pattern_days = (window_df.iloc[-1]['date'] - pattern_start_date).days
                                
                                patterns.append({
                                    'ticker': ticker,
                                    'pattern': 'Double Top',
                                    'type': 'bearish',
                                    'signal': 'SELL',
                                    'current_price': current_price,
                                    'target_price': target,
                                    'stop_loss': stop_loss,
                                    'confidence': self._calculate_confidence(current_price, trough_val, 'bearish'),
                                    'neckline': trough_val,
                                    'formation_days': pattern_days,
                                    'detected_date': df.iloc[-1]['date'],
                                    'risk_reward': abs(target - current_price) / abs(stop_loss - current_price)
                                })
            
            # Double Bottom
            if len(troughs) >= 2:
                for i in range(len(troughs) - 1):
                    trough1, trough2 = troughs[i], troughs[i + 1]
                    
                    # Check if troughs are similar depth (within 2%)
                    if abs(low[trough1] - low[trough2]) / low[trough1] < 0.02:
                        if trough2 - trough1 > 10:
                            peak_val = np.max(high[trough1:trough2+1])
                            current_price = close[-1]
                            
                            if current_price >= peak_val * 0.98:
                                height = peak_val - (low[trough1] + low[trough2]) / 2
                                target = peak_val + height
                                stop_loss = min(low[trough1], low[trough2]) * 0.98
                                
                                pattern_start_date = window_df.iloc[trough1]['date']
                                pattern_days = (window_df.iloc[-1]['date'] - pattern_start_date).days
                                
                                patterns.append({
                                    'ticker': ticker,
                                    'pattern': 'Double Bottom',
                                    'type': 'bullish',
                                    'signal': 'BUY',
                                    'current_price': current_price,
                                    'target_price': target,
                                    'stop_loss': stop_loss,
                                    'confidence': self._calculate_confidence(current_price, peak_val, 'bullish'),
                                    'neckline': peak_val,
                                    'formation_days': pattern_days,
                                    'detected_date': df.iloc[-1]['date'],
                                    'risk_reward': abs(target - current_price) / abs(current_price - stop_loss)
                                })
        
        return patterns
    
    def _detect_triangles(self, df: pd.DataFrame, ticker: str, include_forming: bool = False) -> List[Dict]:
        """Detect Triangle patterns (Ascending, Descending, Symmetrical)."""
        patterns = []
        
        for lookback_days in [60, 120, 180, 250]:
            if len(df) < lookback_days:
                continue
            
            window_df = df.tail(lookback_days).reset_index(drop=True)
            high = window_df['high'].values
            low = window_df['low'].values
            close = window_df['close'].values
            
            # Get highs and lows for trendline fitting
            peaks, _ = self._find_peaks_troughs(pd.Series(high), order=5)
            _, troughs = self._find_peaks_troughs(pd.Series(low), order=5)
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                # Fit trendlines
                if len(peaks) >= 2:
                    high_slope, high_intercept, _, _, _ = linregress(peaks, high[peaks])
                else:
                    high_slope = 0
                
                if len(troughs) >= 2:
                    low_slope, low_intercept, _, _, _ = linregress(troughs, low[troughs])
                else:
                    low_slope = 0
                
                current_price = close[-1]
                
                # Ascending Triangle (bullish)
                if abs(high_slope) < 0.01 and low_slope > 0.01:
                    resistance = np.mean(high[peaks])
                    support_current = low_slope * len(window_df) + low_intercept
                    
                    is_complete = current_price > resistance * 0.95
                    is_forming = current_price > support_current * 1.05 and current_price < resistance * 0.95
                    
                    if is_complete or (include_forming and is_forming):
                        height = resistance - support_current
                        target = resistance + height
                        stop_loss = support_current * 0.98
                        
                        patterns.append({
                            'ticker': ticker,
                            'pattern': 'Ascending Triangle',
                            'type': 'bullish',
                            'signal': 'BUY',
                            'status': 'complete' if is_complete else 'forming',
                            'current_price': current_price,
                            'target_price': target,
                            'stop_loss': stop_loss,
                            'confidence': self._calculate_confidence(current_price, resistance, 'bullish') * (1.0 if is_complete else 0.6),
                            'neckline': resistance,
                            'formation_days': lookback_days,
                            'detected_date': df.iloc[-1]['date'],
                            'risk_reward': abs(target - current_price) / abs(current_price - stop_loss),
                            'trendlines': {
                                'resistance': {'slope': high_slope, 'intercept': high_intercept, 'level': resistance},
                                'support': {'slope': low_slope, 'intercept': low_intercept}
                            },
                            'key_points': {
                                'peaks': [{'index': p, 'price': high[p], 'date': window_df.iloc[p]['date']} for p in peaks[-3:]],
                                'troughs': [{'index': t, 'price': low[t], 'date': window_df.iloc[t]['date']} for t in troughs[-3:]]
                            }
                        })
                
                # Descending Triangle (bearish)
                elif abs(low_slope) < 0.01 and high_slope < -0.01:
                    support = np.mean(low[troughs])
                    resistance_current = high_slope * len(window_df) + high_intercept
                    
                    is_complete = current_price < support * 1.05
                    is_forming = current_price < resistance_current * 0.95 and current_price > support * 1.05
                    
                    if is_complete or (include_forming and is_forming):
                        height = resistance_current - support
                        target = support - height
                        stop_loss = resistance_current * 1.02
                        
                        patterns.append({
                            'ticker': ticker,
                            'pattern': 'Descending Triangle',
                            'type': 'bearish',
                            'signal': 'SELL',
                            'status': 'complete' if is_complete else 'forming',
                            'current_price': current_price,
                            'target_price': target,
                            'stop_loss': stop_loss,
                            'confidence': self._calculate_confidence(current_price, support, 'bearish') * (1.0 if is_complete else 0.6),
                            'neckline': support,
                            'formation_days': lookback_days,
                            'detected_date': df.iloc[-1]['date'],
                            'risk_reward': abs(target - current_price) / abs(stop_loss - current_price),
                            'trendlines': {
                                'resistance': {'slope': high_slope, 'intercept': high_intercept},
                                'support': {'slope': low_slope, 'intercept': low_intercept, 'level': support}
                            },
                            'key_points': {
                                'peaks': [{'index': p, 'price': high[p], 'date': window_df.iloc[p]['date']} for p in peaks[-3:]],
                                'troughs': [{'index': t, 'price': low[t], 'date': window_df.iloc[t]['date']} for t in troughs[-3:]]
                            }
                        })
                
                # Symmetrical Triangle (continuation pattern)
                elif high_slope < -0.01 and low_slope > 0.01:
                    # Determine trend before triangle
                    pre_triangle_trend = 'bullish' if close[0] < close[len(close)//3] else 'bearish'
                    apex_index = int((high_intercept - low_intercept) / (low_slope - high_slope))
                    
                    if apex_index > len(window_df) * 0.7:  # Near apex
                        height = abs(high[peaks[0]] - low[troughs[0]])
                        
                        if pre_triangle_trend == 'bullish':
                            target = current_price + height
                            stop_loss = low[troughs[-1]] * 0.98
                            signal = 'BUY'
                        else:
                            target = current_price - height
                            stop_loss = high[peaks[-1]] * 1.02
                            signal = 'SELL'
                        
                        patterns.append({
                            'ticker': ticker,
                            'pattern': 'Symmetrical Triangle',
                            'type': pre_triangle_trend,
                            'signal': signal,
                            'current_price': current_price,
                            'target_price': target,
                            'stop_loss': stop_loss,
                            'confidence': 0.6,  # Lower confidence for symmetrical
                            'neckline': None,
                            'formation_days': lookback_days,
                            'detected_date': df.iloc[-1]['date'],
                            'risk_reward': abs(target - current_price) / abs(current_price - stop_loss)
                        })
        
        return patterns
    
    def _detect_cup_handle(self, df: pd.DataFrame, ticker: str, include_forming: bool = False) -> List[Dict]:
        """Detect Cup and Handle pattern (bullish)."""
        patterns = []
        
        for lookback_days in [120, 180, 250, 365]:
            if len(df) < lookback_days:
                continue
            
            window_df = df.tail(lookback_days).reset_index(drop=True)
            close = window_df['close'].values
            high = window_df['high'].values
            low = window_df['low'].values
            
            # Find the cup
            cup_start = 0
            cup_bottom_idx = np.argmin(low[:int(len(low)*0.7)])
            
            if cup_bottom_idx > len(low) * 0.2:  # Bottom should be in middle section
                cup_bottom = low[cup_bottom_idx]
                cup_rim = max(high[cup_start], high[cup_bottom_idx])
                
                # Check for U-shape (gradual decline and rise)
                left_decline = (high[cup_start] - cup_bottom) / cup_bottom_idx
                right_rise = (close[-1] - cup_bottom) / (len(close) - cup_bottom_idx)
                
                if left_decline > 0 and right_rise > 0 and abs(left_decline - right_rise) < 0.5:
                    # Look for handle (small pullback near rim)
                    handle_start = int(len(close) * 0.7)
                    handle_low = np.min(low[handle_start:])
                    current_price = close[-1]
                    
                    # Handle should be shallow (not more than 1/3 of cup depth)
                    cup_depth = cup_rim - cup_bottom
                    if (cup_rim - handle_low) < cup_depth * 0.33 and current_price > handle_low * 1.02:
                        target = cup_rim + cup_depth
                        stop_loss = handle_low * 0.98
                        
                        patterns.append({
                            'ticker': ticker,
                            'pattern': 'Cup and Handle',
                            'type': 'bullish',
                            'signal': 'BUY',
                            'current_price': current_price,
                            'target_price': target,
                            'stop_loss': stop_loss,
                            'confidence': self._calculate_confidence(current_price, cup_rim, 'bullish'),
                            'neckline': cup_rim,
                            'formation_days': lookback_days,
                            'detected_date': df.iloc[-1]['date'],
                            'risk_reward': abs(target - current_price) / abs(current_price - stop_loss)
                        })
        
        return patterns
    
    def _detect_flags_pennants(self, df: pd.DataFrame, ticker: str, include_forming: bool = False) -> List[Dict]:
        """Detect Flag and Pennant patterns (continuation patterns)."""
        patterns = []
        
        for lookback_days in [30, 60, 90]:
            if len(df) < lookback_days:
                continue
            
            window_df = df.tail(lookback_days).reset_index(drop=True)
            close = window_df['close'].values
            high = window_df['high'].values
            low = window_df['low'].values
            volume = window_df['volume'].values if 'volume' in window_df.columns else None
            
            # Identify strong move (pole)
            pole_length = int(len(close) * 0.4)
            pole_change = (close[pole_length] - close[0]) / close[0]
            
            # Strong uptrend or downtrend in pole
            if abs(pole_change) > 0.15:  # At least 15% move
                trend = 'bullish' if pole_change > 0 else 'bearish'
                
                # Check for consolidation (flag/pennant)
                consolidation = close[pole_length:]
                consolidation_range = (np.max(consolidation) - np.min(consolidation)) / np.mean(consolidation)
                
                # Flag: parallel consolidation, Pennant: converging consolidation
                if consolidation_range < 0.10:  # Tight consolidation
                    current_price = close[-1]
                    pole_height = abs(close[pole_length] - close[0])
                    
                    if trend == 'bullish':
                        target = current_price + pole_height
                        stop_loss = np.min(consolidation) * 0.98
                        signal = 'BUY'
                        confidence = self._calculate_confidence(current_price, np.max(consolidation[:int(len(consolidation)*0.5)]), 'bullish')
                    else:
                        target = current_price - pole_height
                        stop_loss = np.max(consolidation) * 1.02
                        signal = 'SELL'
                        confidence = self._calculate_confidence(current_price, np.min(consolidation[:int(len(consolidation)*0.5)]), 'bearish')
                    
                    # Check volume (should decrease during consolidation)
                    volume_decreased = True
                    if volume is not None:
                        pole_volume = np.mean(volume[:pole_length])
                        flag_volume = np.mean(volume[pole_length:])
                        volume_decreased = flag_volume < pole_volume
                    
                    if volume_decreased:
                        patterns.append({
                            'ticker': ticker,
                            'pattern': 'Bull Flag' if trend == 'bullish' else 'Bear Flag',
                            'type': trend,
                            'signal': signal,
                            'current_price': current_price,
                            'target_price': target,
                            'stop_loss': stop_loss,
                            'confidence': confidence,
                            'neckline': None,
                            'formation_days': lookback_days,
                            'detected_date': df.iloc[-1]['date'],
                            'risk_reward': abs(target - current_price) / abs(current_price - stop_loss)
                        })
        
        return patterns
    
    def _detect_wedges(self, df: pd.DataFrame, ticker: str, include_forming: bool = False) -> List[Dict]:
        """Detect Rising and Falling Wedge patterns."""
        patterns = []
        
        for lookback_days in [60, 120, 180]:
            if len(df) < lookback_days:
                continue
            
            window_df = df.tail(lookback_days).reset_index(drop=True)
            high = window_df['high'].values
            low = window_df['low'].values
            close = window_df['close'].values
            
            peaks, _ = self._find_peaks_troughs(pd.Series(high), order=5)
            _, troughs = self._find_peaks_troughs(pd.Series(low), order=5)
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                high_slope, high_intercept, _, _, _ = linregress(peaks, high[peaks])
                low_slope, low_intercept, _, _, _ = linregress(troughs, low[troughs])
                
                current_price = close[-1]
                
                # Rising Wedge (bearish) - both lines rising but converging
                if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
                    resistance = high_slope * len(window_df) + high_intercept
                    support = low_slope * len(window_df) + low_intercept
                    
                    if current_price < support * 1.05:
                        height = resistance - support
                        target = current_price - height * 1.5
                        stop_loss = resistance * 1.02
                        
                        patterns.append({
                            'ticker': ticker,
                            'pattern': 'Rising Wedge',
                            'type': 'bearish',
                            'signal': 'SELL',
                            'current_price': current_price,
                            'target_price': target,
                            'stop_loss': stop_loss,
                            'confidence': self._calculate_confidence(current_price, support, 'bearish'),
                            'neckline': support,
                            'formation_days': lookback_days,
                            'detected_date': df.iloc[-1]['date'],
                            'risk_reward': abs(target - current_price) / abs(stop_loss - current_price)
                        })
                
                # Falling Wedge (bullish) - both lines falling but converging
                elif high_slope < 0 and low_slope < 0 and abs(high_slope) < abs(low_slope):
                    resistance = high_slope * len(window_df) + high_intercept
                    support = low_slope * len(window_df) + low_intercept
                    
                    if current_price > resistance * 0.95:
                        height = resistance - support
                        target = current_price + height * 1.5
                        stop_loss = support * 0.98
                        
                        patterns.append({
                            'ticker': ticker,
                            'pattern': 'Falling Wedge',
                            'type': 'bullish',
                            'signal': 'BUY',
                            'current_price': current_price,
                            'target_price': target,
                            'stop_loss': stop_loss,
                            'confidence': self._calculate_confidence(current_price, resistance, 'bullish'),
                            'neckline': resistance,
                            'formation_days': lookback_days,
                            'detected_date': df.iloc[-1]['date'],
                            'risk_reward': abs(target - current_price) / abs(current_price - stop_loss)
                        })
        
        return patterns
    
    def _calculate_confidence(self, current_price: float, breakout_level: float, direction: str) -> float:
        """
        Calculate confidence score based on proximity to breakout and pattern quality.
        
        Args:
            current_price: Current price
            breakout_level: Key level for breakout
            direction: 'bullish' or 'bearish'
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if breakout_level == 0:
            return 0.5
        
        distance = abs(current_price - breakout_level) / breakout_level
        
        if direction == 'bullish':
            if current_price > breakout_level:
                # Already broken out
                confidence = max(0.7, min(0.95, 0.95 - distance * 2))
            else:
                # Approaching breakout
                confidence = max(0.5, min(0.75, 0.75 - distance * 3))
        else:  # bearish
            if current_price < breakout_level:
                # Already broken down
                confidence = max(0.7, min(0.95, 0.95 - distance * 2))
            else:
                # Approaching breakdown
                confidence = max(0.5, min(0.75, 0.75 - distance * 3))
        
        return round(confidence, 2)
    
    def calculate_timeframe_targets(self, pattern: Dict, df: pd.DataFrame) -> Dict:
        """
        Calculate price targets for different timeframes.
        
        Args:
            pattern: Detected pattern dictionary
            df: Historical price data
        
        Returns:
            Dictionary with timeframe targets
        """
        current_price = pattern['current_price']
        target_price = pattern['target_price']
        
        # Calculate expected move per day based on historical volatility
        if len(df) >= 30:
            recent_df = df.tail(30)
            daily_returns = recent_df['close'].pct_change().dropna()
            avg_daily_move = abs(daily_returns.mean())
            std_daily_move = daily_returns.std()
        else:
            avg_daily_move = 0.01
            std_daily_move = 0.02
        
        # 1-3 day target (conservative)
        days_1_3_move = avg_daily_move * 2 * (1 if pattern['type'] == 'bullish' else -1)
        target_1_3_days = current_price * (1 + days_1_3_move)
        
        # 1 month target (more aggressive, but not full pattern target)
        total_move = (target_price - current_price) / current_price
        target_1_month = current_price * (1 + total_move * 0.5)  # 50% of full move in 1 month
        
        return {
            'target_1_3_days': round(target_1_3_days, 2),
            'target_1_month': round(target_1_month, 2),
            'target_full': round(target_price, 2),
            'expected_days_to_target': int(abs(total_move) / avg_daily_move) if avg_daily_move > 0 else None
        }


def rank_patterns_by_quality(patterns: List[Dict]) -> List[Dict]:
    """
    Rank patterns by quality score.
    
    Args:
        patterns: List of detected patterns
    
    Returns:
        Sorted list of patterns with quality scores
    """
    for pattern in patterns:
        score = 0.0
        
        # Confidence weight (40%)
        score += pattern.get('confidence', 0.5) * 0.4
        
        # Risk/Reward weight (30%)
        rr = pattern.get('risk_reward', 1.0)
        if rr >= 2.0:
            score += 0.3
        elif rr >= 1.5:
            score += 0.2
        elif rr >= 1.0:
            score += 0.1
        
        # Formation time weight (20%) - longer formation is stronger
        days = pattern.get('formation_days', 0)
        if days >= 180:
            score += 0.2
        elif days >= 90:
            score += 0.15
        elif days >= 60:
            score += 0.1
        
        # Pattern type weight (10%)
        strong_patterns = ['Head and Shoulders Top', 'Inverse Head and Shoulders', 
                          'Cup and Handle', 'Double Bottom', 'Double Top']
        if pattern.get('pattern') in strong_patterns:
            score += 0.1
        else:
            score += 0.05
        
        pattern['quality_score'] = round(score, 2)
    
    # Sort by quality score descending
    return sorted(patterns, key=lambda x: x.get('quality_score', 0), reverse=True)
