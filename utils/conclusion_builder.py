"""
Conclusion Builder for EMA Signals
Generates actionable buy/sell recommendations and market strategy.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def generate_immediate_buy_signals(df_indicators: pd.DataFrame, top_n: int = 3) -> List[Dict]:
    """
    Generate top immediate buy signals based on EMA analysis.
    
    Args:
        df_indicators: DataFrame with all ticker indicators
        top_n: Number of top buy signals to return
    
    Returns:
        List of dictionaries with buy signal details
    """
    if df_indicators.empty:
        return []
    
    buy_candidates = []
    
    for _, row in df_indicators.iterrows():
        ticker = row.get('ticker', '')
        if not ticker or ticker == 'VNINDEX':
            continue
        
        close = row.get('close', np.nan)
        ema10 = row.get('ema10', np.nan)
        ema20 = row.get('ema20', np.nan)
        ema50 = row.get('ema50', np.nan)
        ema100 = row.get('ema100', np.nan)
        ema200 = row.get('ema200', np.nan)
        
        # Skip if essential data is missing
        if pd.isna(close) or pd.isna(ema20) or pd.isna(ema50):
            continue
        
        # Calculate buy score
        buy_score = 0
        reasons = []
        
        # 1. FOMO Breakout: Price breaking above key EMAs
        if pd.notna(ema50) and close > ema50:
            if pd.notna(ema20) and ema20 > ema50:
                buy_score += 3
                reasons.append("FOMO breakout above EMA50")
        
        # 2. Golden Cross: EMA10 > EMA20 > EMA50
        if all(pd.notna(ema) for ema in [ema10, ema20, ema50]):
            if ema10 > ema20 > ema50:
                buy_score += 3
                reasons.append("Golden EMA alignment")
        
        # 3. Above all major EMAs
        emas_to_check = [ema for ema in [ema20, ema50, ema100, ema200] if pd.notna(ema)]
        if emas_to_check:
            above_all = all(close > ema for ema in emas_to_check)
            if above_all:
                buy_score += 2
                reasons.append("Above all EMAs")
        
        # 4. Increasing momentum (price > EMA10 > EMA20)
        if all(pd.notna(ema) for ema in [ema10, ema20]):
            if close > ema10 > ema20:
                buy_score += 2
                reasons.append("Strong upward momentum")
        
        # 5. Near EMA support (within 2% of EMA20)
        if pd.notna(ema20):
            distance_to_ema20 = ((close - ema20) / ema20) * 100
            if 0 <= distance_to_ema20 <= 2:
                buy_score += 1
                reasons.append("Near EMA20 support")
        
        # 6. EMA convergence (potential breakout)
        if all(pd.notna(ema) for ema in [ema20, ema50, ema100]):
            ema_std = np.std([ema20, ema50, ema100])
            ema_mean = np.mean([ema20, ema50, ema100])
            if ema_mean > 0:
                convergence = (ema_std / ema_mean) * 100
                if convergence < 3:  # Very tight EMAs
                    buy_score += 2
                    reasons.append("EMA convergence - breakout imminent")
        
        # 7. RSI check (if available)
        rsi = row.get('rsi', np.nan)
        if pd.notna(rsi):
            if 40 <= rsi <= 60:
                buy_score += 1
                reasons.append("RSI in neutral zone")
            elif rsi < 40:
                buy_score += 2
                reasons.append("RSI oversold - bounce expected")
        
        # 8. MACD check (if available)
        macd_hist = row.get('macd_hist', np.nan)
        if pd.notna(macd_hist) and macd_hist > 0:
            buy_score += 1
            reasons.append("MACD bullish")
        
        if buy_score >= 5:  # Threshold for consideration
            buy_candidates.append({
                'ticker': ticker,
                'score': buy_score,
                'close': close,
                'reasons': reasons,
                'ema20': ema20,
                'ema50': ema50,
                'rsi': rsi if pd.notna(rsi) else None
            })
    
    # Sort by score and return top N
    buy_candidates.sort(key=lambda x: x['score'], reverse=True)
    return buy_candidates[:top_n]


def generate_immediate_sell_signals(df_indicators: pd.DataFrame, top_n: int = 3) -> List[Dict]:
    """
    Generate top immediate sell signals based on EMA analysis.
    
    Args:
        df_indicators: DataFrame with all ticker indicators
        top_n: Number of top sell signals to return
    
    Returns:
        List of dictionaries with sell signal details
    """
    if df_indicators.empty:
        return []
    
    sell_candidates = []
    
    for _, row in df_indicators.iterrows():
        ticker = row.get('ticker', '')
        if not ticker or ticker == 'VNINDEX':
            continue
        
        close = row.get('close', np.nan)
        ema10 = row.get('ema10', np.nan)
        ema20 = row.get('ema20', np.nan)
        ema50 = row.get('ema50', np.nan)
        ema100 = row.get('ema100', np.nan)
        ema200 = row.get('ema200', np.nan)
        
        # Skip if essential data is missing
        if pd.isna(close) or pd.isna(ema20) or pd.isna(ema50):
            continue
        
        # Calculate sell score
        sell_score = 0
        reasons = []
        
        # 1. Breakdown below EMA50
        if pd.notna(ema50) and close < ema50:
            sell_score += 3
            reasons.append("Breakdown below EMA50")
            
            # Extra penalty if breaking below EMA100
            if pd.notna(ema100) and close < ema100:
                sell_score += 2
                reasons.append("Critical breakdown below EMA100")
        
        # 2. Death Cross: EMA10 < EMA20 < EMA50
        if all(pd.notna(ema) for ema in [ema10, ema20, ema50]):
            if ema10 < ema20 < ema50:
                sell_score += 3
                reasons.append("Death cross - bearish EMA alignment")
        
        # 3. Below all major EMAs
        emas_to_check = [ema for ema in [ema20, ema50, ema100, ema200] if pd.notna(ema)]
        if emas_to_check:
            below_all = all(close < ema for ema in emas_to_check)
            if below_all:
                sell_score += 2
                reasons.append("Below all EMAs")
        
        # 4. Declining momentum (price < EMA10 < EMA20)
        if all(pd.notna(ema) for ema in [ema10, ema20]):
            if close < ema10 < ema20:
                sell_score += 2
                reasons.append("Strong downward momentum")
        
        # 5. Failed resistance at EMA20 (close to but below)
        if pd.notna(ema20):
            distance_to_ema20 = ((close - ema20) / ema20) * 100
            if -3 <= distance_to_ema20 < 0:
                sell_score += 1
                reasons.append("Failed EMA20 resistance")
        
        # 6. Large distance below EMAs (overextended)
        if pd.notna(ema50):
            distance_to_ema50 = ((close - ema50) / ema50) * 100
            if distance_to_ema50 < -10:
                sell_score += 2
                reasons.append("Severely overextended below EMA50")
        
        # 7. RSI check (if available)
        rsi = row.get('rsi', np.nan)
        if pd.notna(rsi):
            if rsi > 70:
                sell_score += 2
                reasons.append("RSI overbought - correction due")
            elif 50 < rsi <= 60:
                sell_score += 1
                reasons.append("RSI weakening")
        
        # 8. MACD check (if available)
        macd_hist = row.get('macd_hist', np.nan)
        if pd.notna(macd_hist) and macd_hist < 0:
            sell_score += 1
            reasons.append("MACD bearish")
        
        if sell_score >= 5:  # Threshold for consideration
            sell_candidates.append({
                'ticker': ticker,
                'score': sell_score,
                'close': close,
                'reasons': reasons,
                'ema20': ema20,
                'ema50': ema50,
                'rsi': rsi if pd.notna(rsi) else None
            })
    
    # Sort by score and return top N
    sell_candidates.sort(key=lambda x: x['score'], reverse=True)
    return sell_candidates[:top_n]


def calculate_market_breadth_summary(df_indicators: pd.DataFrame) -> Dict:
    """
    Calculate market breadth metrics for strategy generation.
    
    Args:
        df_indicators: DataFrame with all ticker indicators
    
    Returns:
        Dictionary with breadth metrics
    """
    if df_indicators.empty:
        return {
            'total_tickers': 0,
            'above_ema50_pct': 0,
            'above_ema200_pct': 0,
            'bullish_alignment_pct': 0,
            'bearish_alignment_pct': 0
        }
    
    total = len(df_indicators)
    
    # Count tickers above key EMAs
    above_ema50 = 0
    above_ema200 = 0
    bullish_alignment = 0
    bearish_alignment = 0
    
    for _, row in df_indicators.iterrows():
        close = row.get('close', np.nan)
        ema50 = row.get('ema50', np.nan)
        ema200 = row.get('ema200', np.nan)
        ema10 = row.get('ema10', np.nan)
        ema20 = row.get('ema20', np.nan)
        
        if pd.notna(close) and pd.notna(ema50):
            if close > ema50:
                above_ema50 += 1
        
        if pd.notna(close) and pd.notna(ema200):
            if close > ema200:
                above_ema200 += 1
        
        # Check alignment
        if all(pd.notna(x) for x in [close, ema10, ema20, ema50]):
            if close > ema10 > ema20 > ema50:
                bullish_alignment += 1
            elif close < ema10 < ema20 < ema50:
                bearish_alignment += 1
    
    return {
        'total_tickers': total,
        'above_ema50': above_ema50,
        'above_ema50_pct': (above_ema50 / total * 100) if total > 0 else 0,
        'above_ema200': above_ema200,
        'above_ema200_pct': (above_ema200 / total * 100) if total > 0 else 0,
        'bullish_alignment': bullish_alignment,
        'bullish_alignment_pct': (bullish_alignment / total * 100) if total > 0 else 0,
        'bearish_alignment': bearish_alignment,
        'bearish_alignment_pct': (bearish_alignment / total * 100) if total > 0 else 0
    }


def generate_market_strategy(
    vnindex_row: Optional[pd.Series],
    breadth_summary: Dict,
    buy_signals: List[Dict],
    sell_signals: List[Dict]
) -> str:
    """
    Generate 1-sentence market strategy based on VNINDEX and breadth.
    
    Args:
        vnindex_row: VNINDEX indicator row
        breadth_summary: Market breadth metrics
        buy_signals: List of buy signal candidates
        sell_signals: List of sell signal candidates
    
    Returns:
        Strategy sentence
    """
    # Default cautious stance
    if vnindex_row is None or breadth_summary['total_tickers'] == 0:
        return "⚠️ Market data insufficient - remain cautious and wait for clearer signals."
    
    # Analyze VNINDEX position
    vnindex_bullish = False
    vnindex_bearish = False
    
    close = vnindex_row.get('close', np.nan)
    ema20 = vnindex_row.get('ema20', np.nan)
    ema50 = vnindex_row.get('ema50', np.nan)
    ema200 = vnindex_row.get('ema200', np.nan)
    
    if all(pd.notna(x) for x in [close, ema50, ema200]):
        if close > ema50 > ema200:
            vnindex_bullish = True
        elif close < ema50 < ema200:
            vnindex_bearish = True
    
    # Analyze breadth
    above_ema50_pct = breadth_summary.get('above_ema50_pct', 0)
    bullish_align_pct = breadth_summary.get('bullish_alignment_pct', 0)
    
    strong_breadth = above_ema50_pct > 60
    weak_breadth = above_ema50_pct < 40
    
    # Generate strategy
    if vnindex_bullish and strong_breadth:
        if len(buy_signals) >= 2:
            return f"🚀 Strong bull market - aggressively buy dips on quality stocks, ride the momentum with {len(buy_signals)} immediate opportunities."
        else:
            return f"📈 Bullish trend intact with {above_ema50_pct:.0f}% stocks above EMA50 - selectively accumulate on pullbacks to EMA20/50."
    
    elif vnindex_bullish and not weak_breadth:
        return f"✅ Market healthy but selective - focus on {len(buy_signals)} strong EMA breakouts, avoid laggards below EMA50."
    
    elif vnindex_bearish and weak_breadth:
        if len(sell_signals) >= 2:
            return f"🔴 Bear market confirmed - cut losses on {len(sell_signals)} weak positions, move to cash or short opportunities."
        else:
            return f"⚠️ Bearish pressure with only {above_ema50_pct:.0f}% above EMA50 - raise cash, tighten stops, wait for market stabilization."
    
    elif vnindex_bearish:
        return f"📉 Index weak but some stocks holding - protect profits, avoid new positions unless {len(buy_signals)} clear EMA50 reclaims occur."
    
    elif weak_breadth:
        return f"⚠️ Narrow market with {above_ema50_pct:.0f}% above EMA50 - be highly selective, focus only on {len(buy_signals)} strongest setups."
    
    else:
        # Mixed/neutral market
        if bullish_align_pct > 30:
            return f"🔄 Transitioning market - {bullish_align_pct:.0f}% bullish alignment suggests bottoming, cautiously accumulate {len(buy_signals)} EMA breakouts."
        else:
            return f"😐 Choppy market lacking clear direction - stay patient, trade only {len(buy_signals)} high-conviction EMA setups with tight stops."


def format_signal_card(signal: Dict, signal_type: str) -> Dict:
    """
    Format signal data for display card.
    
    Args:
        signal: Signal dictionary
        signal_type: 'buy' or 'sell'
    
    Returns:
        Formatted dictionary for display
    """
    ticker = signal.get('ticker', 'N/A')
    score = signal.get('score', 0)
    close = signal.get('close', 0)
    reasons = signal.get('reasons', [])
    rsi = signal.get('rsi')
    
    # Priority label
    if score >= 10:
        priority = "URGENT"
        priority_color = "#F44336"
    elif score >= 7:
        priority = "HIGH"
        priority_color = "#FF9800"
    else:
        priority = "MEDIUM"
        priority_color = "#2196F3"
    
    # Action label
    if signal_type == 'buy':
        action = "BUY"
        action_color = "#4CAF50"
    else:
        action = "SELL"
        action_color = "#F44336"
    
    return {
        'ticker': ticker,
        'action': action,
        'action_color': action_color,
        'priority': priority,
        'priority_color': priority_color,
        'score': score,
        'close': close,
        'reasons': reasons,
        'rsi': rsi,
        'reason_text': " &bull; ".join(reasons[:3])  # Top 3 reasons with HTML entity
    }
