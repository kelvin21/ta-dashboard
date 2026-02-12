# Trading Pattern Scanner

## Overview
The Pattern Scanner identifies classical chart patterns that can form over 6-18 months in Vietnamese stock market data. It provides actionable buy/sell signals with specific target prices and stop loss levels.

## Features

### Pattern Detection
Automatically detects the following classical chart patterns:

#### Reversal Patterns
- **Head and Shoulders Top** (Bearish): Strong reversal after uptrend
- **Inverse Head and Shoulders** (Bullish): Reversal after downtrend  
- **Double Top/Bottom**: Multiple tests of resistance/support levels
- **Triple Top/Bottom**: Even stronger reversal signals
- **Rising Wedge** (Bearish): Converging upward price action
- **Falling Wedge** (Bullish): Converging downward price action

#### Continuation Patterns
- **Bull Flag**: Brief pause in uptrend before continuation
- **Bear Flag**: Brief pause in downtrend before continuation
- **Ascending Triangle** (Bullish): Flat resistance with rising support
- **Descending Triangle** (Bearish): Flat support with declining resistance
- **Symmetrical Triangle**: Continuation of prior trend
- **Cup and Handle** (Bullish): U-shaped recovery with small pullback

### Signal Components

Each detected pattern provides:
1. **Signal Type**: BUY or SELL
2. **Current Price**: Latest closing price
3. **Target Prices**:
   - **1-3 Days**: Conservative short-term target
   - **1 Month**: Medium-term target (~50% of pattern move)
   - **Full Target**: Complete pattern projection
4. **Stop Loss**: Risk management level
5. **Risk/Reward Ratio**: Potential gain vs potential loss
6. **Confidence Score**: Pattern quality (0-100%)
7. **Quality Score**: Overall signal strength (0-100%)
8. **Formation Days**: How long the pattern took to form

### Usage

1. **Navigate** to the Pattern Scanner page (5_📈_Pattern_Scanner.py)

2. **Configure Analysis**:
   - Set analysis period (3-24 months, default 18)
   - Set minimum quality score filter
   - Set minimum risk/reward ratio

3. **Run Analysis**: Click "🔍 Scan All Patterns"
   - Scans all tickers in database
   - Uses parallel processing (10 workers)
   - Takes 2-5 minutes for ~1000 tickers

4. **Review Results**:
   - Filter by signal type (BUY/SELL)
   - Filter by pattern type
   - Sort by quality, risk/reward, potential gain, or confidence
   - View detailed charts for each pattern

## Pattern Detection Algorithm

### Process Flow
1. Load historical price data (6-18 months)
2. Identify local peaks and troughs using scipy
3. Analyze price formations for pattern characteristics
4. Calculate breakout/breakdown levels (necklines)
5. Project target prices using pattern-specific rules
6. Calculate stop loss levels
7. Assign confidence and quality scores
8. Rank patterns by overall quality

### Quality Scoring
Quality score is calculated from:
- **Confidence** (40%): How close to breakout/breakdown
- **Risk/Reward** (30%): Potential gain vs risk
- **Formation Time** (20%): Longer = stronger
- **Pattern Type** (10%): Classic patterns score higher

### Target Calculation

**Head & Shoulders:**
```
Target = Neckline ± Pattern_Height
Pattern_Height = Head - Neckline
```

**Double Top/Bottom:**
```
Target = Support/Resistance ± Pattern_Height
Pattern_Height = Peak - Trough
```

**Triangles:**
```
Target = Breakout_Level ± Triangle_Height
Triangle_Height = Initial_Range
```

**Cup & Handle:**
```
Target = Cup_Rim + Cup_Depth
Cup_Depth = Rim - Bottom
```

**Flags & Pennants:**
```
Target = Current_Price ± Pole_Height
Pole_Height = Initial_Move_Before_Consolidation
```

## Risk Management

### Stop Loss Placement
- **Bullish patterns**: Below most recent significant low
- **Bearish patterns**: Above most recent significant high
- Typically 2-5% from entry price

### Position Sizing
Recommended approach:
```
Risk_Amount = Account_Size × Risk_Percentage (e.g., 1-2%)
Position_Size = Risk_Amount / (Entry_Price - Stop_Loss)
```

### Risk/Reward Guidelines
- Minimum 1.5:1 ratio recommended
- 2:1 or higher preferred
- Consider pattern quality and confidence

## Timeframe Projections

### 1-3 Days Target
- Conservative estimate
- Based on recent volatility (30-day ATR)
- Suitable for day traders and swing traders
- Lower risk, lower reward

### 1 Month Target  
- Medium-term projection
- Approximately 50% of full pattern move
- Suitable for position traders
- Balanced risk/reward

### Full Target
- Complete pattern projection
- May take several months to reach
- For long-term investors
- Higher risk, higher reward

## Technical Implementation

### Files
- `utils/pattern_detection.py`: Pattern detection algorithms
- `pages/5_📈_Pattern_Scanner.py`: Streamlit UI
- Uses scipy for peak detection
- Parallel processing with ThreadPoolExecutor

### Dependencies
```python
scipy>=1.11.0  # For signal processing and regression
numpy>=1.24.0  # Numerical operations
pandas>=2.0.0  # Data manipulation
```

### Database Requirements
Requires OHLCV data with:
- `ticker`: Stock symbol
- `date`: Trading date
- `open`, `high`, `low`, `close`: Price data
- `volume`: Trading volume (optional but recommended)

### Performance
- Analyzes ~1000 tickers in 2-5 minutes
- Uses 10 parallel workers
- Each ticker timeout: 30 seconds
- Processes up to 18 months of data per ticker

## Interpretation Guide

### Signal Confidence
- **High (70-100%)**: Pattern well-formed, near breakout
- **Medium (50-70%)**: Pattern developing, watch for confirmation
- **Low (<50%)**: Weak pattern, proceed with caution

### Quality Score
- **Excellent (80-100%)**: Top opportunities
- **Good (60-80%)**: Solid setups
- **Fair (40-60%)**: Consider with caution
- **Poor (<40%)**: Skip unless other factors support

### Formation Time
- **3-6 months**: Valid but less reliable
- **6-12 months**: Strong patterns
- **12-18+ months**: Very strong, high conviction
- **Too short (<1 month)**: May be false pattern

## Best Practices

1. **Confirmation**: Wait for price to break key levels
2. **Volume**: Higher volume on breakout = stronger signal
3. **Trend Context**: Patterns work best with overall trend
4. **Multiple Timeframes**: Check daily and weekly charts
5. **Market Conditions**: Consider overall market sentiment
6. **Diversification**: Don't put all capital in one signal
7. **Stop Losses**: Always use protective stops
8. **Position Sizing**: Risk only 1-2% per trade

## Limitations

- Patterns can fail (no guarantee)
- Historical patterns don't ensure future performance
- Market conditions change
- News and events can invalidate patterns
- Requires sufficient historical data (min 30 days)
- Quality depends on data accuracy

## Examples

### Example 1: Inverse Head & Shoulders (Bullish)
```
Ticker: VCB
Pattern: Inverse Head and Shoulders
Signal: BUY
Current: 95,000
Target (1-3d): 97,000 (+2.1%)
Target (1m): 100,000 (+5.3%)
Target (Full): 105,000 (+10.5%)
Stop Loss: 92,000 (-3.2%)
R/R: 1:3.3
Quality: 85%
```

**Action**: Buy at current or on breakout above 96,000. Stop at 92,000. Target 105,000.

### Example 2: Head & Shoulders Top (Bearish)
```
Ticker: HPG
Pattern: Head and Shoulders Top
Signal: SELL
Current: 28,500
Target (1-3d): 28,000 (-1.8%)
Target (1m): 26,500 (-7.0%)
Target (Full): 25,000 (-12.3%)
Stop Loss: 29,500 (+3.5%)
R/R: 1:3.5
Quality: 78%
```

**Action**: Sell/short at current or on break below 28,000. Stop at 29,500. Target 25,000.

## References

### Pattern Recognition Resources
- CMC Markets: Stock Charts Trading Guide
- Bulkowski's Pattern Site: Pattern statistics
- Technical Analysis books: Classic pattern theory

### Code Documentation
- `PatternDetector` class: Main detection engine
- `analyze_ticker_patterns()`: Single ticker analysis
- `analyze_all_tickers()`: Batch processing
- `rank_patterns_by_quality()`: Scoring system

## Support

For issues or questions:
1. Check pattern chart visually
2. Verify data quality in database
3. Adjust quality/R-R filters
4. Review pattern confidence scores
5. Check formation period settings

---

**⚠️ Risk Disclaimer**: Trading patterns are not guaranteed to succeed. This tool is for educational purposes. Always perform your own analysis, use proper risk management, and never risk more than you can afford to lose. Past performance does not indicate future results.
