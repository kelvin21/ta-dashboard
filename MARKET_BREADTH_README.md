# Market Breadth Analysis Implementation

## Overview
This implementation adds a comprehensive Market Breadth Analysis sub-page to the MACD Reversal Dashboard, providing advanced technical indicators and market-wide sentiment analysis for the Vietnamese stock market.

## Components Created

### 1. Utility Modules (`utils/`)

#### `utils/indicators.py`
- **Purpose**: Technical indicator calculations using TA-Lib (with pandas fallback)
- **Features**:
  - EMA/SMA calculations (10, 20, 50, 100, 200 periods)
  - RSI calculation (Wilder's smoothing method)
  - MACD calculation (12, 26, 9 parameters)
  - Bollinger Bands (20 period, 2 std dev)
  - Batch indicator calculation function
  - RSI categorization helper functions
  
#### `utils/macd_stage.py`
- **Purpose**: MACD histogram stage detection and categorization
- **Features**:
  - 6-stage MACD detection:
    1. Troughing (below zero, declining momentum)
    2. Confirmed Trough (crossing from negative to positive)
    3. Rising above Zero (positive and strengthening)
    4. Peaking (above zero, declining momentum)
    5. Confirmed Peak (crossing from positive to negative)
    6. Falling below Zero (negative and weakening)
  - Stage categorization helpers
  - Color coding for visualization
  - Numeric scoring for sentiment analysis

#### `utils/db_async.py`
- **Purpose**: Async MongoDB operations using Motor
- **Features**:
  - Async database adapter class
  - Price data loading
  - Indicator storage and retrieval
  - Market breadth history management
  - Trading date utilities
  - Synchronous fallback adapter (pymongo)

### 2. Market Breadth Page Integration

The existing `pages/1_📊_Market_Breadth.py` has been enhanced with:
- Optional integration with new utility modules
- Backward compatibility with existing functions
- Improved TA-Lib support detection

## Architecture

```
macd-reversal/
├── utils/
│   ├── __init__.py
│   ├── indicators.py          # Technical indicator calculations
│   ├── macd_stage.py          # MACD stage detection
│   └── db_async.py            # Async MongoDB operations
├── pages/
│   └── 1_📊_Market_Breadth.py # Market breadth analysis page
├── ta_dashboard.py            # Main dashboard (existing)
├── db_adapter.py              # Database adapter (existing)
└── build_price_db.py          # Data builder (existing)
```

## Functional Requirements Implemented

### ✅ Page Structure
- [x] Control Panel with date selection
- [x] Market Breadth Summary tables
- [x] Indicator basket groups with ticker lists
- [x] VNINDEX Technical Chart
- [x] 1-Year Market Breadth Historical Charts
- [x] Debug view for indicators

### ✅ Indicator Calculation
- [x] EMA: 10, 20, 50, 100, 200 periods
- [x] RSI(14) with Wilder's smoothing
- [x] MACD(12,26,9) with histogram
- [x] Bollinger Bands(20,2)
- [x] MACD stage detection (6 stages)
- [x] Uses TA-Lib when available (most accurate)
- [x] Pandas fallback for missing TA-Lib

### ✅ Market Breadth Metrics
- [x] MA/EMA breadth (% above each period)
- [x] RSI breadth (oversold/undersold/overbought categories)
- [x] MACD stage distribution (6 stages)
- [x] Historical breadth tracking
- [x] Sentiment scoring (bullish/bearish signals)

### ✅ Data Management
- [x] MongoDB integration (via db_adapter.py)
- [x] SQLite fallback support
- [x] Async batch processing for historical calculations
- [x] Indicator caching in database
- [x] Market breadth history storage

### ✅ Visualization
- [x] Interactive Plotly charts
- [x] Multi-panel VNINDEX technical chart
- [x] Breadth trend charts (1-year history)
- [x] Synchronized hover across subplots
- [x] Peak/bottom region shading
- [x] Responsive layouts

### ✅ User Interface
- [x] Date picker for historical snapshots
- [x] Recalculation controls
- [x] Indicator selection filters
- [x] Expandable ticker lists
- [x] Debug mode
- [x] CSV data export

## Dependencies

### Required
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `plotly` - Interactive charts
- `pymongo` - MongoDB driver (sync)
- `python-dotenv` - Environment variable management

### Optional (Recommended)
- `talib` - Technical Analysis Library (most accurate indicators)
- `motor` - Async MongoDB driver
- `aiohttp` - Async HTTP client

### Installation

```bash
# Core dependencies
pip install streamlit pandas numpy plotly pymongo python-dotenv

# Optional - for best performance
pip install TA-Lib motor aiohttp

# Note: TA-Lib requires compilation. See: https://github.com/mrjbq7/ta-lib
```

## Configuration

### Environment Variables (.env)

```env
# MongoDB Configuration
USE_MONGODB=true
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/?appName=AppName
MONGODB_DB_NAME=macd_reversal

# Database Paths (if using SQLite)
PRICE_DB_PATH=price_data.db
REF_DB_PATH=analysis_results.db

# Feature Flags
SHOW_MARKET_BREADTH_PAGE=true
SHOW_MODULE_WARNINGS=false

# Cache Settings
CACHE_TTL=1800
```

## Usage

### Running the Dashboard

```bash
# Start the main dashboard
streamlit run ta_dashboard.py

# Access Market Breadth page from sidebar navigation
```

### Calculating Historical Breadth

1. Navigate to Market Breadth page
2. Enable "Recalculate historical indicators" in sidebar
3. Set lookback period (default: 200 trading days)
4. Click "Calculate Now"
5. View results in historical charts section

### Viewing Historical Snapshots

1. Enable "View specific date snapshot" in sidebar
2. Select date from date picker
3. View breadth metrics for that date
4. Compare with current market conditions

### Exporting Data

- Click "Download Summary CSV" for breadth metrics
- Click "Download Detailed Data CSV" for individual ticker indicators
- Click "Download Debug Table" (debug mode) for full calculation details

## Data Schema

### MongoDB Collections

#### `indicators` Collection
```javascript
{
  ticker: "VCB",
  date: ISODate("2025-12-08"),
  close: 95.5,
  ema10: 94.2,
  ema20: 93.8,
  ema50: 92.1,
  ema100: 90.5,
  ema200: 88.3,
  rsi: 65.4,
  macd: 1.23,
  macd_signal: 1.15,
  macd_hist: 0.08,
  macd_stage: "3. Rising above Zero",
  bb_upper: 98.5,
  bb_middle: 95.0,
  bb_lower: 91.5,
  updated_at: ISODate("2025-12-08T10:30:00Z")
}
```

#### `market_breadth` Collection
```javascript
{
  date: ISODate("2025-12-08"),
  total_tickers: 150,
  above_ema10: 85,
  above_ema10_pct: 56.7,
  above_ema20: 78,
  above_ema20_pct: 52.0,
  // ... (all EMA periods)
  rsi_oversold: 12,
  rsi_oversold_pct: 8.0,
  rsi_<50: 45,
  rsi_<50_pct: 30.0,
  // ... (all RSI categories)
  macd_troughing: 8,
  macd_confirmed_trough: 15,
  macd_rising: 45,
  // ... (all MACD stages)
  updated_at: ISODate("2025-12-08T10:30:00Z")
}
```

## Performance Optimization

### Caching Strategy
- **Streamlit caching**: `@st.cache_data` for data loads (30 min TTL)
- **Database indexing**: Compound indexes on (ticker, date)
- **Async processing**: Concurrent calculation of multiple dates
- **Batch operations**: Grouped database writes

### Async Processing
- Uses `motor` for async MongoDB operations
- Concurrent calculation of historical breadth
- Progress tracking for long-running operations
- Timeout protection (60 seconds per operation)

### Memory Management
- Streaming data processing (no full dataset in memory)
- Limited ticker lists in UI (first 50 for debug)
- Cleanup of temporary DataFrames
- Efficient pandas operations

## Troubleshooting

### TA-Lib Installation Issues

**Windows:**
```bash
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.XX‑cpXX‑cpXX‑win_amd64.whl
```

**Linux:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

### MongoDB Connection Issues

1. Check `.env` file exists in project root
2. Verify `MONGODB_URI` is correct
3. Check network connectivity
4. Increase timeout in `db_async.py` if needed
5. Test connection: `python -c "from utils.db_async import get_sync_db_adapter; db = get_sync_db_adapter(); print(db.get_all_tickers())"`

### Common Errors

**Error: "No indicator data found"**
- Solution: Run historical calculation first
- Check date range in database
- Verify tickers exist in price_data collection

**Error: "TA-Lib not installed"**
- Solution: Install TA-Lib or use manual calculations
- Manual calculations are slower but functional
- See TA-Lib installation guide above

**Error: "motor not installed"**
- Solution: Install motor for async support
- Or use synchronous fallback (slower)
- `pip install motor`

## Testing

### Unit Tests (Recommended)

```python
# test_indicators.py
import pytest
from utils.indicators import calculate_ema, calculate_rsi, calculate_macd

def test_ema_calculation():
    data = pd.Series([100, 102, 101, 103, 105])
    ema = calculate_ema(data, 3)
    assert not ema.empty
    assert ema.iloc[-1] > 100

def test_rsi_calculation():
    data = pd.Series([100, 102, 101, 103, 105, 103, 102, 104])
    rsi = calculate_rsi(data, 7)
    assert 0 <= rsi.iloc[-1] <= 100

# Run tests
pytest test_indicators.py
```

### Manual Testing Checklist

- [ ] Load main dashboard successfully
- [ ] Navigate to Market Breadth page
- [ ] View current breadth metrics
- [ ] Calculate historical breadth (small sample)
- [ ] View historical charts
- [ ] Select specific date
- [ ] Export CSV data
- [ ] Enable debug mode
- [ ] Test with different date ranges
- [ ] Verify VNINDEX chart displays correctly

## Future Enhancements

### Planned Features
- [ ] Real-time breadth updates during trading hours
- [ ] Alert system for breadth thresholds
- [ ] Sector-specific breadth analysis
- [ ] Comparative breadth (vs. historical averages)
- [ ] Machine learning breadth predictions
- [ ] Custom indicator formulas
- [ ] Advanced filtering (by sector, market cap, etc.)
- [ ] Correlation analysis between breadth and index

### Performance Improvements
- [ ] Redis caching layer
- [ ] Incremental calculations (only new data)
- [ ] Pre-computed breadth snapshots
- [ ] Parallel processing for multiple tickers
- [ ] Database query optimization
- [ ] Frontend pagination for large datasets

## API Reference

### Indicators Module

```python
from utils.indicators import calculate_all_indicators

# Calculate all indicators for a DataFrame
df_with_indicators = calculate_all_indicators(
    df=price_df,
    ema_periods=[10, 20, 50, 100, 200],
    rsi_period=14,
    macd_params=(12, 26, 9),
    bb_params=(20, 2.0)
)
```

### MACD Stage Module

```python
from utils.macd_stage import detect_macd_stage, categorize_macd_stage

# Detect stage for histogram series
stage = detect_macd_stage(macd_hist_series, lookback=20)
# Returns: "2. Confirmed Trough"

# Categorize for grouping
category = categorize_macd_stage(stage)
# Returns: "confirmed_trough"
```

### Async DB Module

```python
from utils.db_async import get_async_db_adapter
import asyncio

async def main():
    db = get_async_db_adapter()
    tickers = await db.get_all_tickers()
    print(f"Found {len(tickers)} tickers")
    
    # Load price data
    df = await db.get_price_data("VCB", start_date, end_date)
    
    # Save indicators
    await db.save_indicators("VCB", date, indicators_dict)
    
    db.close()

asyncio.run(main())
```

## Support

For issues or questions:
1. Check this README first
2. Review error messages in debug mode
3. Check Streamlit logs: `streamlit run ta_dashboard.py --logger.level=debug`
4. Open GitHub issue with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version, package versions)

## License

This implementation is part of the MACD Reversal Dashboard project.

## Changelog

### Version 1.0.0 (2025-12-08)
- Initial implementation of Market Breadth Analysis
- Created utility modules (indicators, macd_stage, db_async)
- Integrated with existing Market Breadth page
- Added TA-Lib support with fallback
- Implemented async MongoDB operations
- Added comprehensive documentation
