# Quick Start Guide - Market Breadth Analysis

## Installation

### 1. Install Dependencies

```bash
# Navigate to project directory
cd c:\Users\hadao\OneDrive\Documents\Programming\macd-reversal

# Install core dependencies
pip install streamlit pandas numpy plotly pymongo python-dotenv

# Optional but recommended: TA-Lib for accurate indicators
# Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl

# Optional: async MongoDB support
pip install motor aiohttp
```

### 2. Verify Installation

```powershell
# Test imports
python -c "from utils.indicators import calculate_ema; print('✓ Indicators module OK')"
python -c "from utils.macd_stage import detect_macd_stage; print('✓ MACD stage module OK')"
python -c "from utils.db_async import get_sync_db_adapter; print('✓ DB async module OK')"

# Check TA-Lib
python -c "import talib; print('✓ TA-Lib installed')" 2>$null || echo "⚠ TA-Lib not installed (will use pandas fallback)"
```

## Quick Test

### Test 1: Run Debug Script

```powershell
# Test indicator calculations on a single ticker
python debug_breadth_calculations.py VCB --days 365 --lookback 20
```

Expected output:
```
🔍 Debugging Ticker: VCB
📅 Date Range: 2024-12-08 to 2025-12-08
✅ Loaded XXX bars
...
📈 Latest Values (Bar XXX):
   Date: 2025-12-08
   Close: XX.XX
   MA20:  XX.XX
   ...
```

### Test 2: Launch Dashboard

```powershell
# Start Streamlit
streamlit run ta_dashboard.py
```

Then:
1. Open browser to http://localhost:8501
2. Navigate to "📊 Market Breadth" in sidebar
3. You should see the Market Breadth Analysis page

### Test 3: Calculate Sample Breadth Data

In the Market Breadth page:
1. Enable "Recalculate historical indicators" in sidebar
2. Set "Trading days" to 20 (small test)
3. Click "▶️ Calculate Now"
4. Wait for calculation to complete
5. View historical breadth charts

## Verification Checklist

- [ ] ✓ utils/ directory created with 3 modules
- [ ] ✓ All imports work without errors
- [ ] ✓ debug_breadth_calculations.py runs successfully
- [ ] ✓ Streamlit dashboard launches
- [ ] ✓ Market Breadth page loads
- [ ] ✓ Current breadth metrics display
- [ ] ✓ Indicators calculation completes
- [ ] ✓ Historical charts render
- [ ] ✓ CSV export works

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'utils'"

**Solution:**
```powershell
# Ensure __init__.py exists in utils/
Test-Path "utils\__init__.py"
# Should return True

# If False, create it:
New-Item -ItemType File -Path "utils\__init__.py" -Value "# Utility modules"
```

### Issue: "No indicator data found for YYYY-MM-DD"

**Solution:**
1. Run historical calculation first
2. Check database has price data
3. Verify date range is valid
4. Enable debug mode to see detailed errors

### Issue: TA-Lib import errors

**Solution:**
```powershell
# Check Python version (must be 3.7-3.11)
python --version

# Download correct wheel for your Python version
# Example for Python 3.11, 64-bit:
# TA_Lib-0.4.24-cp311-cp311-win_amd64.whl

# Install
pip install path\to\TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
```

### Issue: MongoDB connection timeout

**Solution:**
1. Check .env file has correct MONGODB_URI
2. Test connection:
   ```powershell
   python -c "from pymongo import MongoClient; client = MongoClient('YOUR_URI'); print('Connected:', client.server_info())"
   ```
3. If timeout, check:
   - Internet connection
   - MongoDB Atlas whitelist (add 0.0.0.0/0 for testing)
   - Firewall settings

## Performance Tips

### For Large Datasets

1. **Enable async processing**
   ```bash
   pip install motor
   ```

2. **Use smaller date ranges initially**
   - Start with 20-50 trading days
   - Gradually increase to 200+ days

3. **Enable caching**
   - Streamlit automatically caches data
   - Clear cache only when needed: Ctrl+C in console, restart

### For Slow Calculations

1. **Check TA-Lib installation**
   - TA-Lib is 10-50x faster than pandas
   - Worth the installation effort

2. **Use batch processing**
   - Calculate multiple dates at once
   - Async mode processes concurrently

3. **Optimize lookback period**
   - 20 bars is usually sufficient for MACD stage detection
   - Reduce to 10 for faster calculations

## Next Steps

After successful installation:

1. **Populate historical data**
   - Run with 200 trading days
   - This enables MA200 calculations

2. **Set up daily updates**
   - Create scheduled task to run calculations
   - Update before market open

3. **Explore features**
   - Historical snapshots
   - VNINDEX technical analysis
   - Ticker list filtering
   - CSV exports

4. **Customize**
   - Adjust indicator periods
   - Modify breadth thresholds
   - Add custom metrics

## Getting Help

If you encounter issues:

1. **Enable debug mode**
   - Check "Show debug info" in sidebar
   - Review detailed error messages

2. **Check logs**
   ```powershell
   # Run with debug logging
   streamlit run ta_dashboard.py --logger.level=debug
   ```

3. **Test components individually**
   ```powershell
   # Test indicator module
   python -c "from utils.indicators import calculate_all_indicators; print('OK')"
   
   # Test database
   python -c "from utils.db_async import get_sync_db_adapter; db = get_sync_db_adapter(); print('Tickers:', len(db.get_all_tickers()))"
   ```

4. **Review documentation**
   - See MARKET_BREADTH_README.md for detailed info
   - Check function docstrings
   - Review code comments

## Success Criteria

Your installation is successful if you can:

✅ Run `python debug_breadth_calculations.py VCB` without errors
✅ Launch Streamlit dashboard and access Market Breadth page
✅ View current market breadth metrics
✅ Calculate historical breadth for at least 20 days
✅ View historical breadth charts
✅ Export data to CSV

Congratulations! Your Market Breadth Analysis system is ready to use.
