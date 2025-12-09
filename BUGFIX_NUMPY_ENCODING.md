# 🐛 Bug Fix: NumPy Type Encoding Error

## Issue

MongoDB was rejecting documents containing numpy data types (`np.int64`, `np.float64`) with error:
```
cannot encode object: np.int64(42), of type: <class 'numpy.int64'>
```

## Root Cause

When calculating market breadth metrics, pandas operations return numpy data types:
- `df.sum()` returns `np.int64`
- Percentage calculations return `np.float64`
- These types cannot be directly serialized to MongoDB (BSON format)

## Solution

### 1. Added Type Conversion Helper Function

**File:** `pages/1_📊_Market_Breadth.py`

```python
def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for MongoDB compatibility.
    
    Args:
        obj: Object that may contain numpy types
    
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    else:
        return obj
```

**Features:**
- Recursively converts nested dictionaries and lists
- Converts `np.int64` → `int`
- Converts `np.float64` → `float`
- Converts `np.ndarray` → `list`
- Converts `pd.Timestamp` → `datetime`

### 2. Updated Synchronous Save Function

**File:** `pages/1_📊_Market_Breadth.py`

```python
def save_market_breadth(date: datetime, breadth_data: dict) -> bool:
    """Save market breadth to database."""
    try:
        # ...
        
        # Convert numpy types to Python native types
        clean_breadth_data = convert_numpy_types(breadth_data)
        
        doc = {
            "date": date,
            **clean_breadth_data,
            "updated_at": datetime.now()
        }
        
        # ... save to MongoDB
```

### 3. Updated Async Save Function

**File:** `pages/1_📊_Market_Breadth.py`

```python
async def calculate_and_save_breadth_for_date_async(...):
    """Calculate and save market breadth for a specific date."""
    try:
        # Calculate breadth
        breadth = calculate_market_breadth(df_indicators)
        
        # Convert numpy types to Python native types
        clean_breadth = convert_numpy_types(breadth)
        
        # Save to database
        success = await db.save_market_breadth(date, clean_breadth)
```

### 4. Updated Async Database Adapter

**File:** `utils/db_async.py`

Added `_convert_numpy_types()` method to AsyncDatabaseAdapter class:

```python
class AsyncDatabaseAdapter:
    async def save_market_breadth(self, date: datetime, breadth_data: Dict) -> bool:
        """Save market breadth calculations to database."""
        try:
            # Convert numpy types to Python native types
            clean_breadth_data = self._convert_numpy_types(breadth_data)
            
            doc = {
                "date": date,
                **clean_breadth_data,
                "updated_at": datetime.now()
            }
            # ... save
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types."""
        # ... same logic as above
```

### 5. Fixed Indicator Saving

**File:** `pages/1_📊_Market_Breadth.py`

```python
def save_indicators_to_db(ticker: str, df: pd.DataFrame) -> bool:
    """Save calculated indicators to database."""
    # ...
    for _, row in df.iterrows():
        if pd.notna(row.get('date')):
            # Convert date to datetime if it's a Timestamp
            date_value = row['date']
            if isinstance(date_value, pd.Timestamp):
                date_value = date_value.to_pydatetime()
            
            indicators = {
                'ticker': ticker.upper(),
                'date': date_value,  # Use converted date
                # ... rest of indicators
            }
```

## Files Modified

1. ✅ `pages/1_📊_Market_Breadth.py`
   - Added `convert_numpy_types()` function
   - Updated `save_market_breadth()`
   - Updated `calculate_and_save_breadth_for_date_async()`
   - Updated `save_indicators_to_db()`

2. ✅ `utils/db_async.py`
   - Added `_convert_numpy_types()` method
   - Updated `save_market_breadth()` method

## Testing

### Before Fix
```
Error saving market breadth: Invalid document {...} | 
cannot encode object: np.int64(42), of type: <class 'numpy.int64'>
```

### After Fix
```python
# Example breadth data before conversion:
{
    'total_tickers': np.int64(130),
    'above_ema10': np.int64(42),
    'above_ema10_pct': np.float64(32.307692),
    ...
}

# After conversion:
{
    'total_tickers': 130,              # int
    'above_ema10': 42,                 # int  
    'above_ema10_pct': 32.307692,      # float
    ...
}
```

### Validation

Run the calculation and verify no encoding errors:
```powershell
streamlit run ta_dashboard.py
```

1. Navigate to Market Breadth page
2. Enable recalculation
3. Click "Calculate Now"
4. Verify no "cannot encode object" errors in console
5. Check MongoDB to confirm data is saved correctly

## Impact

✅ **Fixed**: Market breadth calculation now works without encoding errors  
✅ **Fixed**: Both sync and async save operations handle numpy types  
✅ **Fixed**: Indicator saving also handles pandas Timestamp types  
✅ **Maintained**: No breaking changes to existing functionality  
✅ **Performance**: Minimal overhead (conversion is fast)  

## Prevention

### For Future Development

When saving data to MongoDB, always convert numpy/pandas types:

```python
# ❌ Bad - will fail
collection.insert_one({
    'count': df['column'].sum(),  # Returns np.int64
    'percent': (value / total * 100)  # Returns np.float64
})

# ✅ Good - explicitly convert
collection.insert_one({
    'count': int(df['column'].sum()),
    'percent': float(value / total * 100)
})

# ✅ Best - use helper function for complex objects
data = {'count': df['column'].sum(), 'percent': pct}
clean_data = convert_numpy_types(data)
collection.insert_one(clean_data)
```

### Common NumPy/Pandas Types to Watch

| Type | Convert To |
|------|-----------|
| `np.int64`, `np.int32` | `int()` |
| `np.float64`, `np.float32` | `float()` |
| `np.ndarray` | `.tolist()` |
| `pd.Timestamp` | `.to_pydatetime()` |
| `pd.Series` | `.to_list()` or `.values.tolist()` |

## Related Issues

This fix also resolves potential issues with:
- Saving indicators with numpy-type values
- Async batch processing breadth calculations
- Any future MongoDB save operations with pandas DataFrames

---

**Fixed Date**: December 9, 2025  
**Status**: ✅ Resolved  
**Impact**: Critical (blocking feature)  
**Effort**: Low (1 helper function + 4 call sites)
