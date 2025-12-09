# TA-Lib Installation for Streamlit Cloud

## ✅ Recommended: Runtime Compilation (Implemented)

The project now uses **runtime compilation** which automatically installs TA-Lib when deployed to Streamlit Cloud.

### How It Works

1. **`setup_talib.py`** - Downloads and compiles TA-Lib at runtime (first run only)
2. **`utils/indicators.py`** - Automatically calls setup on Streamlit Cloud
3. **`packages.txt`** - Provides build tools (gcc, make)
4. **`requirements.txt`** - Includes requests for downloading

### Files Required

- ✅ `setup_talib.py` (runtime installer)
- ✅ `packages.txt` (build-essential, gcc, make, wget)
- ✅ `requirements.txt` (requests library)
- ✅ `utils/indicators.py` (automatic setup detection)

### Deployment Steps

1. Push all files to GitHub
2. Deploy to Streamlit Cloud
3. First run takes ~2 minutes (compiling TA-Lib)
4. Subsequent runs are instant (uses cached build)

### Fallback Behavior

If TA-Lib installation fails, the app automatically uses pandas-based indicator calculations (slower but functional).

## Alternative Options

### Option 1: Pre-compiled Binary (Unreliable)

```txt
# requirements_cloud.txt
ta-lib-bin>=0.4.28
```

⚠️ May not work on all Streamlit Cloud configurations.

### Option 2: Manual Build (Current Implementation)

Based on: https://discuss.streamlit.io/t/ta-lib-streamlit-deploy-error/7643/7

- Downloads TA-Lib source from SourceForge
- Compiles using gcc/make
- Installs to `/home/appuser` (user directory)
- Loads shared library at runtime

## Testing Locally

```bash
# Test the setup script
python setup_talib.py

# Run Streamlit
streamlit run ta_dashboard.py
```

## Troubleshooting

### Build Fails on Streamlit Cloud

Check the logs for:
- `gcc` and `make` availability
- Download timeouts (SourceForge)
- Disk space issues

### TA-Lib Not Working After Install

The app will show: "⚠️ TA-Lib not installed. Using pandas fallback"

This is expected and the app will still work with pandas calculations.

### Performance Without TA-Lib

- RSI/MACD calculations: ~10-50x slower
- Still acceptable for most use cases
- Consider reducing historical data range
