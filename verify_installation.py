"""
Market Breadth Implementation Verification Script
Run this to verify all components are installed correctly.
"""
import sys
from pathlib import Path

def print_status(check_name, passed, message=""):
    status = "✅" if passed else "❌"
    print(f"{status} {check_name}")
    if message:
        print(f"   {message}")
    return passed

def main():
    print("=" * 70)
    print("Market Breadth Implementation Verification")
    print("=" * 70)
    print()
    
    all_checks_passed = True
    
    # Check Python version
    print("1. Python Environment")
    print("-" * 70)
    py_version = sys.version_info
    py_ok = py_version.major == 3 and py_version.minor >= 7
    all_checks_passed &= print_status(
        "Python Version", 
        py_ok,
        f"Python {py_version.major}.{py_version.minor}.{py_version.micro} (need 3.7+)"
    )
    print()
    
    # Check file structure
    print("2. File Structure")
    print("-" * 70)
    files_to_check = [
        "utils/__init__.py",
        "utils/indicators.py",
        "utils/macd_stage.py",
        "utils/db_async.py",
        "pages/1_📊_Market_Breadth.py",
        "ta_dashboard.py",
        "db_adapter.py",
        "MARKET_BREADTH_README.md",
        "QUICKSTART.md"
    ]
    
    for file_path in files_to_check:
        exists = Path(file_path).exists()
        all_checks_passed &= print_status(f"File: {file_path}", exists)
    print()
    
    # Check core dependencies
    print("3. Core Dependencies")
    print("-" * 70)
    
    deps = {
        "streamlit": "Streamlit framework",
        "pandas": "Data manipulation",
        "numpy": "Numerical operations",
        "plotly": "Interactive charts",
        "pymongo": "MongoDB driver",
        "dotenv": "Environment variables"
    }
    
    for module, description in deps.items():
        try:
            if module == "dotenv":
                __import__("dotenv")
            else:
                __import__(module)
            all_checks_passed &= print_status(f"{module}", True, description)
        except ImportError:
            all_checks_passed &= print_status(f"{module}", False, f"MISSING - {description}")
    print()
    
    # Check optional dependencies
    print("4. Optional Dependencies")
    print("-" * 70)
    
    optional_deps = {
        "talib": "TA-Lib (10-50x faster indicators)",
        "motor": "Async MongoDB support",
        "aiohttp": "Async HTTP client"
    }
    
    optional_status = {}
    for module, description in optional_deps.items():
        try:
            __import__(module)
            optional_status[module] = True
            print_status(f"{module}", True, description)
        except ImportError:
            optional_status[module] = False
            print_status(f"{module}", False, f"Not installed - {description}")
    print()
    
    # Check utility modules
    print("5. Utility Modules")
    print("-" * 70)
    
    try:
        from utils.indicators import calculate_ema, calculate_rsi, calculate_macd
        all_checks_passed &= print_status("utils.indicators", True, "Indicator calculations")
    except Exception as e:
        all_checks_passed &= print_status("utils.indicators", False, str(e))
    
    try:
        from utils.macd_stage import detect_macd_stage, categorize_macd_stage
        all_checks_passed &= print_status("utils.macd_stage", True, "MACD stage detection")
    except Exception as e:
        all_checks_passed &= print_status("utils.macd_stage", False, str(e))
    
    try:
        from utils.db_async import get_sync_db_adapter
        all_checks_passed &= print_status("utils.db_async", True, "Async database operations")
    except Exception as e:
        all_checks_passed &= print_status("utils.db_async", False, str(e))
    print()
    
    # Check database connectivity (optional)
    print("6. Database Connectivity (Optional)")
    print("-" * 70)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from utils.db_async import get_sync_db_adapter
        db = get_sync_db_adapter()
        tickers = db.get_all_tickers()
        print_status("Database connection", True, f"Connected - {len(tickers)} tickers found")
        db.close()
    except Exception as e:
        print_status("Database connection", False, f"Not configured or error: {str(e)[:50]}")
    print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    if all_checks_passed:
        print("✅ All critical checks passed!")
        print()
        print("Your Market Breadth implementation is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run: streamlit run ta_dashboard.py")
        print("  2. Navigate to '📊 Market Breadth' page")
        print("  3. Calculate historical breadth data")
        print()
    else:
        print("❌ Some checks failed. Please review the errors above.")
        print()
        print("Common fixes:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Ensure all files are in the correct location")
        print("  3. Check Python version (need 3.7+)")
        print()
    
    # Optional enhancements
    if not all(optional_status.values()):
        print("💡 Recommended Enhancements:")
        if not optional_status.get('talib', False):
            print("  • Install TA-Lib for 10-50x faster indicators")
            print("    See QUICKSTART.md for installation instructions")
        if not optional_status.get('motor', False):
            print("  • Install motor for async MongoDB support (faster calculations)")
            print("    Run: pip install motor")
        print()
    
    print("=" * 70)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
