"""
TA-Lib setup for Streamlit Cloud deployment.
This script downloads, compiles, and installs TA-Lib at runtime.
Based on: https://discuss.streamlit.io/t/ta-lib-streamlit-deploy-error/7643/7
"""
import os
import sys
import requests
from pathlib import Path

def setup_talib():
    """
    Download and compile TA-Lib for Streamlit Cloud.
    This only runs once - checks if already installed.
    """
    # Check if already installed
    talib_dir = Path("/tmp/ta-lib")
    lib_file = Path("/home/appuser/lib/libta_lib.so.0")
    
    if talib_dir.exists() and lib_file.exists():
        print("✓ TA-Lib already installed")
        # Add library to environment
        from ctypes import CDLL
        CDLL(str(lib_file))
        return True
    
    print("📦 Installing TA-Lib (first run, takes ~2 minutes)...")
    
    try:
        # Download TA-Lib source
        print("  Downloading TA-Lib source...")
        tar_path = "/tmp/ta-lib-0.4.0-src.tar.gz"
        
        if not Path(tar_path).exists():
            response = requests.get(
                "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz",
                timeout=60
            )
            with open(tar_path, "wb") as f:
                f.write(response.content)
        
        # Save current directory
        original_dir = os.getcwd()
        
        # Extract
        print("  Extracting...")
        os.chdir("/tmp")
        os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz > /dev/null 2>&1")
        
        # Configure and build
        print("  Configuring...")
        os.chdir("/tmp/ta-lib")
        os.system("./configure --prefix=/home/appuser > /dev/null 2>&1")
        
        print("  Compiling (this takes a minute)...")
        os.system("make > /dev/null 2>&1")
        
        print("  Installing...")
        os.system("make install > /dev/null 2>&1")
        
        # Install Python wrapper
        print("  Installing Python wrapper...")
        os.system(
            'pip3 install --no-cache-dir '
            '--global-option=build_ext '
            '--global-option="-L/home/appuser/lib/" '
            '--global-option="-I/home/appuser/include/" '
            'TA-Lib > /dev/null 2>&1'
        )
        
        # Return to original directory
        os.chdir(original_dir)
        
        # Load library
        from ctypes import CDLL
        CDLL(str(lib_file))
        
        print("✓ TA-Lib installation complete!")
        sys.stdout.flush()
        return True
        
    except Exception as e:
        print(f"⚠️ TA-Lib installation failed: {e}")
        print("  Falling back to pandas-based indicators")
        return False

def check_talib_available():
    """Check if TA-Lib is available and working."""
    try:
        import talib
        # Test if it actually works
        import numpy as np
        test_data = np.random.random(100)
        _ = talib.SMA(test_data, 10)
        return True
    except (ImportError, Exception):
        return False

if __name__ == "__main__":
    # Run setup
    setup_talib()
    
    # Test installation
    if check_talib_available():
        print("✓ TA-Lib is working correctly")
    else:
        print("✗ TA-Lib test failed")
