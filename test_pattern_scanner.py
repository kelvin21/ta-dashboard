"""
Test script for Pattern Scanner functionality
Run this after installing scipy: pip install scipy
"""
import sys
from pathlib import Path

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

def test_pattern_detector():
    """Test basic pattern detection functionality."""
    print("Testing Pattern Detector...")
    
    try:
        from utils.pattern_detection import PatternDetector, rank_patterns_by_quality
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        print("✓ Imports successful")
        
        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
        
        # Create a simple head and shoulders pattern
        prices = []
        for i in range(180):
            if i < 30:
                prices.append(100 + i * 0.5)  # Left shoulder rise
            elif i < 60:
                prices.append(115 - (i-30) * 0.3)  # Left shoulder fall
            elif i < 90:
                prices.append(106 + (i-60) * 0.6)  # Head rise
            elif i < 120:
                prices.append(124 - (i-90) * 0.5)  # Head fall
            elif i < 150:
                prices.append(109 + (i-120) * 0.4)  # Right shoulder rise
            else:
                prices.append(121 - (i-150) * 0.3)  # Break down
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 180)
        })
        
        print(f"✓ Created test data: {len(df)} days")
        
        # Initialize detector
        detector = PatternDetector(min_pattern_days=30, max_pattern_days=540)
        print("✓ Initialized PatternDetector")
        
        # Detect patterns
        patterns = detector.detect_all_patterns(df, 'TEST')
        print(f"✓ Pattern detection ran successfully")
        print(f"  Found {len(patterns)} patterns")
        
        if patterns:
            print("\n📊 Detected Patterns:")
            for p in patterns:
                print(f"  - {p['pattern']} ({p['signal']})")
                print(f"    Current: {p['current_price']:.2f}")
                print(f"    Target: {p['target_price']:.2f}")
                print(f"    Stop: {p['stop_loss']:.2f}")
                print(f"    R/R: 1:{p['risk_reward']:.2f}")
                print(f"    Quality: {p.get('quality_score', 0):.2f}")
        
        # Test timeframe targets
        if patterns:
            print("\n⏱️ Testing timeframe calculations...")
            targets = detector.calculate_timeframe_targets(patterns[0], df)
            print(f"  1-3 days: {targets['target_1_3_days']}")
            print(f"  1 month: {targets['target_1_month']}")
            print(f"  Full: {targets['target_full']}")
        
        # Test ranking
        print("\n🏆 Testing pattern ranking...")
        ranked = rank_patterns_by_quality(patterns)
        print(f"  Ranked {len(ranked)} patterns")
        
        print("\n✅ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Please install scipy: pip install scipy")
        return False
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_integration():
    """Test database integration."""
    print("\n" + "="*50)
    print("Testing Database Integration...")
    
    try:
        from utils.db_async import get_sync_db_adapter
        
        db = get_sync_db_adapter()
        print("✓ Database adapter initialized")
        
        tickers = db.get_all_tickers()
        print(f"✓ Found {len(tickers)} tickers in database")
        
        if tickers:
            sample_ticker = tickers[0]
            print(f"  Testing with ticker: {sample_ticker}")
            
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            df = db.get_price_data(sample_ticker, start_date, end_date)
            print(f"✓ Retrieved {len(df)} days of data for {sample_ticker}")
            
            if len(df) >= 30:
                from utils.pattern_detection import PatternDetector
                
                detector = PatternDetector()
                patterns = detector.detect_all_patterns(df, sample_ticker)
                
                print(f"✓ Pattern detection on real data: {len(patterns)} patterns found")
                
                if patterns:
                    print("\n📊 Sample Pattern:")
                    p = patterns[0]
                    print(f"  Ticker: {p['ticker']}")
                    print(f"  Pattern: {p['pattern']}")
                    print(f"  Signal: {p['signal']}")
                    print(f"  Confidence: {p.get('confidence', 0)*100:.0f}%")
            else:
                print("⚠️ Not enough data for pattern detection")
        
        print("\n✅ Database integration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*50)
    print("Pattern Scanner Test Suite")
    print("="*50)
    
    # Run tests
    test1 = test_pattern_detector()
    test2 = test_database_integration()
    
    print("\n" + "="*50)
    if test1 and test2:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check output above.")
    print("="*50)
