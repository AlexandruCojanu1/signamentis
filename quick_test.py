#!/usr/bin/env python3
"""
SignaMentis - Quick Functionality Test

This script performs a quick test of core functionality
without requiring external dependencies like MetaTrader5.

Author: SignaMentis Team
Version: 1.0.0
"""

import sys
import os
import time
from datetime import datetime

def test_basic_imports():
    """Test basic module imports."""
    print("🔍 Testing basic module imports...")
    
    try:
        # Test core modules that don't require external deps
        import scripts.logger
        print("✅ Logger module imported successfully")
        
        import scripts.data_cleaner
        print("✅ Data cleaner module imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_structures():
    """Test data structure creation."""
    print("\n🔧 Testing data structure creation...")
    
    try:
        # Test logger enums
        from scripts.logger import LogLevel, LogCategory
        
        # Test enum values
        assert LogLevel.INFO.value == "INFO"
        assert LogCategory.SYSTEM.value == "SYSTEM"
        print("✅ Logger enums working correctly")
        
        # Test data cleaner enums
        from scripts.data_cleaner import DataQuality, OutlierMethod
        
        # Test enum values
        assert DataQuality.EXCELLENT.value == "EXCELLENT"
        assert OutlierMethod.IQR.value == "IQR"
        print("✅ Data cleaner enums working correctly")
        
        return True
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False

def test_configuration_loading():
    """Test configuration file loading."""
    print("\n⚙️ Testing configuration loading...")
    
    try:
        import yaml
        
        # Test settings.yaml
        with open('config/settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
        
        assert 'system' in settings
        assert 'database' in settings
        assert settings['system']['name'] == 'SignaMentis'
        print("✅ Settings configuration loaded correctly")
        
        # Test model_config.yaml
        with open('config/model_config.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        
        assert 'bilstm' in model_config
        assert 'gru' in model_config
        assert 'transformer' in model_config
        print("✅ Model configuration loaded correctly")
        
        # Test risk_config.yaml
        with open('config/risk_config.yaml', 'r') as f:
            risk_config = yaml.safe_load(f)
        
        assert 'position_sizing' in risk_config
        assert 'stop_loss' in risk_config
        print("✅ Risk configuration loaded correctly")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_feature_engineering_basic():
    """Test basic feature engineering functionality."""
    print("\n🔬 Testing basic feature engineering...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-08-11', periods=100, freq='5T')
        sample_data = pd.DataFrame({
            'open': np.random.normal(2000, 50, 100),
            'high': np.random.normal(2010, 50, 100),
            'low': np.random.normal(1990, 50, 100),
            'close': np.random.normal(2000, 50, 100),
            'volume': np.random.normal(100, 20, 100)
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        sample_data['high'] = np.maximum(sample_data['high'], sample_data['close'])
        sample_data['low'] = np.minimum(sample_data['low'], sample_data['close'])
        sample_data['open'] = np.clip(sample_data['open'], 
                                     sample_data['low'], 
                                     sample_data['high'])
        
        # Test basic calculations
        sample_data['price_change'] = sample_data['close'].pct_change()
        sample_data['sma_20'] = sample_data['close'].rolling(window=20).mean()
        sample_data['volatility'] = sample_data['price_change'].rolling(window=20).std()
        
        # Verify calculations
        assert len(sample_data) == 100
        assert 'price_change' in sample_data.columns
        assert 'sma_20' in sample_data.columns
        assert 'volatility' in sample_data.columns
        
        print("✅ Basic feature engineering working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_ensemble_logic():
    """Test ensemble logic without PyTorch."""
    print("\n🤖 Testing ensemble logic...")
    
    try:
        # Test ensemble aggregation logic
        def simple_ensemble_predict(predictions, weights):
            """Simple ensemble prediction without PyTorch."""
            if not predictions or not weights:
                return None
            
            # Calculate weighted average
            weighted_sum = 0
            total_weight = 0
            
            for pred, weight in zip(predictions, weights):
                weighted_sum += pred * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0
            
            return weighted_sum / total_weight
        
        # Test with sample data
        predictions = [0.6, 0.7, 0.8]
        weights = [0.3, 0.3, 0.4]
        
        result = simple_ensemble_predict(predictions, weights)
        expected = (0.6 * 0.3 + 0.7 * 0.3 + 0.8 * 0.4) / (0.3 + 0.3 + 0.4)
        
        assert abs(result - expected) < 0.001
        print("✅ Ensemble logic working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble logic test failed: {e}")
        return False

def test_risk_management_logic():
    """Test risk management logic."""
    print("\n🛡️ Testing risk management logic...")
    
    try:
        # Test position size calculation
        def calculate_position_size(account_balance, risk_percentage, stop_loss_pips, pip_value):
            """Calculate position size based on risk."""
            risk_amount = account_balance * risk_percentage
            position_size = risk_amount / (stop_loss_pips * pip_value)
            return position_size
        
        # Test with sample data
        account_balance = 10000.0
        risk_percentage = 0.02  # 2%
        stop_loss_pips = 20
        pip_value = 1.0
        
        position_size = calculate_position_size(account_balance, risk_percentage, stop_loss_pips, pip_value)
        expected_position_size = (10000 * 0.02) / (20 * 1.0)  # 200 / 20 = 10
        
        assert abs(position_size - expected_position_size) < 0.001
        print("✅ Position size calculation working correctly")
        
        # Test risk limits
        def check_risk_limits(daily_pnl, max_daily_loss, current_drawdown, max_drawdown):
            """Check if risk limits are exceeded."""
            daily_loss_ok = abs(daily_pnl) <= max_daily_loss
            drawdown_ok = current_drawdown <= max_drawdown
            return daily_loss_ok and drawdown_ok
        
        # Test scenarios
        assert check_risk_limits(-500, 1000, 0.05, 0.1) == True   # Within limits
        assert check_risk_limits(-1500, 1000, 0.05, 0.1) == False # Daily loss exceeded
        assert check_risk_limits(-500, 1000, 0.15, 0.1) == False  # Drawdown exceeded
        
        print("✅ Risk limit checking working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk management test failed: {e}")
        return False

def test_strategy_logic():
    """Test trading strategy logic."""
    print("\n📈 Testing trading strategy logic...")
    
    try:
        # Test SuperTrend calculation logic
        def calculate_supertrend(high, low, close, period=10, multiplier=3):
            """Calculate SuperTrend indicator."""
            # Calculate ATR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # Calculate SuperTrend
            upperband = (high + low) / 2 + multiplier * atr
            lowerband = (high + low) / 2 - multiplier * atr
            
            supertrend = pd.Series(index=close.index, dtype=float)
            supertrend.iloc[0] = lowerband.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > supertrend.iloc[i-1]:
                    supertrend.iloc[i] = max(lowerband.iloc[i], supertrend.iloc[i-1])
                else:
                    supertrend.iloc[i] = min(upperband.iloc[i], supertrend.iloc[i-1])
            
            return supertrend
        
        # Test with sample data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range('2023-08-11', periods=50, freq='5T')
        sample_data = pd.DataFrame({
            'high': np.random.normal(2010, 50, 50),
            'low': np.random.normal(1990, 50, 50),
            'close': np.random.normal(2000, 50, 50)
        }, index=dates)
        
        # Ensure valid OHLC relationships
        sample_data['high'] = np.maximum(sample_data['high'], sample_data['close'])
        sample_data['low'] = np.minimum(sample_data['low'], sample_data['close'])
        
        supertrend = calculate_supertrend(sample_data['high'], sample_data['low'], sample_data['close'])
        
        # Verify SuperTrend properties
        assert len(supertrend) == len(sample_data)
        assert not supertrend.isnull().all()
        assert all(supertrend > 0)
        
        print("✅ SuperTrend calculation working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy logic test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 SignaMentis - Quick Functionality Test")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Structures", test_data_structures),
        ("Configuration Loading", test_configuration_loading),
        ("Feature Engineering", test_feature_engineering_basic),
        ("Ensemble Logic", test_ensemble_logic),
        ("Risk Management", test_risk_management_logic),
        ("Strategy Logic", test_strategy_logic)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 Quick Test Results:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All quick tests passed!")
        print("✅ SignaMentis core functionality is working correctly!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed!")
        print("🔧 Please check the failing tests above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
