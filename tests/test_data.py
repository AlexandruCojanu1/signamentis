#!/usr/bin/env python3
"""
Unit tests for data loading and processing modules.

This module tests the functionality of:
- DataLoader
- DataCleaner
- FeatureEngineer
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add scripts directory to path
sys.path.append('scripts')

from data_loader import DataLoader, DataConfig
from data_cleaner import DataCleaner, create_data_cleaner
from feature_engineering import FeatureEngineer


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DataConfig(
            symbol="XAUUSD",
            timeframe="M5",
            data_source="csv",
            raw_data_path="data/raw",
            processed_data_path="data/processed"
        )
        self.data_loader = DataLoader(self.config)
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'Local time': [
                '11.08.2023 01:00:00.175 GMT+0300',
                '11.08.2023 01:00:00.230 GMT+0300',
                '11.08.2023 01:00:00.285 GMT+0300'
            ],
            'Open': [1912.105, 1912.575, 1912.025],
            'High': [1912.105, 1912.575, 1912.025],
            'Low': [1912.105, 1912.575, 1912.025],
            'Close': [1912.105, 1912.575, 1912.025],
            'Volume': [0, 0, 0]
        })
        
        # Save sample data to temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        self.assertIsNotNone(self.data_loader)
        self.assertEqual(self.data_loader.config.symbol, "XAUUSD")
        self.assertEqual(self.data_loader.config.timeframe, "M5")
    
    def test_csv_data_loading(self):
        """Test CSV data loading functionality."""
        try:
            data = self.data_loader.load_csv_data(self.temp_file.name)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
            self.assertIn('open', data.columns)
            self.assertIn('high', data.columns)
            self.assertIn('low', data.columns)
            self.assertIn('close', data.columns)
            self.assertIn('volume', data.columns)
        except Exception as e:
            self.fail(f"CSV loading failed with error: {e}")
    
    def test_data_validation(self):
        """Test data validation functionality."""
        # Test with valid data
        valid_data = pd.DataFrame({
            'open': [1912.105, 1912.575, 1912.025],
            'high': [1912.105, 1912.575, 1912.025],
            'low': [1912.105, 1912.575, 1912.025],
            'close': [1912.105, 1912.575, 1912.025],
            'volume': [0, 0, 0]
        })
        
        # This should not raise an exception
        try:
            self.data_loader._validate_basic_structure(valid_data)
        except Exception as e:
            self.fail(f"Data validation failed with valid data: {e}")
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns."""
        invalid_data = pd.DataFrame({
            'open': [1912.105, 1912.575, 1912.025],
            'high': [1912.105, 1912.575, 1912.025]
            # Missing low, close, volume
        })
        
        with self.assertRaises(ValueError):
            self.data_loader._validate_basic_structure(invalid_data)


class TestDataCleaner(unittest.TestCase):
    """Test cases for DataCleaner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'outlier_method': 'IQR',
            'outlier_threshold': 3.0,
            'iqr_multiplier': 1.5,
            'min_data_quality': 0.7
        }
        self.cleaner = create_data_cleaner(self.config)
        
        # Create sample data with issues
        dates = pd.date_range('2023-08-11', periods=100, freq='5T')
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(2000, 50, 100),
            'high': np.random.normal(2010, 50, 100),
            'low': np.random.normal(1990, 50, 100),
            'close': np.random.normal(2000, 50, 100),
            'volume': np.random.normal(100, 20, 100)
        })
        
        # Add some data quality issues
        self.sample_data.loc[10:15, 'close'] = np.nan  # Missing values
        self.sample_data.loc[20, 'close'] = 5000  # Outlier
        self.sample_data.loc[30:35] = self.sample_data.iloc[10:15]  # Duplicates
    
    def test_data_cleaner_initialization(self):
        """Test DataCleaner initialization."""
        self.assertIsNotNone(self.cleaner)
        self.assertEqual(self.cleaner.outlier_method.value, 'IQR')
        self.assertEqual(self.cleaner.outlier_threshold, 3.0)
    
    def test_basic_structure_validation(self):
        """Test basic structure validation."""
        try:
            validated_data = self.cleaner._validate_basic_structure(self.sample_data)
            self.assertIsInstance(validated_data, pd.DataFrame)
        except Exception as e:
            self.fail(f"Basic structure validation failed: {e}")
    
    def test_missing_values_handling(self):
        """Test missing values handling."""
        try:
            cleaned_data = self.cleaner._handle_missing_values(self.sample_data)
            self.assertIsInstance(cleaned_data, pd.DataFrame)
            # Check that no NaN values remain in OHLC columns
            ohlc_columns = ['open', 'high', 'low', 'close']
            for col in ohlc_columns:
                self.assertEqual(cleaned_data[col].isnull().sum(), 0)
        except Exception as e:
            self.fail(f"Missing values handling failed: {e}")
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        try:
            # Test IQR method
            outliers = self.cleaner._detect_outliers_iqr(self.sample_data['close'])
            self.assertIsInstance(outliers, pd.Series)
            self.assertEqual(len(outliers), len(self.sample_data))
        except Exception as e:
            self.fail(f"Outlier detection failed: {e}")
    
    def test_data_quality_assessment(self):
        """Test data quality assessment."""
        try:
            quality_report = self.cleaner.assess_data_quality(self.sample_data)
            self.assertIsNotNone(quality_report)
            self.assertIsInstance(quality_report.quality_score, float)
            self.assertGreaterEqual(quality_report.quality_score, 0)
            self.assertLessEqual(quality_report.quality_score, 100)
        except Exception as e:
            self.fail(f"Data quality assessment failed: {e}")
    
    def test_full_cleaning_pipeline(self):
        """Test the complete data cleaning pipeline."""
        try:
            cleaned_data = self.cleaner.clean_data(self.sample_data, "XAUUSD")
            self.assertIsInstance(cleaned_data, pd.DataFrame)
            self.assertGreater(len(cleaned_data), 0)
            
            # Check that data is properly sorted
            self.assertTrue(cleaned_data.index.is_monotonic_increasing)
            
            # Check that no duplicates exist
            self.assertEqual(cleaned_data.duplicated().sum(), 0)
            
        except Exception as e:
            self.fail(f"Full cleaning pipeline failed: {e}")


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature_engineer = FeatureEngineer()
        
        # Create sample OHLCV data
        dates = pd.date_range('2023-08-11', periods=100, freq='5T')
        self.sample_data = pd.DataFrame({
            'open': np.random.normal(2000, 50, 100),
            'high': np.random.normal(2010, 50, 100),
            'low': np.random.normal(1990, 50, 100),
            'close': np.random.normal(2000, 50, 100),
            'volume': np.random.normal(100, 20, 100)
        }, index=dates)
    
    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initialization."""
        self.assertIsNotNone(self.feature_engineer)
    
    def test_technical_indicators_creation(self):
        """Test technical indicators creation."""
        try:
            features = self.feature_engineer.create_features(self.sample_data)
            self.assertIsInstance(features, pd.DataFrame)
            self.assertGreater(len(features.columns), len(self.sample_data.columns))
            
            # Check for specific technical indicators
            expected_indicators = ['sma_20', 'ema_20', 'rsi', 'macd', 'bollinger_upper']
            for indicator in expected_indicators:
                if indicator in features.columns:
                    self.assertIsInstance(features[indicator], pd.Series)
            
        except Exception as e:
            self.fail(f"Technical indicators creation failed: {e}")
    
    def test_moving_averages(self):
        """Test moving averages calculation."""
        try:
            # Test SMA
            sma_20 = self.feature_engineer._add_simple_moving_averages(self.sample_data)
            self.assertIn('sma_20', sma_20.columns)
            self.assertEqual(len(sma_20), len(self.sample_data))
            
            # Test EMA
            ema_20 = self.feature_engineer._add_exponential_moving_averages(self.sample_data)
            self.assertIn('ema_20', ema_20.columns)
            self.assertEqual(len(ema_20), len(self.sample_data))
            
        except Exception as e:
            self.fail(f"Moving averages calculation failed: {e}")
    
    def test_oscillators(self):
        """Test oscillators calculation."""
        try:
            # Test RSI
            rsi_data = self.feature_engineer._add_rsi(self.sample_data)
            self.assertIn('rsi', rsi_data.columns)
            
            # Check RSI bounds (should be between 0 and 100)
            rsi_values = rsi_data['rsi'].dropna()
            self.assertTrue(all(0 <= val <= 100 for val in rsi_values))
            
        except Exception as e:
            self.fail(f"Oscillators calculation failed: {e}")
    
    def test_volatility_features(self):
        """Test volatility features calculation."""
        try:
            volatility_data = self.feature_engineer._add_volatility_features(self.sample_data)
            
            # Check for volatility columns
            volatility_columns = [col for col in volatility_data.columns if 'volatility' in col.lower()]
            self.assertGreater(len(volatility_columns), 0)
            
        except Exception as e:
            self.fail(f"Volatility features calculation failed: {e}")
    
    def test_feature_validation(self):
        """Test feature validation."""
        try:
            features = self.feature_engineer.create_features(self.sample_data)
            
            # Check that no infinite values exist
            numeric_data = features.select_dtypes(include=[np.number])
            self.assertFalse(np.isinf(numeric_data).any().any())
            
            # Check that no extreme outliers exist (beyond 5 standard deviations)
            numeric_features = features.select_dtypes(include=[np.number])
            for col in numeric_features.columns:
                col_data = numeric_features[col].dropna()
                if len(col_data) > 0:
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    if std_val > 0:
                        z_scores = np.abs((col_data - mean_val) / std_val)
                        self.assertTrue(all(z_scores < 5))
            
        except Exception as e:
            self.fail(f"Feature validation failed: {e}")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataLoader))
    test_suite.addTest(unittest.makeSuite(TestDataCleaner))
    test_suite.addTest(unittest.makeSuite(TestFeatureEngineer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running SignaMentis Data Module Tests...")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    print(f"\nTest execution completed with {'success' if success else 'failure'}.")
