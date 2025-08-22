#!/usr/bin/env python3
"""
Unit tests for logger and data cleaner modules.

This module tests the functionality of:
- Log Manager
- Data Cleaner
- Structured Logging
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import tempfile
import json
import logging

# Add scripts directory to path
sys.path.append('scripts')

from logger import LogManager, LogLevel, LogCategory, LogEntry, StructuredFormatter, JSONFormatter
from data_cleaner import DataCleaner, DataQuality, OutlierMethod, DataQualityReport, create_data_cleaner


class TestLogManager(unittest.TestCase):
    """Test cases for Log Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'log_level': 'INFO',
            'log_dir': 'test_logs',
            'max_file_size': 1024 * 1024,  # 1MB
            'backup_count': 3,
            'enable_console': True,
            'enable_file': True,
            'enable_json': False,
            'enable_network': False
        }
        
        self.log_manager = LogManager(self.config)
        
        # Create sample log entry
        self.sample_log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            message='Test log message',
            source='TestModule',
            thread_id=12345,
            process_id=67890,
            data={'test_key': 'test_value'},
            exception=None,
            stack_trace=None
        )
    
    def test_log_manager_initialization(self):
        """Test Log Manager initialization."""
        self.assertIsNotNone(self.log_manager)
        self.assertIsInstance(self.log_manager, LogManager)
        self.assertEqual(self.log_manager.log_level, logging.INFO)
        self.assertEqual(self.log_manager.log_dir, 'test_logs')
        self.assertEqual(self.log_manager.max_file_size, 1024 * 1024)
    
    def test_log_level_enumeration(self):
        """Test log level enumeration."""
        # Test all log levels
        self.assertEqual(LogLevel.DEBUG.value, "DEBUG")
        self.assertEqual(LogLevel.INFO.value, "INFO")
        self.assertEqual(LogLevel.WARNING.value, "WARNING")
        self.assertEqual(LogLevel.ERROR.value, "ERROR")
        self.assertEqual(LogLevel.CRITICAL.value, "CRITICAL")
    
    def test_log_category_enumeration(self):
        """Test log category enumeration."""
        # Test all log categories
        self.assertEqual(LogCategory.SYSTEM.value, "SYSTEM")
        self.assertEqual(LogCategory.TRADING.value, "TRADING")
        self.assertEqual(LogCategory.AI_MODELS.value, "AI_MODELS")
        self.assertEqual(LogCategory.RISK_MANAGEMENT.value, "RISK_MANAGEMENT")
        self.assertEqual(LogCategory.EXECUTION.value, "EXECUTION")
        self.assertEqual(LogCategory.PERFORMANCE.value, "PERFORMANCE")
        self.assertEqual(LogCategory.ALERTS.value, "ALERTS")
        self.assertEqual(LogCategory.API.value, "API")
        self.assertEqual(LogCategory.DATABASE.value, "DATABASE")
        self.assertEqual(LogCategory.NETWORK.value, "NETWORK")
    
    def test_log_entry_creation(self):
        """Test log entry creation."""
        try:
            # Test log entry properties
            self.assertIsInstance(self.sample_log_entry, LogEntry)
            self.assertEqual(self.sample_log_entry.level, LogLevel.INFO)
            self.assertEqual(self.sample_log_entry.category, LogCategory.SYSTEM)
            self.assertEqual(self.sample_log_entry.message, 'Test log message')
            self.assertEqual(self.sample_log_entry.source, 'TestModule')
            self.assertEqual(self.sample_log_entry.thread_id, 12345)
            self.assertEqual(self.sample_log_entry.process_id, 67890)
            
            # Test data field
            self.assertIn('test_key', self.sample_log_entry.data)
            self.assertEqual(self.sample_log_entry.data['test_key'], 'test_value')
            
        except Exception as e:
            self.fail(f"Log entry creation failed: {e}")
    
    def test_logger_retrieval(self):
        """Test logger retrieval functionality."""
        try:
            # Get logger for different categories
            system_logger = self.log_manager.get_logger('system', LogCategory.SYSTEM)
            trading_logger = self.log_manager.get_logger('trading', LogCategory.TRADING)
            ai_logger = self.log_manager.get_logger('ai_models', LogCategory.AI_MODELS)
            
            # Check that loggers are created
            self.assertIsNotNone(system_logger)
            self.assertIsNotNone(trading_logger)
            self.assertIsNotNone(ai_logger)
            
            # Check that loggers are instances of logging.Logger
            self.assertIsInstance(system_logger, logging.Logger)
            self.assertIsInstance(trading_logger, logging.Logger)
            self.assertIsInstance(ai_logger, logging.Logger)
            
        except Exception as e:
            self.fail(f"Logger retrieval failed: {e}")
    
    def test_logging_functionality(self):
        """Test basic logging functionality."""
        try:
            # Get a logger
            test_logger = self.log_manager.get_logger('test', LogCategory.SYSTEM)
            
            # Test different log levels
            test_logger.debug('Debug message')
            test_logger.info('Info message')
            test_logger.warning('Warning message')
            test_logger.error('Error message')
            
            # Check log counts
            self.assertGreaterEqual(self.log_manager.log_counts['INFO'], 1)
            self.assertGreaterEqual(self.log_manager.log_counts['WARNING'], 1)
            self.assertGreaterEqual(self.log_manager.log_counts['ERROR'], 1)
            
        except Exception as e:
            self.fail(f"Logging functionality failed: {e}")
    
    def test_structured_formatter(self):
        """Test structured formatter."""
        try:
            # Create formatter
            formatter = StructuredFormatter(include_timestamp=True, include_thread=True)
            
            # Create a log record
            record = logging.LogRecord(
                name='test',
                level=logging.INFO,
                pathname='test.py',
                lineno=1,
                msg='Test message',
                args=(),
                exc_info=None
            )
            
            # Add custom attributes
            record.category = LogCategory.SYSTEM
            record.source = 'TestModule'
            record.threadName = 'MainThread'
            
            # Format the record
            formatted = formatter.format(record)
            
            # Check that formatting worked
            self.assertIsInstance(formatted, str)
            self.assertIn('Test message', formatted)
            self.assertIn('SYSTEM', formatted)
            self.assertIn('TestModule', formatted)
            
        except Exception as e:
            self.fail(f"Structured formatter failed: {e}")
    
    def test_json_formatter(self):
        """Test JSON formatter."""
        try:
            # Create formatter
            formatter = JSONFormatter()
            
            # Create a log record
            record = logging.LogRecord(
                name='test',
                level=logging.INFO,
                pathname='test.py',
                lineno=1,
                msg='Test message',
                args=(),
                exc_info=None
            )
            
            # Add custom attributes
            record.category = LogCategory.SYSTEM
            record.source = 'TestModule'
            
            # Format the record
            formatted = formatter.format(record)
            
            # Check that formatting worked
            self.assertIsInstance(formatted, str)
            
            # Try to parse as JSON
            parsed = json.loads(formatted)
            self.assertIn('message', parsed)
            self.assertEqual(parsed['message'], 'Test message')
            self.assertEqual(parsed['level'], 'INFO')
            
        except Exception as e:
            self.fail(f"JSON formatter failed: {e}")


class TestDataCleaner(unittest.TestCase):
    """Test cases for Data Cleaner."""
    
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
        """Test Data Cleaner initialization."""
        self.assertIsNotNone(self.cleaner)
        self.assertIsInstance(self.cleaner, DataCleaner)
        self.assertEqual(self.cleaner.outlier_method.value, 'IQR')
        self.assertEqual(self.cleaner.outlier_threshold, 3.0)
        self.assertEqual(self.cleaner.iqr_multiplier, 1.5)
    
    def test_outlier_method_enumeration(self):
        """Test outlier method enumeration."""
        # Test all outlier methods
        self.assertEqual(OutlierMethod.IQR.value, "IQR")
        self.assertEqual(OutlierMethod.ZSCORE.value, "ZSCORE")
        self.assertEqual(OutlierMethod.ISOLATION_FOREST.value, "ISOLATION_FORELER_FACTOR")
        self.assertEqual(OutlierMethod.DBSCAN.value, "DBSCAN")
    
    def test_data_quality_enumeration(self):
        """Test data quality enumeration."""
        # Test all data quality levels
        self.assertEqual(DataQuality.EXCELLENT.value, "EXCELLENT")
        self.assertEqual(DataQuality.GOOD.value, "GOOD")
        self.assertEqual(DataQuality.FAIR.value, "FAIR")
        self.assertEqual(DataQuality.POOR.value, "POOR")
        self.assertEqual(DataQuality.UNUSABLE.value, "UNUSABLE")
    
    def test_basic_structure_validation(self):
        """Test basic structure validation."""
        try:
            # Test with valid data
            validated_data = self.cleaner._validate_basic_structure(self.sample_data)
            self.assertIsInstance(validated_data, pd.DataFrame)
            self.assertEqual(len(validated_data), len(self.sample_data))
            
        except Exception as e:
            self.fail(f"Basic structure validation failed: {e}")
    
    def test_missing_values_handling(self):
        """Test missing values handling."""
        try:
            # Handle missing values
            cleaned_data = self.cleaner._handle_missing_values(self.sample_data)
            
            # Check that data was cleaned
            self.assertIsInstance(cleaned_data, pd.DataFrame)
            
            # Check that no NaN values remain in OHLC columns
            ohlc_columns = ['open', 'high', 'low', 'close']
            for col in ohlc_columns:
                self.assertEqual(cleaned_data[col].isnull().sum(), 0)
            
        except Exception as e:
            self.fail(f"Missing values handling failed: {e}")
    
    def test_outlier_detection_iqr(self):
        """Test IQR outlier detection."""
        try:
            # Test IQR method
            outliers = self.cleaner._detect_outliers_iqr(self.sample_data['close'])
            
            # Check that outliers were detected
            self.assertIsInstance(outliers, pd.Series)
            self.assertEqual(len(outliers), len(self.sample_data))
            
            # Check that outliers are boolean
            self.assertTrue(all(isinstance(x, bool) for x in outliers))
            
        except Exception as e:
            self.fail(f"IQR outlier detection failed: {e}")
    
    def test_outlier_detection_zscore(self):
        """Test Z-score outlier detection."""
        try:
            # Test Z-score method
            outliers = self.cleaner._detect_outliers_zscore(self.sample_data['close'])
            
            # Check that outliers were detected
            self.assertIsInstance(outliers, pd.Series)
            self.assertEqual(len(outliers), len(self.sample_data))
            
            # Check that outliers are boolean
            self.assertTrue(all(isinstance(x, bool) for x in outliers))
            
        except Exception as e:
            self.fail(f"Z-score outlier detection failed: {e}")
    
    def test_duplicate_removal(self):
        """Test duplicate removal."""
        try:
            # Remove duplicates
            cleaned_data = self.cleaner._remove_duplicates(self.sample_data)
            
            # Check that duplicates were removed
            self.assertIsInstance(cleaned_data, pd.DataFrame)
            self.assertLessEqual(len(cleaned_data), len(self.sample_data))
            
            # Check that no duplicates remain
            self.assertEqual(cleaned_data.duplicated().sum(), 0)
            
        except Exception as e:
            self.fail(f"Duplicate removal failed: {e}")
    
    def test_data_quality_assessment(self):
        """Test data quality assessment."""
        try:
            # Assess data quality
            quality_report = self.cleaner.assess_data_quality(self.sample_data)
            
            # Check quality report
            self.assertIsNotNone(quality_report)
            self.assertIsInstance(quality_report, DataQualityReport)
            self.assertIsInstance(quality_report.quality_score, float)
            self.assertGreaterEqual(quality_report.quality_score, 0)
            self.assertLessEqual(quality_report.quality_score, 100)
            
            # Check report fields
            self.assertIsInstance(quality_report.total_rows, int)
            self.assertIsInstance(quality_report.total_columns, int)
            self.assertIsInstance(quality_report.missing_values, dict)
            self.assertIsInstance(quality_report.quality_level, DataQuality)
            
        except Exception as e:
            self.fail(f"Data quality assessment failed: {e}")
    
    def test_full_cleaning_pipeline(self):
        """Test the complete data cleaning pipeline."""
        try:
            # Run full cleaning pipeline
            cleaned_data = self.cleaner.clean_data(self.sample_data, "XAUUSD")
            
            # Check that data was cleaned
            self.assertIsInstance(cleaned_data, pd.DataFrame)
            self.assertGreater(len(cleaned_data), 0)
            
            # Check that data is properly sorted
            self.assertTrue(cleaned_data.index.is_monotonic_increasing)
            
            # Check that no duplicates exist
            self.assertEqual(cleaned_data.duplicated().sum(), 0)
            
            # Check that no NaN values exist in OHLC columns
            ohlc_columns = ['open', 'high', 'low', 'close']
            for col in ohlc_columns:
                if col in cleaned_data.columns:
                    self.assertEqual(cleaned_data[col].isnull().sum(), 0)
            
        except Exception as e:
            self.fail(f"Full cleaning pipeline failed: {e}")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestLogManager))
    test_suite.addTest(unittest.makeSuite(TestDataCleaner))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running SignaMentis Logger & Data Cleaner Tests...")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    print(f"\nTest execution completed with {'success' if success else 'failure'}.")
