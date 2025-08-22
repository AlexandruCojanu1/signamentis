#!/usr/bin/env python3
"""
SignaMentis - Main Test Runner

This script runs all unit tests for the SignaMentis trading system.
It provides a comprehensive test suite covering all major components.

Author: SignaMentis Team
Version: 1.0.0
"""

import unittest
import sys
import os
import time
from datetime import datetime

# Add tests directory to path
sys.path.append('tests')

# Import all test modules
from tests.test_data import TestDataLoader, TestDataCleaner, TestFeatureEngineer
from tests.test_models import TestBiLSTMModel, TestGRUModel, TestTransformerModel, TestLNNModel, TestLTNModel, TestEnsembleManager
from tests.test_strategy import TestSuperTrendStrategy, TestRiskManager, TestBacktester
from tests.test_executor import TestMT5Executor, TestLiveDashboard, TestTradingAPI
from tests.test_logger import TestLogManager, TestDataCleaner as TestDataCleanerLogger


def create_test_suite():
    """Create comprehensive test suite."""
    test_suite = unittest.TestSuite()
    
    print("🔧 Creating test suite...")
    
    # Data Processing Tests
    print("  📊 Adding data processing tests...")
    test_suite.addTest(unittest.makeSuite(TestDataLoader))
    test_suite.addTest(unittest.makeSuite(TestDataCleaner))
    test_suite.addTest(unittest.makeSuite(TestFeatureEngineer))
    
    # AI Models Tests
    print("  🤖 Adding AI models tests...")
    test_suite.addTest(unittest.makeSuite(TestBiLSTMModel))
    test_suite.addTest(unittest.makeSuite(TestGRUModel))
    test_suite.addTest(unittest.makeSuite(TestTransformerModel))
    test_suite.addTest(unittest.makeSuite(TestLNNModel))
    test_suite.addTest(unittest.makeSuite(TestLTNModel))
    test_suite.addTest(unittest.makeSuite(TestEnsembleManager))
    
    # Strategy & Risk Management Tests
    print("  📈 Adding strategy & risk management tests...")
    test_suite.addTest(unittest.makeSuite(TestSuperTrendStrategy))
    test_suite.addTest(unittest.makeSuite(TestRiskManager))
    test_suite.addTest(unittest.makeSuite(TestBacktester))
    
    # Execution & Monitoring Tests
    print("  ⚡ Adding execution & monitoring tests...")
    test_suite.addTest(unittest.makeSuite(TestMT5Executor))
    test_suite.addTest(unittest.makeSuite(TestLiveDashboard))
    test_suite.addTest(unittest.makeSuite(TestTradingAPI))
    
    # Logger & Utilities Tests
    print("  📝 Adding logger & utilities tests...")
    test_suite.addTest(unittest.makeSuite(TestLogManager))
    test_suite.addTest(unittest.makeSuite(TestDataCleanerLogger))
    
    return test_suite


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    try:
        # Try to import coverage module
        import coverage
        
        print("📊 Running tests with coverage...")
        
        # Start coverage measurement
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        test_suite = create_test_suite()
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # Stop coverage measurement
        cov.stop()
        cov.save()
        
        # Generate coverage report
        print("\n📊 Coverage Report:")
        cov.report()
        
        # Save HTML report
        cov.html_report(directory='coverage_html')
        print("📁 HTML coverage report saved to 'coverage_html/' directory")
        
        return result.wasSuccessful()
        
    except ImportError:
        print("⚠️  Coverage module not available. Running tests without coverage...")
        return run_tests_basic()


def run_tests_basic():
    """Run tests without coverage."""
    print("🧪 Running tests without coverage...")
    
    test_suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def run_specific_test_category(category):
    """Run tests for a specific category."""
    test_suite = unittest.TestSuite()
    
    if category.lower() == 'data':
        print("📊 Running data processing tests only...")
        test_suite.addTest(unittest.makeSuite(TestDataLoader))
        test_suite.addTest(unittest.makeSuite(TestDataCleaner))
        test_suite.addTest(unittest.makeSuite(TestFeatureEngineer))
        
    elif category.lower() == 'models':
        print("🤖 Running AI models tests only...")
        test_suite.addTest(unittest.makeSuite(TestBiLSTMModel))
        test_suite.addTest(unittest.makeSuite(TestGRUModel))
        test_suite.addTest(unittest.makeSuite(TestTransformerModel))
        test_suite.addTest(unittest.makeSuite(TestLNNModel))
        test_suite.addTest(unittest.makeSuite(TestLTNModel))
        test_suite.addTest(unittest.makeSuite(TestEnsembleManager))
        
    elif category.lower() == 'strategy':
        print("📈 Running strategy & risk management tests only...")
        test_suite.addTest(unittest.makeSuite(TestSuperTrendStrategy))
        test_suite.addTest(unittest.makeSuite(TestRiskManager))
        test_suite.addTest(unittest.makeSuite(TestBacktester))
        
    elif category.lower() == 'executor':
        print("⚡ Running execution & monitoring tests only...")
        test_suite.addTest(unittest.makeSuite(TestMT5Executor))
        test_suite.addTest(unittest.makeSuite(TestLiveDashboard))
        test_suite.addTest(unittest.makeSuite(TestTradingAPI))
        
    elif category.lower() == 'logger':
        print("📝 Running logger & utilities tests only...")
        test_suite.addTest(unittest.makeSuite(TestLogManager))
        test_suite.addTest(unittest.makeSuite(TestDataCleanerLogger))
        
    else:
        print(f"❌ Unknown test category: {category}")
        return False
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def main():
    """Main test execution function."""
    print("🚀 SignaMentis - Comprehensive Test Suite")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--coverage':
            success = run_tests_with_coverage()
        elif sys.argv[1] == '--category' and len(sys.argv) > 2:
            success = run_specific_test_category(sys.argv[2])
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python run_all_tests.py                    # Run all tests")
            print("  python run_all_tests.py --coverage         # Run tests with coverage")
            print("  python run_all_tests.py --category <cat>   # Run specific category")
            print("  python run_all_tests.py --help             # Show this help")
            print()
            print("Available categories:")
            print("  data      - Data processing tests")
            print("  models    - AI models tests")
            print("  strategy  - Strategy & risk management tests")
            print("  executor  - Execution & monitoring tests")
            print("  logger    - Logger & utilities tests")
            return
        else:
            print(f"❌ Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    else:
        # Run all tests by default
        success = run_tests_basic()
    
    # Print results
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed successfully!")
        print("✅ SignaMentis trading system is ready for production!")
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("🔧 Fix the failing tests before proceeding to production.")
    
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
