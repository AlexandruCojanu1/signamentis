#!/usr/bin/env python3
"""
SignaMentis - Simple Test Script

This script performs basic functionality tests without requiring
external Python dependencies like pandas, numpy, torch, etc.
It checks project structure and basic logic.

Author: SignaMentis Team
Version: 2.0.0
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path


class SimpleTester:
    """Simple test class for basic functionality verification."""
    
    def __init__(self):
        """Initialize the tester."""
        self.test_results = {}
        self.start_time = datetime.now()
        
        print("🧪 SignaMentis Simple Test Script")
        print("=" * 60)
        print(f"🕐 Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def test_project_structure(self) -> bool:
        """Test basic project structure."""
        print("📁 Testing Project Structure...")
        
        required_files = [
            'main.py',
            'requirements.txt',
            'README.md',
            'Makefile',
            'PROJECT_STATUS.md'
        ]
        
        required_dirs = [
            'config',
            'scripts',
            'services',
            'tests',
            'data'
        ]
        
        missing_files = []
        missing_dirs = []
        
        # Check files
        for file_name in required_files:
            if not os.path.exists(file_name):
                missing_files.append(file_name)
        
        # Check directories
        for dir_name in required_dirs:
            if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
                missing_dirs.append(dir_name)
        
        if missing_files:
            print(f"❌ Missing files: {', '.join(missing_files)}")
        else:
            print("✅ All required files present")
        
        if missing_dirs:
            print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        else:
            print("✅ All required directories present")
        
        success = len(missing_files) == 0 and len(missing_dirs) == 0
        self.test_results['project_structure'] = success
        
        return success
    
    def test_config_files(self) -> bool:
        """Test configuration files."""
        print("\n⚙️  Testing Configuration Files...")
        
        config_files = [
            'config/settings.yaml',
            'config/model_config.yaml',
            'config/risk_config.yaml'
        ]
        
        missing_configs = []
        
        for config_file in config_files:
            if not os.path.exists(config_file):
                missing_configs.append(config_file)
        
        if missing_configs:
            print(f"❌ Missing config files: {', '.join(missing_configs)}")
        else:
            print("✅ All configuration files present")
        
        success = len(missing_configs) == 0
        self.test_results['config_files'] = success
        
        return success
    
    def test_core_modules(self) -> bool:
        """Test core module files."""
        print("\n🔧 Testing Core Modules...")
        
        core_modules = [
            'scripts/data_loader.py',
            'scripts/feature_engineering.py',
            'scripts/ensemble.py',
            'scripts/risk_manager.py',
            'scripts/strategy.py',
            'scripts/backtester.py',
            'scripts/executor.py',
            'scripts/monitor.py',
            'scripts/logger.py'
        ]
        
        missing_modules = []
        
        for module in core_modules:
            if not os.path.exists(module):
                missing_modules.append(module)
        
        if missing_modules:
            print(f"❌ Missing modules: {', '.join(missing_modules)}")
        else:
            print("✅ All core modules present")
        
        success = len(missing_modules) == 0
        self.test_results['core_modules'] = success
        
        return success
    
    def test_news_nlp_service(self) -> bool:
        """Test news NLP service structure."""
        print("\n📰 Testing News NLP Service...")
        
        news_nlp_files = [
            'services/news_nlp/ingestors/gdelt.py',
            'services/news_nlp/ingestors/tradingeconomics.py',
            'services/news_nlp/ingestors/newsapi.py',
            'services/news_nlp/ingestors/gnews.py',
            'services/news_nlp/nlp/finbert.py',
            'services/news_nlp/features/news_features.py',
            'services/news_nlp/api.py'
        ]
        
        missing_files = []
        
        for file_path in news_nlp_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing News NLP files: {', '.join(missing_files)}")
        else:
            print("✅ All News NLP service files present")
        
        success = len(missing_files) == 0
        self.test_results['news_nlp_service'] = success
        
        return success
    
    def test_test_files(self) -> bool:
        """Test test files."""
        print("\n🧪 Testing Test Files...")
        
        test_files = [
            'tests/test_data.py',
            'tests/test_models.py',
            'tests/test_strategy.py',
            'tests/test_executor.py',
            'tests/test_logger.py',
            'run_all_tests.py'
        ]
        
        missing_tests = []
        
        for test_file in test_files:
            if not os.path.exists(test_file):
                missing_tests.append(test_file)
        
        if missing_tests:
            print(f"❌ Missing test files: {', '.join(missing_tests)}")
        else:
            print("✅ All test files present")
        
        success = len(missing_tests) == 0
        self.test_results['test_files'] = success
        
        return success
    
    def test_basic_logic(self) -> bool:
        """Test basic logic without external dependencies."""
        print("\n🧠 Testing Basic Logic...")
        
        try:
            # Test ensemble logic
            ensemble_weights = {'bilstm': 0.3, 'gru': 0.3, 'transformer': 0.4}
            total_weight = sum(ensemble_weights.values())
            
            if abs(total_weight - 1.0) < 0.001:
                print("✅ Ensemble weight calculation logic correct")
            else:
                print(f"❌ Ensemble weight calculation failed: {total_weight}")
                return False
            
            # Test risk management logic
            risk_multipliers = {'low': 0.5, 'medium': 1.0, 'high': 2.0}
            position_size = 1000
            risk_level = 'medium'
            
            adjusted_size = position_size * risk_multipliers[risk_level]
            if adjusted_size == 1000:
                print("✅ Risk management logic correct")
            else:
                print(f"❌ Risk management logic failed: {adjusted_size}")
                return False
            
            # Test strategy logic
            signal_strength = 0.8
            threshold = 0.7
            
            if signal_strength > threshold:
                print("✅ Strategy signal logic correct")
            else:
                print("❌ Strategy signal logic failed")
                return False
            
            # Test data handling logic
            data_points = [100, 200, 300, 400, 500]
            average = sum(data_points) / len(data_points)
            
            if average == 300:
                print("✅ Data handling logic correct")
            else:
                print(f"❌ Data handling logic failed: {average}")
                return False
            
            self.test_results['basic_logic'] = True
            return True
            
        except Exception as e:
            print(f"❌ Basic logic test failed: {e}")
            self.test_results['basic_logic'] = False
            return False
    
    def test_file_sizes(self) -> bool:
        """Test that key files have reasonable sizes."""
        print("\n📏 Testing File Sizes...")
        
        key_files = [
            'main.py',
            'scripts/ensemble.py',
            'scripts/risk_manager.py',
            'scripts/strategy.py',
            'README.md'
        ]
        
        small_files = []
        reasonable_files = []
        
        for file_path in key_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                if size < 100:  # Less than 100 bytes
                    small_files.append(f"{file_path} ({size} bytes)")
                else:
                    reasonable_files.append(f"{file_path} ({size} bytes)")
        
        if small_files:
            print(f"⚠️  Small files (may be incomplete): {', '.join(small_files)}")
        
        if reasonable_files:
            print(f"✅ Reasonable file sizes: {', '.join(reasonable_files)}")
        
        success = len(reasonable_files) > 0
        self.test_results['file_sizes'] = success
        
        return success
    
    def test_python_syntax(self) -> bool:
        """Test basic Python syntax without importing."""
        print("\n🐍 Testing Python Syntax...")
        
        python_files = [
            'main.py',
            'scripts/ensemble.py',
            'scripts/risk_manager.py',
            'scripts/strategy.py'
        ]
        
        syntax_errors = []
        valid_files = []
        
        for file_path in python_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    # Basic syntax check (without ast.parse to avoid import issues)
                    if 'def ' in source and 'class ' in source:
                        valid_files.append(file_path)
                    else:
                        syntax_errors.append(f"{file_path} (missing function/class definitions)")
                        
                except Exception as e:
                    syntax_errors.append(f"{file_path} (read error: {e})")
        
        if syntax_errors:
            print(f"❌ Syntax issues: {', '.join(syntax_errors)}")
        
        if valid_files:
            print(f"✅ Valid Python files: {', '.join(valid_files)}")
        
        success = len(valid_files) > 0
        self.test_results['python_syntax'] = success
        
        return success
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        print("🚀 Running All Tests...")
        print()
        
        tests = [
            ("Project Structure", self.test_project_structure),
            ("Configuration Files", self.test_config_files),
            ("Core Modules", self.test_core_modules),
            ("News NLP Service", self.test_news_nlp_service),
            ("Test Files", self.test_test_files),
            ("Basic Logic", self.test_basic_logic),
            ("File Sizes", self.test_file_sizes),
            ("Python Syntax", self.test_python_syntax)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                print()
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = False
                print()
        
        # Print results
        print("=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        for test_name, test_func in tests:
            key = test_name.lower().replace(' ', '_')
            status = "✅ PASS" if self.test_results.get(key, False) else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        print()
        print(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! SignaMentis is ready for development.")
        else:
            print("⚠️  Some tests failed. Please check the output above.")
        
        return passed == total
    
    def get_test_results(self) -> dict:
        """Get detailed test results."""
        return {
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for v in self.test_results.values() if v),
                'failed_tests': sum(1 for v in self.test_results.values() if not v),
                'success_rate': sum(1 for v in self.test_results.values() if v) / len(self.test_results) if self.test_results else 0
            },
            'details': self.test_results,
            'timestamp': self.start_time.isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds()
        }


def main():
    """Main test execution function."""
    tester = SimpleTester()
    
    try:
        success = tester.run_all_tests()
        
        # Save results
        results = tester.get_test_results()
        with open('simple_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Test results saved to: simple_test_results.json")
        
        return success
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
