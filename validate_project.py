#!/usr/bin/env python3
"""
SignaMentis - Project Validation Script

This script validates the project structure, checks for required files,
and performs basic syntax validation without importing external dependencies.

Author: SignaMentis Team
Version: 2.0.0
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime


class ProjectValidator:
    """Project validation and structure checking."""
    
    def __init__(self, project_root: str = "."):
        """Initialize validator with project root."""
        self.project_root = Path(project_root).resolve()
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        self.start_time = datetime.now()
        
        # Required directory structure
        self.required_dirs = [
            "config",
            "scripts",
            "services",
            "services/news_nlp",
            "services/news_nlp/ingestors",
            "services/news_nlp/nlp",
            "services/news_nlp/features",
            "tests",
            "data",
            "logs",
            "docs"
        ]
        
        # Required configuration files
        self.required_config_files = [
            "config/settings.yaml",
            "config/model_config.yaml",
            "config/risk_config.yaml",
            ".env.example"
        ]
        
        # Required Python files
        self.required_python_files = [
            "main.py",
            "requirements.txt",
            "README.md",
            "Makefile",
            "scripts/data_loader.py",
            "scripts/feature_engineering.py",
            "scripts/model_bilstm.py",
            "scripts/model_gru.py",
            "scripts/model_transformer.py",
            "scripts/model_lnn.py",
            "scripts/model_ltn.py",
            "scripts/ensemble.py",
            "scripts/risk_manager.py",
            "scripts/strategy.py",
            "scripts/backtester.py",
            "scripts/executor.py",
            "scripts/monitor.py",
            "scripts/logger.py",
            "scripts/data_cleaner.py",
            "services/api.py"
        ]
        
        # Required news NLP files
        self.required_news_nlp_files = [
            "services/news_nlp/ingestors/gdelt.py",
            "services/news_nlp/ingestors/tradingeconomics.py",
            "services/news_nlp/ingestors/newsapi.py",
            "services/news_nlp/ingestors/gnews.py",
            "services/news_nlp/nlp/finbert.py",
            "services/news_nlp/features/news_features.py",
            "services/news_nlp/api.py"
        ]
        
        # Required test files
        self.required_test_files = [
            "tests/test_data.py",
            "tests/test_models.py",
            "tests/test_strategy.py",
            "tests/test_executor.py",
            "tests/test_logger.py",
            "run_all_tests.py"
        ]
        
        print("🔍 SignaMentis Project Validator")
        print("=" * 60)
        print(f"🕐 Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Project root: {self.project_root}")
        print()
    
    def validate_directory_structure(self) -> bool:
        """Validate project directory structure."""
        print("📁 Validating Directory Structure...")
        
        missing_dirs = []
        existing_dirs = []
        
        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                existing_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)
        
        # Create missing directories
        for dir_path in missing_dirs:
            full_path = self.project_root / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {dir_path}")
                existing_dirs.append(dir_path)
            except Exception as e:
                print(f"❌ Failed to create directory {dir_path}: {e}")
                self.errors.append(f"Directory creation failed: {dir_path}")
        
        print(f"✅ Directory structure: {len(existing_dirs)}/{len(self.required_dirs)} directories exist")
        
        if missing_dirs:
            print(f"⚠️  Missing directories: {', '.join(missing_dirs)}")
            self.warnings.append(f"Missing directories: {missing_dirs}")
        
        self.validation_results['directory_structure'] = {
            'total_required': len(self.required_dirs),
            'existing': len(existing_dirs),
            'missing': len(missing_dirs),
            'success': len(missing_dirs) == 0
        }
        
        return len(missing_dirs) == 0
    
    def validate_config_files(self) -> bool:
        """Validate configuration files."""
        print("\n⚙️  Validating Configuration Files...")
        
        missing_configs = []
        existing_configs = []
        
        for config_file in self.required_config_files:
            full_path = self.project_root / config_file
            if full_path.exists() and full_path.is_file():
                existing_configs.append(config_file)
            else:
                missing_configs.append(config_file)
        
        print(f"✅ Configuration files: {len(existing_configs)}/{len(self.required_config_files)} exist")
        
        if missing_configs:
            print(f"⚠️  Missing config files: {', '.join(missing_configs)}")
            self.warnings.append(f"Missing config files: {missing_configs}")
        
        self.validation_results['config_files'] = {
            'total_required': len(self.required_config_files),
            'existing': len(existing_configs),
            'missing': len(missing_configs),
            'success': len(missing_configs) == 0
        }
        
        return len(missing_configs) == 0
    
    def validate_python_files(self) -> bool:
        """Validate Python source files."""
        print("\n🐍 Validating Python Source Files...")
        
        missing_python = []
        existing_python = []
        syntax_errors = []
        
        for python_file in self.required_python_files:
            full_path = self.project_root / python_file
            if full_path.exists() and full_path.is_file():
                existing_python.append(python_file)
                
                # Check Python syntax
                if python_file.endswith('.py'):
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            source = f.read()
                        ast.parse(source)
                    except SyntaxError as e:
                        syntax_errors.append(f"{python_file}: {e}")
                    except Exception as e:
                        syntax_errors.append(f"{python_file}: {e}")
            else:
                missing_python.append(python_file)
        
        print(f"✅ Python files: {len(existing_python)}/{len(self.required_python_files)} exist")
        
        if missing_python:
            print(f"⚠️  Missing Python files: {', '.join(missing_python)}")
            self.warnings.append(f"Missing Python files: {missing_python}")
        
        if syntax_errors:
            print(f"❌ Syntax errors found: {len(syntax_errors)}")
            for error in syntax_errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(syntax_errors) > 5:
                print(f"   ... and {len(syntax_errors) - 5} more")
            self.errors.extend(syntax_errors)
        
        self.validation_results['python_files'] = {
            'total_required': len(self.required_python_files),
            'existing': len(existing_python),
            'missing': len(missing_python),
            'syntax_errors': len(syntax_errors),
            'success': len(missing_python) == 0 and len(syntax_errors) == 0
        }
        
        return len(missing_python) == 0 and len(syntax_errors) == 0
    
    def validate_news_nlp_files(self) -> bool:
        """Validate news NLP service files."""
        print("\n📰 Validating News NLP Service Files...")
        
        missing_news_nlp = []
        existing_news_nlp = []
        syntax_errors = []
        
        for news_nlp_file in self.required_news_nlp_files:
            full_path = self.project_root / news_nlp_file
            if full_path.exists() and full_path.is_file():
                existing_news_nlp.append(news_nlp_file)
                
                # Check Python syntax
                if news_nlp_file.endswith('.py'):
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            source = f.read()
                        ast.parse(source)
                    except SyntaxError as e:
                        syntax_errors.append(f"{news_nlp_file}: {e}")
                    except Exception as e:
                        syntax_errors.append(f"{news_nlp_file}: {e}")
            else:
                missing_news_nlp.append(news_nlp_file)
        
        print(f"✅ News NLP files: {len(existing_news_nlp)}/{len(self.required_news_nlp_files)} exist")
        
        if missing_news_nlp:
            print(f"⚠️  Missing News NLP files: {', '.join(missing_news_nlp)}")
            self.warnings.append(f"Missing News NLP files: {missing_news_nlp}")
        
        if syntax_errors:
            print(f"❌ Syntax errors found: {len(syntax_errors)}")
            for error in syntax_errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(syntax_errors) > 5:
                print(f"   ... and {len(syntax_errors) - 5} more")
            self.errors.extend(syntax_errors)
        
        self.validation_results['news_nlp_files'] = {
            'total_required': len(self.required_news_nlp_files),
            'existing': len(existing_news_nlp),
            'missing': len(missing_news_nlp),
            'syntax_errors': len(syntax_errors),
            'success': len(missing_news_nlp) == 0 and len(syntax_errors) == 0
        }
        
        return len(missing_news_nlp) == 0 and len(syntax_errors) == 0
    
    def validate_test_files(self) -> bool:
        """Validate test files."""
        print("\n🧪 Validating Test Files...")
        
        missing_tests = []
        existing_tests = []
        syntax_errors = []
        
        for test_file in self.required_test_files:
            full_path = self.project_root / test_file
            if full_path.exists() and full_path.is_file():
                existing_tests.append(test_file)
                
                # Check Python syntax
                if test_file.endswith('.py'):
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            source = f.read()
                        ast.parse(source)
                    except SyntaxError as e:
                        syntax_errors.append(f"{test_file}: {e}")
                    except Exception as e:
                        syntax_errors.append(f"{test_file}: {e}")
            else:
                missing_tests.append(test_file)
        
        print(f"✅ Test files: {len(existing_tests)}/{len(self.required_test_files)} exist")
        
        if missing_tests:
            print(f"⚠️  Missing test files: {', '.join(missing_tests)}")
            self.warnings.append(f"Missing test files: {missing_tests}")
        
        if syntax_errors:
            print(f"❌ Syntax errors found: {len(syntax_errors)}")
            for error in syntax_errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(syntax_errors) > 5:
                print(f"   ... and {len(syntax_errors) - 5} more")
            self.errors.extend(syntax_errors)
        
        self.validation_results['test_files'] = {
            'total_required': len(self.required_test_files),
            'existing': len(existing_tests),
            'missing': len(missing_tests),
            'syntax_errors': len(syntax_errors),
            'success': len(missing_tests) == 0 and len(syntax_errors) == 0
        }
        
        return len(missing_tests) == 0 and len(syntax_errors) == 0
    
    def validate_requirements(self) -> bool:
        """Validate requirements.txt file."""
        print("\n📦 Validating Requirements.txt...")
        
        requirements_path = self.project_root / "requirements.txt"
        
        if not requirements_path.exists():
            print("❌ requirements.txt not found")
            self.errors.append("requirements.txt not found")
            self.validation_results['requirements'] = {'success': False, 'error': 'File not found'}
            return False
        
        try:
            with open(requirements_path, 'r', encoding='utf-8') as f:
                requirements = f.read()
            
            # Basic validation
            lines = requirements.strip().split('\n')
            valid_lines = [line for line in lines if line.strip() and not line.startswith('#')]
            
            print(f"✅ Requirements.txt: {len(valid_lines)} packages listed")
            
            # Check for critical packages
            critical_packages = ['torch', 'pandas', 'numpy', 'fastapi', 'uvicorn']
            missing_critical = []
            
            for package in critical_packages:
                if not any(package in line for line in valid_lines):
                    missing_critical.append(package)
            
            if missing_critical:
                print(f"⚠️  Missing critical packages: {', '.join(missing_critical)}")
                self.warnings.append(f"Missing critical packages: {missing_critical}")
            
            self.validation_results['requirements'] = {
                'total_packages': len(valid_lines),
                'missing_critical': len(missing_critical),
                'success': True
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading requirements.txt: {e}")
            self.errors.append(f"Requirements.txt error: {e}")
            self.validation_results['requirements'] = {'success': False, 'error': str(e)}
            return False
    
    def validate_docker_files(self) -> bool:
        """Validate Docker-related files."""
        print("\n🐳 Validating Docker Files...")
        
        docker_files = ['Dockerfile', 'docker-compose.yml', 'services/news_nlp/Dockerfile']
        existing_docker = []
        missing_docker = []
        
        for docker_file in docker_files:
            full_path = self.project_root / docker_file
            if full_path.exists() and full_path.is_file():
                existing_docker.append(docker_file)
            else:
                missing_docker.append(docker_file)
        
        print(f"✅ Docker files: {len(existing_docker)}/{len(docker_files)} exist")
        
        if missing_docker:
            print(f"⚠️  Missing Docker files: {', '.join(missing_docker)}")
            self.warnings.append(f"Missing Docker files: {missing_docker}")
        
        self.validation_results['docker_files'] = {
            'total_required': len(docker_files),
            'existing': len(existing_docker),
            'missing': len(missing_docker),
            'success': len(missing_docker) == 0
        }
        
        return len(missing_docker) == 0
    
    def validate_project_structure(self) -> bool:
        """Validate overall project structure."""
        print("\n🏗️  Validating Overall Project Structure...")
        
        # Check for key project files
        key_files = [
            'main.py',
            'README.md',
            'Makefile',
            'PROJECT_STATUS.md'
        ]
        
        existing_key = []
        missing_key = []
        
        for key_file in key_files:
            full_path = self.project_root / key_file
            if full_path.exists() and full_path.is_file():
                existing_key.append(key_file)
            else:
                missing_key.append(key_file)
        
        print(f"✅ Key project files: {len(existing_key)}/{len(key_files)} exist")
        
        if missing_key:
            print(f"⚠️  Missing key files: {', '.join(missing_key)}")
            self.warnings.append(f"Missing key files: {missing_key}")
        
        # Check for data directories
        data_dirs = ['data', 'data/news_raw', 'data/processed', 'data/processed/news']
        existing_data_dirs = []
        missing_data_dirs = []
        
        for data_dir in data_dirs:
            full_path = self.project_root / data_dir
            if full_path.exists() and full_path.is_dir():
                existing_data_dirs.append(data_dir)
            else:
                missing_data_dirs.append(data_dir)
        
        print(f"✅ Data directories: {len(existing_data_dirs)}/{len(data_dirs)} exist")
        
        if missing_data_dirs:
            print(f"⚠️  Missing data directories: {', '.join(missing_data_dirs)}")
            self.warnings.append(f"Missing data directories: {missing_data_dirs}")
        
        self.validation_results['project_structure'] = {
            'key_files': {
                'total': len(key_files),
                'existing': len(existing_key),
                'missing': len(missing_key)
            },
            'data_dirs': {
                'total': len(data_dirs),
                'existing': len(existing_data_dirs),
                'missing': len(missing_data_dirs)
            },
            'success': len(missing_key) == 0 and len(missing_data_dirs) == 0
        }
        
        return len(missing_key) == 0 and len(missing_data_dirs) == 0
    
    def run_validation(self) -> bool:
        """Run complete project validation."""
        print("🚀 Starting Project Validation...")
        print()
        
        validation_steps = [
            ("Directory Structure", self.validate_directory_structure),
            ("Configuration Files", self.validate_config_files),
            ("Python Source Files", self.validate_python_files),
            ("News NLP Service Files", self.validate_news_nlp_files),
            ("Test Files", self.validate_test_files),
            ("Requirements", self.validate_requirements),
            ("Docker Files", self.validate_docker_files),
            ("Project Structure", self.validate_project_structure)
        ]
        
        passed = 0
        total = len(validation_steps)
        
        for step_name, step_func in validation_steps:
            try:
                if step_func():
                    passed += 1
                print()
            except Exception as e:
                print(f"❌ {step_name} validation crashed: {e}")
                self.errors.append(f"{step_name} validation failed: {e}")
                print()
        
        # Print summary
        print("=" * 60)
        print("📊 Validation Results Summary")
        print("=" * 60)
        
        for step_name, step_func in validation_steps:
            key = step_name.lower().replace(' ', '_')
            status = "✅ PASS" if self.validation_results.get(key, {}).get('success', False) else "❌ FAIL"
            print(f"{step_name}: {status}")
        
        print()
        print(f"Total: {passed}/{total} validation steps passed")
        
        if self.errors:
            print(f"\n❌ Errors found: {len(self.errors)}")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(self.errors) > 5:
                print(f"   ... and {len(self.errors) - 5} more")
        
        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                print(f"   - {warning}")
            if len(self.warnings) > 5:
                print(f"   ... and {len(self.warnings) - 5} more")
        
        if passed == total and not self.errors:
            print("\n🎉 All validations passed! Project structure is valid.")
        elif passed == total:
            print("\n⚠️  All validations passed but there are errors to fix.")
        else:
            print("\n❌ Some validations failed. Please check the output above.")
        
        return passed == total and not self.errors
    
    def get_validation_results(self) -> Dict[str, Any]:
        """Get detailed validation results."""
        return {
            'summary': {
                'total_steps': len(self.validation_results),
                'passed_steps': sum(1 for v in self.validation_results.values() if v.get('success', False)),
                'failed_steps': sum(1 for v in self.validation_results.values() if not v.get('success', False)),
                'success_rate': sum(1 for v in self.validation_results.values() if v.get('success', False)) / len(self.validation_results) if self.validation_results else 0
            },
            'details': self.validation_results,
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': self.start_time.isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds()
        }


def main():
    """Main validation function."""
    validator = ProjectValidator()
    
    try:
        success = validator.run_validation()
        
        # Save results
        results = validator.get_validation_results()
        with open('project_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Validation results saved to: project_validation_results.json")
        
        return success
        
    except Exception as e:
        print(f"❌ Validation execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
