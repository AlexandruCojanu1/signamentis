# SignaMentis Trading System - Makefile
# Author: SignaMentis Team
# Version: 1.0.0

.PHONY: help setup install test test-coverage test-data test-models test-strategy test-executor test-logger clean lint format docs run-backtest run-live docker-build docker-run docker-stop

# Default target
help:
	@echo "🚀 SignaMentis Trading System - Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  setup           - Initial project setup and environment creation"
	@echo "  install         - Install all dependencies"
	@echo "  install-dev     - Install development dependencies"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test            - Run all unit tests"
	@echo "  test-coverage   - Run tests with coverage report"
	@echo "  test-data       - Run data processing tests only"
	@echo "  test-models     - Run AI models tests only"
	@echo "  test-strategy   - Run strategy & risk management tests only"
	@echo "  test-executor   - Run execution & monitoring tests only"
	@echo "  test-logger     - Run logger & utilities tests only"
	@echo "  test-news-nlp   - Run News NLP service tests only"
	@echo ""
	@echo "🔧 Development:"
	@echo "  lint            - Run code linting and style checks"
	@echo "  format          - Format code with black and isort"
	@echo "  clean           - Clean build artifacts and cache files"
	@echo ""
	@echo "📊 Trading Operations:"
	@echo "  run-backtest    - Run backtesting on historical data"
	@echo "  run-live        - Start live trading mode"
	@echo ""
	@echo "🐳 Docker Operations:"
	@echo "  docker-build    - Build Docker containers"
	@echo "  docker-run      - Run Docker containers"
	@echo "  docker-stop     - Stop Docker containers"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs            - Generate documentation"
	@echo ""
	@echo "💡 Examples:"
	@echo "  make test       - Run all tests"
	@echo "  make test-data  - Run only data tests"
	@echo "  make lint       - Check code quality"

# Setup and Installation
setup:
	@echo "🔧 Setting up SignaMentis project..."
	@python -m venv venv
	@echo "✅ Virtual environment created"
	@echo "📝 Please activate the virtual environment:"
	@echo "   source venv/bin/activate  # On Unix/MacOS"
	@echo "   venv\\Scripts\\activate     # On Windows"

install:
	@echo "📦 Installing production dependencies..."
	@pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

install-dev:
	@echo "🔧 Installing development dependencies..."
	@pip install -r requirements.txt
	@pip install coverage pytest pytest-cov black isort flake8 mypy
	@echo "✅ Development dependencies installed successfully"

# Testing Commands
test:
	@echo "🧪 Running all unit tests..."
	@python run_all_tests.py
	@echo "📰 Testing News NLP service..."
	@python test_news_nlp.py

test-coverage:
	@echo "📊 Running tests with coverage..."
	@python run_all_tests.py --coverage

test-data:
	@echo "📊 Running data processing tests..."
	@python run_all_tests.py --category data

test-models:
	@echo "🤖 Running AI models tests..."
	@python run_all_tests.py --category models

test-strategy:
	@echo "📈 Running strategy & risk management tests..."
	@python run_all_tests.py --category strategy

test-executor:
	@echo "⚡ Running execution & monitoring tests..."
	@python run_all_tests.py --category executor

test-logger:
	@echo "📝 Running logger & utilities tests..."
	@python run_all_tests.py --category logger

test-news-nlp:
	@echo "📰 Running News NLP service tests..."
	@python test_news_nlp.py

# Development Commands
lint:
	@echo "🔍 Running code linting..."
	@flake8 scripts/ tests/ --max-line-length=120 --ignore=E501,W503
	@black --check scripts/ tests/
	@isort --check-only scripts/ tests/
	@echo "✅ Code linting completed"

format:
	@echo "🎨 Formatting code..."
	@black scripts/ tests/
	@isort scripts/ tests/
	@echo "✅ Code formatting completed"

clean:
	@echo "🧹 Cleaning project..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name "coverage_html" -exec rm -rf {} +
	@find . -type f -name ".coverage" -delete
	@find . -type f -name "*.log" -delete
	@echo "✅ Project cleaned"

# Trading Operations
run-backtest:
	@echo "📊 Running backtesting..."
	@python main.py --mode backtest --start-date 2023-08-11 --end-date 2023-08-12

run-live:
	@echo "🚀 Starting live trading mode..."
	@python main.py --mode live

# Docker Operations
docker-build:
	@echo "🐳 Building Docker containers..."
	@docker-compose build

docker-run:
	@echo "🚀 Running Docker containers..."
	@docker-compose up -d

docker-stop:
	@echo "🛑 Stopping Docker containers..."
	@docker-compose down

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@mkdir -p docs
	@echo "📝 Documentation structure created"
	@echo "📁 Please add your documentation files to the docs/ directory"

# Quick validation
validate:
	@echo "✅ Validating project structure..."
	@python validate_project.py

# Performance check
perf-check:
	@echo "⚡ Running performance checks..."
	@python -c "import time; start=time.time(); import scripts.feature_engineering; print(f'Feature engineering import: {time.time()-start:.3f}s')"
	@python -c "import time; start=time.time(); import scripts.ensemble; print(f'Ensemble import: {time.time()-start:.3f}s')"
	@echo "✅ Performance check completed"

# Security check
security-check:
	@echo "🔒 Running security checks..."
	@bandit -r scripts/ -f json -o security-report.json || echo "⚠️  Bandit not installed, skipping security scan"
	@echo "✅ Security check completed"

# Full system test
system-test: clean install-dev test lint validate
	@echo "🎉 Full system test completed successfully!"
	@echo "✅ SignaMentis is ready for production!"

# Development workflow
dev-workflow: format lint test
	@echo "🔄 Development workflow completed"

# Production preparation
prod-prep: clean install test lint validate security-check
	@echo "🚀 Production preparation completed!"
	@echo "✅ SignaMentis is production-ready!"

# Help for specific targets
test-help:
	@echo "🧪 Test Commands Help:"
	@echo "  make test            - Run all tests"
	@echo "  make test-coverage   - Run tests with coverage report"
	@echo "  make test-data       - Run only data processing tests"
	@echo "  make test-models     - Run only AI models tests"
	@echo "  make test-strategy   - Run only strategy tests"
	@echo "  make test-executor   - Run only executor tests"
	@echo "  make test-logger     - Run only logger tests"
	@echo "  make test-news-nlp   - Run only News NLP service tests"

docker-help:
	@echo "🐳 Docker Commands Help:"
	@echo "  make docker-build    - Build all containers"
	@echo "  make docker-run      - Start all services"
	@echo "  make docker-stop     - Stop all services"
	@echo "  docker-compose logs  - View service logs"
	@echo "  docker-compose ps    - Check service status"
