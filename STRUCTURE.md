# 📁 SignaMentis Project Structure

## Overview
This document describes the organized structure of the SignaMentis trading system.

## Directory Structure

```
SignaMentis/
├── 📁 config/                  # Configuration files
│   ├── settings.yaml          # Main system settings
│   ├── model_config.yaml      # AI model configurations
│   ├── risk_config.yaml       # Risk management settings
│   └── news_nlp.yaml         # News sentiment config
│
├── 📁 scripts/                # Core trading modules
│   ├── data_loader.py         # Data ingestion
│   ├── feature_engineering.py # Feature creation
│   ├── model_*.py             # AI models (6 models)
│   ├── ensemble.py            # Model combination
│   ├── strategy.py            # Trading strategy
│   ├── risk_manager.py        # Risk management
│   ├── backtester*.py         # Backtesting engines
│   └── mt5_connector.py       # Trading execution
│
├── 📁 data/                   # Data storage
│   ├── raw/                   # Raw market data
│   ├── processed/             # Processed features
│   ├── external/              # Large external files
│   ├── archive/               # Archived databases
│   └── splits/                # Data splits
│
├── 📁 tests/                  # Testing suite
│   ├── test_*.py              # Unit tests
│   ├── manual/                # Manual test scripts
│   └── integration/           # Integration tests
│
├── 📁 services/               # External services
│   ├── api.py                 # Main API service
│   ├── news_nlp/              # News sentiment service
│   └── telemetry.py           # Monitoring
│
├── 📁 mlops/                  # MLOps components
│   ├── mlflow_setup.md        # MLflow configuration
│   ├── optuna_tuner.py        # Hyperparameter tuning
│   └── feast/                 # Feature store
│
├── 📁 docker/                 # Containerization
│   ├── nginx/                 # Reverse proxy
│   ├── prometheus.yml         # Monitoring config
│   └── mongo-init.js          # Database init
│
├── 📁 validation/             # Cross-validation
│   └── purged_kfold.py        # Time series CV
│
├── 📁 messaging/              # Message queues
│   ├── redis_bus.py           # Redis messaging
│   └── rabbit_bus.py          # RabbitMQ messaging
│
├── 📁 telemetry/              # Monitoring
│   └── prometheus_metrics.py  # Metrics collection
│
├── 📁 executor/               # Trade execution
│   └── metaapi_executor.py    # MetaAPI connector
│
├── 📁 qa/                     # Quality assurance
│   └── gx/                    # Great Expectations
│
├── 📁 logs/                   # Log files (ignored)
├── 📁 results/                # Analysis results
├── 📁 models/                 # Saved models
└── 📁 notebooks/              # Jupyter notebooks
```

## Key Files

### Configuration
- `config/settings.yaml` - Main system configuration
- `config/model_config.yaml` - AI model hyperparameters
- `config/risk_config.yaml` - Risk management rules

### Core Scripts
- `main.py` - Main entry point
- `scripts/ensemble.py` - AI model combination
- `scripts/strategy.py` - SuperTrend trading strategy
- `scripts/risk_manager.py` - Risk management system

### AI Models (6 total)
- `scripts/model_bilstm.py` - Bidirectional LSTM
- `scripts/model_gru.py` - Gated Recurrent Unit
- `scripts/model_transformer.py` - Transformer architecture
- `scripts/model_lnn.py` - Liquid Neural Network
- `scripts/model_ltn.py` - Logical Tensor Network
- `scripts/model_random_forest.py` - Random Forest (NEW)

### Testing
- `run_all_tests.py` - Main test runner
- `scripts/comprehensive_test.py` - System validation
- `tests/test_*.py` - Unit tests for each component

### Deployment
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Multi-service deployment
- `requirements.txt` - Python dependencies
- `Makefile` - Build automation

## Data Flow

1. **Raw Data** → `data/raw/` (CSV files from MT5/external)
2. **Feature Engineering** → `data/processed/` (Engineered features)
3. **Model Training** → `models/saved/` (Trained AI models)
4. **Backtesting** → `results/` (Performance analysis)
5. **Live Trading** → `logs/` (Trading activity logs)

## File Management Rules

### Ignored Files (.gitignore)
- Log files (`*.log`)
- Temporary results (`backtest_results_*.json`)
- Database files (`*.db`)
- Large data files (`*.csv`, `*.parquet`)
- Python cache (`__pycache__/`)

### Cleanup Schedule
- **Daily**: Clean temporary logs and results
- **Weekly**: Archive old processed data
- **Monthly**: Backup models and configurations

## Best Practices

1. **Keep raw data in `data/external/`** for large files
2. **Use `data/processed/`** for engineered features
3. **Store results in `results/`** for analysis
4. **Place manual tests in `tests/manual/`**
5. **Use `logs/`** for runtime logging
6. **Archive old data in `data/archive/`**

## Maintenance

To clean up the project:
```bash
# Remove temporary files
make clean

# Archive old data
python scripts/cleanup.py --archive

# Validate structure
python tests/manual/validate_project.py
```

This structure ensures:
- ✅ Clear separation of concerns
- ✅ Easy navigation and maintenance
- ✅ Efficient version control
- ✅ Scalable architecture
- ✅ Production-ready organization
