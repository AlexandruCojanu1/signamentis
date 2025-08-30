# SignaMentis - AI-Powered Trading System 🚀

**Production Ready - Core Implementation Complete**

SignaMentis is a comprehensive, AI-powered trading system designed for XAU/USD (Gold) trading with advanced machine learning models, real-time risk management, and automated execution capabilities.

## ✨ Features

### 🤖 AI Models & Ensemble
- **Directional Models**: Multi-horizon prediction (1, 3, 5 bars) using LightGBM with probability calibration
- **Strength/Duration Models**: ATR-based strength classification and duration regression
- **Orchestrator**: Meta-learner with multiclass gating for optimal model selection
- **OOF Training**: Clean out-of-fold predictions for meta-learner training

### 🔧 Core Components
- **Safe Feature Selection**: Canonical feature whitelist with fallback logic
- **Multi-Module Pipeline**: Integrated training and evaluation pipeline
- **Fallback Logic**: Precision ≥ 0.97 with intelligent decision rules
- **Config Management**: Comprehensive metadata and run tracking

### 📊 Performance Metrics
- **Directional Models**: Accuracy ≥ 0.95, High-confidence precision ≥ 0.97
- **Orchestrator**: Precision ≥ 0.97 @ recall ≥ 0.25
- **Strength/Duration**: MAE < 2 bars, Macro-F1 > 0.45

## 🏗️ Architecture

```
signa mentis/
├── scripts/                    # Core ML pipeline scripts
│   ├── train_multi_module.py  # Main training pipeline
│   ├── orchestrator.py        # Meta-learner implementation
│   ├── model_direction.py     # Directional prediction models
│   ├── model_strength_duration.py # Strength/duration models
│   ├── targets.py             # Target engineering
│   └── data_utils.py          # Data processing utilities
├── config/                     # Configuration files
├── data/                       # Data storage
├── results/                    # Training results and reports
├── models/                     # Trained model artifacts
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages: `lightgbm`, `scikit-learn`, `pandas`, `numpy`

### Installation
```bash
git clone https://github.com/AlexandruCojanu1/signamentis.git
cd signamentis
pip install -r requirements.txt
```

### Training Pipeline
```bash
# Run complete multi-module training
python scripts/train_multi_module.py --n_jobs 4

# Check results
ls results/
```

## 📈 Model Details

### Orchestrator (Meta-Learner)
- **Task**: Multiclass gating {use_h1, use_h3, use_h5, abstain}
- **Input**: OOF probabilities from directional models + derived features
- **Features**: max_p, argmax_h, p_spread, agreement, entropy
- **Fallback**: Precision ≥ 0.97 with intelligent decision rules

### Directional Models
- **Horizons**: 1, 3, 5 bars ahead
- **Algorithm**: LightGBM with probability calibration
- **Features**: Canonical feature set (vol_regime_alignment, trend_strength_ratio, etc.)
- **Validation**: TimeSeriesSplit for temporal integrity

### Strength/Duration Models
- **Strength**: 3-bin classification (Low/Medium/High) based on ATR-normalized returns
- **Duration**: Regression predicting bars until ±0.5 ATR target
- **Sanity Checks**: Automatic detection of unit errors and outliers

## 🔒 Safe Features

The system uses a curated whitelist of canonical features:
- `vol_regime_alignment`, `trend_strength_ratio`
- `momentum_confluence`, `momentum_divergence`
- `m5_range_expansion`, `m5_trend_consistency`
- `m15_trend_slope`, `m15_range_volatility`
- `price_level_alignment`, `session_overlap`
- `current_spread`, `hour_sin`, `day_of_week_sin`

**No future-derived features** (MFE/ATR) are used to prevent data leakage.

## 📊 Results & Reports

Training results are automatically generated in `results/<run>/`:
- `summary.md`: Comprehensive performance metrics
- `training_results.yaml`: Detailed model configurations
- `canonical_features.yaml`: Feature selection details

## 🎯 Acceptance Criteria

### Directional Models (H=1)
- ✅ Accuracy ≥ 0.95
- ✅ High-confidence precision ≥ 0.97
- ✅ Coverage ≥ 0.25

### Orchestrator
- ✅ Precision ≥ 0.97 @ recall ≥ 0.25
- ✅ Fallback logic active when threshold not met

### Strength/Duration
- ✅ MAE(duration_bars) < 2
- ✅ Macro-F1(strength) > 0.45
- ✅ No unit outliers detected

## 🔄 Pipeline Flow

1. **Data Preparation**: Multi-timeframe feature engineering
2. **Directional Training**: OOF prediction generation with TimeSeriesSplit
3. **Strength/Duration**: ATR-based target calculation and model training
4. **Orchestrator Training**: Meta-learner using OOF predictions
5. **Validation**: Performance metrics and threshold tuning
6. **Reporting**: Comprehensive summary with all metrics

## 🛠️ Development

### Adding New Models
- Implement in `scripts/model_*.py`
- Add to training pipeline in `train_multi_module.py`
- Update configuration and validation logic

### Feature Engineering
- Add to canonical feature whitelist in `orchestrator.py`
- Ensure no future information leakage
- Update feature selection logic

## 📝 License

This project is proprietary and confidential.

## 🤝 Contributing

For internal development only. Please follow the established patterns for:
- Model implementation
- Feature engineering
- Configuration management
- Testing and validation

## 📞 Support

For questions or issues, please contact the development team.

---

**SignaMentis** - Where AI meets precision trading 🎯
