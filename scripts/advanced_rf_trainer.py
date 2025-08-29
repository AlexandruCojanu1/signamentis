#!/usr/bin/env python3
"""
Advanced Random Forest Trainer for SignaMentis

This module implements advanced training strategy for XAU/USD prediction:
- Multi-timeframe features (M5 + M15)
- Optimized hyperparameters for XAU/USD volatility
- Ensemble strategy with multiple models
- Advanced validation with session awareness
- Confidence-based trading decisions

Author: SignaMentis Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import warnings
import yaml
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRandomForestTrainer:
    """
    Advanced Random Forest trainer optimized for XAU/USD trading.
    
    Features:
    - Multi-timeframe learning (M5 + M15)
    - Optimized hyperparameters for forex volatility
    - Ensemble strategy with confidence scoring
    - Session-aware validation
    - Feature importance analysis
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the advanced trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_importances = {}
        self.training_history = {}
        
        # Initialize components
        self.scaler = RobustScaler()
        self.calibrator = None
        
        logger.info("AdvancedRandomForestTrainer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load training configuration."""
        default_config = {
            # Model hyperparameters optimized for XAU/USD
            'random_forest': {
                'n_estimators': 250,
                'max_depth': 12,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'class_weight': 'balanced_subsample',
                'criterion': 'entropy',
                'bootstrap': True,
                'oob_score': True,
                'random_state': 42,
                'n_jobs': -1,
                'max_leaf_nodes': 100,
                'min_impurity_decrease': 0.001
            },
            
            # Ensemble configuration (reduced for small dataset)
            'ensemble': {
                'n_models': 3,
                'feature_subset_ratio': 0.9,
                'voting_method': 'soft',
                'confidence_threshold': 0.6
            },
            
            # Validation configuration (reduced for small dataset)
            'validation': {
                'cv_folds': 3,
                'test_size': 0.2,
                'validation_size': 0.2,
                'session_aware': False,  # Disabled for small dataset
                'time_series_split': True
            },
            
            # Feature selection
            'feature_selection': {
                'method': 'importance',
                'max_features': 50,
                'min_importance': 0.005,
                'correlation_threshold': 0.95
            },
            
            # Trading parameters
            'trading': {
                'confidence_threshold': 0.65,
                'risk_per_trade': 0.02,
                'max_positions': 3,
                'session_filters': ['london', 'newyork']
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge with default config
                for key in user_config:
                    if key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
        
        return default_config
    
    def load_multi_timeframe_data(self, m5_file: str, m15_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare multi-timeframe data.
        
        Args:
            m5_file: Path to M5 features file (can be same as m15_file for multi-timeframe features)
            m15_file: Path to M15 features file
            
        Returns:
            Tuple of (combined_features, targets)
        """
        logger.info("Loading multi-timeframe data...")
        
        # Load multi-timeframe features
        features_data = pd.read_csv(m5_file)
        
        logger.info(f"Multi-timeframe features: {features_data.shape}")
        
        return features_data, features_data
    
    def _create_multi_timeframe_features(self, m5_subset: pd.DataFrame, m15_timestamp: pd.Timestamp) -> Dict:
        """
        Create multi-timeframe features from M5 data for M15 prediction.
        
        Args:
            m5_subset: Last 3 M5 bars (15 minutes of data)
            m15_timestamp: M15 timestamp for alignment
            
        Returns:
            Dictionary of multi-timeframe features
        """
        features = {'timestamp': m15_timestamp}
        
        # M5 aggregated features (last 15 minutes)
        features['m5_open_first'] = m5_subset['open'].iloc[0]
        features['m5_close_last'] = m5_subset['close'].iloc[-1]
        features['m5_high_max'] = m5_subset['high'].max()
        features['m5_low_min'] = m5_subset['low'].min()
        features['m5_volume_sum'] = m5_subset['volume'].sum()
        features['m5_spread_mean'] = m5_subset['spread'].mean()
        
        # M5 price action over 15 minutes
        features['m5_price_change'] = features['m5_close_last'] - features['m5_open_first']
        features['m5_price_change_pct'] = features['m5_price_change'] / features['m5_open_first']
        features['m5_range'] = features['m5_high_max'] - features['m5_low_min']
        
        # M5 volatility features
        m5_returns = m5_subset['close'].pct_change().dropna()
        features['m5_volatility'] = m5_returns.std() if len(m5_returns) > 1 else 0
        features['m5_max_move'] = m5_returns.abs().max() if len(m5_returns) > 0 else 0
        
        # M5 trend features
        features['m5_trend_strength'] = (features['m5_close_last'] - features['m5_open_first']) / features['m5_range'] if features['m5_range'] > 0 else 0
        features['m5_body_ratio'] = abs(features['m5_price_change']) / features['m5_range'] if features['m5_range'] > 0 else 0
        
        # M5 technical indicators (latest values)
        latest_m5 = m5_subset.iloc[-1]
        
        # Copy key technical indicators from latest M5 bar
        technical_features = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'atr',
            'bb_upper', 'bb_lower', 'bb_position', 'supertrend', 'supertrend_direction',
            'macd', 'macd_signal', 'stoch_k', 'stoch_d'
        ]
        
        for feature in technical_features:
            if feature in latest_m5:
                features[f'm5_{feature}'] = latest_m5[feature]
        
        # M5 session and time features
        features['m5_hour'] = latest_m5.get('hour', m15_timestamp.hour)
        features['m5_session_asia'] = latest_m5.get('session_asia', 0)
        features['m5_session_london'] = latest_m5.get('session_london', 0)
        features['m5_session_newyork'] = latest_m5.get('session_newyork', 0)
        features['m5_high_vol_period'] = latest_m5.get('high_vol_period', 0)
        
        # M5 momentum features
        if len(m5_subset) >= 3:
            features['m5_momentum_3'] = m5_subset['close'].iloc[-1] - m5_subset['close'].iloc[0]
            features['m5_acceleration'] = (m5_subset['close'].iloc[-1] - m5_subset['close'].iloc[-2]) - (m5_subset['close'].iloc[-2] - m5_subset['close'].iloc[-3])
        
        return features
    
    def prepare_features_and_targets(self, features_df: pd.DataFrame, targets_df: pd.DataFrame, target_col: str = 'target_direction_1') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and targets for training.
        
        Args:
            features_df: Features DataFrame
            targets_df: Targets DataFrame (same as features_df for multi-timeframe)
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) for training
        """
        logger.info("Preparing features and targets...")
        
        # For multi-timeframe features, targets are in the same DataFrame
        if target_col in features_df.columns:
            data = features_df.copy()
        else:
            raise ValueError(f"Target column '{target_col}' not found in features DataFrame")
        
        # Remove timestamp and target columns for features
        feature_cols = [col for col in data.columns if col != 'timestamp' and not col.startswith('target_')]
        X = data[feature_cols].copy()
        
        # Get target
        y = data[target_col].copy()
        
        # Remove rows with NaN targets first
        valid_target_mask = ~y.isna()
        X = X[valid_target_mask]
        y = y[valid_target_mask]
        
        # Fill NaN features with 0 instead of removing rows
        X = X.fillna(0)
        
        # Alternative: Remove features with too many NaN values
        nan_threshold = 0.5  # Remove features with >50% NaN
        nan_ratio = X.isna().mean()
        features_to_keep = nan_ratio[nan_ratio <= nan_threshold].index
        X = X[features_to_keep]
        
        logger.info(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def create_time_series_splits(self, X: pd.DataFrame, y: pd.Series, features_df: pd.DataFrame) -> List[Tuple]:
        """
        Create time-series splits with session awareness.
        
        Args:
            X: Features
            y: Targets  
            features_df: Original features with timestamps
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        logger.info("Creating time-series splits...")
        
        n_splits = self.config['validation']['cv_folds']
        
        if self.config['validation']['session_aware']:
            # Create session-aware splits
            splits = []
            timestamps = features_df.loc[X.index, 'timestamp']
            
            # Sort by timestamp
            sorted_idx = timestamps.sort_values().index
            
            # Create splits
            n_samples = len(sorted_idx)
            split_size = n_samples // (n_splits + 1)
            
            for i in range(n_splits):
                train_end = (i + 1) * split_size
                val_start = train_end
                val_end = min(val_start + split_size, n_samples)
                
                train_idx = sorted_idx[:train_end]
                val_idx = sorted_idx[val_start:val_end]
                
                if len(val_idx) > 0:
                    splits.append((train_idx, val_idx))
            
            logger.info(f"Created {len(splits)} session-aware splits")
            return splits
        else:
            # Standard time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            return list(tscv.split(X))
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, cv_splits: List) -> Dict:
        """
        Optimize hyperparameters using GridSearch.
        
        Args:
            X: Features
            y: Targets
            cv_splits: Cross-validation splits
            
        Returns:
            Best hyperparameters
        """
        logger.info("Optimizing hyperparameters...")
        
        # Define parameter grid for XAU/USD optimization (smaller grid for small dataset)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [6, 8],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5],
            'class_weight': ['balanced']
        }
        
        # Base model
        base_model = RandomForestClassifier(
            criterion=self.config['random_forest']['criterion'],
            bootstrap=self.config['random_forest']['bootstrap'],
            oob_score=self.config['random_forest']['oob_score'],
            random_state=self.config['random_forest']['random_state'],
            n_jobs=self.config['random_forest']['n_jobs']
        )
        
        # Grid search with custom scoring
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_splits,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        logger.info(f"Best hyperparameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series, cv_splits: List, best_params: Dict) -> Dict:
        """
        Train ensemble of Random Forest models.
        
        Args:
            X: Features
            y: Targets
            cv_splits: Cross-validation splits
            best_params: Optimized hyperparameters
            
        Returns:
            Dictionary of trained models and metrics
        """
        logger.info("Training ensemble of Random Forest models...")
        
        n_models = self.config['ensemble']['n_models']
        feature_subset_ratio = self.config['ensemble']['feature_subset_ratio']
        
        models = {}
        cv_scores = {}
        feature_importances = {}
        
        for i in range(n_models):
            logger.info(f"Training model {i+1}/{n_models}")
            
            # Create feature subset
            n_features = int(len(X.columns) * feature_subset_ratio)
            np.random.seed(42 + i)  # Different seed for each model
            selected_features = np.random.choice(X.columns, size=n_features, replace=False)
            X_subset = X[selected_features]
            
            # Create model with best parameters
            model = RandomForestClassifier(
                **best_params,
                random_state=42 + i  # Different seed for each model
            )
            
            # Cross-validation
            cv_scores_model = []
            for train_idx, val_idx in cv_splits:
                X_train_cv, X_val_cv = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train and validate
                model.fit(X_train_cv, y_train_cv)
                y_pred = model.predict(X_val_cv)
                score = accuracy_score(y_val_cv, y_pred)
                cv_scores_model.append(score)
            
            # Train final model on full training set
            train_idx = cv_splits[-1][0]  # Use last split's training set
            X_train_final = X_subset.iloc[train_idx]
            y_train_final = y.iloc[train_idx]
            
            model.fit(X_train_final, y_train_final)
            
            # Store model and metrics
            models[f'model_{i}'] = {
                'model': model,
                'features': selected_features,
                'scaler': RobustScaler().fit(X_train_final)
            }
            
            cv_scores[f'model_{i}'] = {
                'mean_cv_score': np.mean(cv_scores_model),
                'std_cv_score': np.std(cv_scores_model),
                'cv_scores': cv_scores_model
            }
            
            # Feature importance
            importance_dict = dict(zip(selected_features, model.feature_importances_))
            feature_importances[f'model_{i}'] = importance_dict
            
            logger.info(f"Model {i+1} CV score: {np.mean(cv_scores_model):.4f} ± {np.std(cv_scores_model):.4f}")
        
        return {
            'models': models,
            'cv_scores': cv_scores,
            'feature_importances': feature_importances
        }
    
    def calibrate_probabilities(self, models: Dict, X: pd.DataFrame, y: pd.Series, cv_splits: List) -> Dict:
        """
        Calibrate model probabilities for better confidence estimates.
        
        Args:
            models: Dictionary of trained models
            X: Features
            y: Targets
            cv_splits: Cross-validation splits
            
        Returns:
            Dictionary of calibrated models
        """
        logger.info("Calibrating model probabilities...")
        
        calibrated_models = {}
        
        for model_name, model_info in models.items():
            logger.info(f"Calibrating {model_name}")
            
            model = model_info['model']
            features = model_info['features']
            scaler = model_info['scaler']
            
            X_subset = X[features]
            
            # Use validation split for calibration
            val_idx = cv_splits[-1][1]
            X_val = X_subset.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # Scale features
            X_val_scaled = scaler.transform(X_val)
            
            # Calibrate probabilities
            calibrator = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            calibrator.fit(X_val_scaled, y_val)
            
            calibrated_models[model_name] = {
                'model': calibrator,
                'features': features,
                'scaler': scaler
            }
        
        return calibrated_models
    
    def evaluate_ensemble(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate ensemble performance.
        
        Args:
            models: Dictionary of calibrated models
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating ensemble performance...")
        
        # Individual model predictions
        individual_predictions = {}
        individual_probabilities = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            features = model_info['features']
            scaler = model_info['scaler']
            
            X_test_subset = X_test[features]
            X_test_scaled = scaler.transform(X_test_subset)
            
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
            
            individual_predictions[model_name] = y_pred
            individual_probabilities[model_name] = y_prob
        
        # Ensemble predictions (soft voting)
        ensemble_probabilities = np.mean(list(individual_probabilities.values()), axis=0)
        ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
        
        # Confidence scores
        confidence_scores = np.max(ensemble_probabilities, axis=1)
        
        # Evaluation metrics
        metrics = {
            'accuracy': accuracy_score(y_test, ensemble_predictions),
            'precision': precision_score(y_test, ensemble_predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, ensemble_predictions, average='weighted', zero_division=0),
            'f1': f1_score(y_test, ensemble_predictions, average='weighted', zero_division=0),
            'auc': roc_auc_score(y_test, ensemble_probabilities[:, 1]) if len(np.unique(y_test)) == 2 else 0
        }
        
        # Confidence-based metrics
        confidence_threshold = self.config['ensemble']['confidence_threshold']
        high_confidence_mask = confidence_scores >= confidence_threshold
        
        if high_confidence_mask.sum() > 0:
            metrics['high_confidence_accuracy'] = accuracy_score(
                y_test[high_confidence_mask], 
                ensemble_predictions[high_confidence_mask]
            )
            metrics['high_confidence_samples'] = high_confidence_mask.sum()
            metrics['high_confidence_ratio'] = high_confidence_mask.mean()
        else:
            metrics['high_confidence_accuracy'] = 0
            metrics['high_confidence_samples'] = 0
            metrics['high_confidence_ratio'] = 0
        
        logger.info(f"Ensemble Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"High Confidence Accuracy: {metrics['high_confidence_accuracy']:.4f}")
        logger.info(f"High Confidence Ratio: {metrics['high_confidence_ratio']:.2%}")
        
        return {
            'metrics': metrics,
            'predictions': ensemble_predictions,
            'probabilities': ensemble_probabilities,
            'confidence_scores': confidence_scores,
            'individual_predictions': individual_predictions,
            'individual_probabilities': individual_probabilities
        }
    
    def save_models_and_results(self, models: Dict, results: Dict, output_dir: str = "models") -> str:
        """
        Save trained models and results.
        
        Args:
            models: Dictionary of trained models
            results: Training and evaluation results
            output_dir: Output directory
            
        Returns:
            Path to saved models directory
        """
        logger.info("Saving models and results...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        models_dir = Path(output_dir) / f"advanced_rf_ensemble_{timestamp}"
        models_dir.mkdir(exist_ok=True)
        
        # Save individual models
        for model_name, model_info in models.items():
            model_file = models_dir / f"{model_name}.joblib"
            joblib.dump(model_info, model_file)
        
        # Save configuration
        config_file = models_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Save results
        results_file = models_dir / "training_results.yaml"
        # Convert numpy arrays to lists for YAML serialization
        serializable_results = self._make_serializable(results)
        with open(results_file, 'w') as f:
            yaml.dump(serializable_results, f, default_flow_style=False)
        
        logger.info(f"Models and results saved to: {models_dir}")
        return str(models_dir)
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def train_advanced_ensemble(self, m5_file: str, m15_file: str, output_dir: str = "models") -> str:
        """
        Main training pipeline for advanced ensemble.
        
        Args:
            m5_file: Path to M5 features file
            m15_file: Path to M15 features file
            output_dir: Output directory for models
            
        Returns:
            Path to saved models directory
        """
        logger.info("🚀 Starting Advanced Random Forest Ensemble Training")
        logger.info("=" * 60)
        
        # Step 1: Load multi-timeframe data
        features_df, targets_df = self.load_multi_timeframe_data(m5_file, m15_file)
        
        # Step 2: Prepare features and targets
        X, y = self.prepare_features_and_targets(features_df, targets_df, 'target_direction_1')
        
        # Step 3: Create time-series splits
        cv_splits = self.create_time_series_splits(X, y, features_df)
        
        # Step 4: Optimize hyperparameters
        best_params = self.optimize_hyperparameters(X, y, cv_splits)
        
        # Step 5: Train ensemble models
        ensemble_results = self.train_ensemble_models(X, y, cv_splits, best_params)
        
        # Step 6: Calibrate probabilities
        calibrated_models = self.calibrate_probabilities(ensemble_results['models'], X, y, cv_splits)
        
        # Step 7: Evaluate on test set
        test_idx = cv_splits[-1][1]  # Use last validation split as test
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        evaluation_results = self.evaluate_ensemble(calibrated_models, X_test, y_test)
        
        # Step 8: Combine all results
        final_results = {
            'best_hyperparameters': best_params,
            'ensemble_training': ensemble_results,
            'evaluation': evaluation_results,
            'dataset_info': {
                'total_samples': len(X),
                'n_features': len(X.columns),
                'test_samples': len(X_test),
                'target_distribution': y.value_counts().to_dict()
            }
        }
        
        # Step 9: Save models and results
        models_dir = self.save_models_and_results(calibrated_models, final_results, output_dir)
        
        logger.info("=" * 60)
        logger.info("✅ Advanced Random Forest Ensemble Training Completed")
        logger.info(f"📊 Final Accuracy: {evaluation_results['metrics']['accuracy']:.4f}")
        logger.info(f"🎯 High Confidence Accuracy: {evaluation_results['metrics']['high_confidence_accuracy']:.4f}")
        logger.info(f"💼 Models saved to: {models_dir}")
        logger.info("=" * 60)
        
        return models_dir


if __name__ == "__main__":
    # Test the advanced trainer
    trainer = AdvancedRandomForestTrainer()
    
    # Example usage with multi-timeframe features
    m5_features = "data/processed/multi_timeframe_features_with_targets.csv"
    m15_features = "data/processed/multi_timeframe_features_with_targets.csv"  # Same file for multi-timeframe
    
    try:
        models_dir = trainer.train_advanced_ensemble(m5_features, m15_features)
        print(f"🎉 Training completed successfully!")
        print(f"📁 Models saved to: {models_dir}")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
