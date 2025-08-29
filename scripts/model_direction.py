#!/usr/bin/env python3
"""
Directional model for SignaMentis AI Trading System.
Implements high-confidence directional prediction with LightGBM and calibration.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectionalModel:
    """High-confidence directional prediction model."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize DirectionalModel with configuration."""
        self.config = config or self._default_config()
        self.model = None
        self.calibrator = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.best_params = None
        self.calibration_threshold = None
        self.high_confidence_threshold = None
        
    def _default_config(self) -> Dict:
        """Default configuration for directional model."""
        return {
            'algorithm': 'lightgbm',
            'class_weight': 'balanced_subsample',
            'early_stopping_rounds': 100,
            'cv_folds': 5,
            'optuna_timeout_sec': 3600,
            'min_precision_val': 0.97,
            'min_recall_val': 0.25,
            'n_jobs': 4,
            'random_state': 42
        }
    
    def prepare_features(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (features, target)
        """
        # Get feature columns (exclude timestamp/targets and future-derived engineered cols)
        forbidden = {'estimated_atr','mfe_long','mfe_short','mfe_long_atr','mfe_short_atr'}
        feature_cols = [
            col for col in df.columns
            if not col.startswith('target_') and col != 'timestamp' and col not in forbidden
        ]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Store feature columns for consistency
        self.feature_columns = sorted(feature_cols)
        X = X[self.feature_columns]
        
        # Remove features with too many NaNs
        nan_threshold = 0.3
        nan_ratio = X.isna().mean()
        features_to_keep = nan_ratio[nan_ratio <= nan_threshold].index
        X = X[features_to_keep]
        
        # Update feature columns
        self.feature_columns = list(features_to_keep)
        
        # Fill remaining NaNs
        X = X.fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Remove rows with NaN targets
        valid_target_mask = ~y.isna()
        X = X[valid_target_mask]
        y = y[valid_target_mask]
        
        logger.info(f"Features prepared: {X.shape[1]} features, {len(y)} samples")
        
        return X, y
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Optimize hyperparameters using TimeSeriesSplit cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Best hyperparameters
        """
        logger.info("Optimizing hyperparameters...")
        
        # Create time series splits
        cv_folds = self.config.get('cv_folds', 5)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Define parameter grid for LightGBM
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'min_child_samples': [20, 50, 100],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid, max_combinations=30)
        
        best_score = 0
        best_params = None
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}")
            
            # Set common parameters
            params.update({
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'class_weight': 'balanced',  # LightGBM supports balanced_subsample, but use balanced for compatibility
                'random_state': self.config.get('random_state', 42),
                'n_jobs': self.config.get('n_jobs', 4),
                'verbose': -1
            })
            
            # Cross-validate
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Handle infinite values
                X_train_fold = X_train_fold.replace([np.inf, -np.inf], 0)
                X_val_fold = X_val_fold.replace([np.inf, -np.inf], 0)
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train_fold)
                X_val_scaled = self.scaler.transform(X_val_fold)
                
                # Train model
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_train_scaled, y_train_fold,
                    eval_set=[(X_val_scaled, y_val_fold)],
                    callbacks=[lgb.early_stopping(self.config.get('early_stopping_rounds', 100), verbose=False)]
                )
                
                # Predict
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                score = roc_auc_score(y_val_fold, y_pred_proba)
                cv_scores.append(score)
            
            # Calculate mean CV score
            mean_score = np.mean(cv_scores)
            logger.info(f"CV Score: {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
                logger.info(f"New best score: {best_score:.4f}")
        
        self.best_params = best_params
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")
        
        return best_params
    
    def _generate_param_combinations(self, param_grid: Dict, max_combinations: int = 30) -> List[Dict]:
        """Generate parameter combinations for grid search."""
        import itertools
        
        # Get all possible combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        # Limit to max_combinations
        if len(combinations) > max_combinations:
            # Randomly sample combinations
            random_state = self.config.get('random_state', 42)
            np.random.seed(random_state)
            selected_indices = np.random.choice(len(combinations), max_combinations, replace=False)
            combinations = [combinations[i] for i in selected_indices]
        
        # Convert to list of dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the directional model with best hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.info("Training directional model...")
        
        if self.best_params is None:
            raise ValueError("Must call optimize_hyperparameters first")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train final model
        self.model = lgb.LGBMClassifier(**self.best_params)
        self.model.fit(X_scaled, y)
        
        logger.info("Model training completed")
    
    def calibrate_probabilities(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Calibrate model probabilities using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info("Calibrating probabilities...")
        
        if self.model is None:
            raise ValueError("Must train model first")
        
        # Scale validation features
        X_val_scaled = self.scaler.transform(X_val)
        
        # Get raw probabilities
        raw_probs = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Create calibration plot data
        self._create_calibration_plot(raw_probs, y_val)
        
        # Calibrate using isotonic regression
        self.calibrator = CalibratedClassifierCV(
            estimator=self.model,
            cv='prefit',
            method='isotonic'
        )
        
        # Fit calibrator on validation data
        self.calibrator.fit(X_val_scaled, y_val)
        
        # Get calibrated probabilities
        calibrated_probs = self.calibrator.predict_proba(X_val_scaled)[:, 1]
        
        # Find high-confidence threshold for 97% precision
        self._find_high_confidence_threshold(calibrated_probs, y_val)
        
        logger.info("Probability calibration completed")
    
    def _create_calibration_plot(self, raw_probs: np.ndarray, y_true: pd.Series) -> None:
        """Create calibration plot data for validation."""
        from sklearn.calibration import calibration_curve
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, raw_probs, n_bins=10
        )
        
        # Store for later plotting
        self.calibration_data = {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'raw_probs': raw_probs,
            'y_true': y_true
        }
    
    def _find_high_confidence_threshold(self, calibrated_probs: np.ndarray, y_true: pd.Series) -> None:
        """Find threshold for high-confidence predictions (≥97% precision)."""
        # Sort probabilities and find threshold
        sorted_probs = np.sort(calibrated_probs)[::-1]
        
        # Calculate precision at different thresholds
        thresholds = []
        precisions = []
        recalls = []
        coverages = []
        
        for threshold in np.arange(0.5, 0.99, 0.01):
            high_conf_mask = calibrated_probs >= threshold
            if high_conf_mask.sum() > 0:
                y_pred_high_conf = (calibrated_probs >= threshold).astype(int)
                y_true_high_conf = y_true[high_conf_mask]
                y_pred_high_conf = y_pred_high_conf[high_conf_mask]
                
                precision = precision_score(y_true_high_conf, y_pred_high_conf, zero_division=0)
                recall = recall_score(y_true_high_conf, y_pred_high_conf, zero_division=0)
                coverage = high_conf_mask.mean()
                
                thresholds.append(threshold)
                precisions.append(precision)
                recalls.append(recall)
                coverages.append(coverage)
        
        # Find threshold meeting precision requirement
        target_precision = self.config['min_precision_val']
        target_recall = self.config['min_recall_val']
        
        valid_thresholds = []
        for i, (thresh, prec, rec, cov) in enumerate(zip(thresholds, precisions, recalls, coverages)):
            if prec >= target_precision and rec >= target_recall:
                valid_thresholds.append((thresh, prec, rec, cov))
        
        if valid_thresholds:
            # Choose threshold with highest coverage
            best_threshold, best_prec, best_rec, best_cov = max(valid_thresholds, key=lambda x: x[3])
            
            self.high_confidence_threshold = best_threshold
            self.calibration_threshold = best_threshold
            
            logger.info(f"High-confidence threshold: {best_threshold:.3f}")
            logger.info(f"Precision: {best_prec:.3f}, Recall: {best_rec:.3f}, Coverage: {best_cov:.3f}")
        else:
            logger.warning(f"Could not find threshold meeting requirements: Precision ≥ {target_precision}, Recall ≥ {target_recall}")
            # Use default threshold
            self.high_confidence_threshold = 0.8
            self.calibration_threshold = 0.8
    
    def predict(self, X: pd.DataFrame, return_confidence: bool = False) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions (and confidence scores if requested)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not set")
        
        # Align features (add missing columns as 0 to avoid KeyError)
        missing_cols = [c for c in self.feature_columns if c not in X.columns]
        if missing_cols:
            for c in missing_cols:
                X[c] = 0
        X_aligned = X[self.feature_columns].copy().replace([np.inf, -np.inf], 0).fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X_aligned)
        
        if self.calibrator is not None:
            # Use calibrated probabilities
            probs = self.calibrator.predict_proba(X_scaled)[:, 1]
        else:
            # Use raw probabilities
            probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # Make predictions
        predictions = (probs >= 0.5).astype(int)
        
        if return_confidence:
            return predictions, probs
        else:
            return predictions
    
    def predict_high_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make high-confidence predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, confidence_scores, high_conf_mask)
        """
        if self.high_confidence_threshold is None:
            raise ValueError("High-confidence threshold not set. Run calibration first.")
        
        predictions, confidence_scores = self.predict(X, return_confidence=True)
        
        # Identify high-confidence predictions
        high_conf_mask = confidence_scores >= self.high_confidence_threshold
        
        return predictions, confidence_scores, high_conf_mask
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, split_name: str = "test") -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True targets
            split_name: Name of the data split
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {split_name} set...")
        
        # Get predictions
        predictions, confidence_scores = self.predict(X, return_confidence=True)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1': f1_score(y, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y, confidence_scores)
        }
        
        # High-confidence metrics
        if self.high_confidence_threshold is not None:
            high_conf_mask = confidence_scores >= self.high_confidence_threshold
            
            if high_conf_mask.sum() > 0:
                y_high_conf = y[high_conf_mask]
                pred_high_conf = predictions[high_conf_mask]
                conf_high_conf = confidence_scores[high_conf_mask]
                
                high_conf_metrics = {
                    'high_conf_precision': precision_score(y_high_conf, pred_high_conf, zero_division=0),
                    'high_conf_recall': recall_score(y_high_conf, pred_high_conf, zero_division=0),
                    'high_conf_coverage': high_conf_mask.mean(),
                    'high_conf_count': high_conf_mask.sum()
                }
                metrics.update(high_conf_metrics)
        
        # Log metrics
        logger.info(f"{split_name.capitalize()} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, output_dir: str, model_name: str) -> None:
        """Save trained model and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.calibrator is not None:
            joblib.dump(self.calibrator, output_path / f"{model_name}_calibrated.joblib")
        else:
            joblib.dump(self.model, output_path / f"{model_name}.joblib")
        
        # Save scaler
        joblib.dump(self.scaler, output_path / f"{model_name}_scaler.joblib")
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'algorithm': self.config.get('algorithm', 'lightgbm'),
            'best_params': self.best_params,
            'feature_columns': self.feature_columns,
            'high_confidence_threshold': self.high_confidence_threshold,
            'calibration_threshold': self.calibration_threshold,
            'training_date': datetime.now().isoformat(),
            'config': self.config
        }
        
        with open(output_path / f"{model_name}_metadata.yaml", 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        logger.info(f"Model saved to: {output_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load trained model from file."""
        model_path = Path(model_path)
        
        # Load model
        if (model_path.parent / f"{model_path.stem}_calibrated.joblib").exists():
            self.calibrator = joblib.load(model_path.parent / f"{model_path.stem}_calibrated.joblib")
            self.model = self.calibrator.estimator_
        else:
            self.model = joblib.load(model_path)
        
        # Load scaler
        self.scaler = joblib.load(model_path.parent / f"{model_path.stem}_scaler.joblib")
        
        # Load metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.high_confidence_threshold = metadata['high_confidence_threshold']
            self.calibration_threshold = metadata['calibration_threshold']
            self.best_params = metadata['best_params']
        
        logger.info(f"Model loaded from: {model_path}")


if __name__ == "__main__":
    # Test the DirectionalModel class
    model = DirectionalModel()
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)
    
    print("Sample data created successfully")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Test model training pipeline
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y_series = pd.Series(y)
    
    print("\nTesting model pipeline...")
    
    # Optimize hyperparameters
    best_params = model.optimize_hyperparameters(X_df, y_series)
    
    # Train model
    model.train_model(X_df, y_series)
    
    # Make predictions
    predictions = model.predict(X_df)
    print(f"Predictions shape: {predictions.shape}")
    
    print("Model pipeline test completed successfully!")
