#!/usr/bin/env python3
"""
Random Forest Baseline Trainer for SignaMentis

This module implements a Random Forest baseline model for XAU/USD trading
with essential features, proper data splits, probability calibration,
and comprehensive evaluation.
"""

import sys
import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestBaselineTrainer:
    """
    Random Forest baseline trainer for XAU/USD direction prediction.
    
    Features:
    - Essential technical indicators (ATR, SuperTrend, SMA/EMA, RSI)
    - Time-based features (session, time-of-day)
    - Market features (spread, volume)
    - Feature selection based on importance
    - Probability calibration
    - Comprehensive evaluation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Random Forest baseline trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Model parameters
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 10)
        self.min_samples_split = self.config.get('min_samples_split', 5)
        self.min_samples_leaf = self.config.get('min_samples_leaf', 2)
        self.random_state = self.config.get('random_state', 42)
        
        # Training parameters
        self.calibrate_probabilities = self.config.get('calibrate_probabilities', True)
        self.feature_selection = self.config.get('feature_selection', False)  # Temporarily disable for compatibility
        self.min_feature_importance = self.config.get('min_feature_importance', 0.01)
        self.cv_folds = self.config.get('cv_folds', 5)
        
        # Initialize components
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_importance = None
        self.performance_metrics = {}
        
        logger.info("RandomForestBaselineTrainer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f).get('random_forest', {})
        else:
            # Default configuration - optimized to prevent overfitting
            return {
                'n_estimators': 100,
                'max_depth': 8,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': 42,
                'calibrate_probabilities': True,
                'feature_selection': True,
                'min_feature_importance': 0.005,
                'cv_folds': 5,
                'bootstrap': True,
                'oob_score': True,
                'class_weight': 'balanced',
                'criterion': 'gini',
                'max_leaf_nodes': 50,
                'min_impurity_decrease': 0.001
            }
    
    def load_splits(self, splits_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, validation, and test splits.
        
        Args:
            splits_dir: Directory containing data splits
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Loading data splits from {splits_dir}")
        
        splits_path = Path(splits_dir)
        
        # Load splits
        train_df = pd.read_csv(splits_path / "train.csv")
        val_df = pd.read_csv(splits_path / "val.csv")
        test_df = pd.read_csv(splits_path / "test.csv")
        
        logger.info(f"Data splits loaded:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and targets from dataframe.
        
        Args:
            df: Input dataframe with features and targets
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Define feature columns (exclude timestamp and target columns)
        feature_cols = [col for col in df.columns if not col.startswith('target_') and col != 'timestamp']
        
        # Extract features and target
        X = df[feature_cols].values
        y = df['target_direction'].values  # Binary classification: 0=DOWN, 1=UP
        
        logger.info(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        return X, y, feature_cols
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train Random Forest model with optimized parameters to prevent overfitting.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained Random Forest model
        """
        logger.info("Training Random Forest model with optimized parameters")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create optimized Random Forest
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            bootstrap=self.config.get('bootstrap', True),
            oob_score=self.config.get('oob_score', True),
            class_weight=self.config.get('class_weight', 'balanced'),
            criterion=self.config.get('criterion', 'gini'),
            max_leaf_nodes=self.config.get('max_leaf_nodes', 50),
            min_impurity_decrease=self.config.get('min_impurity_decrease', 0.001),
            max_features=self.config.get('max_features', 'sqrt'),
            n_jobs=self.config.get('n_jobs', -1)
        )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Log training results
        oob_score = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else 'N/A'
        logger.info(f"Random Forest training completed:")
        logger.info(f"  OOB Score: {oob_score}")
        logger.info(f"  Feature Importance: {self.model.feature_importances_.shape}")
        
        return self.model
    
    def _calibrate_probabilities(self, X_val: np.ndarray, y_val: np.ndarray) -> CalibratedClassifierCV:
        """
        Calibrate model probabilities using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Calibrated classifier
        """
        logger.info("Starting probability calibration")
        
        logger.info("Calibrating model probabilities")
        
        # Create calibrated classifier
        calibrated_model = CalibratedClassifierCV(
            self.model, 
            cv=self.cv_folds, 
            method='isotonic'
        )
        
        # Fit calibration
        calibrated_model.fit(X_val, y_val)
        
        # Store calibrated model
        self.calibrated_model = calibrated_model
        
        logger.info("Probability calibration completed")
        return calibrated_model
    
    def select_features(self, X_train: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Select most important features.
        
        Args:
            X_train: Training features
            feature_names: Feature names
            
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        if not self.feature_selection:
            logger.info("Feature selection disabled")
            return X_train, feature_names
        
        logger.info("Selecting most important features")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Select features above threshold
        selected_features = feature_importance_df[
            feature_importance_df['importance'] >= self.min_feature_importance
        ]
        
        selected_feature_names = selected_features['feature'].tolist()
        selected_indices = [feature_names.index(name) for name in selected_feature_names]
        
        # Select features from X
        X_selected = X_train[:, selected_indices]
        
        logger.info(f"Feature selection completed:")
        logger.info(f"  Original features: {len(feature_names)}")
        logger.info(f"  Selected features: {len(selected_feature_names)}")
        logger.info(f"  Threshold: {self.min_feature_importance}")
        
        # Store feature importance
        self.feature_importance = feature_importance_df
        
        return X_selected, selected_feature_names
    
    def evaluate_model(self, 
                      X_val: np.ndarray, 
                      y_val: np.ndarray, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance on validation and test sets.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info("Evaluating model performance")
        
        # Use calibrated model if available
        model_to_evaluate = self.calibrated_model if self.calibrated_model else self.model
        
        # Validation performance
        y_val_pred = model_to_evaluate.predict(X_val)
        y_val_proba = model_to_evaluate.predict_proba(X_val)
        
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_report = classification_report(y_val, y_val_pred, output_dict=True)
        
        # Test performance
        y_test_pred = model_to_evaluate.predict(X_test)
        y_test_proba = model_to_evaluate.predict_proba(X_test)
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Store performance metrics
        self.performance_metrics = {
            'validation': {
                'accuracy': val_accuracy,
                'classification_report': val_report,
                'predictions': y_val_pred,
                'probabilities': y_val_proba
            },
            'test': {
                'accuracy': test_accuracy,
                'classification_report': test_report,
                'predictions': y_test_pred,
                'probabilities': y_test_proba
            }
        }
        
        logger.info("Model evaluation completed:")
        logger.info(f"  Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        
        return self.performance_metrics
    
    def generate_feature_importance_report(self, output_dir: str = "results") -> str:
        """
        Generate feature importance report.
        
        Args:
            output_dir: Output directory for report
            
        Returns:
            Path to generated report
        """
        if self.feature_importance is None:
            logger.warning("No feature importance data available")
            return None
        
        logger.info("Generating feature importance report")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report
        report = {
            'model_type': 'RandomForestClassifier',
            'generation_date': timestamp,
            'feature_importance': {
                'top_10_features': self.feature_importance.head(10).to_dict('records'),
                'all_features': self.feature_importance.to_dict('records'),
                'summary': {
                    'total_features': len(self.feature_importance),
                    'importance_threshold': self.min_feature_importance,
                    'selected_features': len(self.feature_importance[
                        self.feature_importance['importance'] >= self.min_feature_importance
                    ])
                }
            }
        }
        
        # Save report
        report_path = Path(output_dir) / f"rf_feature_importance_{timestamp}.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"Feature importance report saved: {report_path}")
        return str(report_path)
    
    def save_model(self, output_dir: str = "models") -> str:
        """
        Save trained model and metadata.
        
        Args:
            output_dir: Output directory for model files
            
        Returns:
            Path to saved model
        """
        logger.info("Saving Random Forest model")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main model
        model_path = Path(output_dir) / f"rf_baseline_{timestamp}.joblib"
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector
        }, model_path)
        
        # Save calibrated model if available
        calibrated_path = None
        if self.calibrated_model:
            calibrated_path = Path(output_dir) / f"rf_baseline_calibrated_{timestamp}.joblib"
            joblib.dump(self.calibrated_model, calibrated_path)
        
        # Save metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'training_date': timestamp,
            'parameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'random_state': self.random_state
            },
            'performance': self.performance_metrics,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None
        }
        
        metadata_path = Path(output_dir) / f"rf_baseline_metadata_{timestamp}.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        logger.info(f"Model saved successfully:")
        logger.info(f"  Main model: {model_path}")
        if calibrated_path:
            logger.info(f"  Calibrated model: {calibrated_path}")
        logger.info(f"  Metadata: {metadata_path}")
        
        return str(model_path)
    
    def train_baseline(self, splits_dir: str, output_dir: str = "results") -> Dict:
        """
        Complete Random Forest baseline training pipeline.
        
        Args:
            splits_dir: Directory containing data splits
            output_dir: Output directory for results
            
        Returns:
            Dictionary with training results and file paths
        """
        logger.info("Starting Random Forest baseline training pipeline")
        
        # 1. Load splits
        train_df, val_df, test_df = self.load_splits(splits_dir)
        
        # 2. Prepare features
        X_train, y_train, train_features = self.prepare_features(train_df)
        X_val, y_val, val_features = self.prepare_features(val_df)
        X_test, y_test, test_features = self.prepare_features(test_df)
        
        # 3. Train model
        self.train_model(X_train, y_train)
        
        # 4. Calibrate probabilities (before feature selection to avoid scaler mismatch)
        if self.calibrate_probabilities:
            self._calibrate_probabilities(X_val, y_val)
        
        # 5. Select features (after calibration)
        X_train_selected, selected_features = self.select_features(X_train, train_features)
        X_val_selected, _ = self.select_features(X_val, val_features)
        X_test_selected, _ = self.select_features(X_test, test_features)
        
        # 6. Evaluate model
        performance = self.evaluate_model(X_val_selected, y_val, X_test_selected, y_test)
        
        # 7. Generate reports
        feature_report_path = self.generate_feature_importance_report(output_dir)
        
        # 8. Save model
        model_path = self.save_model()
        
        # 9. Compile results
        results = {
            'model_path': model_path,
            'feature_report_path': feature_report_path,
            'performance': performance,
            'selected_features': selected_features,
            'training_summary': {
                'train_samples': len(X_train_selected),
                'val_samples': len(X_val_selected),
                'test_samples': len(X_test_selected),
                'total_features': len(selected_features),
                'validation_accuracy': performance['validation']['accuracy'],
                'test_accuracy': performance['test']['accuracy']
            }
        }
        
        logger.info("Random Forest baseline training pipeline completed successfully")
        return results


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    # Example usage
    trainer = RandomForestBaselineTrainer()
    
    # Train baseline
    results = trainer.train_baseline(
        splits_dir="data/splits",
        output_dir="results"
    )
    
    print("Random Forest baseline training completed!")
    print(f"Results: {results}")