#!/usr/bin/env python3
"""
Advanced Random Forest Trainer for 3-Year XAU/USD Dataset

This module implements advanced Random Forest training with:
- Multi-timeframe features (162K+ samples)
- Time series cross-validation
- Ensemble strategies
- Hyperparameter optimization
- Probability calibration
- Feature importance analysis

Author: SignaMentis Team
Version: 3.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import joblib
import gc

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support
)
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, f_classif

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRandomForestTrainer3Years:
    """
    Advanced Random Forest trainer optimized for 3+ year datasets.
    
    Features:
    - Time series cross-validation
    - Ensemble strategies
    - Hyperparameter optimization
    - Memory-efficient processing
    - Comprehensive evaluation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the advanced Random Forest trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.models = {}
        self.feature_importance = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.validation_results = {}
        self.test_results = {}
        
        logger.info("AdvancedRandomForestTrainer3Years initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration for 3-year training."""
        return {
            'target_variables': {
                'direction_1': 'target_direction_1',  # 1 bar ahead
                'direction_3': 'target_direction_3',  # 3 bars ahead
                'direction_5': 'target_direction_5'   # 5 bars ahead
            },
            'data_splits': {
                'train_end': '2024-06-01',  # Train on 2021-2024
                'validation_end': '2025-02-01',  # Validate on 2024-2025
                'test_start': '2025-02-01'  # Test on 2025
            },
            'feature_engineering': {
                'variance_threshold': 0.01,
                'k_best_features': 40,
                'use_feature_selection': True,
                'scaling_method': 'robust'  # 'standard' or 'robust'
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample']
            },
            'cross_validation': {
                'n_splits': 5,
                'test_size': 0.2,
                'gap': 100  # Gap between train and validation
            },
            'ensemble': {
                'n_models': 5,
                'feature_subset_ratio': 0.8,
                'voting_method': 'soft',
                'confidence_threshold': 0.6
            },
            'memory_optimization': {
                'chunk_size': 10000,
                'use_dtype_optimization': True,
                'cleanup_memory': True
            }
        }
    
    def load_features_data(self, features_file: str) -> pd.DataFrame:
        """
        Load multi-timeframe features data.
        
        Args:
            features_file: Path to features CSV file
            
        Returns:
            Features DataFrame
        """
        logger.info(f"Loading features from: {features_file}")
        
        # Load data
        features_df = pd.read_csv(features_file)
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        
        # Sort by timestamp
        features_df = features_df.sort_values('timestamp').reset_index(drop=True)
        
        # Memory optimization
        if self.config['memory_optimization']['use_dtype_optimization']:
            features_df = self._optimize_dtypes(features_df)
        
        logger.info(f"Loaded features: {features_df.shape}")
        logger.info(f"Date range: {features_df['timestamp'].min()} to {features_df['timestamp'].max()}")
        
        return features_df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency."""
        # Optimize numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'timestamp':
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def create_target_variables(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for prediction.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            DataFrame with target variables added
        """
        logger.info("Creating target variables...")
        
        # Get current price column
        if 'current_close' not in features_df.columns:
            logger.warning("No 'current_close' column found. Using 'close' if available.")
            price_col = 'close' if 'close' in features_df.columns else 'current_close'
        else:
            price_col = 'current_close'
        
        # Calculate future returns
        future_returns = features_df[price_col].shift(-1).pct_change(1)
        
        # Create direction targets (1 = up, 0 = down)
        features_df['target_direction_1'] = (future_returns > 0).astype(int)
        features_df['target_direction_3'] = (features_df[price_col].shift(-3) > features_df[price_col]).astype(int)
        features_df['target_direction_5'] = (features_df[price_col].shift(-5) > features_df[price_col]).astype(int)
        
        # Create return targets
        features_df['target_return_1'] = future_returns
        features_df['target_return_3'] = features_df[price_col].pct_change(3).shift(-3)
        features_df['target_return_5'] = features_df[price_col].pct_change(5).shift(-5)
        
        # Create volatility targets
        features_df['target_volatility_1'] = features_df[price_col].pct_change().rolling(3).std().shift(-1)
        features_df['target_volatility_3'] = features_df[price_col].pct_change().rolling(3).std().shift(-3)
        
        # Store original feature columns before removing NaN targets
        self.original_feature_columns = [col for col in features_df.columns 
                                       if not col.startswith('target_') and col != 'timestamp']
        
        # Remove rows with NaN targets (end of dataset)
        features_df = features_df.dropna(subset=['target_direction_1', 'target_direction_3', 'target_direction_5'])
        
        logger.info(f"Target variables created. Final shape: {features_df.shape}")
        logger.info(f"Original feature columns: {len(self.original_feature_columns)}")
        
        return features_df
    
    def split_data_by_time(self, features_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data by time periods for train/validation/test.
        
        Args:
            features_df: Features DataFrame with targets
            
        Returns:
            Dictionary with train/validation/test splits
        """
        logger.info("Splitting data by time periods...")
        
        # Parse split dates
        train_end = pd.to_datetime(self.config['data_splits']['train_end'])
        validation_end = pd.to_datetime(self.config['data_splits']['validation_end'])
        test_start = pd.to_datetime(self.config['data_splits']['test_start'])
        
        # Create splits
        train_data = features_df[features_df['timestamp'] <= train_end].copy()
        validation_data = features_df[
            (features_df['timestamp'] > train_end) & 
            (features_df['timestamp'] <= validation_end)
        ].copy()
        test_data = features_df[features_df['timestamp'] > test_start].copy()
        
        logger.info(f"Train: {len(train_data):,} samples ({train_data['timestamp'].min()} to {train_data['timestamp'].max()})")
        logger.info(f"Validation: {len(validation_data):,} samples ({validation_data['timestamp'].min()} to {validation_data['timestamp'].max()})")
        logger.info(f"Test: {len(test_data):,} samples ({test_data['timestamp'].min()} to {test_data['timestamp'].max()})")
        
        return {
            'train': train_data,
            'validation': validation_data,
            'test': test_data
        }
    
    def prepare_features_and_targets(self, data: pd.DataFrame, target_col: str, is_training: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and targets for training.
        
        Args:
            data: Input data
            target_col: Target column name
            is_training: Whether this is training data (to determine feature selection)
            
        Returns:
            Tuple of (features, targets)
        """
        if is_training:
            # For training: use all available features and select best ones
            feature_cols = [col for col in data.columns 
                           if not col.startswith('target_') and col != 'timestamp']
            
            X = data[feature_cols].copy()
            y = data[target_col].copy()
            
            # Remove features with too many NaNs
            nan_threshold = 0.3
            nan_ratio = X.isna().mean()
            features_to_keep = nan_ratio[nan_ratio <= nan_threshold].index
            X = X[features_to_keep]
            
            # Store selected features for consistency
            self.selected_feature_columns = list(features_to_keep)
            
        else:
            # For validation/test: use same features as training
            if not hasattr(self, 'selected_feature_columns'):
                raise ValueError("Model must be trained first to get feature columns")
            
            # Ensure all required features exist
            missing_features = set(self.selected_feature_columns) - set(data.columns)
            if missing_features:
                logger.warning(f"Missing features in data: {missing_features}")
                # Add missing features with 0 values
                for feature in missing_features:
                    data[feature] = 0
            
            X = data[self.selected_feature_columns].copy()
            y = data[target_col].copy()
        
        # Fill remaining NaNs
        X = X.fillna(0)
        
        # Remove rows with NaN targets
        valid_target_mask = ~y.isna()
        X = X[valid_target_mask]
        y = y[valid_target_mask]
        
        logger.info(f"Features prepared: {X.shape[1]} features, {len(y)} samples")
        
        return X, y
    
    def create_time_series_splits(self, n_splits: int = 5, gap: int = 100) -> TimeSeriesSplit:
        """
        Create time series cross-validation splits.
        
        Args:
            n_splits: Number of splits
            gap: Gap between train and validation
            
        Returns:
            TimeSeriesSplit object
        """
        return TimeSeriesSplit(
            n_splits=n_splits,
            test_size=int(0.2 * 1000),  # 20% of data
            gap=gap
        )
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Optimize Random Forest hyperparameters using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with best parameters
        """
        logger.info("Optimizing hyperparameters...")
        
        # Create base Random Forest
        base_rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # Create time series splits
        tscv = self.create_time_series_splits(
            n_splits=self.config['cross_validation']['n_splits'],
            gap=self.config['cross_validation']['gap']
        )
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=self.config['random_forest'],
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series, 
                           best_params: Dict, target_name: str) -> Dict:
        """
        Train ensemble of Random Forest models.
        
        Args:
            X: Feature matrix
            y: Target vector
            best_params: Best hyperparameters
            target_name: Name of target variable
            
        Returns:
            Dictionary with trained models and metadata
        """
        logger.info(f"Training ensemble for {target_name}...")
        
        n_models = self.config['ensemble']['n_models']
        feature_subset_ratio = self.config['ensemble']['feature_subset_ratio']
        
        models = []
        feature_subsets = []
        
        # Create feature subsets
        n_features = X.shape[1]
        n_subset_features = int(n_features * feature_subset_ratio)
        
        for i in range(n_models):
            # Random feature subset
            feature_indices = np.random.choice(n_features, n_subset_features, replace=False)
            X_subset = X.iloc[:, feature_indices]
            
            # Train model
            model = RandomForestClassifier(
                **best_params,
                random_state=42 + i,
                n_jobs=-1,
                verbose=0
            )
            
            model.fit(X_subset, y)
            models.append(model)
            feature_subsets.append(feature_indices)
            
            logger.info(f"Trained model {i+1}/{n_models}")
        
        # Store models and metadata
        self.models[target_name] = {
            'models': models,
            'feature_subsets': feature_subsets,
            'best_params': best_params,
            'n_features': n_features,
            'n_subset_features': n_subset_features
        }
        
        logger.info(f"Ensemble training completed for {target_name}")
        
        return self.models[target_name]
    
    def calibrate_probabilities(self, X: pd.DataFrame, y: pd.Series, 
                              target_name: str) -> None:
        """
        Calibrate probabilities for ensemble models.
        
        Args:
            X: Feature matrix
            y: Target vector
            target_name: Name of target variable
        """
        logger.info(f"Calibrating probabilities for {target_name}...")
        
        if target_name not in self.models:
            raise ValueError(f"No models found for {target_name}")
        
        model_info = self.models[target_name]
        calibrated_models = []
        
        # Calibrate each model
        for i, (model, feature_indices) in enumerate(zip(model_info['models'], model_info['feature_subsets'])):
            X_subset = X.iloc[:, feature_indices]
            
            # Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(
                estimator=model,
                cv=3,
                method='isotonic'
            )
            
            calibrated_model.fit(X_subset, y)
            calibrated_models.append(calibrated_model)
            
            logger.info(f"Calibrated model {i+1}/{len(model_info['models'])}")
        
        # Update models with calibrated versions
        self.models[target_name]['calibrated_models'] = calibrated_models
        
        logger.info(f"Probability calibration completed for {target_name}")
    
    def evaluate_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                        target_name: str, split_name: str = "validation") -> Dict:
        """
        Evaluate ensemble performance.
        
        Args:
            X: Feature matrix
            y: Target vector
            target_name: Name of target variable
            split_name: Name of data split
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating ensemble for {target_name} on {split_name}...")
        
        if target_name not in self.models:
            raise ValueError(f"No models found for {target_name}")
        
        model_info = self.models[target_name]
        
        # Get predictions from all models
        predictions = []
        probabilities = []
        
        for i, (model, feature_indices) in enumerate(zip(model_info['models'], model_info['feature_subsets'])):
            X_subset = X.iloc[:, feature_indices]
            
            # Get predictions
            pred = model.predict(X_subset)
            prob = model.predict_proba(X_subset)[:, 1] if hasattr(model, 'predict_proba') else pred
            
            predictions.append(pred)
            probabilities.append(prob)
        
        # Ensemble predictions
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Voting
        if self.config['ensemble']['voting_method'] == 'hard':
            ensemble_pred = np.mean(predictions, axis=0) > 0.5
        else:  # soft voting
            ensemble_pred = np.mean(probabilities, axis=0) > 0.5
        
        # Calculate metrics
        accuracy = accuracy_score(y, ensemble_pred)
        
        # Classification report
        report = classification_report(y, ensemble_pred, output_dict=True)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y, np.mean(probabilities, axis=0))
        except:
            roc_auc = 0.5
        
        # Store results
        results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': report,
            'predictions': ensemble_pred,
            'probabilities': np.mean(probabilities, axis=0)
        }
        
        if split_name == "validation":
            self.validation_results[target_name] = results
        elif split_name == "test":
            self.test_results[target_name] = results
        
        logger.info(f"{split_name.capitalize()} results for {target_name}:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def train_all_targets(self, features_file: str) -> Dict:
        """
        Train models for all target variables.
        
        Args:
            features_file: Path to features CSV file
            
        Returns:
            Dictionary with training results
        """
        logger.info("🚀 Starting comprehensive training for all targets...")
        logger.info("=" * 60)
        
        try:
            # Load features
            features_df = self.load_features_data(features_file)
            
            # Create target variables
            features_df = self.create_target_variables(features_df)
            
            # Split data by time
            data_splits = self.split_data_by_time(features_df)
            
            training_results = {}
            
            # Train for each target variable
            for target_key, target_col in self.config['target_variables'].items():
                logger.info(f"\n🎯 Training for target: {target_key} ({target_col})")
                logger.info("-" * 40)
                
                # Prepare training data
                X_train, y_train = self.prepare_features_and_targets(data_splits['train'], target_col, is_training=True)
                
                if len(X_train) == 0:
                    logger.warning(f"No training data available for {target_key}")
                    continue
                
                # Optimize hyperparameters
                best_params = self.optimize_hyperparameters(X_train, y_train)
                
                # Train ensemble
                model_info = self.train_ensemble_model(X_train, y_train, best_params, target_key)
                
                # Calibrate probabilities
                self.calibrate_probabilities(X_train, y_train, target_key)
                
                # Evaluate on validation set
                if len(data_splits['validation']) > 0:
                    X_val, y_val = self.prepare_features_and_targets(data_splits['validation'], target_col, is_training=False)
                    if len(X_val) > 0:
                        val_results = self.evaluate_ensemble(X_val, y_val, target_key, "validation")
                
                # Evaluate on test set
                if len(data_splits['test']) > 0:
                    X_test, y_test = self.prepare_features_and_targets(data_splits['test'], target_col, is_training=False)
                    if len(X_test) > 0:
                        test_results = self.evaluate_ensemble(X_test, y_test, target_key, "test")
                
                training_results[target_key] = {
                    'model_info': model_info,
                    'validation_results': self.validation_results.get(target_key, {}),
                    'test_results': self.test_results.get(target_key, {})
                }
                
                logger.info(f"✅ Training completed for {target_key}")
            
            # Final summary
            logger.info("\n" + "=" * 60)
            logger.info("🎉 COMPREHENSIVE TRAINING COMPLETED!")
            logger.info("=" * 60)
            
            for target_key in training_results:
                val_acc = training_results[target_key]['validation_results'].get('accuracy', 0)
                test_acc = training_results[target_key]['test_results'].get('accuracy', 0)
                logger.info(f"{target_key}: Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def save_models(self, output_dir: str = "models") -> Dict:
        """
        Save trained models and results.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary with saved file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # Save models
        for target_name, model_info in self.models.items():
            model_file = Path(output_dir) / f"rf_ensemble_{target_name}_{timestamp}.joblib"
            joblib.dump(model_info, model_file)
            saved_files[f'model_{target_name}'] = str(model_file)
        
        # Save validation results
        val_file = Path(output_dir) / f"validation_results_{timestamp}.joblib"
        joblib.dump(self.validation_results, val_file)
        saved_files['validation_results'] = str(val_file)
        
        # Save test results
        test_file = Path(output_dir) / f"test_results_{timestamp}.joblib"
        joblib.dump(self.test_results, test_file)
        saved_files['test_results'] = str(test_file)
        
        logger.info(f"Models and results saved to: {output_dir}")
        
        return saved_files


if __name__ == "__main__":
    # Test the advanced Random Forest trainer for 3-year dataset
    trainer = AdvancedRandomForestTrainer3Years()
    
    # Features file path
    features_file = "data/processed/multi_timeframe_features_3years_20250823_104329.csv"
    
    try:
        # Train all targets
        results = trainer.train_all_targets(features_file)
        
        # Save models
        saved_files = trainer.save_models()
        
        print(f"\n🎯 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Models saved: {len(saved_files)} files")
        print(f"📁 Output directory: models/")
        
        # Print final results
        for target_key in results:
            val_acc = results[target_key]['validation_results'].get('accuracy', 0)
            test_acc = results[target_key]['test_results'].get('accuracy', 0)
            print(f"{target_key}: Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
