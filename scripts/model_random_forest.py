"""
SignaMentis Random Forest Model Module

This module implements a Random Forest model for XAU/USD price prediction.
The model provides robust, interpretable predictions and integrates seamlessly
with the SignaMentis ensemble system.

Author: SignaMentis Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import warnings
import pickle
from pathlib import Path
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    Random Forest model for XAU/USD price prediction.
    
    Features:
    - Robust classification for direction prediction
    - Regression for price target prediction
    - Feature importance analysis
    - Ensemble integration ready
    - Interpretable predictions
    """
    
    def __init__(self, 
                 input_size: int,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[str, int, float]] = 'sqrt',
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize the Random Forest model.
        
        Args:
            input_size: Number of input features
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider for best split
            random_state: Random seed for reproducibility
            n_jobs: Number of jobs for parallel processing
        """
        self.input_size = input_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize models
        self.direction_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            oob_score=True,  # Out-of-bag score for validation
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.price_regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            oob_score=True
        )
        
        self.confidence_regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            oob_score=True
        )
        
        # Scaler for features
        self.feature_scaler = StandardScaler()
        
        # Scaler for price targets
        self.price_scaler = StandardScaler()
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Model performance metrics
        self.performance_metrics = {
            'direction_accuracy': 0.0,
            'price_rmse': 0.0,
            'confidence_rmse': 0.0,
            'oob_score_direction': 0.0,
            'oob_score_price': 0.0,
            'oob_score_confidence': 0.0
        }
        
        logger.info(f"Random Forest model initialized with {n_estimators} trees")
    
    def prepare_data(self, 
                    df: pd.DataFrame,
                    sequence_length: int = 100,
                    target_column: str = 'close',
                    feature_columns: Optional[List[str]] = None,
                    test_size: float = 0.2,
                    val_size: float = 0.2) -> Tuple[np.ndarray, ...]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            sequence_length: Length of input sequences
            target_column: Name of target column
            feature_columns: List of feature column names
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_direction_train, y_direction_val, 
                     y_direction_test, y_price_train, y_price_val, y_price_test,
                     y_confidence_train, y_confidence_val, y_confidence_test)
        """
        logger.info("Preparing data for Random Forest training...")
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Ensure target column is included
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Prepare features and target
        X = df[feature_columns].values
        y_price = df[target_column].values
        
        # Create direction labels (1 for up, 0 for down)
        y_direction = (df[target_column].diff() > 0).astype(int)
        y_direction = y_direction.fillna(0)
        
        # Create confidence labels (based on price movement magnitude)
        price_changes = df[target_column].pct_change().abs()
        y_confidence = np.clip(price_changes * 100, 0, 1)  # Scale to 0-1
        y_confidence = y_confidence.fillna(0)
        
        # Remove rows with NaN values
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y_direction) | np.isnan(y_confidence))
        X = X[valid_indices]
        y_direction = y_direction[valid_indices]
        y_price = y_price[valid_indices]
        y_confidence = y_confidence[valid_indices]
        
        # Create sequences (for Random Forest, we'll use the last sequence_length features)
        X_sequences, y_direction_sequences, y_price_sequences, y_confidence_sequences = [], [], [], []
        
        for i in range(sequence_length, len(X)):
            # For Random Forest, flatten the sequence into a single feature vector
            sequence_features = X[i-sequence_length:i].flatten()
            X_sequences.append(sequence_features)
            y_direction_sequences.append(y_direction[i])
            y_price_sequences.append(y_price[i])
            y_confidence_sequences.append(y_confidence[i])
        
        X_sequences = np.array(X_sequences)
        y_direction_sequences = np.array(y_direction_sequences)
        y_price_sequences = np.array(y_price_sequences)
        y_confidence_sequences = np.array(y_confidence_sequences)
        
        # Split into train/val/test
        X_temp, X_test, y_direction_temp, y_direction_test, y_price_temp, y_price_test, y_confidence_temp, y_confidence_test = train_test_split(
            X_sequences, y_direction_sequences, y_price_sequences, y_confidence_sequences,
            test_size=test_size, random_state=self.random_state, stratify=y_direction_sequences
        )
        
        X_train, X_val, y_direction_train, y_direction_val, y_price_train, y_price_val, y_confidence_train, y_confidence_val = train_test_split(
            X_temp, y_direction_temp, y_price_temp, y_confidence_temp,
            test_size=val_size, random_state=self.random_state, stratify=y_direction_temp
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Scale price targets
        y_price_train_scaled = self.price_scaler.fit_transform(y_price_train.reshape(-1, 1)).flatten()
        y_price_val_scaled = self.price_scaler.transform(y_price_val.reshape(-1, 1)).flatten()
        y_price_test_scaled = self.price_scaler.transform(y_price_test.reshape(-1, 1)).flatten()
        
        logger.info(f"Data prepared: {len(X_train_scaled)} train, {len(X_val_scaled)} val, {len(X_test_scaled)} test samples")
        logger.info(f"Feature dimension: {X_train_scaled.shape[1]}")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled,
                y_direction_train, y_direction_val, y_direction_test,
                y_price_train_scaled, y_price_val_scaled, y_price_test_scaled,
                y_confidence_train, y_confidence_val, y_confidence_test)
    
    def train(self, 
              X_train: np.ndarray,
              y_direction_train: np.ndarray,
              y_price_train: np.ndarray,
              y_confidence_train: np.ndarray,
              X_val: np.ndarray,
              y_direction_val: np.ndarray,
              y_price_val: np.ndarray,
              y_confidence_val: np.ndarray) -> Dict:
        """
        Train the Random Forest models.
        
        Args:
            Training and validation data arrays
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting Random Forest training...")
        
        # Validate data dimensions
        if len(X_train) != len(y_direction_train) or len(X_train) != len(y_price_train) or len(X_train) != len(y_confidence_train):
            raise ValueError(f"Training data length mismatch: X_train={len(X_train)}, y_direction={len(y_direction_train)}, y_price={len(y_price_train)}, y_confidence={len(y_confidence_train)}")
        
        if len(X_val) != len(y_direction_val) or len(X_val) != len(y_price_val) or len(X_val) != len(y_confidence_val):
            raise ValueError(f"Validation data length mismatch: X_val={len(X_val)}, y_direction={len(y_direction_val)}, y_price={len(y_price_val)}, y_confidence={len(y_confidence_val)}")
        
        # Train direction classifier
        logger.info("Training direction classifier...")
        self.direction_classifier.fit(X_train, y_direction_train)
        
        # Train price regressor
        logger.info("Training price regressor...")
        self.price_regressor.fit(X_train, y_price_train)
        
        # Train confidence regressor
        logger.info("Training confidence regressor...")
        self.confidence_regressor.fit(X_train, y_confidence_train)
        
        # Evaluate on validation set
        self._evaluate_models(X_val, y_direction_val, y_price_val, y_confidence_val)
        
        # Store feature importance
        self._store_feature_importance()
        
        logger.info("Random Forest training completed!")
        
        return self.performance_metrics
    
    def _evaluate_models(self, 
                        X_val: np.ndarray,
                        y_direction_val: np.ndarray,
                        y_price_val: np.ndarray,
                        y_confidence_val: np.ndarray):
        """Evaluate models on validation set."""
        
        # Direction prediction
        y_direction_pred = self.direction_classifier.predict(X_val)
        direction_accuracy = accuracy_score(y_direction_val, y_direction_pred)
        
        # Price prediction
        y_price_pred = self.price_regressor.predict(X_val)
        price_rmse = np.sqrt(mean_squared_error(y_price_val, y_price_pred))
        
        # Confidence prediction
        y_confidence_pred = self.confidence_regressor.predict(X_val)
        confidence_rmse = np.sqrt(mean_squared_error(y_confidence_val, y_confidence_pred))
        
        # Update performance metrics
        self.performance_metrics.update({
            'direction_accuracy': direction_accuracy,
            'price_rmse': price_rmse,
            'confidence_rmse': confidence_rmse,
            'oob_score_direction': self.direction_classifier.oob_score_,
            'oob_score_price': self.price_regressor.oob_score_,
            'oob_score_confidence': self.confidence_regressor.oob_score_
        })
        
        logger.info(f"Validation Results:")
        logger.info(f"  Direction Accuracy: {direction_accuracy:.4f}")
        logger.info(f"  Price RMSE: {price_rmse:.4f}")
        logger.info(f"  Confidence RMSE: {confidence_rmse:.4f}")
        logger.info(f"  OOB Score Direction: {self.direction_classifier.oob_score_:.4f}")
        logger.info(f"  OOB Score Price: {self.price_regressor.oob_score_:.4f}")
        logger.info(f"  OOB Score Confidence: {self.confidence_regressor.oob_score_:.4f}")
    
    def _store_feature_importance(self):
        """Store feature importance for analysis."""
        self.feature_importance = {
            'direction': self.direction_classifier.feature_importances_,
            'price': self.price_regressor.feature_importances_,
            'confidence': self.confidence_regressor.feature_importances_
        }
        
        logger.info("Feature importance stored")
    
    def predict(self, X: np.ndarray) -> Dict:
        """
        Make predictions on new data.
        
        Args:
            X: Input features array
            
        Returns:
            Dictionary with predictions
        """
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Check if model is trained
        if not hasattr(self.direction_classifier, 'classes_'):
            # Return dummy predictions if not trained
            n_samples = X.shape[0]
            dummy_probs = np.array([[0.5, 0.5] for _ in range(n_samples)])
            dummy_direction = np.zeros(n_samples, dtype=int)
            dummy_price = np.full(n_samples, 2000.0)  # Default price
            dummy_confidence = np.full(n_samples, 0.5)  # Default confidence
            
            return {
                'direction_probabilities': dummy_probs,
                'predicted_direction': dummy_direction,
                'price_targets': dummy_price,
                'confidence': dummy_confidence
            }
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Get predictions
        direction_probs = self.direction_classifier.predict_proba(X_scaled)
        predicted_direction = self.direction_classifier.predict(X_scaled)
        price_target_scaled = self.price_regressor.predict(X_scaled)
        confidence = self.confidence_regressor.predict(X_scaled)
        
        # Inverse transform price predictions
        price_target_original = self.price_scaler.inverse_transform(price_target_scaled.reshape(-1, 1))
        
        return {
            'direction_probabilities': direction_probs,
            'predicted_direction': predicted_direction,
            'price_targets': price_target_original.flatten(),
            'confidence': confidence
        }
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importance for each target
        """
        if not self.feature_importance:
            return {}
        
        # Create feature names (assuming sequential naming)
        feature_names = [f"feature_{i}" for i in range(len(self.feature_importance['direction']))]
        
        importance_dict = {}
        for target, importance in self.feature_importance.items():
            # Create tuples of (feature_name, importance_score)
            feature_importance_tuples = list(zip(feature_names, importance))
            # Sort by importance (descending)
            feature_importance_tuples.sort(key=lambda x: x[1], reverse=True)
            # Take top N
            importance_dict[target] = feature_importance_tuples[:top_n]
        
        return importance_dict
    
    def save_model(self, filepath: str):
        """Save the trained model and scalers."""
        save_dict = {
            'direction_classifier': self.direction_classifier,
            'price_regressor': self.price_regressor,
            'confidence_regressor': self.confidence_regressor,
            'feature_scaler': self.feature_scaler,
            'price_scaler': self.price_scaler,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics,
            'model_config': {
                'input_size': self.input_size,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
            }
        }
        
        joblib.dump(save_dict, filepath)
        logger.info(f"Random Forest model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model and scalers."""
        checkpoint = joblib.load(filepath)
        
        # Load models
        self.direction_classifier = checkpoint['direction_classifier']
        self.price_regressor = checkpoint['price_regressor']
        self.confidence_regressor = checkpoint['confidence_regressor']
        
        # Load scalers
        self.feature_scaler = checkpoint['feature_scaler']
        self.price_scaler = checkpoint['price_scaler']
        
        # Load additional data
        self.feature_importance = checkpoint.get('feature_importance', {})
        self.performance_metrics = checkpoint.get('performance_metrics', {})
        
        logger.info(f"Random Forest model loaded from {filepath}")


class RandomForestTrainer:
    """
    Trainer class for the Random Forest model.
    
    Handles data preparation, training, validation, and model saving.
    """
    
    def __init__(self, 
                 model: RandomForestModel,
                 config: Optional[Dict] = None):
        """
        Initialize the trainer.
        
        Args:
            model: Random Forest model instance
            config: Training configuration
        """
        self.model = model
        self.config = config or {}
        
        # Training parameters
        self.sequence_length = self.config.get('sequence_length', 100)
        self.test_size = self.config.get('test_size', 0.2)
        self.val_size = self.config.get('val_size', 0.2)
        
        # Cross-validation
        self.cv_folds = self.config.get('cv_folds', 5)
        
        logger.info(f"Random Forest trainer initialized")
    
    def prepare_data(self, 
                    df: pd.DataFrame,
                    sequence_length: int = 100,
                    target_column: str = 'close',
                    feature_columns: Optional[List[str]] = None,
                    test_size: float = 0.2,
                    val_size: float = 0.2) -> Tuple[np.ndarray, ...]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            sequence_length: Length of input sequences
            target_column: Name of target column
            feature_columns: List of feature column names
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            
        Returns:
            Tuple of training data arrays
        """
        return self.model.prepare_data(
            df, sequence_length, target_column, feature_columns, test_size, val_size
        )
    
    def train(self, 
              X_train: np.ndarray,
              y_direction_train: np.ndarray,
              y_price_train: np.ndarray,
              y_confidence_train: np.ndarray,
              X_val: np.ndarray,
              y_direction_val: np.ndarray,
              y_price_val: np.ndarray,
              y_confidence_val: np.ndarray) -> Dict:
        """
        Train the Random Forest model.
        
        Args:
            Training and validation data arrays
            
        Returns:
            Training history dictionary
        """
        return self.model.train(
            X_train, y_direction_train, y_price_train, y_confidence_train,
            X_val, y_direction_val, y_price_val, y_confidence_val
        )
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        self.model.load_model(filepath)


# Convenience function for quick model creation
def create_random_forest_model(input_size: int, config: Optional[Dict] = None) -> RandomForestModel:
    """
    Create a Random Forest model with default or custom configuration.
    
    Args:
        input_size: Number of input features
        config: Model configuration dictionary
        
    Returns:
        RandomForestModel instance
    """
    default_config = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    if config:
        # Filter only model-specific parameters
        model_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
                       'max_features', 'random_state', 'n_jobs']
        filtered_config = {k: v for k, v in config.items() if k in model_params}
        default_config.update(filtered_config)
    
    return RandomForestModel(input_size=input_size, **default_config)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
    
    sample_data = pd.DataFrame({
        'open': 2000 + np.random.randn(1000).cumsum(),
        'high': 2000 + np.random.randn(1000).cumsum() + 5,
        'low': 2000 + np.random.randn(1000).cumsum() - 5,
        'close': 2000 + np.random.randn(1000).cumsum(),
        'volume': np.random.randint(100, 1000, 1000),
        'rsi': np.random.uniform(0, 100, 1000),
        'macd': np.random.randn(1000),
        'atr': np.random.uniform(1, 10, 1000)
    }, index=dates)
    
    # Create model
    input_size = 4  # rsi, macd, atr, volume
    model = create_random_forest_model(input_size)
    
    # Create trainer
    trainer = RandomForestTrainer(model)
    
    # Prepare data
    data_tensors = trainer.prepare_data(
        sample_data,
        sequence_length=50,
        feature_columns=['rsi', 'macd', 'atr', 'volume']
    )
    
    # Train model
    history = trainer.train(*data_tensors[:8])  # First 8 tensors for training
    
    # Make predictions
    X_test = data_tensors[2]  # Test features
    predictions = model.predict(X_test[:10])  # First 10 test samples
    
    print(f"Random Forest model trained successfully!")
    print(f"Predictions shape: {predictions['direction_probabilities'].shape}")
    print(f"Sample predictions: {predictions['predicted_direction'][:5]}")
    print(f"Sample confidence: {predictions['confidence'][:5]}")
    
    # Show feature importance
    importance = model.get_feature_importance(top_n=10)
    print(f"\nTop 10 Feature Importance:")
    for target, features in importance.items():
        print(f"\n{target.upper()}:")
        for feature_name, importance_score in features:
            print(f"  {feature_name}: {importance_score:.4f}")
