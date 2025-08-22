"""
Test suite for Random Forest model.

Tests the Random Forest model functionality including:
- Model initialization
- Data preparation
- Training
- Prediction
- Feature importance
- Model saving/loading

Author: SignaMentis Team
Version: 1.0.0
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.model_random_forest import (
    RandomForestModel,
    RandomForestTrainer,
    create_random_forest_model
)


class TestRandomForestModel(unittest.TestCase):
    """Test cases for RandomForestModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
        
        self.sample_data = pd.DataFrame({
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
        self.input_size = 4
        self.model = create_random_forest_model(self.input_size)
        
        # Create trainer
        self.trainer = RandomForestTrainer(self.model)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.n_estimators, 100)
        self.assertIsNotNone(self.model.direction_classifier)
        self.assertIsNotNone(self.model.price_regressor)
        self.assertIsNotNone(self.model.confidence_regressor)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.sequence_length, 100)
        self.assertEqual(self.trainer.test_size, 0.2)
        self.assertEqual(self.trainer.val_size, 0.2)
    
    def test_data_preparation(self):
        """Test data preparation."""
        data_tensors = self.trainer.prepare_data(
            self.sample_data,
            sequence_length=50,
            feature_columns=['rsi', 'macd', 'atr', 'volume']
        )
        
        # Check that we get the expected number of tensors
        self.assertEqual(len(data_tensors), 12)
        
        # Check shapes
        X_train, X_val, X_test = data_tensors[0], data_tensors[1], data_tensors[2]
        self.assertEqual(len(X_train), len(data_tensors[3]))  # y_direction_train
        self.assertEqual(len(X_val), len(data_tensors[4]))    # y_direction_val
        self.assertEqual(len(X_test), len(data_tensors[5]))   # y_direction_test
        
        # Check feature dimensions
        self.assertEqual(X_train.shape[1], 50 * 4)  # sequence_length * num_features
    
    def test_model_training(self):
        """Test model training."""
        # Prepare data
        data_tensors = self.trainer.prepare_data(
            self.sample_data,
            sequence_length=50,
            feature_columns=['rsi', 'macd', 'atr', 'volume']
        )
        
        # Train model - data_tensors order: [X_train, X_val, X_test, y_direction_train, y_direction_val, y_direction_test, y_price_train, y_price_val, y_price_test, y_confidence_train, y_confidence_val, y_confidence_test]
        history = self.trainer.train(
            data_tensors[0],  # X_train
            data_tensors[3],  # y_direction_train
            data_tensors[6],  # y_price_train
            data_tensors[9],  # y_confidence_train
            data_tensors[1],  # X_val
            data_tensors[4],  # y_direction_val
            data_tensors[7],  # y_price_val
            data_tensors[10]  # y_confidence_val
        )
        
        # Check that training completed
        self.assertIsNotNone(history)
        self.assertIn('direction_accuracy', history)
        self.assertIn('price_rmse', history)
        self.assertIn('confidence_rmse', history)
        
        # Check that accuracy is reasonable
        self.assertGreater(history['direction_accuracy'], 0.0)
        self.assertLessEqual(history['direction_accuracy'], 1.0)
    
    def test_prediction(self):
        """Test model prediction."""
        # Prepare and train
        data_tensors = self.trainer.prepare_data(
            self.sample_data,
            sequence_length=50,
            feature_columns=['rsi', 'macd', 'atr', 'volume']
        )
        
        self.trainer.train(
            data_tensors[0],  # X_train
            data_tensors[3],  # y_direction_train
            data_tensors[6],  # y_price_train
            data_tensors[9],  # y_confidence_train
            data_tensors[1],  # X_val
            data_tensors[4],  # y_direction_val
            data_tensors[7],  # y_price_val
            data_tensors[10]  # y_confidence_val
        )
        
        # Make predictions
        X_test = data_tensors[2]  # Test features
        predictions = self.model.predict(X_test[:10])
        
        # Check prediction structure
        self.assertIn('direction_probabilities', predictions)
        self.assertIn('predicted_direction', predictions)
        self.assertIn('price_targets', predictions)
        self.assertIn('confidence', predictions)
        
        # Check shapes
        self.assertEqual(predictions['direction_probabilities'].shape[0], 10)
        self.assertEqual(predictions['predicted_direction'].shape[0], 10)
        self.assertEqual(predictions['price_targets'].shape[0], 10)
        self.assertEqual(predictions['confidence'].shape[0], 10)
        
        # Check probability values
        for probs in predictions['direction_probabilities']:
            self.assertAlmostEqual(np.sum(probs), 1.0, places=5)
            self.assertTrue(np.all(probs >= 0))
            self.assertTrue(np.all(probs <= 1))
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        # Prepare and train
        data_tensors = self.trainer.prepare_data(
            self.sample_data,
            sequence_length=50,
            feature_columns=['rsi', 'macd', 'atr', 'volume']
        )
        
        self.trainer.train(
            data_tensors[0],  # X_train
            data_tensors[3],  # y_direction_train
            data_tensors[6],  # y_price_train
            data_tensors[9],  # y_confidence_train
            data_tensors[1],  # X_val
            data_tensors[4],  # y_direction_val
            data_tensors[7],  # y_price_val
            data_tensors[10]  # y_confidence_val
        )
        
        # Get feature importance
        importance = self.model.get_feature_importance(top_n=10)
        
        # Check structure
        self.assertIn('direction', importance)
        self.assertIn('price', importance)
        self.assertIn('confidence', importance)
        
        # Check that we get the requested number of features
        self.assertLessEqual(len(importance['direction']), 10)
        self.assertLessEqual(len(importance['price']), 10)
        self.assertLessEqual(len(importance['confidence']), 10)
        
        # Check that importance scores are sorted
        direction_scores = [score for _, score in importance['direction']]
        self.assertEqual(direction_scores, sorted(direction_scores, reverse=True))
    
    def test_model_saving_loading(self):
        """Test model saving and loading."""
        # Prepare and train
        data_tensors = self.trainer.prepare_data(
            self.sample_data,
            sequence_length=50,
            feature_columns=['rsi', 'macd', 'atr', 'volume']
        )
        
        self.trainer.train(
            data_tensors[0],  # X_train
            data_tensors[3],  # y_direction_train
            data_tensors[6],  # y_price_train
            data_tensors[9],  # y_confidence_train
            data_tensors[1],  # X_val
            data_tensors[4],  # y_direction_val
            data_tensors[7],  # y_price_val
            data_tensors[10]  # y_confidence_val
        )
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            save_path = tmp_file.name
        
        try:
            self.trainer.save_model(save_path)
            self.assertTrue(os.path.exists(save_path))
            
            # Create new model and load
            new_model = create_random_forest_model(self.input_size)
            new_model.load_model(save_path)
            
            # Test that predictions are the same
            X_test = data_tensors[2][:5]
            original_preds = self.model.predict(X_test)
            loaded_preds = new_model.predict(X_test)
            
            # Check that predictions are identical
            np.testing.assert_array_equal(
                original_preds['predicted_direction'],
                loaded_preds['predicted_direction']
            )
            
        finally:
            # Clean up
            if os.path.exists(save_path):
                os.unlink(save_path)
    
    def test_custom_configuration(self):
        """Test model creation with custom configuration."""
        custom_config = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 123
        }
        
        custom_model = create_random_forest_model(self.input_size, custom_config)
        
        self.assertEqual(custom_model.n_estimators, 200)
        self.assertEqual(custom_model.max_depth, 10)
        self.assertEqual(custom_model.min_samples_split, 10)
        self.assertEqual(custom_model.min_samples_leaf, 5)
        self.assertEqual(custom_model.random_state, 123)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.trainer.prepare_data(empty_df)
        
        # Test with missing target column
        df_no_target = self.sample_data.drop('close', axis=1)
        with self.assertRaises(ValueError):
            self.trainer.prepare_data(df_no_target, target_column='close')
        
        # Test with invalid sequence length
        with self.assertRaises(ValueError):
            self.trainer.prepare_data(self.sample_data, sequence_length=0)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        # Prepare and train
        data_tensors = self.trainer.prepare_data(
            self.sample_data,
            sequence_length=50,
            feature_columns=['rsi', 'macd', 'atr', 'volume']
        )
        
        history = self.trainer.train(
            data_tensors[0],  # X_train
            data_tensors[3],  # y_direction_train
            data_tensors[6],  # y_price_train
            data_tensors[9],  # y_confidence_train
            data_tensors[1],  # X_val
            data_tensors[4],  # y_direction_val
            data_tensors[7],  # y_price_val
            data_tensors[10]  # y_confidence_val
        )
        
        # Check all metrics are present
        required_metrics = [
            'direction_accuracy', 'price_rmse', 'confidence_rmse',
            'oob_score_direction', 'oob_score_price', 'oob_score_confidence'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, history)
            self.assertIsInstance(history[metric], (int, float))
        
        # Check OOB scores are reasonable
        self.assertGreaterEqual(history['oob_score_direction'], 0.0)
        self.assertLessEqual(history['oob_score_direction'], 1.0)
        # OOB score for regression can be negative (R² score)
        self.assertGreaterEqual(history['oob_score_price'], -1.0)
        self.assertLessEqual(history['oob_score_price'], 1.0)
        self.assertGreaterEqual(history['oob_score_confidence'], -1.0)
        self.assertLessEqual(history['oob_score_confidence'], 1.0)


class TestRandomForestIntegration(unittest.TestCase):
    """Test Random Forest integration with ensemble system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='5T')
        
        self.sample_data = pd.DataFrame({
            'open': 2000 + np.random.randn(500).cumsum(),
            'high': 2000 + np.random.randn(500).cumsum() + 5,
            'low': 2000 + np.random.randn(500).cumsum() - 5,
            'close': 2000 + np.random.randn(500).cumsum(),
            'volume': np.random.randint(100, 1000, 500),
            'rsi': np.random.uniform(0, 100, 500),
            'macd': np.random.randn(500),
            'atr': np.random.uniform(1, 10, 500)
        }, index=dates)
    
    def test_ensemble_compatibility(self):
        """Test that Random Forest is compatible with ensemble system."""
        # Create model
        model = create_random_forest_model(4)
        
        # Check that model has required methods
        required_methods = ['predict', 'get_feature_importance']
        for method in required_methods:
            self.assertTrue(hasattr(model, method))
            self.assertTrue(callable(getattr(model, method)))
        
        # Check that predict returns expected structure
        # Prepare some test data
        test_features = np.random.randn(10, 200)  # 10 samples, 200 features (50*4)
        predictions = model.predict(test_features)
        
        required_keys = ['direction_probabilities', 'predicted_direction', 'price_targets', 'confidence']
        for key in required_keys:
            self.assertIn(key, predictions)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
