#!/usr/bin/env python3
"""
Unit tests for AI models and ensemble.

This module tests the functionality of:
- BiLSTM Model
- GRU Model
- Transformer Model
- LNN Model
- LTN Model
- Ensemble Manager
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add scripts directory to path
sys.path.append('scripts')

from model_bilstm import BiLSTMModel, BiLSTMTrainer, create_bilstm_model
from model_gru import GRUModel, GRUTrainer, create_gru_model
from model_transformer import TransformerModel, TransformerTrainer, create_transformer_model
from model_lnn import LNNModel, LNNTrainer, create_lnn_model
from model_ltn import LTNModel, LTNTrainer, create_ltn_model
from ensemble import EnsembleManager, EnsembleAggregator, create_ensemble


class TestBiLSTMModel(unittest.TestCase):
    """Test cases for BiLSTM model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 50
        self.hidden_size = 128
        self.num_layers = 2
        self.num_classes = 2
        self.batch_size = 32
        self.seq_length = 100
        
        self.model = create_bilstm_model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )
        
        # Create sample input data
        self.sample_input = torch.randn(self.batch_size, self.seq_length, self.input_size)
    
    def test_model_initialization(self):
        """Test BiLSTM model initialization."""
        self.assertIsInstance(self.model, BiLSTMModel)
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.num_classes, self.num_classes)
    
    def test_model_forward_pass(self):
        """Test BiLSTM model forward pass."""
        try:
            # Forward pass
            direction_probs, price_target, confidence = self.model(self.sample_input)
            
            # Check output shapes
            self.assertEqual(direction_probs.shape, (self.batch_size, self.num_classes))
            self.assertEqual(price_target.shape, (self.batch_size, 1))
            self.assertEqual(confidence.shape, (self.batch_size, 1))
            
            # Check output values
            self.assertTrue(torch.allclose(direction_probs.sum(dim=1), torch.ones(self.batch_size)))
            self.assertTrue(torch.all(direction_probs >= 0))
            self.assertTrue(torch.all(confidence >= 0) and torch.all(confidence <= 1))
            
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 0)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertGreater(trainable_params, 0)
    
    def test_trainer_initialization(self):
        """Test BiLSTM trainer initialization."""
        trainer = BiLSTMTrainer(self.model)
        self.assertIsInstance(trainer, BiLSTMTrainer)
        self.assertIsInstance(trainer.model, BiLSTMModel)


class TestGRUModel(unittest.TestCase):
    """Test cases for GRU model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 50
        self.hidden_size = 128
        self.num_layers = 2
        self.num_classes = 2
        self.batch_size = 32
        self.seq_length = 100
        
        self.model = create_gru_model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )
        
        # Create sample input data
        self.sample_input = torch.randn(self.batch_size, self.seq_length, self.input_size)
    
    def test_model_initialization(self):
        """Test GRU model initialization."""
        self.assertIsInstance(self.model, GRUModel)
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.num_classes, self.num_classes)
    
    def test_model_forward_pass(self):
        """Test GRU model forward pass."""
        try:
            # Forward pass
            direction_probs, price_target, confidence = self.model(self.sample_input)
            
            # Check output shapes
            self.assertEqual(direction_probs.shape, (self.batch_size, self.num_classes))
            self.assertEqual(price_target.shape, (self.batch_size, 1))
            self.assertEqual(confidence.shape, (self.batch_size, 1))
            
            # Check output values
            self.assertTrue(torch.allclose(direction_probs.sum(dim=1), torch.ones(self.batch_size)))
            self.assertTrue(torch.all(direction_probs >= 0))
            self.assertTrue(torch.all(confidence >= 0) and torch.all(confidence <= 1))
            
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
    
    def test_trainer_initialization(self):
        """Test GRU trainer initialization."""
        trainer = GRUTrainer(self.model)
        self.assertIsInstance(trainer, GRUTrainer)
        self.assertIsInstance(trainer.model, GRUModel)


class TestTransformerModel(unittest.TestCase):
    """Test cases for Transformer model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 50
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 2
        self.num_classes = 2
        self.batch_size = 32
        self.seq_length = 100
        
        self.model = create_transformer_model(
            input_size=self.input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )
        
        # Create sample input data
        self.sample_input = torch.randn(self.batch_size, self.seq_length, self.input_size)
    
    def test_model_initialization(self):
        """Test Transformer model initialization."""
        self.assertIsInstance(self.model, TransformerModel)
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.d_model, self.d_model)
        self.assertEqual(self.model.nhead, self.nhead)
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.num_classes, self.num_classes)
    
    def test_model_forward_pass(self):
        """Test Transformer model forward pass."""
        try:
            # Forward pass
            direction_probs, price_target, confidence = self.model(self.sample_input)
            
            # Check output shapes
            self.assertEqual(direction_probs.shape, (self.batch_size, self.num_classes))
            self.assertEqual(price_target.shape, (self.batch_size, 1))
            self.assertEqual(confidence.shape, (self.batch_size, 1))
            
            # Check output values
            self.assertTrue(torch.allclose(direction_probs.sum(dim=1), torch.ones(self.batch_size)))
            self.assertTrue(torch.all(direction_probs >= 0))
            self.assertTrue(torch.all(confidence >= 0) and torch.all(confidence <= 1))
            
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
    
    def test_positional_encoding(self):
        """Test positional encoding functionality."""
        pos_encoding = self.model.pos_encoder
        self.assertIsInstance(pos_encoding, nn.Module)
        
        # Test positional encoding output
        test_input = torch.randn(1, 50, self.d_model)
        encoded = pos_encoding(test_input)
        self.assertEqual(encoded.shape, test_input.shape)
    
    def test_trainer_initialization(self):
        """Test Transformer trainer initialization."""
        trainer = TransformerTrainer(self.model)
        self.assertIsInstance(trainer, TransformerTrainer)
        self.assertIsInstance(trainer.model, TransformerModel)


class TestLNNModel(unittest.TestCase):
    """Test cases for LNN model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 50
        self.hidden_size = 128
        self.num_layers = 2
        self.num_classes = 2
        self.batch_size = 32
        self.seq_length = 100
        
        self.model = create_lnn_model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )
        
        # Create sample input data
        self.sample_input = torch.randn(self.batch_size, self.seq_length, self.input_size)
    
    def test_model_initialization(self):
        """Test LNN model initialization."""
        self.assertIsInstance(self.model, LNNModel)
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.num_classes, self.num_classes)
    
    def test_model_forward_pass(self):
        """Test LNN model forward pass."""
        try:
            # Forward pass
            direction_probs, price_target, confidence = self.model(self.sample_input)
            
            # Check output shapes
            self.assertEqual(direction_probs.shape, (self.batch_size, self.num_classes))
            self.assertEqual(price_target.shape, (self.batch_size, 1))
            self.assertEqual(confidence.shape, (self.batch_size, 1))
            
            # Check output values
            self.assertTrue(torch.allclose(direction_probs.sum(dim=1), torch.ones(self.batch_size)))
            self.assertTrue(torch.all(direction_probs >= 0))
            self.assertTrue(torch.all(confidence >= 0) and torch.all(confidence <= 1))
            
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
    
    def test_trainer_initialization(self):
        """Test LNN trainer initialization."""
        trainer = LNNTrainer(self.model)
        self.assertIsInstance(trainer, LNNTrainer)
        self.assertIsInstance(trainer.model, LNNModel)


class TestLTNModel(unittest.TestCase):
    """Test cases for LTN model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 50
        self.hidden_size = 128
        self.num_layers = 2
        self.num_rules = 8
        self.num_classes = 2
        self.batch_size = 32
        self.seq_length = 100
        
        self.model = create_ltn_model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_rules=self.num_rules,
            num_classes=self.num_classes
        )
        
        # Create sample input data
        self.sample_input = torch.randn(self.batch_size, self.seq_length, self.input_size)
    
    def test_model_initialization(self):
        """Test LTN model initialization."""
        self.assertIsInstance(self.model, LTNModel)
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.num_rules, self.num_rules)
        self.assertEqual(self.model.num_classes, self.num_classes)
    
    def test_model_forward_pass(self):
        """Test LTN model forward pass."""
        try:
            # Forward pass
            direction_probs, price_target, confidence = self.model(self.sample_input)
            
            # Check output shapes
            self.assertEqual(direction_probs.shape, (self.batch_size, self.num_classes))
            self.assertEqual(price_target.shape, (self.batch_size, 1))
            self.assertEqual(confidence.shape, (self.batch_size, 1))
            
            # Check output values
            self.assertTrue(torch.allclose(direction_probs.sum(dim=1), torch.ones(self.batch_size)))
            self.assertTrue(torch.all(direction_probs >= 0))
            self.assertTrue(torch.all(confidence >= 0) and torch.all(confidence <= 1))
            
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
    
    def test_trainer_initialization(self):
        """Test LTN trainer initialization."""
        trainer = LTNTrainer(self.model)
        self.assertIsInstance(trainer, LTNTrainer)
        self.assertIsInstance(trainer.model, LTNModel)


class TestEnsembleManager(unittest.TestCase):
    """Test cases for Ensemble Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ensemble = create_ensemble()
        
        # Create sample models
        self.models = {
            'bilstm': create_bilstm_model(input_size=50, hidden_size=64, num_classes=2),
            'gru': create_gru_model(input_size=50, hidden_size=64, num_classes=2),
            'transformer': create_transformer_model(input_size=50, d_model=64, num_classes=2)
        }
        
        # Create sample predictions
        self.sample_predictions = {
            'bilstm': {
                'direction_probs': torch.tensor([[0.6, 0.4], [0.3, 0.7]]),
                'price_target': torch.tensor([[2000.0], [2005.0]]),
                'confidence': torch.tensor([[0.8], [0.9]])
            },
            'gru': {
                'direction_probs': torch.tensor([[0.7, 0.3], [0.4, 0.6]]),
                'price_target': torch.tensor([[2001.0], [2004.0]]),
                'confidence': torch.tensor([[0.85], [0.75]])
            },
            'transformer': {
                'direction_probs': torch.tensor([[0.5, 0.5], [0.2, 0.8]]),
                'price_target': torch.tensor([[2002.0], [2003.0]]),
                'confidence': torch.tensor([[0.7], [0.95]])
            }
        }
    
    def test_ensemble_initialization(self):
        """Test Ensemble Manager initialization."""
        self.assertIsInstance(self.ensemble, EnsembleManager)
        self.assertIsInstance(self.ensemble.aggregator, EnsembleAggregator)
    
    def test_model_registration(self):
        """Test model registration functionality."""
        # Register models
        for name, model in self.models.items():
            self.ensemble.register_model(name, model, weight=1.0)
        
        # Check registered models
        self.assertEqual(len(self.ensemble.models), len(self.models))
        for name in self.models.keys():
            self.assertIn(name, self.ensemble.models)
    
    def test_weight_assignment(self):
        """Test weight assignment functionality."""
        # Register models with different weights
        self.ensemble.register_model('bilstm', self.models['bilstm'], weight=0.4)
        self.ensemble.register_model('gru', self.models['gru'], weight=0.3)
        self.ensemble.register_model('transformer', self.models['transformer'], weight=0.3)
        
        # Check weights
        total_weight = sum(self.ensemble.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction functionality."""
        # Register models
        for name, model in self.models.items():
            self.ensemble.register_model(name, model, weight=1.0/len(self.models))
        
        # Test prediction aggregation
        try:
            aggregated_prediction = self.ensemble.predict(self.sample_predictions)
            
            # Check output structure
            self.assertIn('direction_probs', aggregated_prediction)
            self.assertIn('price_target', aggregated_prediction)
            self.assertIn('confidence', aggregated_prediction)
            
            # Check output shapes
            self.assertEqual(aggregated_prediction['direction_probs'].shape, (2, 2))
            self.assertEqual(aggregated_prediction['price_target'].shape, (2, 1))
            self.assertEqual(aggregated_prediction['confidence'].shape, (2, 1))
            
        except Exception as e:
            self.fail(f"Ensemble prediction failed: {e}")
    
    def test_confidence_weighted_aggregation(self):
        """Test confidence-weighted aggregation."""
        # Register models with equal weights
        for name, model in self.models.items():
            self.ensemble.register_model(name, model, weight=1.0/len(self.models))
        
        # Test confidence-weighted aggregation
        try:
            weighted_prediction = self.ensemble.predict_confidence_weighted(self.sample_predictions)
            
            # Check output structure
            self.assertIn('direction_probs', weighted_prediction)
            self.assertIn('price_target', weighted_prediction)
            self.assertIn('confidence', weighted_prediction)
            
        except Exception as e:
            self.fail(f"Confidence-weighted prediction failed: {e}")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBiLSTMModel))
    test_suite.addTest(unittest.makeSuite(TestGRUModel))
    test_suite.addTest(unittest.makeSuite(TestTransformerModel))
    test_suite.addTest(unittest.makeSuite(TestLNNModel))
    test_suite.addTest(unittest.makeSuite(TestLTNModel))
    test_suite.addTest(unittest.makeSuite(TestEnsembleManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running SignaMentis AI Models Tests...")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    print(f"\nTest execution completed with {'success' if success else 'failure'}.")
