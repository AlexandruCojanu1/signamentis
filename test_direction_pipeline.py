#!/usr/bin/env python3
"""
Test script to validate the direction prediction pipeline.
"""

import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from scripts.binance_data_fetcher import BinanceDataFetcher
from scripts.feature_engineer import FeatureEngineer
from scripts.label_engineer import LabelEngineer
from scripts.sequence_builder import SequenceBuilder
from scripts.transformer_bilstm_model import TransformerBiLSTMClassifier

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_data_fetching():
    """Test data fetching from Binance."""
    print("Testing data fetching...")
    fetcher = BinanceDataFetcher()
    
    # Test with small amount of data
    df = fetcher.download_historical_data('SOLUSDT', '15m', years=0.01)  # ~3.65 days
    
    assert len(df) > 0, "No data fetched"
    assert 'close' in df.columns, "Missing close column"
    assert df.index.tz is not None, "Index should be timezone aware"
    print(f"✓ Fetched {len(df)} rows from {df.index[0]} to {df.index[-1]}")

def test_feature_engineering():
    """Test feature engineering."""
    print("Testing feature engineering...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='15T', tz='UTC')
    np.random.seed(42)
    
    data = {
        'open': 100 + np.random.randn(1000).cumsum() * 0.1,
        'high': None,
        'low': None,
        'close': None,
        'volume': np.random.lognormal(10, 1, 1000)
    }
    
    # Make realistic OHLC
    data['close'] = data['open'] + np.random.randn(1000) * 0.05
    data['high'] = np.maximum(data['open'], data['close']) + np.random.exponential(0.02, 1000)
    data['low'] = np.minimum(data['open'], data['close']) - np.random.exponential(0.02, 1000)
    
    df = pd.DataFrame(data, index=dates)
    
    config = {
        'volatility_window': 48,
        'ema_periods': [20, 50],
        'rsi_period': 14,
        'macd_params': [12, 26, 9],
        'atr_period': 14
    }
    
    engineer = FeatureEngineer(config)
    features = engineer.compute_features(df)
    
    assert len(features) > 0, "No features generated"
    assert 'rsi' in features.columns, "Missing RSI feature"
    assert not features.isnull().all().any(), "All-NaN columns present"
    print(f"✓ Generated {features.shape[1]} features, {len(features)} valid rows")

def test_label_engineering():
    """Test label engineering."""
    print("Testing label engineering...")
    
    # Use same sample data
    dates = pd.date_range('2024-01-01', periods=500, freq='15T', tz='UTC')
    np.random.seed(42)
    
    data = {
        'open': 100 + np.random.randn(500).cumsum() * 0.1,
        'high': None,
        'low': None,
        'close': None,
        'volume': np.random.lognormal(10, 1, 500)
    }
    
    data['close'] = data['open'] + np.random.randn(500) * 0.05
    data['high'] = np.maximum(data['open'], data['close']) + np.random.exponential(0.02, 500)
    data['low'] = np.minimum(data['open'], data['close']) - np.random.exponential(0.02, 500)
    
    df = pd.DataFrame(data, index=dates)
    
    label_engineer = LabelEngineer(k=0.25)
    features, labels = label_engineer.create_labels(df)
    
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 3, f"Expected 3 classes, got {unique_labels}"
    assert set(unique_labels) == {0, 1, 2}, f"Expected classes [0,1,2], got {unique_labels}"
    print(f"✓ Generated labels with distribution: {np.bincount(labels)}")

def test_sequence_building():
    """Test sequence building."""
    print("Testing sequence building...")
    
    # Create sample features and labels
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feat_{i}' for i in range(n_features)],
        index=pd.date_range('2024-01-01', periods=n_samples, freq='15T', tz='UTC')
    )
    labels = pd.Series(np.random.randint(0, 3, n_samples), index=features.index)
    
    seq_builder = SequenceBuilder(seq_len=64, val_ratio=0.2)
    X_train, y_train, X_val, y_val = seq_builder.prepare_sequences(features, labels)
    
    assert X_train.shape[0] > 0, "No training sequences"
    assert X_val.shape[0] > 0, "No validation sequences"
    assert X_train.shape[1] == 64, f"Wrong sequence length: {X_train.shape[1]}"
    assert X_train.shape[2] == n_features, f"Wrong feature count: {X_train.shape[2]}"
    
    # Test scaler fitting
    assert seq_builder.scaler.mean_ is not None, "Scaler not fitted"
    print(f"✓ Generated sequences: train {X_train.shape}, val {X_val.shape}")

def test_model_forward():
    """Test model forward pass."""
    print("Testing model forward pass...")
    
    config = {
        'd_model': 64,
        'num_layers': 2,
        'nhead': 4,
        'lstm_hidden': 64,
        'dropout': 0.1,
        'n_classes': 3
    }
    
    model = TransformerBiLSTMClassifier(n_features=10, config=config)
    
    # Test forward pass
    batch_size, seq_len, n_features = 8, 32, 10
    x = torch.randn(batch_size, seq_len, n_features)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (batch_size, 3), f"Wrong output shape: {logits.shape}"
    
    # Test softmax sums to 1
    probs = torch.softmax(logits, dim=1)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums)), "Probabilities don't sum to 1"
    
    print(f"✓ Model forward pass successful: {logits.shape}")

def run_all_tests():
    """Run all unit tests."""
    setup_logging()
    
    print("Running Direction Pipeline Tests")
    print("=" * 40)
    
    try:
        test_data_fetching()
        test_feature_engineering() 
        test_label_engineering()
        test_sequence_building()
        test_model_forward()
        
        print("=" * 40)
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
