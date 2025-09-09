#!/usr/bin/env python3
"""
Training script for SOL/USDT 15m direction prediction using Transformer + BiLSTM.
"""

import argparse
import os
import json
import logging
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# Import our modules
from scripts.binance_data_fetcher import BinanceDataFetcher
from scripts.feature_engineer import FeatureEngineer
from scripts.label_engineer import LabelEngineer
from scripts.sequence_builder import SequenceBuilder
from scripts.trainer import DirectionTrainer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SOL/USDT 15m direction classifier')
    
    # Data parameters
    parser.add_argument('--years', type=int, default=3, help='Years of historical data')
    parser.add_argument('--k', type=float, default=0.25, help='ATR threshold multiplier')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    parser.add_argument('--d-model', type=int, default=128, help='Transformer model dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Transformer layers')
    parser.add_argument('--nhead', type=int, default=4, help='Transformer attention heads')
    parser.add_argument('--lstm-hidden', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Technical parameters
    parser.add_argument('--mixed-precision', choices=['auto', 'on', 'off'], default='auto',
                       help='Mixed precision training')
    
    # I/O parameters
    parser.add_argument('--save-dir', type=str, default='./checkpoints_direction_15m',
                       help='Directory to save artifacts')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--fast-dev-run', action='store_true', 
                       help='Fast dev run on 10k samples')
    
    return parser.parse_args()

def create_config(args):
    """Create configuration dictionary from arguments."""
    return {
        # Data config
        'years': args.years,
        'k': args.k,
        'val_ratio': args.val_ratio,
        'seq_len': args.seq_len,
        
        # Feature engineering config
        'volatility_window': 48,
        'ema_periods': [20, 50, 200],
        'rsi_period': 14,
        'macd_params': [12, 26, 9],
        'atr_period': 14,
        
        # Model config
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'nhead': args.nhead,
        'lstm_hidden': args.lstm_hidden,
        'dropout': args.dropout,
        'n_classes': 3,
        
        # Training config
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'label_smoothing': args.label_smoothing,
        'mixed_precision': args.mixed_precision,
        'seed': args.seed,
        
        # I/O config
        'save_dir': args.save_dir,
        'resume': args.resume,
        'fast_dev_run': args.fast_dev_run
    }

def sanity_checks(features, labels, seq_builder, label_engineer):
    """Run sanity checks on the pipeline."""
    logging.info("Running sanity checks...")
    
    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    assert len(unique_labels) == 3, f"Expected 3 classes, got {len(unique_labels)}"
    logging.info("✓ All three classes present in labels")
    
    # Check sequence alignment (spot check)
    # This is a simplified check - in practice you'd want more thorough validation
    assert len(features) == len(labels), "Features and labels length mismatch"
    logging.info("✓ Features and labels aligned")
    
    # Check scaler fitting
    assert seq_builder.scaler.mean_ is not None, "Scaler not fitted"
    logging.info("✓ Scaler fitted on training data")
    
    logging.info("All sanity checks passed!")

def main():
    """Main training function."""
    setup_logging()
    args = parse_args()
    config = create_config(args)
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Starting training with config: {config}")
    
    try:
        # 1. Download data
        logging.info("Downloading historical data...")
        fetcher = BinanceDataFetcher()
        df = fetcher.download_historical_data('SOLUSDT', '15m', config['years'])
        
        if config['fast_dev_run']:
            df = df.tail(10000)
            logging.info("Fast dev run: using last 10k rows")
        
        logging.info(f"Data span: {df.index[0]} to {df.index[-1]} ({len(df)} rows)")
        
        # 2. Feature engineering
        logging.info("Engineering features...")
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.compute_features(df)
        
        # 3. Label engineering
        logging.info("Creating labels...")
        label_engineer = LabelEngineer(config['k'])
        features, labels = label_engineer.create_labels(features)
        
        # 4. Sequence preparation
        logging.info("Preparing sequences...")
        seq_builder = SequenceBuilder(config['seq_len'], config['val_ratio'])
        X_train, y_train, X_val, y_val = seq_builder.prepare_sequences(features, labels)
        
        # 5. Sanity checks
        sanity_checks(features, labels, seq_builder, label_engineer)
        
        # 6. Model training
        logging.info("Training model...")
        trainer = DirectionTrainer(config)
        trainer.prepare_model(X_train.shape[-1])
        
        # Resume if requested
        if config['resume']:
            checkpoint_path = os.path.join(config['save_dir'], 'model.pt')
            if os.path.exists(checkpoint_path):
                logging.info("Resuming from checkpoint...")
                checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.best_val_acc = checkpoint['best_val_acc']
                trainer.best_epoch = checkpoint['epoch']
                logging.info(f"Resumed from epoch {trainer.best_epoch} with val_acc={trainer.best_val_acc:.4f}")
        
        # Train
        training_results = trainer.train(X_train, y_train, X_val, y_val, config['save_dir'])
        
        # 7. Load best model and calibrate
        logging.info("Loading best model and calibrating probabilities...")
        checkpoint = torch.load(os.path.join(config['save_dir'], 'model.pt'), 
                               map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        temperature = trainer.calibrate_temperature(X_val, y_val)
        
        # 8. Final evaluation
        logging.info("Final evaluation...")
        eval_results = trainer.evaluate(X_val, y_val, label_engineer.get_label_map())
        
        # 9. Save artifacts
        logging.info("Saving artifacts...")
        
        # Preprocessing metadata
        preproc_meta = {
            'feature_cols': seq_builder.feature_cols,
            'scaler_mean': seq_builder.scaler.mean_.tolist(),
            'scaler_scale': seq_builder.scaler.scale_.tolist(),
            'seq_len': config['seq_len'],
            'label_map': label_engineer.get_label_map(),
            'k': config['k'],
            'temperature': temperature,
            'training_config': config,
            'data_span': {
                'start': str(df.index[0]),
                'end': str(df.index[-1]),
                'total_rows': len(df)
            },
            'training_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(config['save_dir'], 'preproc_and_meta.json'), 'w') as f:
            json.dump(preproc_meta, f, indent=2)
        
        # Training summary
        summary = {
            **training_results,
            **eval_results,
            'temperature': temperature
        }
        
        with open(os.path.join(config['save_dir'], 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Simple README
        readme_content = f"""# SOL/USDT 15m Direction Classifier

## Training Results
- Best Validation Accuracy: {training_results['best_val_acc']:.4f}
- Best Epoch: {training_results['best_epoch']}
- Macro F1 Score: {eval_results['f1_macro']:.4f}
- Temperature: {temperature:.3f}

## Usage

### Training
```bash
python train_direction_15m_sol.py --epochs 50 --batch-size 256
```

### Inference
```bash
python infer_direction_15m_sol.py --save-dir {config['save_dir']}
```

## Files
- `model.pt`: Best model checkpoint
- `preproc_and_meta.json`: Preprocessing parameters and metadata
- `training_summary.json`: Complete training and evaluation results
- `training_metrics.csv`: Per-epoch training metrics
- `config.json`: Training configuration
"""
        
        with open(os.path.join(config['save_dir'], 'README.md'), 'w') as f:
            f.write(readme_content)
        
        logging.info(f"Training completed successfully!")
        logging.info(f"Best validation accuracy: {training_results['best_val_acc']:.4f}")
        logging.info(f"Artifacts saved to: {config['save_dir']}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    main()
