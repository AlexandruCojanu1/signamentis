#!/usr/bin/env python3
"""
Inference script for SOL/USDT 15m direction prediction.
"""

import argparse
import json
import logging
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from scripts.binance_data_fetcher import BinanceDataFetcher
from scripts.feature_engineer import FeatureEngineer
from scripts.sequence_builder import SequenceBuilder
from scripts.transformer_bilstm_model import TransformerBiLSTMClassifier, TemperatureScaling

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Infer SOL/USDT 15m direction')
    
    parser.add_argument('--save-dir', type=str, required=True,
                       help='Directory containing trained model artifacts')
    parser.add_argument('--from-csv', type=str, default=None,
                       help='Use local CSV file instead of live data')
    
    return parser.parse_args()

def load_artifacts(save_dir):
    """Load all training artifacts."""
    # Load metadata
    with open(os.path.join(save_dir, 'preproc_and_meta.json'), 'r') as f:
        meta = json.load(f)
    
    # Load model config
    with open(os.path.join(save_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    return meta, config

def load_model(save_dir, meta, config, device):
    """Load trained model."""
    # Initialize model
    n_features = len(meta['feature_cols'])
    model = TransformerBiLSTMClassifier(n_features, config)
    
    # Load checkpoint
    checkpoint_path = os.path.join(save_dir, 'model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load temperature scaler
    temp_scaler = TemperatureScaling()
    temp_scaler.temperature.data = torch.tensor([meta['temperature']])
    temp_scaler.to(device)
    
    return model, temp_scaler

def prepare_sequence_builder(meta):
    """Prepare sequence builder from metadata."""
    seq_builder = SequenceBuilder(seq_len=meta['seq_len'])
    seq_builder.feature_cols = meta['feature_cols']
    
    # Recreate scaler
    from sklearn.preprocessing import StandardScaler
    seq_builder.scaler = StandardScaler()
    seq_builder.scaler.mean_ = np.array(meta['scaler_mean'])
    seq_builder.scaler.scale_ = np.array(meta['scaler_scale'])
    seq_builder.scaler.n_features_in_ = len(meta['feature_cols'])
    
    return seq_builder

def get_latest_data(from_csv=None, years=1):
    """Get latest data for inference."""
    if from_csv:
        logging.info(f"Loading data from CSV: {from_csv}")
        df = pd.read_csv(from_csv, index_col=0, parse_dates=True)
        if not df.index.tz:
            df.index = df.index.tz_localize('UTC')
    else:
        logging.info("Fetching latest data from Binance...")
        fetcher = BinanceDataFetcher()
        df = fetcher.download_historical_data('SOLUSDT', '15m', years)
    
    return df

def main():
    """Main inference function."""
    setup_logging()
    args = parse_args()
    
    try:
        # Load artifacts
        logging.info(f"Loading artifacts from {args.save_dir}...")
        meta, config = load_artifacts(args.save_dir)
        
        # Setup device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        logging.info(f"Using device: {device}")
        
        # Load model
        logging.info("Loading trained model...")
        model, temp_scaler = load_model(args.save_dir, meta, config, device)
        
        # Prepare sequence builder
        seq_builder = prepare_sequence_builder(meta)
        
        # Get latest data
        df = get_latest_data(args.from_csv)
        logging.info(f"Data span: {df.index[0]} to {df.index[-1]} ({len(df)} rows)")
        
        # Feature engineering
        logging.info("Engineering features...")
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.compute_features(df)
        
        # Prepare sequence for inference
        logging.info("Preparing sequence...")
        sequence = seq_builder.transform_for_inference(features)
        
        # Make prediction
        logging.info("Making prediction...")
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).to(device)
            logits = model(sequence_tensor)
            
            # Apply temperature scaling
            calibrated_logits = temp_scaler(logits)
            probs = torch.softmax(calibrated_logits, dim=1)
            
            pred_class = logits.argmax(dim=1).item()
            pred_probs = probs.cpu().numpy()[0]
        
        # Format results
        label_map = meta['label_map']
        pred_label = label_map[str(pred_class)]
        
        # Get input window timestamps
        seq_len = meta['seq_len']
        input_start = features.index[-seq_len] if len(features) >= seq_len else features.index[0]
        input_end = features.index[-1]
        
        # Output results
        print("\n" + "="*50)
        print("SOL/USDT 15m Direction Prediction")
        print("="*50)
        print(f"Prediction: {pred_label}")
        print(f"Confidence: {pred_probs[pred_class]:.3f}")
        print(f"\nClass Probabilities:")
        for i, (class_idx, class_name) in enumerate(label_map.items()):
            print(f"  {class_name}: {pred_probs[int(class_idx)]:.3f}")
        print(f"\nInput Window:")
        print(f"  Start: {input_start}")
        print(f"  End:   {input_end}")
        print(f"  Length: {len(features) if len(features) < seq_len else seq_len} timesteps")
        print("="*50)
        
        # Also return as dict for programmatic use
        result = {
            'prediction': pred_label,
            'class_id': pred_class,
            'probabilities': {
                label_map[str(i)]: float(pred_probs[i]) for i in range(len(pred_probs))
            },
            'confidence': float(pred_probs[pred_class]),
            'input_window': {
                'start': str(input_start),
                'end': str(input_end),
                'length': len(features) if len(features) < seq_len else seq_len
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save result
        result_path = os.path.join(args.save_dir, 'latest_prediction.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logging.info(f"Prediction saved to: {result_path}")
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise

if __name__ == '__main__':
    main()
