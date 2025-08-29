#!/usr/bin/env python3
"""
Data utilities for SignaMentis AI Trading System.
Handles feature consistency, UTC timestamps, and data hygiene.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataUtils:
    """Utilities for consistent data handling and feature management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize DataUtils with configuration."""
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.canonical_features: Optional[List[str]] = None
        self.feature_order: Optional[List[str]] = None
        
    def _default_config(self) -> Dict:
        """Default configuration for data handling."""
        return {
            'general': {
                'seed': 42,
                'n_jobs': 4,
                'horizon_bars': 6
            },
            'data': {
                'timestamp_col': 'timestamp',
                'required_features': [
                    'current_close', 'current_high', 'current_low', 'current_open',
                    'm5_atr', 'm5_supertrend', 'm5_sma_20', 'm5_ema_20',
                    'm5_rsi', 'm5_bb_upper', 'm5_bb_lower', 'm5_macd',
                    'm15_atr', 'm15_supertrend', 'm15_sma_20', 'm15_ema_20',
                    'm15_rsi', 'm15_bb_upper', 'm15_bb_lower', 'm15_macd'
                ]
            },
            'splits': {
                'train_end': "2024-06-01T00:00:00Z",
                'val_end': "2025-02-01T00:00:00Z",
                'test_end': "2025-08-16T23:45:00Z"
            }
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
            return self._default_config()
    
    def load_features(self, path: str) -> pd.DataFrame:
        """
        Load features CSV with proper UTC timestamp handling.
        
        Args:
            path: Path to features CSV file
            
        Returns:
            DataFrame with UTC tz-aware timestamps and sorted by time
        """
        logger.info(f"Loading features from: {path}")
        
        # Load CSV
        df = pd.read_csv(path)
        logger.info(f"Loaded raw data: {df.shape}")
        
        # Handle timestamp column
        timestamp_col = self.config['data']['timestamp_col']
        if timestamp_col not in df.columns:
            # Try common alternatives
            alternatives = ['Local time', 'local_time', 'time', 'date']
            for alt in alternatives:
                if alt in df.columns:
                    df = df.rename(columns={alt: timestamp_col})
                    logger.info(f"Renamed '{alt}' to '{timestamp_col}'")
                    break
            else:
                raise ValueError(f"Timestamp column '{timestamp_col}' not found. Available: {list(df.columns)}")
        
        # Convert to UTC tz-aware timestamps
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='mixed', utc=True)
        
        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates(subset=[timestamp_col])
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} duplicate timestamps")
        
        # Data hygiene: drop all-NaN columns
        df = self._drop_all_nan_columns(df)
        
        # Validate required features
        self._validate_required_features(df)
        
        logger.info(f"Final data shape: {df.shape}")
        logger.info(f"Date range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
        
        return df
    
    def _drop_all_nan_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are all NaN and log what was dropped."""
        initial_cols = len(df.columns)
        
        # Find columns with all NaN values
        all_nan_cols = df.columns[df.isna().all()].tolist()
        
        if all_nan_cols:
            logger.warning(f"Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
            df = df.drop(columns=all_nan_cols)
        
        # Find columns with >90% NaN values
        high_nan_cols = []
        for col in df.columns:
            nan_ratio = df[col].isna().mean()
            if nan_ratio > 0.9:
                high_nan_cols.append((col, nan_ratio))
        
        if high_nan_cols:
            logger.warning(f"Columns with >90% NaN values:")
            for col, ratio in high_nan_cols:
                logger.warning(f"  {col}: {ratio:.1%} NaN")
        
        final_cols = len(df.columns)
        if final_cols < initial_cols:
            logger.info(f"Dropped {initial_cols - final_cols} problematic columns")
        
        return df
    
    def _validate_required_features(self, df: pd.DataFrame) -> None:
        """Validate that required features are present."""
        required = self.config['data']['required_features']
        missing = [feat for feat in required if feat not in df.columns]
        
        if missing:
            logger.warning(f"Missing required features: {missing}")
            logger.warning(f"Available features: {list(df.columns)}")
        else:
            logger.info("All required features present")
    
    def align_feature_columns(self, df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Align feature columns to ensure consistency across train/val/test.
        
        Args:
            df: Input DataFrame
            required_cols: List of required feature columns (if None, use canonical)
            
        Returns:
            DataFrame with aligned feature columns
        """
        if required_cols is None:
            if self.canonical_features is None:
                raise ValueError("Canonical features not set. Call set_canonical_features first.")
            required_cols = self.canonical_features
        
        # Ensure all required columns exist
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.warning(f"Adding missing columns with 0 values: {missing_cols}")
            for col in missing_cols:
                df[col] = 0
        
        # Reorder columns to match canonical order
        df = df[required_cols + [col for col in df.columns if col not in required_cols]]
        
        logger.info(f"Aligned features: {len(required_cols)} columns")
        return df
    
    def set_canonical_features(self, df: pd.DataFrame) -> None:
        """
        Set canonical feature list from training data.
        
        Args:
            df: Training DataFrame to extract canonical features from
        """
        # Get feature columns (exclude timestamp and targets)
        feature_cols = [col for col in df.columns 
                       if not col.startswith('target_') and col != self.config['data']['timestamp_col']]
        
        self.canonical_features = sorted(feature_cols)
        self.feature_order = self.canonical_features.copy()
        
        logger.info(f"Set canonical features: {len(self.canonical_features)} columns")
        logger.debug(f"Feature order: {self.canonical_features[:5]}...")
    
    def split_by_time(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data by time periods for reproducible train/val/test splits.
        
        Args:
            df: Input DataFrame with timestamp column
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        timestamp_col = self.config['data']['timestamp_col']
        
        # Parse split dates
        train_end = pd.to_datetime(self.config['splits']['train_end'], utc=True)
        val_end = pd.to_datetime(self.config['splits']['val_end'], utc=True)
        test_end = pd.to_datetime(self.config['splits']['test_end'], utc=True)
        
        # Split data
        train_mask = df[timestamp_col] < train_end
        val_mask = (df[timestamp_col] >= train_end) & (df[timestamp_col] < val_end)
        test_mask = (df[timestamp_col] >= val_end) & (df[timestamp_col] <= test_end)
        
        splits = {
            'train': df[train_mask].copy(),
            'validation': df[val_mask].copy(),
            'test': df[test_mask].copy()
        }
        
        # Log split statistics
        for split_name, split_data in splits.items():
            if len(split_data) > 0:
                logger.info(f"{split_name.capitalize()}: {len(split_data):,} samples "
                          f"({split_data[timestamp_col].min()} to {split_data[timestamp_col].max()})")
            else:
                logger.warning(f"{split_name.capitalize()}: No data")
        
        return splits
    
    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract feature matrix (exclude timestamp and targets).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Feature matrix DataFrame
        """
        if self.canonical_features is None:
            raise ValueError("Canonical features not set. Call set_canonical_features first.")
        
        return df[self.canonical_features].copy()
    
    def save_canonical_features(self, output_path: str) -> None:
        """Save canonical feature list to file."""
        if self.canonical_features is None:
            raise ValueError("Canonical features not set")
        
        feature_info = {
            'canonical_features': self.canonical_features,
            'feature_count': len(self.canonical_features),
            'timestamp_col': self.config['data']['timestamp_col'],
            'created_at': pd.Timestamp.now(tz='UTC').isoformat()
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(feature_info, f, default_flow_style=False)
        
        logger.info(f"Saved canonical features to: {output_path}")


if __name__ == "__main__":
    # Test the DataUtils class
    data_utils = DataUtils()
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2021-01-01', periods=100, freq='5T', tz='UTC'),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    print("Sample data created successfully")
    print(f"Shape: {sample_data.shape}")
    print(f"Columns: {list(sample_data.columns)}")
