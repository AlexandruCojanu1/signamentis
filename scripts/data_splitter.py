"""
SignaMentis Data Splitter Module

This module creates proper train/validation/test splits for time series data
without time leakage, covering the last 3 years of data.

Author: SignaMentis Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Time series data splitter for XAU/USD trading data.
    
    Features:
    - No time leakage between splits
    - Covers last 3 years of data
    - Proper train/validation/test proportions
    - Session-aware splitting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data splitter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Split configuration
        self.train_years = self.config.get('train_years', 2)
        self.val_months = self.config.get('val_months', 6)
        self.test_months = self.config.get('test_months', 6)
        
        # Minimum data requirements
        self.min_train_samples = self.config.get('min_train_samples', 10000)
        self.min_val_samples = self.config.get('min_val_samples', 2000)
        self.min_test_samples = self.config.get('min_test_samples', 2000)
        
        # Session-aware splitting
        self.session_aware = self.config.get('session_aware', True)
        
        logger.info("DataSplitter initialized")
    
    def analyze_data_timeline(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the timeline of the dataset.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            Dictionary with timeline analysis
        """
        logger.info("Analyzing data timeline")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Timeline analysis
        timeline = {
            'start_date': df_sorted['timestamp'].min(),
            'end_date': df_sorted['timestamp'].max(),
            'total_duration': df_sorted['timestamp'].max() - df_sorted['timestamp'].min(),
            'total_records': len(df_sorted),
            'timeframe': self._detect_timeframe(df_sorted),
            'sessions_coverage': self._analyze_sessions(df_sorted)
        }
        
        logger.info(f"Timeline analysis:")
        logger.info(f"  Start: {timeline['start_date']}")
        logger.info(f"  End: {timeline['end_date']}")
        logger.info(f"  Duration: {timeline['total_duration']}")
        logger.info(f"  Records: {timeline['total_records']}")
        logger.info(f"  Timeframe: {timeline['timeframe']}")
        
        return timeline
    
    def _detect_timeframe(self, df: pd.DataFrame) -> str:
        """Detect the timeframe of the data."""
        if len(df) < 2:
            return "unknown"
        
        time_diff = df['timestamp'].diff().dropna()
        median_diff = time_diff.median()
        
        if median_diff <= pd.Timedelta(minutes=1):
            return "M1"
        elif median_diff <= pd.Timedelta(minutes=5):
            return "M5"
        elif median_diff <= pd.Timedelta(minutes=15):
            return "M15"
        elif median_diff <= pd.Timedelta(minutes=30):
            return "M30"
        elif median_diff <= pd.Timedelta(hours=1):
            return "H1"
        else:
            return "H4+"
    
    def _analyze_sessions(self, df: pd.DataFrame) -> Dict:
        """Analyze session coverage in the data."""
        if 'session' not in df.columns:
            return {'sessions_available': False}
        
        session_counts = df['session'].value_counts()
        total_records = len(df)
        
        session_coverage = {
            'sessions_available': True,
            'session_distribution': (session_counts / total_records * 100).to_dict(),
            'total_sessions': len(session_counts)
        }
        
        return session_coverage
    
    def create_time_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based splits without leakage.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating time-based splits")
        
        # Ensure timestamp is datetime and sorted
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate total duration
        total_duration = df_sorted['timestamp'].max() - df_sorted['timestamp'].min()
        total_days = total_duration.total_seconds() / (24 * 3600)
        
        logger.info(f"Total data duration: {total_days:.1f} days")
        
        # Adjust split strategy based on data size
        if total_days < 30:  # Less than 1 month
            # Use simple 60/20/20 split
            logger.info("Using simple 60/20/20 split for small dataset")
            
            total_records = len(df_sorted)
            train_size = int(total_records * 0.6)
            val_size = int(total_records * 0.2)
            
            train_df = df_sorted.iloc[:train_size].copy()
            val_df = df_sorted.iloc[train_size:train_size + val_size].copy()
            test_df = df_sorted.iloc[train_size + val_size:].copy()
            
        else:
            # Use time-based splits
            end_date = df_sorted['timestamp'].max()
            
            # Test set: last 6 months
            test_start = end_date - pd.DateOffset(months=self.test_months)
            
            # Validation set: 6 months before test
            val_start = test_start - pd.DateOffset(months=self.val_months)
            
            # Training set: everything before validation
            train_end = val_start
            
            logger.info(f"Split dates:")
            logger.info(f"  Train: start to {train_end}")
            logger.info(f"  Validation: {val_start} to {test_start}")
            logger.info(f"  Test: {test_start} to {end_date}")
            
            # Create splits
            train_df = df_sorted[df_sorted['timestamp'] < train_end].copy()
            val_df = df_sorted[
                (df_sorted['timestamp'] >= val_start) & 
                (df_sorted['timestamp'] < test_start)
            ].copy()
            test_df = df_sorted[df_sorted['timestamp'] >= test_start].copy()
        
        # Validate split sizes
        self._validate_split_sizes(train_df, val_df, test_df)
        
        logger.info(f"Split sizes:")
        logger.info(f"  Train: {len(train_df)} records")
        logger.info(f"  Validation: {len(val_df)} records")
        logger.info(f"  Test: {len(test_df)} records")
        
        return train_df, val_df, test_df
    
    def _validate_split_sizes(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Validate that splits meet minimum size requirements."""
        if len(train_df) < self.min_train_samples:
            raise ValueError(f"Training set too small: {len(train_df)} < {self.min_train_samples}")
        
        if len(val_df) < self.min_val_samples:
            raise ValueError(f"Validation set too small: {len(val_df)} < {self.min_val_samples}")
        
        if len(test_df) < self.min_test_samples:
            raise ValueError(f"Test set too small: {len(test_df)} < {self.min_test_samples}")
    
    def create_session_aware_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create session-aware splits to ensure all sessions are represented.
        
        Args:
            df: DataFrame with session column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not self.session_aware or 'session' not in df.columns:
            logger.info("Session-aware splitting disabled or session data not available")
            return self.create_time_splits(df)
        
        logger.info("Creating session-aware splits")
        
        # Get unique sessions
        sessions = df['session'].unique()
        logger.info(f"Available sessions: {sessions}")
        
        # Check if we have enough data for session-aware splitting
        total_records = len(df)
        min_records_per_session = 1000  # Minimum records per session for splitting
        
        if total_records < min_records_per_session * len(sessions):
            logger.info("Not enough data for session-aware splitting, using simple time-based split")
            return self.create_time_splits(df)
        
        # Create splits for each session
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for session in sessions:
            session_df = df[df['session'] == session].copy()
            
            if len(session_df) < 100:  # Skip sessions with too few records
                logger.warning(f"Session {session} has too few records: {len(session_df)}")
                continue
            
            # Create time splits for this session
            try:
                session_train, session_val, session_test = self.create_time_splits(session_df)
                
                train_dfs.append(session_train)
                val_dfs.append(session_val)
                test_dfs.append(session_test)
            except ValueError as e:
                logger.warning(f"Failed to split session {session}: {e}")
                continue
        
        # If no valid splits, fall back to simple time-based split
        if not train_dfs:
            logger.info("No valid session splits created, using simple time-based split")
            return self.create_time_splits(df)
        
        # Combine splits
        train_df = pd.concat(train_dfs, ignore_index=True).sort_values('timestamp')
        val_df = pd.concat(val_dfs, ignore_index=True).sort_values('timestamp')
        test_df = pd.concat(val_dfs, ignore_index=True).sort_values('timestamp')
        
        logger.info(f"Session-aware splits created:")
        logger.info(f"  Train: {len(train_df)} records")
        logger.info(f"  Validation: {len(val_df)} records")
        logger.info(f"  Test: {len(test_df)} records")
        
        return train_df, val_df, test_df
    
    def create_manifest(self, 
                       train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame) -> Dict:
        """
        Create a manifest file documenting the splits.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            
        Returns:
            Dictionary with split manifest
        """
        logger.info("Creating split manifest")
        
        manifest = {
            'split_info': {
                'train_years': self.train_years,
                'val_months': self.val_months,
                'test_months': self.test_months,
                'session_aware': self.session_aware
            },
            'data_info': {
                'total_records': len(train_df) + len(val_df) + len(test_df),
                'train_records': len(train_df),
                'val_records': len(val_df),
                'test_records': len(test_df)
            },
            'timeline': {
                'train_start': train_df['timestamp'].min().isoformat(),
                'train_end': train_df['timestamp'].max().isoformat(),
                'val_start': val_df['timestamp'].min().isoformat(),
                'val_end': val_df['timestamp'].max().isoformat(),
                'test_start': test_df['timestamp'].min().isoformat(),
                'test_end': test_df['timestamp'].max().isoformat()
            },
            'session_distribution': {}
        }
        
        # Add session distribution if available
        if 'session' in train_df.columns:
            for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                manifest['session_distribution'][split_name] = split_df['session'].value_counts().to_dict()
        
        # Add feature information
        feature_columns = [col for col in train_df.columns if col not in ['timestamp', 'session']]
        manifest['features'] = {
            'total_features': len(feature_columns),
            'feature_columns': feature_columns
        }
        
        logger.info("Split manifest created")
        return manifest
    
    def save_splits(self, 
                   train_df: pd.DataFrame, 
                   val_df: pd.DataFrame, 
                   test_df: pd.DataFrame,
                   manifest: Dict,
                   output_dir: str = "data/splits") -> Dict[str, str]:
        """
        Save all splits and manifest to disk.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            manifest: Split manifest
            output_dir: Output directory path
            
        Returns:
            Dictionary with file paths
        """
        logger.info(f"Saving splits to {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        train_path = Path(output_dir) / f"train_{timestamp}.parquet"
        val_path = Path(output_dir) / f"validation_{timestamp}.parquet"
        test_path = Path(output_dir) / f"test_{timestamp}.parquet"
        manifest_path = Path(output_dir) / f"manifest_{timestamp}.yaml"
        
        # Save DataFrames
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        # Save manifest
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
        
        logger.info("Splits saved successfully:")
        logger.info(f"  Train: {train_path}")
        logger.info(f"  Validation: {val_path}")
        logger.info(f"  Test: {test_path}")
        logger.info(f"  Manifest: {manifest_path}")
        
        return {
            'train_path': str(train_path),
            'val_path': str(val_path),
            'test_path': str(test_path),
            'manifest_path': str(manifest_path),
            'output_dir': output_dir
        }
    
    def split_data(self, 
                  df: pd.DataFrame, 
                  output_dir: str = "data/splits") -> Dict[str, str]:
        """
        Complete data splitting pipeline.
        
        Args:
            df: Input DataFrame with features
            output_dir: Output directory for splits
            
        Returns:
            Dictionary with file paths and manifest
        """
        logger.info("Starting complete data splitting pipeline")
        
        # 1. Analyze data timeline
        timeline = self.analyze_data_timeline(df)
        
        # 2. Create splits
        if self.session_aware and 'session' in df.columns:
            train_df, val_df, test_df = self.create_session_aware_splits(df)
        else:
            train_df, val_df, test_df = self.create_time_splits(df)
        
        # 3. Create manifest
        manifest = self.create_manifest(train_df, val_df, test_df)
        
        # 4. Save splits
        file_paths = self.save_splits(train_df, val_df, test_df, manifest, output_dir)
        
        # 5. Add timeline info to file paths
        file_paths['timeline'] = timeline
        
        logger.info("Data splitting pipeline completed successfully")
        return file_paths


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from scripts.data_normalizer import DataNormalizer
    from scripts.essential_features import EssentialFeatureEngineer
    
    # Initialize components
    normalizer = DataNormalizer()
    feature_engineer = EssentialFeatureEngineer()
    splitter = DataSplitter()
    
    # Load and normalize data
    m5_clean, m15_clean, norm_report = normalizer.normalize_data(
        m5_path="data/external/XAUUSD_Tickbar_5_BID_11.08.2023-11.08.2023.csv"
    )
    
    # Create features
    features_df = feature_engineer.create_essential_features(m5_clean, m15_clean)
    
    # Split data
    split_paths = splitter.split_data(features_df)
    
    print("Data splitting completed!")
    print(f"Split files: {split_paths}")
