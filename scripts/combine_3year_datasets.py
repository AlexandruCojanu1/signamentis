#!/usr/bin/env python3
"""
Combine 3-Year XAU/USD Datasets for Advanced Training

This script combines the 4 separate datasets into unified M5 and M15 datasets:
- M5: 2021-2024 + 2024-2025 = 4.6 years of data
- M15: 2021-2024 + 2024-2025 = 4.6 years of data

Author: SignaMentis Team
Version: 3.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetCombiner:
    """
    Combines multiple XAU/USD datasets into unified datasets for training.
    """
    
    def __init__(self):
        """Initialize the dataset combiner."""
        self.combined_m5 = None
        self.combined_m15 = None
        logger.info("DatasetCombiner initialized")
    
    def load_and_combine_m5_data(self) -> pd.DataFrame:
        """
        Load and combine M5 datasets.
        
        Returns:
            Combined M5 DataFrame
        """
        logger.info("Loading and combining M5 datasets...")
        
        # Load M5 datasets
        m5_2021_2024 = pd.read_csv('./XAUUSD_Candlestick_5_M_BID_01.01.2021-01.01.2024.csv')
        m5_2024_2025 = pd.read_csv('./XAUUSD_Candlestick_5_M_BID_01.01.2024-16.08.2025.csv')
        
        logger.info(f"M5 2021-2024: {m5_2021_2024.shape[0]:,} rows")
        logger.info(f"M5 2024-2025: {m5_2024_2025.shape[0]:,} rows")
        
        # Convert timestamps
        m5_2021_2024['timestamp'] = pd.to_datetime(m5_2021_2024['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
        m5_2024_2025['timestamp'] = pd.to_datetime(m5_2024_2025['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
        
        # Standardize column names
        m5_2021_2024 = m5_2021_2024.rename(columns={
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        m5_2024_2025 = m5_2024_2025.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low', 
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add spread column (typical for FX data)
        m5_2021_2024['spread'] = 0.0003  # 3 pips typical spread
        m5_2024_2025['spread'] = 0.0003
        
        # Combine datasets
        combined_m5 = pd.concat([m5_2021_2024, m5_2024_2025], ignore_index=True)
        
        # Sort by timestamp
        combined_m5 = combined_m5.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        combined_m5 = combined_m5.drop_duplicates(subset=['timestamp'])
        
        # Select final columns
        final_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']
        combined_m5 = combined_m5[final_columns]
        
        logger.info(f"Combined M5: {combined_m5.shape[0]:,} rows")
        logger.info(f"Date range: {combined_m5['timestamp'].min()} to {combined_m5['timestamp'].max()}")
        
        self.combined_m5 = combined_m5
        return combined_m5
    
    def load_and_combine_m15_data(self) -> pd.DataFrame:
        """
        Load and combine M15 datasets.
        
        Returns:
            Combined M15 DataFrame
        """
        logger.info("Loading and combining M15 datasets...")
        
        # Load M15 datasets
        m15_2021_2024 = pd.read_csv('./XAUUSD_Candlestick_15_M_BID_01.01.2021-01.01.2024.csv')
        m15_2024_2025 = pd.read_csv('./XAUUSD_Candlestick_15_M_BID_01.01.2024-16.08.2025.csv')
        
        logger.info(f"M15 2021-2024: {m15_2021_2024.shape[0]:,} rows")
        logger.info(f"M15 2024-2025: {m15_2024_2025.shape[0]:,} rows")
        
        # Convert timestamps
        m15_2021_2024['timestamp'] = pd.to_datetime(m15_2021_2024['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
        m15_2024_2025['timestamp'] = pd.to_datetime(m15_2024_2025['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
        
        # Standardize column names
        m15_2021_2024 = m15_2021_2024.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        m15_2024_2025 = m15_2024_2025.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add spread column
        m15_2021_2024['spread'] = 0.0003
        m15_2024_2025['spread'] = 0.0003
        
        # Combine datasets
        combined_m15 = pd.concat([m15_2021_2024, m15_2024_2025], ignore_index=True)
        
        # Sort by timestamp
        combined_m15 = combined_m15.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        combined_m15 = combined_m15.drop_duplicates(subset=['timestamp'])
        
        # Select final columns
        final_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']
        combined_m15 = combined_m15[final_columns]
        
        logger.info(f"Combined M15: {combined_m15.shape[0]:,} rows")
        logger.info(f"Date range: {combined_m15['timestamp'].min()} to {combined_m15['timestamp'].max()}")
        
        self.combined_m15 = combined_m15
        return combined_m15
    
    def validate_combined_data(self) -> Dict:
        """
        Validate the combined datasets.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating combined datasets...")
        
        validation_results = {}
        
        # M5 validation
        if self.combined_m5 is not None:
            m5_validation = {
                'total_rows': len(self.combined_m5),
                'date_range': {
                    'start': self.combined_m5['timestamp'].min(),
                    'end': self.combined_m5['timestamp'].max(),
                    'duration_days': (self.combined_m5['timestamp'].max() - self.combined_m5['timestamp'].min()).days
                },
                'duplicates': self.combined_m5.duplicated(subset=['timestamp']).sum(),
                'null_counts': self.combined_m5.isnull().sum().to_dict(),
                'ohlc_validation': self._validate_ohlc(self.combined_m5),
                'memory_usage_mb': self.combined_m5.memory_usage(deep=True).sum() / 1024 / 1024
            }
            validation_results['m5'] = m5_validation
        
        # M15 validation
        if self.combined_m15 is not None:
            m15_validation = {
                'total_rows': len(self.combined_m15),
                'date_range': {
                    'start': self.combined_m15['timestamp'].min(),
                    'end': self.combined_m15['timestamp'].max(),
                    'duration_days': (self.combined_m15['timestamp'].max() - self.combined_m15['timestamp'].min()).days
                },
                'duplicates': self.combined_m15.duplicated(subset=['timestamp']).sum(),
                'null_counts': self.combined_m15.isnull().sum().to_dict(),
                'ohlc_validation': self._validate_ohlc(self.combined_m15),
                'memory_usage_mb': self.combined_m15.memory_usage(deep=True).sum() / 1024 / 1024
            }
            validation_results['m15'] = m15_validation
        
        logger.info("Validation completed")
        return validation_results
    
    def _validate_ohlc(self, df: pd.DataFrame) -> Dict:
        """Validate OHLC relationships."""
        total_rows = len(df)
        
        # OHLC validation rules
        high_low_valid = (df['high'] >= df['low']).sum()
        open_close_valid = ((df['open'] >= df['low']) & (df['open'] <= df['high'])).sum()
        close_valid = ((df['close'] >= df['low']) & (df['close'] <= df['high'])).sum()
        
        return {
            'high_low_valid': high_low_valid,
            'high_low_valid_pct': high_low_valid / total_rows * 100,
            'open_valid': open_close_valid,
            'open_valid_pct': open_close_valid / total_rows * 100,
            'close_valid': close_valid,
            'close_valid_pct': close_valid / total_rows * 100
        }
    
    def save_combined_datasets(self, output_dir: str = "data/combined") -> Dict:
        """
        Save combined datasets to files.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary with file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # Save M5 data
        if self.combined_m5 is not None:
            m5_file = Path(output_dir) / f"XAUUSD_M5_combined_2021_2025_{timestamp}.csv"
            self.combined_m5.to_csv(m5_file, index=False)
            saved_files['m5'] = str(m5_file)
            logger.info(f"M5 data saved to: {m5_file}")
        
        # Save M15 data
        if self.combined_m15 is not None:
            m15_file = Path(output_dir) / f"XAUUSD_M15_combined_2021_2025_{timestamp}.csv"
            self.combined_m15.to_csv(m15_file, index=False)
            saved_files['m15'] = str(m15_file)
            logger.info(f"M15 data saved to: {m15_file}")
        
        return saved_files
    
    def combine_all_datasets(self) -> Dict:
        """
        Main method to combine all datasets.
        
        Returns:
            Dictionary with combined datasets and validation results
        """
        logger.info("🚀 Starting dataset combination process...")
        logger.info("=" * 60)
        
        try:
            # Combine M5 data
            m5_data = self.load_and_combine_m5_data()
            
            # Combine M15 data
            m15_data = self.load_and_combine_m15_data()
            
            # Validate combined data
            validation_results = self.validate_combined_data()
            
            # Save combined datasets
            saved_files = self.save_combined_datasets()
            
            # Final summary
            summary = {
                'm5_summary': {
                    'total_rows': len(m5_data),
                    'date_range': f"{m5_data['timestamp'].min()} to {m5_data['timestamp'].max()}",
                    'duration_years': (m5_data['timestamp'].max() - m5_data['timestamp'].min()).days / 365
                },
                'm15_summary': {
                    'total_rows': len(m15_data),
                    'date_range': f"{m15_data['timestamp'].min()} to {m15_data['timestamp'].max()}",
                    'duration_years': (m15_data['timestamp'].max() - m15_data['timestamp'].min()).days / 365
                },
                'validation': validation_results,
                'saved_files': saved_files
            }
            
            logger.info("=" * 60)
            logger.info("✅ Dataset combination completed successfully!")
            logger.info(f"📊 M5: {len(m5_data):,} rows")
            logger.info(f"📊 M15: {len(m15_data):,} rows")
            logger.info("=" * 60)
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Dataset combination failed: {e}")
            raise


if __name__ == "__main__":
    # Test the dataset combiner
    combiner = DatasetCombiner()
    
    try:
        results = combiner.combine_all_datasets()
        
        print("\n🎯 FINAL RESULTS:")
        print("=" * 40)
        print(f"M5 Dataset: {results['m5_summary']['total_rows']:,} rows")
        print(f"M15 Dataset: {results['m15_summary']['total_rows']:,} rows")
        print(f"Duration: {results['m5_summary']['duration_years']:.1f} years")
        print(f"Files saved: {list(results['saved_files'].values())}")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        raise
