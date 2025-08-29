#!/usr/bin/env python3
"""
Target engineering for SignaMentis AI Trading System.
Creates directional, strength, duration, and price target labels.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TargetEngineer:
    """Engineers various target variables for multi-task learning."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize TargetEngineer with configuration."""
        self.config = config or self._default_config()
        self.horizon_bars = self.config.get('horizon_bars', 6)
        self.price_levels_atr = self.config.get('price_levels_atr', [0.5, 1.0, 1.5])
        
    def _default_config(self) -> Dict:
        """Default configuration for target engineering."""
        return {
            'horizon_bars': 6,  # ~30 minutes for M5
            'price_levels_atr': [0.5, 1.0, 1.5],
            'strength_thresholds': {
                'weak': 0.5,
                'medium': 1.5
            },
            'duration_bins': [2, 6, 12]  # [0-2, 3-6, 7-12, not_reached]
        }
    
    def create_all_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all target variables for the dataset.
        
        Args:
            df: Input DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with all target variables added
        """
        logger.info("Creating comprehensive target variables...")
        
        # Ensure we have required columns
        required_cols = ['current_close', 'current_high', 'current_low', 'current_open']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create directional targets
        df = self._create_directional_targets(df)
        
        # Create strength and duration targets
        df = self._create_strength_duration_targets(df)
        
        # Create price target labels
        df = self._create_price_target_labels(df)
        
        # Remove rows with NaN targets (end of dataset)
        target_cols = [col for col in df.columns if col.startswith('target_')]
        initial_rows = len(df)
        df = df.dropna(subset=target_cols)
        
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} rows with NaN targets")
        
        logger.info(f"Target creation completed. Final shape: {df.shape}")
        logger.info(f"Target columns: {[col for col in df.columns if col.startswith('target_')]}")
        
        return df
    
    def _create_directional_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create directional classification targets."""
        logger.info("Creating directional targets...")
        
        close_col = 'current_close'
        
        # Calculate future returns
        future_returns_1 = df[close_col].shift(-1).pct_change(1)
        future_returns_3 = df[close_col].shift(-3).pct_change(3)
        future_returns_5 = df[close_col].shift(-5).pct_change(5)
        
        # Create direction targets (1 = up, 0 = down)
        df['target_direction_1'] = (future_returns_1 > 0).astype(int)
        df['target_direction_3'] = (future_returns_3 > 0).astype(int)
        df['target_direction_5'] = (future_returns_5 > 0).astype(int)
        
        # Create return targets
        df['target_return_1'] = future_returns_1
        df['target_return_3'] = future_returns_3
        df['target_return_5'] = future_returns_5
        
        logger.info("Directional targets created")
        return df
    
    def _create_strength_duration_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create strength and duration targets based on ATR estimate and future bars.
        Strength: 3 bins from relative amplitude in 5 bars: abs(return_5)/ATR_estimat
        Duration: bars to hit ±0.5 ATR from current_close, capped at 5; if not hit -> 5
        """
        logger.info("Creating strength and duration targets...")
        
        close_col = 'current_close'
        high_col = 'current_high'
        low_col = 'current_low'
        
        # Prefer provided ATR columns; else estimate ATR (ATR_estimat)
        atr_col = None
        for col in ['m5_atr', 'm15_atr', 'atr']:
            if col in df.columns:
                atr_col = col
                break
        if atr_col is None:
            logger.warning("No ATR column found. Estimating ATR from returns.")
            returns = df[close_col].pct_change()
            df['ATR_estimat'] = returns.rolling(20).std() * df[close_col]
            atr_col = 'ATR_estimat'
        
        # Strength: relative amplitude over next 5 bars
        future_close_5 = df[close_col].shift(-5)
        ret_5 = (future_close_5 - df[close_col]) / df[close_col]
        rel_ampl = (ret_5.abs()) / (df[atr_col] / df[close_col]).replace(0, np.nan)
        # Bins: Low: 0–0.5; Med: 0.5–1.0; High: >1.0
        strength_numeric = pd.cut(rel_ampl, bins=[-np.inf, 0.5, 1.0, np.inf], labels=[0,1,2]).astype('float')
        df['target_strength'] = strength_numeric
        
        # Duration: bars to hit ±0.5 ATR, capped at 5 (compute in bar indices)
        duration_bars = pd.Series(index=df.index, dtype=float)
        for i in range(len(df) - 5):
            atr_norm = (df[atr_col].iloc[i] / df[close_col].iloc[i]) if not pd.isna(df[atr_col].iloc[i]) and df[close_col].iloc[i] != 0 else np.nan
            if pd.isna(atr_norm):
                duration_bars.iloc[i] = np.nan
                continue
            up_target = df[close_col].iloc[i] * (1 + 0.5 * atr_norm)
            down_target = df[close_col].iloc[i] * (1 - 0.5 * atr_norm)
            fut_high = df[high_col].iloc[i+1:i+6]
            fut_low = df[low_col].iloc[i+1:i+6]
            up_hit = (fut_high >= up_target)
            down_hit = (fut_low <= down_target)
            hit_any = up_hit | down_hit
            if hit_any.any():
                # first index position corresponds to bars ahead (1..5)
                first_hit_pos = np.argmax(hit_any.values) + 1
                duration_bars.iloc[i] = float(first_hit_pos)
            else:
                duration_bars.iloc[i] = 5.0
        # Fill remaining tail with NaN
        df['target_duration_bars'] = duration_bars
        
        # Duration classification bins: {1}, {2}, {3–4}, {5}
        dur_bins = pd.Series(index=df.index, dtype=float)
        for i in range(len(df)):
            d = df['target_duration_bars'].iloc[i]
            if pd.isna(d):
                dur_bins.iloc[i] = np.nan
            elif d <= 1:
                dur_bins.iloc[i] = 0
            elif d <= 2:
                dur_bins.iloc[i] = 1
            elif d <= 4:
                dur_bins.iloc[i] = 2
            else:
                dur_bins.iloc[i] = 3
        df['target_duration_bins'] = dur_bins
        
        logger.info("Strength and duration targets created")
        return df
    
    def _calculate_mfe_targets(self, df: pd.DataFrame, atr_col: str) -> pd.DataFrame:
        """Deprecated in new labeling; keep no-op for compatibility."""
        return df
    
    def _calculate_strength_targets(self, df: pd.DataFrame, atr_col: str) -> pd.DataFrame:
        """Deprecated in new labeling; handled in _create_strength_duration_targets."""
        return df
    
    def _calculate_duration_targets(self, df: pd.DataFrame, atr_col: str) -> pd.DataFrame:
        """Deprecated in new labeling; handled in _create_strength_duration_targets."""
        return df
    
    def _create_price_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price target hitting probability labels."""
        logger.info("Creating price target labels...")
        
        # Get ATR column
        atr_col = None
        for col in ['m5_atr', 'm15_atr', 'atr', 'estimated_atr']:
            if col in df.columns:
                atr_col = col
                break
        
        if atr_col is None:
            logger.warning("No ATR column found. Skipping price target labels.")
            return df
        
        close_col = 'current_close'
        high_col = 'current_high'
        low_col = 'current_low'
        
        # Create hitting probability labels for each ATR level
        for level_multiplier in self.price_levels_atr:
            level_name = f"atr_{level_multiplier:.1f}".replace('.', '_')
            
            # Calculate target price levels
            target_up = df[close_col] * (1 + level_multiplier * df[atr_col] / df[close_col])
            target_down = df[close_col] * (1 - level_multiplier * df[atr_col] / df[close_col])
            
            # Check if targets are hit within horizon
            hit_up = pd.Series(index=df.index, dtype=float)
            hit_down = pd.Series(index=df.index, dtype=float)
            
            for i in range(len(df) - self.horizon_bars):
                if pd.isna(df['target_direction_1'].iloc[i]):
                    hit_up.iloc[i] = np.nan
                    hit_down.iloc[i] = np.nan
                    continue
                
                # Look ahead horizon_bars
                future_high = df[high_col].iloc[i+1:i+1+self.horizon_bars].max()
                future_low = df[low_col].iloc[i+1:i+1+self.horizon_bars].min()
                
                # Check if up target is hit
                hit_up.iloc[i] = (future_high >= target_up.iloc[i]).astype(int)
                
                # Check if down target is hit
                hit_down.iloc[i] = (future_low <= target_down.iloc[i]).astype(int)
            
            # Store hitting probabilities
            df[f'target_hit_up_{level_name}'] = hit_up
            df[f'target_hit_down_{level_name}'] = hit_down
        
        logger.info("Price target labels created")
        return df
    
    def get_target_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for all target variables."""
        target_cols = [col for col in df.columns if col.startswith('target_')]
        
        summary = {}
        for col in target_cols:
            if col in df.columns:
                target_data = df[col].dropna()
                if len(target_data) > 0:
                    if target_data.dtype in ['int64', 'float64']:
                        if col.startswith('target_direction_') or col.startswith('target_hit_'):
                            # Binary targets
                            summary[col] = {
                                'type': 'binary',
                                'count': len(target_data),
                                'positive_rate': target_data.mean(),
                                'unique_values': sorted(target_data.unique())
                            }
                        elif col.startswith('target_strength') or col.startswith('target_duration_bins'):
                            # Categorical targets
                            summary[col] = {
                                'type': 'categorical',
                                'count': len(target_data),
                                'value_counts': target_data.value_counts().to_dict(),
                                'unique_values': sorted(target_data.unique())
                            }
                        else:
                            # Continuous targets
                            summary[col] = {
                                'type': 'continuous',
                                'count': len(target_data),
                                'mean': target_data.mean(),
                                'std': target_data.std(),
                                'min': target_data.min(),
                                'max': target_data.max()
                            }
        
        return summary


if __name__ == "__main__":
    # Test the TargetEngineer class
    target_engineer = TargetEngineer()
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', periods=1000, freq='5T', tz='UTC')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'current_close': 1800 + np.cumsum(np.random.randn(1000) * 0.1),
        'current_high': 1800 + np.cumsum(np.random.randn(1000) * 0.1) + np.random.uniform(0, 2, 1000),
        'current_low': 1800 + np.cumsum(np.random.randn(1000) * 0.1) - np.random.uniform(0, 2, 1000),
        'current_open': 1800 + np.cumsum(np.random.randn(1000) * 0.1),
        'm5_atr': np.random.uniform(0.5, 2.0, 1000),
        'm5_supertrend': np.random.randn(1000),
        'm5_sma_20': 1800 + np.cumsum(np.random.randn(1000) * 0.05),
        'm5_ema_20': 1800 + np.cumsum(np.random.randn(1000) * 0.05),
        'm5_rsi': np.random.uniform(20, 80, 1000),
        'm5_bb_upper': 1800 + np.cumsum(np.random.randn(1000) * 0.05) + 2,
        'm5_bb_lower': 1800 + np.cumsum(np.random.randn(1000) * 0.05) - 2,
        'm5_macd': np.random.randn(1000)
    })
    
    print("Sample data created successfully")
    print(f"Shape: {sample_data.shape}")
    
    # Create targets
    sample_data_with_targets = target_engineer.create_all_targets(sample_data)
    
    print(f"Data with targets shape: {sample_data_with_targets.shape}")
    
    # Get target summary
    summary = target_engineer.get_target_summary(sample_data_with_targets)
    
    print("\nTarget Summary:")
    for target, info in summary.items():
        print(f"\n{target}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
