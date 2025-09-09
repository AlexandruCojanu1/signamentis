#!/usr/bin/env python3
"""
Multi-Timeframe Feature Engineering for 3-Year XAU/USD Dataset

This module creates advanced features by combining M5 and M15 data:
- M5 micro-structure features (461K+ bars)
- M15 macro-trend features (162K+ bars)
- Cross-timeframe momentum and volatility
- Session-aware features across 4.6 years
- Advanced technical confluences

Author: SignaMentis Team
Version: 3.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import gc

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTimeframeFeatureEngineer3Years:
    """
    Advanced multi-timeframe feature engineering for 3+ year datasets.
    
    Optimized for large datasets with efficient memory management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the multi-timeframe feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        logger.info("MultiTimeframeFeatureEngineer3Years initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration for 3-year multi-timeframe features."""
        return {
            'lookback_periods': {
                'm5_bars': 12,  # 1 hour of M5 data
                'm15_bars': 8,  # 2 hours of M15 data
                'min_bars': 3   # Minimum bars required
            },
            'technical_indicators': {
                'momentum_periods': [3, 6, 12, 24, 48],
                'volatility_periods': [6, 12, 24, 48, 96],
                'trend_periods': [24, 48, 96, 192]
            },
            'session_definitions': {
                'asia': (0, 8),
                'london': (8, 16),
                'newyork': (13, 21),
                'overlap_london_ny': (13, 16)
            },
            'volatility_thresholds': {
                'low': 0.2,
                'medium': 0.5,
                'high': 0.8
            },
            'memory_optimization': {
                'chunk_size': 50000,  # Process in chunks
                'use_dtype_optimization': True,
                'cleanup_memory': True
            }
        }
    
    def load_combined_data(self, m5_file: str, m15_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the combined 3-year datasets.
        
        Args:
            m5_file: Path to combined M5 data file
            m15_file: Path to combined M15 data file
            
        Returns:
            Tuple of (m5_data, m15_data) DataFrames
        """
        logger.info("Loading combined 3-year datasets...")
        
        # Load data with optimized dtypes
        m5_data = pd.read_csv(m5_file)
        m15_data = pd.read_csv(m15_file)
        
        # Convert timestamps
        m5_data['timestamp'] = pd.to_datetime(m5_data['timestamp'])
        m15_data['timestamp'] = pd.to_datetime(m15_data['timestamp'])
        
        # Sort by timestamp
        m5_data = m5_data.sort_values('timestamp').reset_index(drop=True)
        m15_data = m15_data.sort_values('timestamp').reset_index(drop=True)
        
        # Optimize memory usage
        if self.config['memory_optimization']['use_dtype_optimization']:
            m5_data = self._optimize_dtypes(m5_data)
            m15_data = self._optimize_dtypes(m15_data)
        
        logger.info(f"M5 data: {m5_data.shape[0]:,} bars from {m5_data['timestamp'].min()} to {m5_data['timestamp'].max()}")
        logger.info(f"M15 data: {m15_data.shape[0]:,} bars from {m15_data['timestamp'].min()} to {m15_data['timestamp'].max()}")
        
        return m5_data, m15_data
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency."""
        # Optimize numeric columns
        for col in ['open', 'high', 'low', 'close', 'spread']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize volume column
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
        
        return df
    
    def create_m5_microstructure_features(self, m5_subset: pd.DataFrame) -> Dict:
        """
        Create microstructure features from M5 data.
        
        Args:
            m5_subset: Subset of M5 data (last N bars)
            
        Returns:
            Dictionary of microstructure features
        """
        features = {}
        
        if len(m5_subset) < 3:
            return self._empty_microstructure_features()
        
        # Price action features
        features['m5_price_momentum_3'] = m5_subset['close'].iloc[-1] - m5_subset['close'].iloc[-4] if len(m5_subset) >= 4 else 0
        features['m5_price_momentum_6'] = m5_subset['close'].iloc[-1] - m5_subset['close'].iloc[-7] if len(m5_subset) >= 7 else 0
        features['m5_price_momentum_12'] = m5_subset['close'].iloc[-1] - m5_subset['close'].iloc[-13] if len(m5_subset) >= 13 else 0
        features['m5_price_acceleration'] = features['m5_price_momentum_3'] - (m5_subset['close'].iloc[-4] - m5_subset['close'].iloc[-7]) if len(m5_subset) >= 7 else 0
        
        # Volatility features
        m5_returns = m5_subset['close'].pct_change().dropna()
        features['m5_volatility_3'] = m5_returns.tail(3).std() if len(m5_returns) >= 3 else 0
        features['m5_volatility_6'] = m5_returns.tail(6).std() if len(m5_returns) >= 6 else 0
        features['m5_volatility_12'] = m5_returns.tail(12).std() if len(m5_returns) >= 12 else 0
        features['m5_max_move_3'] = m5_returns.tail(3).abs().max() if len(m5_returns) >= 3 else 0
        
        # Range features
        features['m5_avg_range_3'] = (m5_subset['high'] - m5_subset['low']).tail(3).mean()
        features['m5_avg_range_6'] = (m5_subset['high'] - m5_subset['low']).tail(6).mean() if len(m5_subset) >= 6 else features['m5_avg_range_3']
        features['m5_avg_range_12'] = (m5_subset['high'] - m5_subset['low']).tail(12).mean() if len(m5_subset) >= 12 else features['m5_avg_range_6']
        features['m5_range_expansion'] = features['m5_avg_range_3'] / features['m5_avg_range_12'] if features['m5_avg_range_12'] > 0 else 1
        
        # Trend consistency
        price_changes = m5_subset['close'].diff().dropna()
        features['m5_trend_consistency'] = (price_changes > 0).sum() / len(price_changes) if len(price_changes) > 0 else 0.5
        
        # Volume analysis
        features['m5_volume_trend'] = np.corrcoef(range(len(m5_subset)), m5_subset['volume'])[0, 1] if len(m5_subset) > 1 else 0
        features['m5_volume_avg'] = m5_subset['volume'].mean()
        
        # Spread analysis
        features['m5_spread_avg'] = m5_subset['spread'].mean()
        features['m5_spread_volatility'] = m5_subset['spread'].std()
        features['m5_spread_trend'] = np.corrcoef(range(len(m5_subset)), m5_subset['spread'])[0, 1] if len(m5_subset) > 1 and m5_subset['spread'].std() > 0 else 0
        
        # Support/Resistance levels
        recent_highs = m5_subset['high'].tail(12)
        recent_lows = m5_subset['low'].tail(12)
        current_price = m5_subset['close'].iloc[-1]
        
        features['m5_resistance_distance'] = (recent_highs.max() - current_price) / current_price if current_price > 0 else 0
        features['m5_support_distance'] = (current_price - recent_lows.min()) / current_price if current_price > 0 else 0
        
        return features
    
    def create_m15_macro_features(self, m15_subset: pd.DataFrame) -> Dict:
        """
        Create macro-trend features from M15 data.
        
        Args:
            m15_subset: Subset of M15 data (last N bars)
            
        Returns:
            Dictionary of macro features
        """
        features = {}
        
        if len(m15_subset) < 2:
            return self._empty_macro_features()
        
        # Long-term momentum
        features['m15_momentum_4h'] = m15_subset['close'].iloc[-1] - m15_subset['close'].iloc[-17] if len(m15_subset) >= 17 else 0  # 4 hours
        features['m15_momentum_8h'] = m15_subset['close'].iloc[-1] - m15_subset['close'].iloc[-33] if len(m15_subset) >= 33 else 0   # 8 hours
        features['m15_momentum_1d'] = m15_subset['close'].iloc[-1] - m15_subset['close'].iloc[-97] if len(m15_subset) >= 97 else 0   # 1 day
        
        # Trend strength
        if len(m15_subset) >= 16:
            # Linear regression slope over last 4 hours
            x = np.arange(len(m15_subset.tail(16)))
            y = m15_subset['close'].tail(16).values
            features['m15_trend_slope'] = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
        else:
            features['m15_trend_slope'] = 0
        
        # Volatility regime
        m15_returns = m15_subset['close'].pct_change().dropna()
        if len(m15_returns) >= 16:
            rolling_vol = m15_returns.rolling(16).std()
            features['m15_vol_regime'] = rolling_vol.iloc[-1] / rolling_vol.mean() if rolling_vol.mean() > 0 else 1
        else:
            features['m15_vol_regime'] = 1
        
        # Range analysis
        ranges = m15_subset['high'] - m15_subset['low']
        features['m15_avg_range'] = ranges.mean()
        features['m15_range_volatility'] = ranges.std()
        features['m15_current_range_percentile'] = (ranges.iloc[-1] - ranges.mean()) / ranges.std() if ranges.std() > 0 else 0
        
        # Moving average relationships (if available)
        features['m15_price_vs_sma20'] = 0  # Will be calculated later if needed
        features['m15_price_vs_sma50'] = 0
        features['m15_sma20_vs_sma50'] = 0
        
        return features
    
    def create_cross_timeframe_features(self, m5_subset: pd.DataFrame, m15_subset: pd.DataFrame) -> Dict:
        """
        Create features that combine M5 and M15 analysis.
        
        Args:
            m5_subset: M5 data subset
            m15_subset: M15 data subset
            
        Returns:
            Dictionary of cross-timeframe features
        """
        features = {}
        
        if len(m5_subset) < 3 or len(m15_subset) < 2:
            return self._empty_cross_timeframe_features()
        
        # Timeframe momentum confluence
        m5_short_momentum = m5_subset['close'].iloc[-1] - m5_subset['close'].iloc[-4] if len(m5_subset) >= 4 else 0
        m15_momentum = m15_subset['close'].iloc[-1] - m15_subset['close'].iloc[-2] if len(m15_subset) >= 2 else 0
        
        features['momentum_confluence'] = 1 if (m5_short_momentum > 0 and m15_momentum > 0) or (m5_short_momentum < 0 and m15_momentum < 0) else 0
        features['momentum_divergence'] = 1 if (m5_short_momentum > 0 and m15_momentum < 0) or (m5_short_momentum < 0 and m15_momentum > 0) else 0
        
        # Volatility alignment
        m5_vol = m5_subset['close'].pct_change().std() if len(m5_subset) > 1 else 0
        m15_vol = m15_subset['close'].pct_change().std() if len(m15_subset) > 1 else 0
        
        features['vol_m5_vs_m15'] = m5_vol / m15_vol if m15_vol > 0 else 1
        features['vol_regime_alignment'] = 1 if features['vol_m5_vs_m15'] > 0.8 and features['vol_m5_vs_m15'] < 1.2 else 0
        
        # Price level confluence
        m5_current = m5_subset['close'].iloc[-1]
        m15_current = m15_subset['close'].iloc[-1]
        
        features['price_level_diff'] = abs(m5_current - m15_current) / m15_current if m15_current > 0 else 0
        features['price_level_alignment'] = 1 if features['price_level_diff'] < 0.001 else 0
        
        # Trend strength comparison
        m5_trend_strength = abs(m5_short_momentum) / m5_subset['close'].iloc[-1] if m5_subset['close'].iloc[-1] > 0 else 0
        m15_trend_strength = abs(m15_momentum) / m15_subset['close'].iloc[-1] if m15_subset['close'].iloc[-1] > 0 else 0
        
        features['trend_strength_ratio'] = m5_trend_strength / m15_trend_strength if m15_trend_strength > 0 else 1
        
        return features
    
    def create_session_features(self, timestamp: pd.Timestamp, m5_subset: pd.DataFrame) -> Dict:
        """
        Create session-aware features.
        
        Args:
            timestamp: Current timestamp
            m5_subset: M5 data subset
            
        Returns:
            Dictionary of session features
        """
        features = {}
        hour = timestamp.hour
        
        # Current session
        features['session_asia'] = 1 if 0 <= hour < 8 else 0
        features['session_london'] = 1 if 8 <= hour < 16 else 0
        features['session_newyork'] = 1 if 13 <= hour < 21 else 0
        features['session_overlap'] = 1 if 13 <= hour < 16 else 0
        
        # Session transitions (high volatility periods)
        features['session_transition'] = 1 if hour in [0, 8, 13, 16, 21] else 0
        
        # Session-specific volatility
        if len(m5_subset) >= 6:
            recent_volatility = m5_subset['close'].pct_change().tail(6).std()
            
            # Historical session volatility (approximation)
            session_vol_multipliers = {
                'asia': 0.8,
                'london': 1.2,
                'newyork': 1.3,
                'overlap': 1.5
            }
            
            if features['session_asia']:
                expected_vol = recent_volatility * session_vol_multipliers['asia']
            elif features['session_overlap']:
                expected_vol = recent_volatility * session_vol_multipliers['overlap']
            elif features['session_london']:
                expected_vol = recent_volatility * session_vol_multipliers['london']
            elif features['session_newyork']:
                expected_vol = recent_volatility * session_vol_multipliers['newyork']
            else:
                expected_vol = recent_volatility
            
            features['session_vol_anomaly'] = recent_volatility / expected_vol if expected_vol > 0 else 1
        else:
            features['session_vol_anomaly'] = 1
        
        # Time of day effects
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week effects
        day_of_week = timestamp.dayofweek
        features['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        features['is_weekend'] = 1 if day_of_week >= 5 else 0
        
        # Month effects
        month = timestamp.month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        return features
    
    def _empty_microstructure_features(self) -> Dict:
        """Return empty microstructure features."""
        return {
            'm5_price_momentum_3': 0, 'm5_price_momentum_6': 0, 'm5_price_momentum_12': 0,
            'm5_price_acceleration': 0, 'm5_volatility_3': 0, 'm5_volatility_6': 0,
            'm5_volatility_12': 0, 'm5_max_move_3': 0, 'm5_avg_range_3': 0,
            'm5_avg_range_6': 0, 'm5_avg_range_12': 0, 'm5_range_expansion': 1,
            'm5_trend_consistency': 0.5, 'm5_volume_trend': 0, 'm5_volume_avg': 0,
            'm5_spread_avg': 0, 'm5_spread_volatility': 0, 'm5_spread_trend': 0,
            'm5_resistance_distance': 0, 'm5_support_distance': 0
        }
    
    def _empty_macro_features(self) -> Dict:
        """Return empty macro features."""
        return {
            'm15_momentum_4h': 0, 'm15_momentum_8h': 0, 'm15_momentum_1d': 0,
            'm15_trend_slope': 0, 'm15_vol_regime': 1, 'm15_avg_range': 0,
            'm15_range_volatility': 0, 'm15_current_range_percentile': 0,
            'm15_price_vs_sma20': 0, 'm15_price_vs_sma50': 0, 'm15_sma20_vs_sma50': 0
        }
    
    def _empty_cross_timeframe_features(self) -> Dict:
        """Return empty cross-timeframe features."""
        return {
            'momentum_confluence': 0, 'momentum_divergence': 0,
            'vol_m5_vs_m15': 1, 'vol_regime_alignment': 0,
            'price_level_diff': 0, 'price_level_alignment': 0,
            'trend_strength_ratio': 1
        }
    
    def create_multi_timeframe_features_3years(self, m5_data: pd.DataFrame, m15_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive multi-timeframe features for 3-year dataset.
        
        Args:
            m5_data: M5 OHLCV data (461K+ rows)
            m15_data: M15 OHLCV data (162K+ rows)
            
        Returns:
            DataFrame with multi-timeframe features
        """
        logger.info("Creating comprehensive multi-timeframe features for 3-year dataset...")
        
        features_list = []
        m5_lookback = self.config['lookback_periods']['m5_bars']
        m15_lookback = self.config['lookback_periods']['m15_bars']
        chunk_size = self.config['memory_optimization']['chunk_size']
        
        # Get M15 timestamps for alignment
        m15_timestamps = m15_data['timestamp'].values
        total_timestamps = len(m15_timestamps)
        
        logger.info(f"Processing {total_timestamps:,} M15 timestamps...")
        
        # Process in chunks for memory efficiency
        for chunk_start in range(0, total_timestamps, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_timestamps)
            chunk_timestamps = m15_timestamps[chunk_start:chunk_end]
            
            logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_timestamps-1)//chunk_size + 1}: {chunk_start:,} to {chunk_end:,}")
            
            for i, m15_timestamp in enumerate(chunk_timestamps):
                # Get M5 data up to (but not including) the M15 timestamp
                m5_mask = m5_data['timestamp'] < m15_timestamp
                m5_available = m5_data[m5_mask]
                
                # Get recent M5 data
                m5_subset = m5_available.tail(m5_lookback) if len(m5_available) >= 3 else m5_available
                
                # Get recent M15 data
                global_idx = chunk_start + i
                m15_subset = m15_data.iloc[:global_idx+1].tail(m15_lookback) if global_idx >= 0 else m15_data.iloc[:1]
                
                if len(m5_subset) < 3:  # Need minimum M5 data
                    continue
                
                # Create feature dictionary
                features = {'timestamp': m15_timestamp}
                
                # Add M5 microstructure features
                features.update(self.create_m5_microstructure_features(m5_subset))
                
                # Add M15 macro features
                features.update(self.create_m15_macro_features(m15_subset))
                
                # Add cross-timeframe features
                features.update(self.create_cross_timeframe_features(m5_subset, m15_subset))
                
                # Add session features
                features.update(self.create_session_features(pd.to_datetime(m15_timestamp), m5_subset))
                
                # Add latest technical indicators from M5
                if len(m5_subset) > 0:
                    latest_m5 = m5_subset.iloc[-1]
                    technical_features = [
                        'open', 'high', 'low', 'close', 'volume', 'spread'
                    ]
                    
                    for feature in technical_features:
                        if feature in latest_m5:
                            features[f'current_{feature}'] = latest_m5[feature]
                
                features_list.append(features)
            
            # Memory cleanup after each chunk
            if self.config['memory_optimization']['cleanup_memory']:
                gc.collect()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        logger.info(f"Created multi-timeframe features: {features_df.shape}")
        logger.info(f"Feature columns: {len([col for col in features_df.columns if col != 'timestamp' and not col.startswith('target_')])}")
        
        return features_df
    
    def save_features(self, features_df: pd.DataFrame, output_dir: str = "data/processed") -> str:
        """
        Save multi-timeframe features to file.
        
        Args:
            features_df: Features DataFrame
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(output_dir) / f"multi_timeframe_features_3years_{timestamp}.csv"
        
        features_df.to_csv(output_file, index=False)
        logger.info(f"Multi-timeframe features saved to: {output_file}")
        
        return str(output_file)


if __name__ == "__main__":
    # Test the multi-timeframe feature engineer for 3-year dataset
    engineer = MultiTimeframeFeatureEngineer3Years()
    
    # Load combined data
    m5_file = "data/combined/XAUUSD_M5_combined_2021_2025_20250823_101029.csv"
    m15_file = "data/combined/XAUUSD_M15_combined_2021_2025_20250823_101029.csv"
    
    try:
        m5_data, m15_data = engineer.load_combined_data(m5_file, m15_file)
        features_df = engineer.create_multi_timeframe_features_3years(m5_data, m15_data)
        output_file = engineer.save_features(features_df)
        
        print(f"✅ Multi-timeframe features for 3-year dataset created successfully!")
        print(f"📊 Features shape: {features_df.shape}")
        print(f"📁 Saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Feature creation failed: {e}")
        raise
