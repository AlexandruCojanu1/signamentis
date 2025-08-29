"""
SignaMentis Essential Features Module

This module creates essential features for the Random Forest baseline:
- ATR, SuperTrend, SMA/EMA, RSI
- Time-of-day, session, spread features
- M15 aligned with M5 data

Author: SignaMentis Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EssentialFeatureEngineer:
    """
    Essential feature engineering for Random Forest baseline.
    
    Creates core technical indicators and time-based features
    that are essential for XAU/USD price prediction.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the essential feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Technical indicator parameters
        self.atr_period = self.config.get('atr_period', 14)
        self.supertrend_period = self.config.get('supertrend_period', 10)
        self.supertrend_multiplier = self.config.get('supertrend_multiplier', 3.0)
        self.sma_periods = self.config.get('sma_periods', [20, 50, 200])
        self.ema_periods = self.config.get('ema_periods', [12, 26])
        self.rsi_period = self.config.get('rsi_period', 14)
        
        # Feature alignment
        self.align_m15_on_m5 = self.config.get('align_m15_on_m5', True)
        
        logger.info("EssentialFeatureEngineer initialized")
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            ATR series
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Calculate SuperTrend indicator.
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period for SuperTrend
            multiplier: ATR multiplier
            
        Returns:
            DataFrame with SuperTrend columns
        """
        # Calculate ATR
        atr = self.calculate_atr(df, period)
        
        # Basic upper and lower bands
        hl2 = (df['high'] + df['low']) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        # Final upper and lower bands
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        
        # SuperTrend calculation
        supertrend = pd.Series(index=df.index, dtype=float)
        supertrend.iloc[0] = final_upper.iloc[0]
        
        for i in range(1, len(df)):
            # Upper band
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or df['close'].iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
            
            # Lower band
            if basic_lower.iloc[i] > final_lower.iloc[i-1] or df['close'].iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
            
            # SuperTrend
            if supertrend.iloc[i-1] == final_upper.iloc[i-1] and df['close'].iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
            elif supertrend.iloc[i-1] == final_upper.iloc[i-1] and df['close'].iloc[i] > final_upper.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
            elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and df['close'].iloc[i] >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
            elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and df['close'].iloc[i] < final_lower.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
        
        # SuperTrend signal (1 for uptrend, -1 for downtrend)
        supertrend_signal = np.where(df['close'] > supertrend, 1, -1)
        
        # Fill NaN values with forward fill
        supertrend = supertrend.fillna(method='ffill')
        final_upper = final_upper.fillna(method='ffill')
        final_lower = final_lower.fillna(method='ffill')
        
        # If still NaN, fill with close price
        supertrend = supertrend.fillna(df['close'])
        final_upper = final_upper.fillna(df['close'])
        final_lower = final_lower.fillna(df['close'])
        
        return pd.DataFrame({
            'supertrend': supertrend,
            'supertrend_signal': supertrend_signal,
            'supertrend_upper': final_upper,
            'supertrend_lower': final_lower
        })
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Simple and Exponential Moving Averages.
        
        Args:
            df: DataFrame with close prices
            
        Returns:
            DataFrame with MA columns
        """
        df_ma = df.copy()
        
        # Simple Moving Averages
        for period in self.sma_periods:
            df_ma[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in self.ema_periods:
            df_ma[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # MA crossovers
        if 20 in self.sma_periods and 50 in self.sma_periods:
            df_ma['sma_20_50_cross'] = np.where(
                df_ma['sma_20'] > df_ma['sma_50'], 1, -1
            )
        
        if 12 in self.ema_periods and 26 in self.ema_periods:
            df_ma['ema_12_26_cross'] = np.where(
                df_ma['ema_12'] > df_ma['ema_26'], 1, -1
            )
        
        return df_ma
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            df: DataFrame with close prices
            period: RSI period
            
        Returns:
            RSI series
        """
        close = df['close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with price features
        """
        df_price = df.copy()
        
        # Price changes
        df_price['price_change'] = df['close'] - df['open']
        df_price['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        
        # High-Low range
        df_price['hl_range'] = df['high'] - df['low']
        df_price['hl_range_pct'] = (df['high'] - df['low']) / df['low'] * 100
        
        # Body size
        df_price['body_size'] = np.abs(df['close'] - df['open'])
        df_price['body_size_pct'] = np.abs(df['close'] - df['open']) / df['open'] * 100
        
        # Upper and lower shadows
        df_price['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df_price['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Volatility (rolling standard deviation)
        df_price['volatility_20'] = df['close'].rolling(window=20).std()
        df_price['volatility_50'] = df['close'].rolling(window=50).std()
        
        # Price momentum
        df_price['momentum_5'] = df['close'] - df['close'].shift(5)
        df_price['momentum_10'] = df['close'] - df['close'].shift(10)
        
        return df_price
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume features
        """
        df_vol = df.copy()
        
        # Volume moving averages
        df_vol['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df_vol['volume_sma_50'] = df['volume'].rolling(window=50).mean()
        
        # Volume ratio
        df_vol['volume_ratio'] = df['volume'] / df_vol['volume_sma_20']
        
        # Volume momentum
        df_vol['volume_momentum'] = df['volume'] - df['volume'].shift(1)
        
        # High volume indicator
        df_vol['high_volume'] = (df['volume'] > df_vol['volume_sma_20'] * 1.5).astype(int)
        
        return df_vol
    
    def calculate_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate spread-based features.
        
        Args:
            df: DataFrame with spread data
            
        Returns:
            DataFrame with spread features
        """
        df_spread = df.copy()
        
        # Spread moving averages
        df_spread['spread_sma_20'] = df['spread'].rolling(window=20).mean()
        df_spread['spread_sma_50'] = df['spread'].rolling(window=50).mean()
        
        # Spread ratio
        df_spread['spread_ratio'] = df['spread'] / df_spread['spread_sma_20']
        
        # High spread indicator
        df_spread['high_spread'] = (df['spread'] > df_spread['spread_sma_20'] * 1.5).astype(int)
        
        # Spread momentum
        df_spread['spread_momentum'] = df['spread'] - df['spread'].shift(1)
        
        return df_spread
    
    def align_m15_with_m5(self, m5_df: pd.DataFrame, m15_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align M15 features with M5 data at M15 bar close.
        
        Args:
            m5_df: M5 DataFrame
            m15_df: M15 DataFrame
            
        Returns:
            M5 DataFrame with M15 features aligned
        """
        logger.info("Aligning M15 features with M5 data")
        
        # Ensure timestamps are sorted
        m5_df = m5_df.sort_values('timestamp').reset_index(drop=True)
        m15_df = m15_df.sort_values('timestamp').reset_index(drop=True)
        
        # Create M15 timestamp mapping
        m15_timestamps = m15_df['timestamp'].tolist()
        
        # Initialize M15 feature columns in M5 DataFrame
        m5_aligned = m5_df.copy()
        
        # Add M15 features
        m15_features = ['supertrend', 'supertrend_signal', 'sma_200', 'ema_26', 'rsi_14']
        
        for feature in m15_features:
            if feature in m15_df.columns:
                m5_aligned[f'm15_{feature}'] = np.nan
        
        # Align features at M15 bar close
        for i, m5_row in m5_aligned.iterrows():
            m5_timestamp = m5_row['timestamp']
            
            # Find the most recent M15 bar that has closed
            m15_bars = m15_df[m15_df['timestamp'] <= m5_timestamp]
            
            if not m15_bars.empty:
                latest_m15 = m15_bars.iloc[-1]
                
                # Fill M15 features
                for feature in m15_features:
                    if feature in m15_df.columns:
                        m5_aligned.loc[i, f'm15_{feature}'] = latest_m15[feature]
        
        # Forward fill M15 features to avoid NaN gaps
        m15_feature_cols = [col for col in m5_aligned.columns if col.startswith('m15_')]
        m5_aligned[m15_feature_cols] = m5_aligned[m15_feature_cols].fillna(method='ffill')
        
        logger.info(f"M15 features aligned: {len(m15_feature_cols)} features")
        return m5_aligned
    
    def create_essential_features(self, 
                                m5_df: pd.DataFrame, 
                                m15_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create all essential features for Random Forest baseline.
        
        Args:
            m5_df: M5 DataFrame with OHLCV data
            m15_df: M15 DataFrame (optional, will derive if not provided)
            
        Returns:
            DataFrame with all essential features
        """
        logger.info("Creating essential features for Random Forest baseline")
        
        # Start with M5 data
        df_features = m5_df.copy()
        
        # 1. Technical indicators
        logger.info("Calculating technical indicators...")
        
        # ATR
        df_features['atr'] = self.calculate_atr(df_features, self.atr_period)
        logger.info(f"ATR calculated, NaN count: {df_features['atr'].isna().sum()}")
        
        # SuperTrend
        supertrend_data = self.calculate_supertrend(
            df_features, self.supertrend_period, self.supertrend_multiplier
        )
        df_features = pd.concat([df_features, supertrend_data], axis=1)
        logger.info(f"SuperTrend calculated, NaN count: {df_features['supertrend'].isna().sum()}")
        
        # Moving Averages
        df_features = self.calculate_moving_averages(df_features)
        logger.info(f"Moving averages calculated, NaN count: {df_features[['sma_20', 'sma_50', 'sma_200']].isna().sum().sum()}")
        
        # RSI
        df_features['rsi'] = self.calculate_rsi(df_features, self.rsi_period)
        logger.info(f"RSI calculated, NaN count: {df_features['rsi'].isna().sum()}")
        
        # Check for NaN issues
        total_nan = df_features.isna().sum().sum()
        logger.info(f"Total NaN values after technical indicators: {total_nan}")
        
        # 2. Price features
        logger.info("Calculating price features...")
        df_features = self.calculate_price_features(df_features)
        logger.info(f"Price features calculated, NaN count: {df_features[['price_change', 'hl_range', 'body_size']].isna().sum().sum()}")
        
        # 3. Volume features
        logger.info("Calculating volume features...")
        df_features = self.calculate_volume_features(df_features)
        logger.info(f"Volume features calculated, NaN count: {df_features[['volume_sma_20', 'volume_sma_50']].isna().sum().sum()}")
        
        # 4. Spread features
        logger.info("Calculating spread features...")
        df_features = self.calculate_spread_features(df_features)
        logger.info(f"Spread features calculated, NaN count: {df_features[['spread_sma_20', 'spread_sma_50']].isna().sum().sum()}")
        
        # Fill remaining NaN values
        logger.info("Filling remaining NaN values...")
        
        # Forward fill then backward fill
        df_features = df_features.fillna(method='ffill')
        df_features = df_features.fillna(method='bfill')
        
        # If still NaN, fill with 0 for numeric columns
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_columns] = df_features[numeric_columns].fillna(0)
        
        # Final NaN check
        total_nan = df_features.isna().sum().sum()
        logger.info(f"Total NaN values after filling: {total_nan}")
        
        # Check which columns still have NaN
        nan_columns = df_features.columns[df_features.isna().any()].tolist()
        if nan_columns:
            logger.warning(f"Columns with remaining NaN: {nan_columns}")
            for col in nan_columns:
                nan_count = df_features[col].isna().sum()
                logger.warning(f"  {col}: {nan_count} NaN values")
        
        # 5. M15 alignment if requested
        if self.align_m15_on_m5 and m15_df is not None:
            logger.info("Aligning M15 features...")
            df_features = self.align_m15_with_m5(df_features, m15_df)
        
        # 6. Create target variables
        logger.info("Creating target variables...")
        
        # Direction (1 for up, 0 for down)
        df_features['target_direction'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
        
        # Price target (next bar close)
        df_features['target_price'] = df_features['close'].shift(-1)
        
        # Confidence (based on price movement magnitude)
        price_changes = df_features['close'].pct_change().abs()
        df_features['target_confidence'] = np.clip(price_changes * 100, 0, 1)
        
        # Check NaN in target variables
        target_nan = df_features[['target_direction', 'target_price', 'target_confidence']].isna().sum()
        logger.info(f"NaN in target variables: {target_nan.to_dict()}")
        
        # Fill NaN in target variables (first row will have NaN due to shift)
        df_features['target_direction'] = df_features['target_direction'].fillna(0)
        df_features['target_price'] = df_features['target_price'].fillna(df_features['close'])
        df_features['target_confidence'] = df_features['target_confidence'].fillna(0)
        
        # Remove rows with NaN values (due to shifting)
        initial_rows = len(df_features)
        df_features = df_features.dropna()
        final_rows = len(df_features)
        
        logger.info(f"Target variables created: {initial_rows} -> {final_rows} rows after removing NaN")
        
        if final_rows == 0:
            logger.error("No valid rows after creating target variables!")
            logger.error("Check for issues in feature calculation")
            return df_features
        
        logger.info(f"Essential features created: {len(df_features.columns)} total columns")
        logger.info(f"Feature columns: {list(df_features.columns)}")
        
        return df_features
    
    def save_features(self, 
                     df_features: pd.DataFrame, 
                     output_dir: str = "data/processed") -> str:
        """
        Save features to processed directory.
        
        Args:
            df_features: DataFrame with features
            output_dir: Output directory path
            
        Returns:
            Path to saved file
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"XAUUSD_features_{timestamp}.parquet"
        filepath = Path(output_dir) / filename
        
        # Save features
        df_features.to_parquet(filepath, index=False)
        
        logger.info(f"Features saved to: {filepath}")
        return str(filepath)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from scripts.data_normalizer import DataNormalizer
    
    # Initialize
    normalizer = DataNormalizer()
    feature_engineer = EssentialFeatureEngineer()
    
    # Normalize data
    m5_clean, m15_clean, report = normalizer.normalize_data(
        m5_path="data/external/XAUUSD_Tickbar_5_BID_11.08.2023-11.08.2023.csv"
    )
    
    # Create features
    features_df = feature_engineer.create_essential_features(m5_clean, m15_clean)
    
    # Save features
    features_path = feature_engineer.save_features(features_df)
    
    print("Essential features created successfully!")
    print(f"Features shape: {features_df.shape}")
    print(f"Features saved to: {features_path}")
