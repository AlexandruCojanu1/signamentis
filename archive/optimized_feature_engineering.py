#!/usr/bin/env python3
"""
SignaMentis Optimized Feature Engineering Module

This module creates comprehensive features for XAU/USD price prediction:
- Advanced technical indicators
- Price action features
- Time-based features
- Volatility features
- Market microstructure features
- Multiple target variables

Author: SignaMentis Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
import yaml
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedFeatureEngineer:
    """
    Optimized feature engineering for XAU/USD Random Forest baseline.
    
    Creates comprehensive features while preventing overfitting and data leakage.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the optimized feature engineer.
        
        Args:
            config_path: Path to feature configuration file
        """
        self.config = self._load_config(config_path)
        self.scaler = RobustScaler()
        self.feature_selector = None
        
        logger.info("OptimizedFeatureEngineer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load feature engineering configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'technical_indicators': {
                    'sma': {'periods': [10, 20, 50, 100, 200], 'enabled': True},
                    'ema': {'periods': [12, 26, 50], 'enabled': True},
                    'rsi': {'period': 14, 'enabled': True},
                    'atr': {'period': 14, 'enabled': True},
                    'supertrend': {'period': 10, 'multiplier': 3.0, 'enabled': True}
                },
                'price_features': {
                    'price_change': True,
                    'price_change_pct': True,
                    'high_low_range': True,
                    'body_size': True
                },
                'time_features': {
                    'hour_of_day': True,
                    'day_of_week': True,
                    'session_asia': True,
                    'session_london': True,
                    'session_newyork': True
                },
                'target_variables': {
                    'direction_1': True,
                    'direction_3': True,
                    'direction_5': True,
                    'price_1': True,
                    'price_3': True,
                    'price_5': True,
                    'return_1': True,
                    'return_3': True,
                    'return_5': True,
                    'volatility_1': True,
                    'volatility_3': True
                }
            }
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        logger.info("Calculating technical indicators...")
        
        # SMA
        if self.config['technical_indicators']['sma']['enabled']:
            for period in self.config['technical_indicators']['sma']['periods']:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # EMA
        if self.config['technical_indicators']['ema']['enabled']:
            for period in self.config['technical_indicators']['ema']['periods']:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        if self.config['technical_indicators']['rsi']['enabled']:
            df['rsi'] = self._calculate_rsi(df['close'], self.config['technical_indicators']['rsi']['period'])
        
        # ATR
        if self.config['technical_indicators']['atr']['enabled']:
            df['atr'] = self._calculate_atr(df, self.config['technical_indicators']['atr']['period'])
        
        # SuperTrend
        if self.config['technical_indicators']['supertrend']['enabled']:
            supertrend_data = self._calculate_supertrend(
                df, 
                self.config['technical_indicators']['supertrend']['period'],
                self.config['technical_indicators']['supertrend']['multiplier']
            )
            df = pd.concat([df, supertrend_data], axis=1)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        
        return df
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price action features."""
        logger.info("Calculating price features...")
        
        # Basic price features
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        df['high_low_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Advanced price features
        df['price_momentum'] = df['close'].diff(5)
        df['price_acceleration'] = df['price_momentum'].diff()
        df['price_reversal'] = ((df['close'] > df['open']) & (df['close'].shift() < df['open'].shift())).astype(int)
        
        # Candle patterns
        df['doji'] = (abs(df['close'] - df['open']) <= (df['high'] - df['low']) * 0.1).astype(int)
        df['hammer'] = ((df['lower_shadow'] > df['body_size'] * 2) & (df['upper_shadow'] < df['body_size'] * 0.5)).astype(int)
        df['shooting_star'] = ((df['upper_shadow'] > df['body_size'] * 2) & (df['lower_shadow'] < df['body_size'] * 0.5)).astype(int)
        
        # Support/Resistance levels
        df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance_1'] = 2 * df['pivot_point'] - df['low']
        df['support_1'] = 2 * df['pivot_point'] - df['high']
        
        return df
    
    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features."""
        logger.info("Calculating time features...")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Cyclical features (better for ML)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Trading sessions
        df['session_asia'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['session_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['session_newyork'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['session_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        
        # Market hours
        df['market_open'] = (df['hour'] == 0).astype(int)
        df['market_close'] = (df['hour'] == 23).astype(int)
        df['weekend_effect'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features."""
        logger.info("Calculating volatility features...")
        
        # Realized volatility
        returns = df['close'].pct_change()
        for period in [5, 10, 20]:
            df[f'realized_vol_{period}'] = returns.rolling(window=period).std() * np.sqrt(period * 24 * 12)  # Annualized
        
        # Volatility regimes
        vol_20 = df['realized_vol_20']
        df['volatility_regime'] = pd.cut(vol_20, bins=3, labels=['low', 'medium', 'high'])
        df['high_vol_period'] = (vol_20 > vol_20.quantile(0.8)).astype(int)
        df['low_vol_period'] = (vol_20 < vol_20.quantile(0.2)).astype(int)
        
        # Volatility clusters
        df['vol_cluster_5'] = (df['realized_vol_5'] > df['realized_vol_5'].rolling(20).mean()).astype(int)
        df['vol_cluster_10'] = (df['realized_vol_10'] > df['realized_vol_10'].rolling(20).mean()).astype(int)
        
        return df
    
    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features."""
        logger.info("Calculating microstructure features...")
        
        # Spread features
        df['spread_sma_20'] = df['spread'].rolling(20).mean()
        df['spread_volatility'] = df['spread'].rolling(20).std()
        df['spread_regime'] = pd.cut(df['spread'], bins=3, labels=['low', 'medium', 'high'])
        
        # Volume features (even though volume is 0, we can create proxy features)
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_sma_50'] = df['volume'].rolling(50).mean()
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction."""
        logger.info("Creating target variables...")
        
        # Direction targets
        if self.config['target_variables']['direction_1']:
            df['target_direction_1'] = (df['close'].shift(-1) > df['close']).astype(int)
        if self.config['target_variables']['direction_3']:
            df['target_direction_3'] = (df['close'].shift(-3) > df['close']).astype(int)
        if self.config['target_variables']['direction_5']:
            df['target_direction_5'] = (df['close'].shift(-5) > df['close']).astype(int)
        
        # Price targets
        if self.config['target_variables']['price_1']:
            df['target_price_1'] = df['close'].shift(-1)
        if self.config['target_variables']['price_3']:
            df['target_price_3'] = df['close'].shift(-3)
        if self.config['target_variables']['price_5']:
            df['target_price_5'] = df['close'].shift(-5)
        
        # Return targets
        if self.config['target_variables']['return_1']:
            df['target_return_1'] = df['close'].pct_change().shift(-1)
        if self.config['target_variables']['return_3']:
            df['target_return_3'] = df['close'].pct_change(3).shift(-3)
        if self.config['target_variables']['return_5']:
            df['target_return_5'] = df['close'].pct_change(5).shift(-5)
        
        # Volatility targets
        if self.config['target_variables']['volatility_1']:
            df['target_volatility_1'] = df['close'].pct_change().rolling(5).std().shift(-1)
        if self.config['target_variables']['volatility_3']:
            df['target_volatility_3'] = df['close'].pct_change().rolling(5).std().shift(-3)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently."""
        logger.info("Handling missing values...")
        
        # Forward fill for technical indicators
        technical_cols = [col for col in df.columns if any(indicator in col for indicator in ['sma_', 'ema_', 'rsi', 'atr', 'bb_', 'macd', 'stoch'])]
        df[technical_cols] = df[technical_cols].fillna(method='ffill')
        
        # Backward fill for remaining
        df = df.fillna(method='bfill')
        
        # Fill remaining with 0 for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Check final NaN count
        total_nan = df.isna().sum().sum()
        logger.info(f"Total NaN values after handling: {total_nan}")
        
        return df
    
    def feature_selection(self, df: pd.DataFrame, target_col: str = 'target_direction_1') -> pd.DataFrame:
        """Perform feature selection to prevent overfitting."""
        logger.info("Performing feature selection...")
        
        # Remove timestamp and target columns, keep only numeric features
        feature_cols = [col for col in df.columns if not col.startswith('target_') and col != 'timestamp' and df[col].dtype in ['int64', 'float64']]
        X = df[feature_cols]
        y = df[target_col].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        logger.info(f"Feature selection: {len(feature_cols)} numeric features")
        
        # Remove low variance features
        variance_selector = VarianceThreshold(threshold=0.01)
        X_var_selected = variance_selector.fit_transform(X)
        var_selected_features = X.columns[variance_selector.get_support()].tolist()
        logger.info(f"Variance selection: {len(feature_cols)} -> {len(var_selected_features)} features")
        
        # Select top k features based on F-statistic
        k = min(50, len(var_selected_features))  # Max 50 features
        k_best_selector = SelectKBest(score_func=f_classif, k=k)
        X_k_selected = k_best_selector.fit_transform(X[var_selected_features], y)
        k_selected_features = [var_selected_features[i] for i in k_best_selector.get_support(indices=True)]
        logger.info(f"K-best selection: {len(var_selected_features)} -> {len(k_selected_features)} features")
        
        # Store selector for later use
        self.feature_selector = k_best_selector
        
        # Return selected features
        selected_cols = ['timestamp'] + k_selected_features + [col for col in df.columns if col.startswith('target_')]
        return df[selected_cols]
    
    def create_features(self, input_file: str, output_dir: str = "data/processed") -> str:
        """Create all features for the dataset."""
        logger.info(f"Creating features from {input_file}")
        
        # Load data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records")
        
        # Calculate all feature categories
        df = self.calculate_technical_indicators(df)
        df = self.calculate_price_features(df)
        df = self.calculate_time_features(df)
        df = self.calculate_volatility_features(df)
        df = self.calculate_microstructure_features(df)
        df = self.create_target_variables(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature selection
        df = self.feature_selection(df)
        
        # Save features
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(output_dir) / f"optimized_features_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Features created: {len(df)} records -> {output_file}")
        logger.info(f"Final features: {len(df.columns)} columns")
        
        return str(output_file)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR indicator."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Calculate SuperTrend indicator."""
        atr = self._calculate_atr(df, period)
        hl2 = (df['high'] + df['low']) / 2
        
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        
        for i in range(1, len(df)):
            final_upper.iloc[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < final_upper.iloc[i-1] or df['close'].iloc[i-1] > final_upper.iloc[i-1] else final_upper.iloc[i-1]
            final_lower.iloc[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > final_lower.iloc[i-1] or df['close'].iloc[i-1] < final_lower.iloc[i-1] else final_lower.iloc[i-1]
        
        supertrend = pd.Series(index=df.index, dtype=float)
        supertrend.iloc[0] = final_upper.iloc[0]
        
        for i in range(1, len(df)):
            if supertrend.iloc[i-1] == final_upper.iloc[i-1] and df['close'].iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
            elif supertrend.iloc[i-1] == final_upper.iloc[i-1] and df['close'].iloc[i] > final_upper.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
            elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and df['close'].iloc[i] >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
            elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and df['close'].iloc[i] < final_lower.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
        
        supertrend_direction = (df['close'] > supertrend).astype(int)
        
        return pd.DataFrame({
            'supertrend': supertrend,
            'supertrend_direction': supertrend_direction
        })
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator."""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent


if __name__ == "__main__":
    # Example usage
    feature_engineer = OptimizedFeatureEngineer()
    
    # Create features
    output_file = feature_engineer.create_features(
        input_file="SignaMentis/data/clean/XAUUSD_M5_normalized_20250822_223746_cleaned_20250822_223746.csv",
        output_dir="SignaMentis/data/processed"
    )
    
    print(f"Features created successfully: {output_file}")
