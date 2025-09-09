import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config: Dict):
        self.config = config
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for the dataframe."""
        data = df.copy()
        
        # Price & volume features
        data['volume'] = data['volume']
        
        # Returns
        data['return_1'] = data['close'].pct_change()
        data['log_return_1'] = np.log(data['close'] / data['close'].shift(1))
        
        # Volatility
        vol_window = self.config.get('volatility_window', 48)
        data['volatility'] = data['return_1'].rolling(vol_window).std()
        
        # EMAs
        ema_periods = self.config.get('ema_periods', [20, 50, 200])
        for period in ema_periods:
            data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            data[f'price_ema_{period}_ratio'] = data['close'] / data[f'ema_{period}']
            
        # RSI
        rsi_period = self.config.get('rsi_period', 14)
        data['rsi'] = self._compute_rsi(data['close'], rsi_period)
        
        # MACD
        macd_fast, macd_slow, macd_signal = self.config.get('macd_params', [12, 26, 9])
        macd_line, macd_signal_line, macd_hist = self._compute_macd(
            data['close'], macd_fast, macd_slow, macd_signal
        )
        data['macd_line'] = macd_line
        data['macd_signal'] = macd_signal_line
        data['macd_histogram'] = macd_hist
        
        # ATR
        atr_period = self.config.get('atr_period', 14)
        data['atr'] = self._compute_atr(data, atr_period)
        
        # Spread
        data['spread'] = (data['high'] - data['low']) / data['close']
        
        # Volume pct change
        data['volume_pct_change'] = data['volume'].pct_change()
        
        # Keep only feature columns
        feature_cols = [col for col in data.columns if col in [
            'open', 'high', 'low', 'close', 'volume', 'return_1', 'log_return_1',
            'volatility', 'rsi', 'macd_line', 'macd_signal', 'macd_histogram',
            'atr', 'spread', 'volume_pct_change'
        ] + [f'ema_{p}' for p in ema_periods] + [f'price_ema_{p}_ratio' for p in ema_periods]]
        
        result = data[feature_cols].copy()
        
        # Clean data: replace infinities and handle outliers
        result = self._clean_data(result)
        
        # Drop NaNs
        result.dropna(inplace=True)
        
        logger.info(f"Generated {len(feature_cols)} features, {len(result)} valid rows")
        return result
        
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling infinities and extreme outliers."""
        cleaned = df.copy()
        
        # Replace infinities with NaN
        cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Cap extreme outliers at 99.9th percentile
        for col in cleaned.columns:
            if cleaned[col].dtype in ['float64', 'float32']:
                q_low = cleaned[col].quantile(0.001)
                q_high = cleaned[col].quantile(0.999)
                cleaned[col] = cleaned[col].clip(lower=q_low, upper=q_high)
        
        return cleaned
        
    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _compute_macd(self, prices: pd.Series, fast: int, slow: int, signal: int):
        """Compute MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
        
    def _compute_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Compute ATR indicator."""
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        return true_range.rolling(window=period).mean()
