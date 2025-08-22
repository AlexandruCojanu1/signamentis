"""
SignaMentis Feature Engineering Module

This module creates comprehensive features for XAU/USD trading including:
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, SuperTrend)
- Market structure features (Support/Resistance, Fibonacci, Pivot Points)
- Time-based features (Session indicators, Holiday detection)
- Volatility features (GARCH, Regime detection, Clustering)
- Advanced features (Order blocks, Fair value gaps, Liquidity zones)

Author: SignaMentis Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    holidays = None
    HOLIDAYS_AVAILABLE = False
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for XAU/USD trading data.
    
    Creates technical indicators, market structure features, time features,
    volatility features, and advanced trading features.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary for feature engineering
        """
        self.config = config or {}
        self.features_created = []
        self.feature_stats = {}
        
        # Initialize holiday calendar for major markets
        self.holiday_calendars = {
            'US': holidays.US(),
            'UK': holidays.GB(),
            'EU': holidays.EU()
        }
        
        logger.info("FeatureEngineer initialized")
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all available features for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with all features added
        """
        logger.info("Creating all features...")
        
        # Make a copy to avoid modifying original
        df_features = df.copy()
        
        # 1. Technical Indicators
        df_features = self.create_technical_indicators(df_features)
        
        # 2. Market Structure Features
        df_features = self.create_market_structure_features(df_features)
        
        # 3. Time Features
        df_features = self.create_time_features(df_features)
        
        # 4. Volatility Features
        df_features = self.create_volatility_features(df_features)
        
        # 5. Advanced Features
        df_features = self.create_advanced_features(df_features)
        
        # 6. Clean and validate features
        df_features = self._clean_features(df_features)
        
        # 7. Calculate feature statistics
        self._calculate_feature_stats(df_features)
        
        logger.info(f"Created {len(self.features_created)} features")
        return df_features
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        logger.info("Creating technical indicators...")
        
        # Moving Averages
        df = self._add_moving_averages(df)
        
        # RSI
        df = self._add_rsi(df)
        
        # MACD
        df = self._add_macd(df)
        
        # Bollinger Bands
        df = self._add_bollinger_bands(df)
        
        # ATR (Average True Range)
        df = self._add_atr(df)
        
        # SuperTrend
        df = self._add_supertrend(df)
        
        # Stochastic
        df = self._add_stochastic(df)
        
        # Williams %R
        df = self._add_williams_r(df)
        
        # CCI (Commodity Channel Index)
        df = self._add_cci(df)
        
        # ADX (Average Directional Index)
        df = self._add_adx(df)
        
        # Parabolic SAR
        df = self._add_parabolic_sar(df)
        
        # Ichimoku Cloud
        df = self._add_ichimoku(df)
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving averages."""
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            # Simple Moving Average
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Exponential Moving Average
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # Weighted Moving Average
            weights = np.arange(1, period + 1)
            df[f'wma_{period}'] = df['close'].rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
            
            # Moving Average Crossovers
            if period > 20:
                df[f'sma_cross_{period}_20'] = df[f'sma_{period}'] - df['sma_20']
                df[f'ema_cross_{period}_20'] = df[f'ema_{period}'] - df['ema_20']
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI levels
        df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
        df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
        
        return df
    
    def _add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator."""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        
        df['macd_line'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd_line'].ewm(span=signal).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # MACD crossovers
        df['macd_bullish_cross'] = ((df['macd_line'] > df['macd_signal']) & 
                                   (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_bearish_cross'] = ((df['macd_line'] < df['macd_signal']) & 
                                   (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Add Bollinger Bands."""
        df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        
        df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (bb_std * std_dev)
        df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (bb_std * std_dev)
        
        # Bollinger Band width and %B
        df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
        df[f'bb_percent_b_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / df[f'bb_width_{period}']
        
        # Bollinger Band squeeze
        df[f'bb_squeeze_{period}'] = (df[f'bb_width_{period}'] < df[f'bb_width_{period}'].rolling(window=period).mean() * 0.5).astype(int)
        
        return df
    
    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add ATR (Average True Range) indicator."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        # ATR-based volatility
        df[f'atr_volatility_{period}'] = df[f'atr_{period}'] / df['close']
        
        return df
    
    def _add_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Add SuperTrend indicator."""
        atr = df[f'atr_{period}']
        
        # Basic upper and lower bands
        basic_upper = (df['high'] + df['low']) / 2 + (multiplier * atr)
        basic_lower = (df['high'] + df['low']) / 2 - (multiplier * atr)
        
        # Final upper and lower bands
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        
        for i in range(1, len(df)):
            final_upper.iloc[i] = (basic_upper.iloc[i] if basic_upper.iloc[i] < final_upper.iloc[i-1] or df['close'].iloc[i-1] > final_upper.iloc[i-1] 
                                  else final_upper.iloc[i-1])
            final_lower.iloc[i] = (basic_lower.iloc[i] if basic_lower.iloc[i] > final_lower.iloc[i-1] or df['close'].iloc[i-1] < final_lower.iloc[i-1] 
                                  else final_lower.iloc[i-1])
        
        df[f'supertrend_{period}'] = final_upper
        df[f'supertrend_direction_{period}'] = np.where(df['close'] > final_upper, 1, -1)
        
        # SuperTrend signals
        df[f'supertrend_bullish_{period}'] = ((df[f'supertrend_direction_{period}'] == 1) & 
                                             (df[f'supertrend_direction_{period}'].shift(1) == -1)).astype(int)
        df[f'supertrend_bearish_{period}'] = ((df[f'supertrend_direction_{period}'] == -1) & 
                                             (df[f'supertrend_direction_{period}'].shift(1) == 1)).astype(int)
        
        return df
    
    def _add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic oscillator."""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        df[f'stoch_k_{k_period}'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        df[f'stoch_d_{k_period}'] = df[f'stoch_k_{k_period}'].rolling(window=d_period).mean()
        
        # Stochastic levels
        df[f'stoch_overbought_{k_period}'] = (df[f'stoch_k_{k_period}'] > 80).astype(int)
        df[f'stoch_oversold_{k_period}'] = (df[f'stoch_k_{k_period}'] < 20).astype(int)
        
        return df
    
    def _add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R indicator."""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        df[f'williams_r_{period}'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        return df
    
    def _add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add CCI (Commodity Channel Index)."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return df
    
    def _add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add ADX (Average Directional Index)."""
        # Directional Movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / df[f'atr_{period}'])
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / df[f'atr_{period}'])
        
        df[f'adx_{period}'] = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(window=period).mean()
        df[f'plus_di_{period}'] = plus_di
        df[f'minus_di_{period}'] = minus_di
        
        return df
    
    def _add_parabolic_sar(self, df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.DataFrame:
        """Add Parabolic SAR indicator."""
        # Simplified Parabolic SAR implementation
        df['psar'] = df['close'].copy()
        df['psar_direction'] = 1  # 1 for long, -1 for short
        
        for i in range(1, len(df)):
            if df['psar_direction'].iloc[i-1] == 1:
                df['psar'].iloc[i] = min(df['psar'].iloc[i-1], df['low'].iloc[i-1])
                if df['close'].iloc[i] < df['psar'].iloc[i]:
                    df['psar_direction'].iloc[i] = -1
                    df['psar'].iloc[i] = df['high'].iloc[i-1]
            else:
                df['psar'].iloc[i] = max(df['psar'].iloc[i-1], df['high'].iloc[i-1])
                if df['close'].iloc[i] > df['psar'].iloc[i]:
                    df['psar_direction'].iloc[i] = 1
                    df['psar'].iloc[i] = df['low'].iloc[i-1]
        
        return df
    
    def _add_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku Cloud indicators."""
        # Tenkan-sen (Conversion Line)
        period9 = 9
        df['tenkan_sen'] = (df['high'].rolling(window=period9).max() + 
                           df['low'].rolling(window=period9).min()) / 2
        
        # Kijun-sen (Base Line)
        period26 = 26
        df['kijun_sen'] = (df['high'].rolling(window=period26).max() + 
                          df['low'].rolling(window=period26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(period26)
        
        # Senkou Span B (Leading Span B)
        period52 = 52
        df['senkou_span_b'] = ((df['high'].rolling(window=period52).max() + 
                               df['low'].rolling(window=period52).min()) / 2).shift(period26)
        
        # Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-period26)
        
        return df
    
    def create_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market structure features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with market structure features
        """
        logger.info("Creating market structure features...")
        
        # Support and Resistance
        df = self._add_support_resistance(df)
        
        # Fibonacci Retracement
        df = self._add_fibonacci_levels(df)
        
        # Pivot Points
        df = self._add_pivot_points(df)
        
        # Order Blocks
        df = self._add_order_blocks(df)
        
        # Fair Value Gaps
        df = self._add_fair_value_gaps(df)
        
        # Liquidity Zones
        df = self._add_liquidity_zones(df)
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add support and resistance levels."""
        # Find local highs and lows
        df['local_high'] = df['high'].rolling(window=window, center=True).max()
        df['local_low'] = df['low'].rolling(window=window, center=True).min()
        
        # Support and resistance levels
        df['resistance_level'] = np.where(df['high'] == df['local_high'], df['high'], np.nan)
        df['support_level'] = np.where(df['low'] == df['local_low'], df['low'], np.nan)
        
        # Distance to nearest support/resistance
        df['distance_to_resistance'] = df['resistance_level'] - df['close']
        df['distance_to_support'] = df['close'] - df['support_level']
        
        return df
    
    def _add_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
        """Add Fibonacci retracement levels."""
        # Find swing high and low
        df['swing_high'] = df['high'].rolling(window=lookback).max()
        df['swing_low'] = df['low'].rolling(window=lookback).min()
        
        # Fibonacci levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for level in fib_levels:
            df[f'fib_{int(level*1000)}'] = df['swing_low'] + level * (df['swing_high'] - df['swing_low'])
        
        return df
    
    def _add_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pivot point levels."""
        # Daily pivot points (assuming daily data)
        df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        
        df['r1'] = 2 * df['pivot_point'] - df['low']
        df['s1'] = 2 * df['pivot_point'] - df['high']
        df['r2'] = df['pivot_point'] + (df['high'] - df['low'])
        df['s2'] = df['pivot_point'] - (df['high'] - df['low'])
        
        return df
    
    def _add_order_blocks(self, df: pd.DataFrame, min_size: float = 0.5) -> pd.DataFrame:
        """Add order block detection."""
        # Simplified order block detection
        df['candle_size'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        
        # Order block criteria
        df['order_block'] = ((df['candle_size'] > df['candle_size'].rolling(window=20).mean() * min_size) &
                            (df['body_size'] > df['body_size'].rolling(window=20).mean() * 0.8)).astype(int)
        
        return df
    
    def _add_fair_value_gaps(self, df: pd.DataFrame, min_gap: float = 0.1) -> pd.DataFrame:
        """Add fair value gap detection."""
        # Fair value gap: gap between previous high and current low (or vice versa)
        df['fvg_up'] = np.where(df['low'] > df['high'].shift(1), 1, 0)
        df['fvg_down'] = np.where(df['high'] < df['low'].shift(1), 1, 0)
        
        # Gap size
        df['fvg_size_up'] = np.where(df['fvg_up'] == 1, df['low'] - df['high'].shift(1), 0)
        df['fvg_size_down'] = np.where(df['fvg_down'] == 1, df['low'].shift(1) - df['high'], 0)
        
        return df
    
    def _add_liquidity_zones(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Add liquidity zone detection."""
        # High volume areas
        volume_ma = df['volume'].rolling(window=20).mean()
        df['high_volume_zone'] = (df['volume'] > volume_ma * threshold).astype(int)
        
        # Price levels with high volume
        df['liquidity_level'] = np.where(df['high_volume_zone'] == 1, df['close'], np.nan)
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            pd.DataFrame: DataFrame with time features
        """
        logger.info("Creating time features...")
        
        # Basic time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Session indicators
        df = self._add_session_indicators(df)
        
        # Holiday indicators
        df = self._add_holiday_indicators(df)
        
        # Time cycles
        df = self._add_time_cycles(df)
        
        return df
    
    def _add_session_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading session indicators."""
        # Major trading sessions (UTC times)
        df['asia_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        df['london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        df['ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
        df['overlap_session'] = ((df.index.hour >= 13) & (df.index.hour < 16)).astype(int)
        
        # Session strength (based on volume)
        df['session_volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        return df
    
    def _add_holiday_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday indicators for major markets."""
        # US holidays
        us_holidays = self.holiday_calendars['US']
        df['us_holiday'] = df.index.date.isin(us_holidays).astype(int)
        
        # UK holidays
        uk_holidays = self.holiday_calendars['UK']
        df['uk_holiday'] = df.index.date.isin(uk_holidays).astype(int)
        
        # EU holidays
        eu_holidays = self.holiday_calendars['EU']
        df['eu_holiday'] = df.index.date.isin(eu_holidays).astype(int)
        
        # Any holiday
        df['any_holiday'] = (df['us_holiday'] | df['uk_holiday'] | df['eu_holiday']).astype(int)
        
        return df
    
    def _add_time_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time cycle features."""
        # Day of week cycles
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Hour of day cycles
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Month cycles
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with volatility features
        """
        logger.info("Creating volatility features...")
        
        # Realized volatility
        df = self._add_realized_volatility(df)
        
        # GARCH volatility
        df = self._add_garch_volatility(df)
        
        # Volatility regime
        df = self._add_volatility_regime(df)
        
        # Volatility clustering
        df = self._add_volatility_clustering(df)
        
        return df
    
    def _add_realized_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add realized volatility measures."""
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Realized volatility (rolling standard deviation of returns)
        df[f'realized_vol_{window}'] = df['returns'].rolling(window=window).std()
        
        # Parkinson volatility (using high-low range)
        log_hl_ratio = np.log(df['high'] / df['low'])
        parkinson_vol = np.sqrt((1 / (4 * np.log(2))) * (log_hl_ratio ** 2).rolling(window=window).mean())
        df[f'parkinson_vol_{window}'] = parkinson_vol
        
        # Garman-Klass volatility
        log_hl_ratio_sq = (np.log(df['high'] / df['low'])) ** 2
        log_co_ratio_sq = (np.log(df['close'] / df['open'])) ** 2
        gk_vol = np.sqrt((0.5 * log_hl_ratio_sq - (2 * np.log(2) - 1) * log_co_ratio_sq).rolling(window=window).mean())
        df[f'garman_klass_vol_{window}'] = gk_vol
        
        return df
    
    def _add_garch_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add GARCH-based volatility estimation."""
        # Simplified GARCH(1,1) implementation
        returns = df['returns'].fillna(0)
        
        # Initialize GARCH parameters
        omega = 0.000001
        alpha = 0.1
        beta = 0.8
        
        # Initialize variance
        variance = returns.rolling(window=window).var().fillna(returns.var())
        
        # GARCH variance
        for i in range(1, len(df)):
            if i >= window:
                variance.iloc[i] = omega + alpha * returns.iloc[i-1]**2 + beta * variance.iloc[i-1]
        
        df[f'garch_vol_{window}'] = np.sqrt(variance)
        
        return df
    
    def _add_volatility_regime(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add volatility regime detection."""
        # Volatility regime based on ATR
        atr_ma = df[f'atr_20'].rolling(window=window).mean()
        atr_std = df[f'atr_20'].rolling(window=window).std()
        
        # Regime classification
        df['vol_regime'] = np.where(df[f'atr_20'] > atr_ma + atr_std, 'high',
                                   np.where(df[f'atr_20'] < atr_ma - atr_std, 'low', 'normal'))
        
        # Regime encoding
        regime_map = {'low': 0, 'normal': 1, 'high': 2}
        df['vol_regime_encoded'] = df['vol_regime'].map(regime_map)
        
        return df
    
    def _add_volatility_clustering(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add volatility clustering features."""
        # Volatility clustering (autocorrelation of squared returns)
        returns_squared = df['returns'] ** 2
        
        df[f'vol_clustering_{window}'] = returns_squared.rolling(window=window).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0
        )
        
        # Volatility persistence
        df[f'vol_persistence_{window}'] = returns_squared.rolling(window=window).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0
        )
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced trading features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with advanced features
        """
        logger.info("Creating advanced features...")
        
        # Price action patterns
        df = self._add_price_patterns(df)
        
        # Market microstructure
        df = self._add_market_microstructure(df)
        
        # Sentiment indicators
        df = self._add_sentiment_indicators(df)
        
        # Risk metrics
        df = self._add_risk_metrics(df)
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action patterns."""
        # Doji pattern
        body_size = abs(df['close'] - df['open'])
        wick_size = df['high'] - df['low']
        df['doji'] = (body_size < wick_size * 0.1).astype(int)
        
        # Hammer pattern
        body_mid = (df['open'] + df['close']) / 2
        lower_wick = np.minimum(df['open'], df['close']) - df['low']
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        df['hammer'] = ((lower_wick > body_size * 2) & (upper_wick < body_size * 0.5)).astype(int)
        
        # Shooting star pattern
        df['shooting_star'] = ((upper_wick > body_size * 2) & (lower_wick < body_size * 0.5)).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = ((df['open'] < df['close'].shift(1)) & 
                                  (df['close'] > df['open'].shift(1)) &
                                  (df['open'] < df['open'].shift(1)) &
                                  (df['close'] > df['close'].shift(1))).astype(int)
        
        df['bearish_engulfing'] = ((df['open'] > df['close'].shift(1)) & 
                                  (df['close'] < df['open'].shift(1)) &
                                  (df['open'] > df['open'].shift(1)) &
                                  (df['close'] < df['close'].shift(1))).astype(int)
        
        return df
    
    def _add_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Bid-ask spread approximation
        df['spread_approx'] = (df['high'] - df['low']) / df['close']
        
        # Volume-price relationship
        df['volume_price_trend'] = (df['volume'] * df['returns']).rolling(window=20).sum()
        
        # Money flow index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(window=14).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=14).sum()
        
        df['money_flow_index'] = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return df
    
    def _add_sentiment_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment indicators."""
        # Fear and Greed index approximation
        df['fear_greed'] = (df['rsi_14'] + df['stoch_k_14'] + (100 - df['williams_r_14'])) / 3
        
        # Market momentum
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        
        # Trend strength
        df['trend_strength'] = abs(df['ema_20'] - df['ema_50']) / df['ema_20']
        
        return df
    
    def _add_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk metrics."""
        # Value at Risk (VaR)
        returns = df['returns'].fillna(0)
        df['var_95'] = returns.rolling(window=252).quantile(0.05)
        
        # Expected Shortfall (Conditional VaR)
        df['expected_shortfall'] = returns.rolling(window=252).apply(
            lambda x: x[x <= x.quantile(0.05)].mean() if len(x) > 0 else 0
        )
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        df['drawdown'] = (cumulative_returns - rolling_max) / rolling_max
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features."""
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill some features
        forward_fill_cols = [col for col in df.columns if 'level' in col or 'zone' in col]
        df[forward_fill_cols] = df[forward_fill_cols].fillna(method='ffill')
        
        # Backward fill remaining NaNs
        df = df.fillna(method='bfill')
        
        # Remove rows with any remaining NaNs
        df = df.dropna()
        
        return df
    
    def _calculate_feature_stats(self, df: pd.DataFrame) -> None:
        """Calculate statistics for created features."""
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        for col in feature_cols:
            if col in df.columns:
                self.feature_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'null_count': df[col].isnull().sum()
                }
    
    def get_feature_summary(self) -> Dict:
        """Get summary of created features."""
        return {
            'total_features': len(self.features_created),
            'feature_list': self.features_created,
            'feature_stats': self.feature_stats
        }
    
    def select_features(self, df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """
        Select specific features from the dataset.
        
        Args:
            df: DataFrame with all features
            feature_list: List of feature names to select
            
        Returns:
            pd.DataFrame: DataFrame with selected features
        """
        # Always include OHLCV columns
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        available_features = [col for col in feature_list if col in df.columns]
        
        selected_cols = base_cols + available_features
        return df[selected_cols]


# Convenience function for quick feature engineering
def create_features(df: pd.DataFrame, features: str = "all") -> pd.DataFrame:
    """
    Convenience function to quickly create features.
    
    Args:
        df: DataFrame with OHLCV data
        features: Type of features to create ("all", "technical", "market_structure", "time", "volatility")
        
    Returns:
        pd.DataFrame: DataFrame with features added
    """
    engineer = FeatureEngineer()
    
    if features == "all":
        return engineer.create_all_features(df)
    elif features == "technical":
        return engineer.create_technical_indicators(df)
    elif features == "market_structure":
        return engineer.create_market_structure_features(df)
    elif features == "time":
        return engineer.create_time_features(df)
    elif features == "volatility":
        return engineer.create_volatility_features(df)
    else:
        raise ValueError(f"Unknown feature type: {features}")


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': 2000 + np.random.randn(1000).cumsum(),
        'high': 2000 + np.random.randn(1000).cumsum() + 5,
        'low': 2000 + np.random.randn(1000).cumsum() - 5,
        'close': 2000 + np.random.randn(1000).cumsum(),
        'volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    # Create features
    engineer = FeatureEngineer()
    df_with_features = engineer.create_all_features(sample_data)
    
    print(f"Original columns: {list(sample_data.columns)}")
    print(f"Features added: {len(df_with_features.columns) - len(sample_data.columns)}")
    print(f"Total columns: {len(df_with_features.columns)}")
    
    # Show feature summary
    summary = engineer.get_feature_summary()
    print(f"\nFeature Summary:")
    print(f"Total features created: {summary['total_features']}")
    print(f"Feature categories: {list(summary['feature_stats'].keys())[:10]}...")
