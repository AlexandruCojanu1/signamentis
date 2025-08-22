"""
SignaMentis Data Loader Module

This module handles loading, caching, and managing XAU/USD historical data
from various sources including CSV files, MetaTrader 5, and external APIs.

Author: SignaMentis Team
Version: 1.0.0
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import MetaTrader5 as mt5
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading operations."""
    symbol: str = "XAUUSD"
    timeframe: str = "M5"
    data_source: str = "csv"
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    start_date: str = "2004-01-01"
    end_date: str = "2025-12-31"
    update_frequency: str = "5m"
    cache_duration_hours: int = 24
    max_retries: int = 3
    retry_delay_seconds: int = 5


class DataLoader:
    """
    Comprehensive data loader for XAU/USD trading data.
    
    Supports multiple data sources:
    - CSV files (local storage)
    - MetaTrader 5 API
    - External APIs (Yahoo Finance, Alpha Vantage)
    - Real-time streaming data
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the data loader.
        
        Args:
            config: Data configuration object. If None, uses defaults.
        """
        self.config = config or DataConfig()
        self.cache = {}
        self.cache_timestamps = {}
        self.mt5_connected = False
        self._setup_directories()
        self._connect_mt5()
        
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.config.raw_data_path,
            self.config.processed_data_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def _connect_mt5(self) -> bool:
        """
        Connect to MetaTrader 5 terminal.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            if not mt5.initialize():
                logger.warning("Failed to initialize MT5")
                return False
            
            # Test connection
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning("Failed to get MT5 account info")
                return False
            
            self.mt5_connected = True
            logger.info(f"Connected to MT5: {account_info.login}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            self.mt5_connected = False
            return False
    
    def _get_mt5_timeframe(self, timeframe: str) -> int:
        """
        Convert string timeframe to MT5 timeframe constant.
        
        Args:
            timeframe: String representation of timeframe (e.g., 'M5', 'H1', 'D1')
            
        Returns:
            int: MT5 timeframe constant
        """
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        return timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
    
    def load_csv_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Load CSV with specific handling for new data format
            df = pd.read_csv(file_path, **kwargs)
            
            # Standardize column names for new data format
            column_mapping = {
                'Local time': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Convert timestamp to datetime (handle GMT+0300 format)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S.%f GMT%z', errors='coerce')
                
                # If the above format fails, try alternative parsing
                if df['timestamp'].isnull().all():
                    # Try parsing without timezone info
                    df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace(' GMT+0300', ''), 
                                                    format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                df.index = pd.to_datetime(df.index)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV")
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            # Remove rows with NaN values
            df = df.dropna()
            
            logger.info(f"Successfully loaded CSV data: {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV data from {file_path}: {e}")
            raise
    
    def load_mt5_data(self, 
                      symbol: str,
                      timeframe: str,
                      start_date: datetime,
                      end_date: datetime,
                      max_bars: int = 1000000) -> pd.DataFrame:
        """
        Load data from MetaTrader 5.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: Timeframe string (e.g., 'M5')
            start_date: Start date for data
            end_date: End date for data
            max_bars: Maximum number of bars to retrieve
            
        Returns:
            pd.DataFrame: OHLCV data from MT5
        """
        if not self.mt5_connected:
            raise ConnectionError("MT5 not connected. Please check connection.")
        
        try:
            mt5_timeframe = self._get_mt5_timeframe(timeframe)
            
            # Convert dates to MT5 format
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Get rates from MT5
            rates = mt5.copy_rates_range(
                symbol, 
                mt5_timeframe, 
                start_dt, 
                end_dt
            )
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data received from MT5 for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # Rename columns to standard format
            column_mapping = {
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
                'spread': 'spread',
                'real_volume': 'real_volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Select only OHLCV columns
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in ohlcv_columns if col in df.columns]
            df = df[available_columns]
            
            # Sort by datetime
            df.sort_index(inplace=True)
            
            logger.info(f"Successfully loaded MT5 data: {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading MT5 data: {e}")
            raise
    
    async def load_external_api_data(self, 
                                   symbol: str,
                                   start_date: str,
                                   end_date: str,
                                   api_key: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from external APIs (Yahoo Finance, Alpha Vantage, etc.).
        
        Args:
            symbol: Trading symbol
            start_date: Start date string
            end_date: End date string
            api_key: API key if required
            
        Returns:
            pd.DataFrame: OHLCV data from external API
        """
        try:
            # Try Yahoo Finance first (no API key required)
            df = await self._load_yahoo_finance_data(symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                return df
            
            # Fallback to Alpha Vantage if API key provided
            if api_key:
                df = await self._load_alpha_vantage_data(symbol, start_date, end_date, api_key)
                if df is not None and not df.empty:
                    return df
            
            logger.warning("No external API data available")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading external API data: {e}")
            return pd.DataFrame()
    
    async def _load_yahoo_finance_data(self, 
                                     symbol: str,
                                     start_date: str,
                                     end_date: str) -> Optional[pd.DataFrame]:
        """Load data from Yahoo Finance."""
        try:
            import yfinance as yf
            
            # Convert XAUUSD to GC=F for Yahoo Finance
            yahoo_symbol = "GC=F" if symbol.upper() == "XAUUSD" else symbol
            
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval="5m")
            
            if df.empty:
                return None
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                return None
            
            # Add volume column if missing
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            logger.info(f"Successfully loaded Yahoo Finance data: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Yahoo Finance data: {e}")
            return None
    
    async def _load_alpha_vantage_data(self, 
                                     symbol: str,
                                     start_date: str,
                                     end_date: str,
                                     api_key: str) -> Optional[pd.DataFrame]:
        """Load data from Alpha Vantage API."""
        try:
            # Alpha Vantage intraday data endpoint
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': '5min',
                'apikey': api_key,
                'outputsize': 'full'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if 'Time Series (5min)' not in data:
                        return None
                    
                    # Convert to DataFrame
                    time_series = data['Time Series (5min)']
                    df = pd.DataFrame.from_dict(time_series, orient='index')
                    
                    # Rename columns
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    
                    # Convert to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Filter by date range
                    df.index = pd.to_datetime(df.index)
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    df = df.loc[mask]
                    
                    # Sort by datetime
                    df.sort_index(inplace=True)
                    
                    logger.info(f"Successfully loaded Alpha Vantage data: {len(df)} rows")
                    return df
                    
        except Exception as e:
            logger.error(f"Error loading Alpha Vantage data: {e}")
            return None
    
    def get_data(self, 
                symbol: str = None,
                timeframe: str = None,
                start_date: Union[str, datetime] = None,
                end_date: Union[str, datetime] = None,
                use_cache: bool = True) -> pd.DataFrame:
        """
        Main method to get data from the best available source.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            pd.DataFrame: Requested data
        """
        # Use config defaults if not specified
        symbol = symbol or self.config.symbol
        timeframe = timeframe or self.config.timeframe
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        
        # Convert dates to datetime if strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{start_date.date()}_{end_date.date()}"
        if use_cache and cache_key in self.cache:
            cache_age = datetime.now() - self.cache_timestamps[cache_key]
            if cache_age.total_seconds() < self.config.cache_duration_hours * 3600:
                logger.info(f"Using cached data for {cache_key}")
                return self.cache[cache_key].copy()
        
        # Try different data sources in order of preference
        df = None
        
        # 1. Try MT5 first (if connected)
        if self.mt5_connected:
            try:
                df = self.load_mt5_data(symbol, timeframe, start_date, end_date)
                if not df.empty:
                    logger.info("Data loaded from MT5")
            except Exception as e:
                logger.warning(f"MT5 data loading failed: {e}")
        
        # 2. Try CSV files
        if df is None or df.empty:
            csv_files = self._find_csv_files(symbol, timeframe, start_date, end_date)
            if csv_files:
                try:
                    df = self._load_best_csv_file(csv_files, start_date, end_date)
                    if not df.empty:
                        logger.info("Data loaded from CSV")
                except Exception as e:
                    logger.warning(f"CSV data loading failed: {e}")
        
        # 3. Try external APIs as last resort
        if df is None or df.empty:
            try:
                df = asyncio.run(self.load_external_api_data(
                    symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                ))
                if not df.empty:
                    logger.info("Data loaded from external API")
            except Exception as e:
                logger.warning(f"External API data loading failed: {e}")
        
        if df is None or df.empty:
            raise ValueError(f"Could not load data for {symbol} from any source")
        
        # Cache the data
        if use_cache:
            self.cache[cache_key] = df.copy()
            self.cache_timestamps[cache_key] = datetime.now()
        
        return df
    
    def _find_csv_files(self, 
                       symbol: str,
                       timeframe: str,
                       start_date: datetime,
                       end_date: datetime) -> List[str]:
        """Find relevant CSV files in the raw data directory."""
        csv_files = []
        raw_path = Path(self.config.raw_data_path)
        
        # Look for CSV files that might contain the requested data
        for file_path in raw_path.glob("*.csv"):
            if symbol.lower() in file_path.name.lower():
                csv_files.append(str(file_path))
        
        return sorted(csv_files)
    
    def _load_best_csv_file(self, 
                           csv_files: List[str],
                           start_date: datetime,
                           end_date: datetime) -> pd.DataFrame:
        """Load the best matching CSV file based on date range."""
        best_file = None
        best_coverage = 0
        
        for csv_file in csv_files:
            try:
                df = self.load_csv_data(csv_file)
                if df.empty:
                    continue
                
                # Calculate coverage
                file_start = df.index.min()
                file_end = df.index.max()
                
                coverage_start = max(start_date, file_start)
                coverage_end = min(end_date, file_end)
                
                if coverage_start < coverage_end:
                    coverage = (coverage_end - coverage_start).total_seconds()
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_file = csv_file
                        
            except Exception as e:
                logger.warning(f"Error checking CSV file {csv_file}: {e}")
                continue
        
        if best_file:
            df = self.load_csv_data(best_file)
            # Filter to requested date range
            mask = (df.index >= start_date) & (df.index <= end_date)
            return df.loc[mask]
        
        return pd.DataFrame()
    
    def save_data(self, 
                 df: pd.DataFrame,
                 symbol: str,
                 timeframe: str,
                 file_format: str = "csv") -> str:
        """
        Save data to file.
        
        Args:
            df: DataFrame to save
            symbol: Trading symbol
            timeframe: Timeframe
            file_format: Output format (csv, parquet, hdf5)
            
        Returns:
            str: Path to saved file
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timeframe}_{timestamp}.{file_format}"
            file_path = os.path.join(self.config.processed_data_path, filename)
            
            if file_format.lower() == "csv":
                df.to_csv(file_path, index=True)
            elif file_format.lower() == "parquet":
                df.to_parquet(file_path, index=True)
            elif file_format.lower() == "hdf5":
                df.to_hdf(file_path, key='data', mode='w', index=True)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Data saved to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def get_latest_data(self, 
                       symbol: str = None,
                       timeframe: str = None,
                       bars: int = 1000) -> pd.DataFrame:
        """
        Get the most recent data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            bars: Number of bars to retrieve
            
        Returns:
            pd.DataFrame: Latest data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=bars * 5 / 1440)  # Approximate
        
        return self.get_data(symbol, timeframe, start_date, end_date)
    
    def cleanup_cache(self, max_age_hours: int = 24) -> None:
        """
        Clean up old cached data.
        
        Args:
            max_age_hours: Maximum age of cached data in hours
        """
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, timestamp in self.cache_timestamps.items():
            age = current_time - timestamp
            if age.total_seconds() > max_age_hours * 3600:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
            del self.cache_timestamps[key]
        
        logger.info(f"Cleaned up {len(keys_to_remove)} cached entries")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Get information about the loaded data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict: Data information
        """
        info = {
            'rows': len(df),
            'columns': list(df.columns),
            'date_range': {
                'start': df.index.min().isoformat() if not df.empty else None,
                'end': df.index.max().isoformat() if not df.empty else None
            },
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicates': df.index.duplicated().sum()
        }
        
        return info
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.mt5_connected:
            mt5.shutdown()
            logger.info("MT5 connection closed")


# Convenience function for quick data loading
def load_xau_data(start_date: str = "2024-01-01",
                  end_date: str = "2025-01-01",
                  timeframe: str = "M5") -> pd.DataFrame:
    """
    Convenience function to quickly load XAU/USD data.
    
    Args:
        start_date: Start date string
        end_date: End date string
        timeframe: Timeframe string
        
    Returns:
        pd.DataFrame: XAU/USD data
    """
    loader = DataLoader()
    return loader.get_data("XAUUSD", timeframe, start_date, end_date)


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load data for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        df = loader.get_data(start_date=start_date, end_date=end_date)
        print(f"Loaded {len(df)} bars of data")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        
        # Save processed data
        file_path = loader.save_data(df, "XAUUSD", "M5")
        print(f"Data saved to: {file_path}")
        
    except Exception as e:
        print(f"Error: {e}")
