import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BinanceDataFetcher:
    def __init__(self, base_url: str = "https://api.binance.com"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def fetch_klines(self, symbol: str, interval: str, start_time: int, end_time: int, 
                    limit: int = 1000) -> List[List]:
        """Fetch klines from Binance API with retry logic."""
        url = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
                    
    def download_historical_data(self, symbol: str, interval: str, years: int = 3) -> pd.DataFrame:
        """Download historical klines data for specified years."""
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=years * 365)
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        all_klines = []
        current_start = start_ms
        
        logger.info(f"Downloading {symbol} {interval} data from {start_time} to {end_time}")
        
        while current_start < end_ms:
            klines = self.fetch_klines(symbol, interval, current_start, end_ms)
            if not klines:
                break
                
            all_klines.extend(klines)
            current_start = klines[-1][6] + 1  # Next start time
            
            time.sleep(0.1)  # Rate limiting
            
            if len(all_klines) % 10000 == 0:
                logger.info(f"Downloaded {len(all_klines)} records...")
                
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Keep only necessary columns
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Convert to proper types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # Set index and sort
        df.set_index('open_time', inplace=True)
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        logger.info(f"Downloaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
        return df
