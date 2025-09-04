#!/usr/bin/env python3
"""
Binance Public Data downloader for SOLUSDT 15m klines with volume.

Downloads monthly CSVs from Binance Vision and assembles a single CSV with
columns: timestamp, open, high, low, close, volume.

Reference: `https://github.com/binance/binance-public-data.git`
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import zipfile
import io
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"


def download_month(symbol: str, interval: str, year: int, month: int) -> pd.DataFrame:
    url = BASE_URL.format(symbol=symbol, interval=interval, year=year, month=month)
    logger.info(f"Downloading {url}")
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        logger.warning(f"Skip {year}-{month:02d}: HTTP {r.status_code}")
        return pd.DataFrame()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    if not names:
        return pd.DataFrame()
    with z.open(names[0]) as f:
        df = pd.read_csv(f, header=None)
    # Binance kline columns per docs
    df.columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ]
    # Convert to desired schema
    df_out = pd.DataFrame({
        'timestamp': pd.to_datetime(df['open_time'], unit='ms', utc=True),
        'open': df['open'].astype(float),
        'high': df['high'].astype(float),
        'low': df['low'].astype(float),
        'close': df['close'].astype(float),
        'volume': df['volume'].astype(float)
    })
    return df_out


def assemble(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    months = pd.period_range(start=start_dt, end=end_dt, freq='M')
    frames = []
    for p in months:
        df = download_month(symbol, interval, p.year, p.month)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    return data


def main():
    parser = argparse.ArgumentParser(description="Download SOLUSDT 15m klines from Binance Vision")
    parser.add_argument('--symbol', default='SOLUSDT')
    parser.add_argument('--interval', default='15m')
    parser.add_argument('--start', required=True, help='YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='YYYY-MM-DD')
    parser.add_argument('--out', default='data/processed/solusdt_15m.csv')
    args = parser.parse_args()

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    df = assemble(args.symbol, args.interval, args.start, args.end)
    if df.empty:
        logger.error("No data downloaded. Check date range or availability.")
        return
    df.to_csv(args.out, index=False)
    logger.info(f"Saved {len(df):,} rows to {args.out}")


if __name__ == '__main__':
    main()


