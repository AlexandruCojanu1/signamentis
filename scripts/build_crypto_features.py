#!/usr/bin/env python3
"""
Build multi-timeframe features for SOLUSDT using downloaded Binance OHLCV.

Inputs:
  - data/processed/solusdt_5m.csv
  - data/processed/solusdt_15m.csv

Output:
  - data/processed/multi_timeframe_features_crypto_15m.csv
"""

import argparse
from pathlib import Path
import logging
import pandas as pd

from multi_timeframe_features import MultiTimeframeFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m5', default='data/processed/solusdt_5m.csv')
    parser.add_argument('--m15', default='data/processed/solusdt_15m.csv')
    parser.add_argument('--out', default='data/processed/multi_timeframe_features_crypto_15m.csv')
    args = parser.parse_args()

    eng = MultiTimeframeFeatureEngineer()
    m5_df, m15_df = eng.load_and_prepare_data(args.m5, args.m15)

    feats = eng.create_multi_timeframe_features(m5_df, m15_df)
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    feats.to_csv(args.out, index=False)
    logger.info(f"Saved features: {args.out} shape={feats.shape}")


if __name__ == '__main__':
    main()


