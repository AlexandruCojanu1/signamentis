import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class LabelEngineer:
    def __init__(self, k: float = 0.25):
        self.k = k
        self.label_map = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}
        
    def create_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create tri-class labels for direction prediction."""
        data = df.copy()
        
        # Compute future close (next bar)
        data['future_close'] = data['close'].shift(-1)
        
        # Compute future return
        data['future_ret'] = (data['future_close'] - data['close']) / data['close']
        
        # Compute ATR threshold
        atr_period = 14
        if 'atr' not in data.columns:
            # Compute ATR if not already present
            high_low = data['high'] - data['low']
            high_close_prev = np.abs(data['high'] - data['close'].shift(1))
            low_close_prev = np.abs(data['low'] - data['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            data['atr'] = true_range.rolling(window=atr_period).mean()
            
        data['threshold'] = (data['atr'] / data['close']) * self.k
        
        # Assign labels
        conditions = [
            data['future_ret'] < -data['threshold'],  # DOWN
            (data['future_ret'] >= -data['threshold']) & (data['future_ret'] <= data['threshold']),  # SIDEWAYS
            data['future_ret'] > data['threshold']  # UP
        ]
        choices = [0, 1, 2]
        data['label'] = np.select(conditions, choices, default=1)
        
        # Remove rows with NaN labels (last row)
        valid_mask = ~data['label'].isna() & ~data['future_ret'].isna()
        data = data[valid_mask].copy()
        
        # Drop helper columns
        features = data.drop(['future_close', 'future_ret', 'threshold', 'label'], axis=1)
        labels = data['label'].astype(int)
        
        # Log class distribution
        class_counts = labels.value_counts().sort_index()
        logger.info(f"Label distribution: {dict(class_counts)}")
        for idx, count in class_counts.items():
            logger.info(f"  {self.label_map[idx]}: {count} ({count/len(labels)*100:.1f}%)")
            
        return features, labels
        
    def get_label_map(self) -> Dict[int, str]:
        """Return the label mapping."""
        return self.label_map.copy()
