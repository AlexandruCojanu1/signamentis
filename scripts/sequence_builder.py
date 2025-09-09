import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class SequenceBuilder:
    def __init__(self, seq_len: int = 128, val_ratio: float = 0.2):
        self.seq_len = seq_len
        self.val_ratio = val_ratio
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def prepare_sequences(self, features: pd.DataFrame, labels: pd.Series) -> Tuple:
        """Prepare sequences for training."""
        # Store feature columns
        self.feature_cols = list(features.columns)
        
        # Time-ordered split
        split_idx = int(len(features) * (1 - self.val_ratio))
        
        X_train_df = features.iloc[:split_idx]
        y_train = labels.iloc[:split_idx]
        X_val_df = features.iloc[split_idx:]
        y_val = labels.iloc[split_idx:]
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train_df)
        X_val_scaled = self.scaler.transform(X_val_df)
        
        # Build sequences
        X_train_seq, y_train_seq = self._build_sequences(X_train_scaled, y_train.values)
        X_val_seq, y_val_seq = self._build_sequences(X_val_scaled, y_val.values)
        
        logger.info(f"Training sequences: {X_train_seq.shape}, labels: {y_train_seq.shape}")
        logger.info(f"Validation sequences: {X_val_seq.shape}, labels: {y_val_seq.shape}")
        
        # Log class distribution for train/val
        train_unique, train_counts = np.unique(y_train_seq, return_counts=True)
        val_unique, val_counts = np.unique(y_val_seq, return_counts=True)
        
        logger.info("Training class distribution:")
        for cls, count in zip(train_unique, train_counts):
            logger.info(f"  Class {cls}: {count} ({count/len(y_train_seq)*100:.1f}%)")
            
        logger.info("Validation class distribution:")
        for cls, count in zip(val_unique, val_counts):
            logger.info(f"  Class {cls}: {count} ({count/len(y_val_seq)*100:.1f}%)")
        
        return X_train_seq, y_train_seq, X_val_seq, y_val_seq
        
    def _build_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build overlapping sequences."""
        sequences = []
        targets = []
        
        for i in range(self.seq_len, len(X)):
            sequences.append(X[i-self.seq_len:i])
            targets.append(y[i])
            
        return np.array(sequences), np.array(targets)
        
    def transform_for_inference(self, features: pd.DataFrame) -> np.ndarray:
        """Transform features for inference."""
        # Ensure same feature order
        features_aligned = features[self.feature_cols]
        
        # Scale
        scaled = self.scaler.transform(features_aligned)
        
        # Take last seq_len timesteps
        if len(scaled) >= self.seq_len:
            sequence = scaled[-self.seq_len:]
            return sequence.reshape(1, self.seq_len, -1)
        else:
            # Pad with zeros if not enough history
            padded = np.zeros((self.seq_len, scaled.shape[1]))
            padded[-len(scaled):] = scaled
            return padded.reshape(1, self.seq_len, -1)
