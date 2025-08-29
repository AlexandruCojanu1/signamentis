#!/usr/bin/env python3
"""
Strength & Duration models for SignaMentis AI Trading System.
Implements:
- Strength (3-class classification)
- Duration bins (multi-class classification)
- Duration bars (regression)

All models use LightGBM with compact hyperparameters and optional calibration.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import yaml
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, balanced_accuracy_score, mean_absolute_error
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrengthDurationModel:
    """Train and serve predictions for strength and duration models."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()

        self.scaler = StandardScaler()

        # Models
        self.model_strength: Optional[Any] = None
        self.model_duration_bins: Optional[Any] = None
        self.model_duration_bars: Optional[Any] = None

        # Calibrators
        self.cal_strength: Optional[CalibratedClassifierCV] = None
        self.cal_duration_bins: Optional[CalibratedClassifierCV] = None

        self.feature_columns: Optional[list] = None

    def _default_config(self) -> Dict:
        return {
            'cv_folds': 5,
            'random_state': 42,
            'n_jobs': 4,
            'early_stopping_rounds': 100,
            'calibrate': True
        }

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare features and targets from a dataframe containing engineered targets."""
        # Allowed feature space: exclude future-dependent engineered cols
        forbidden = {'estimated_atr','mfe_long','mfe_short','mfe_long_atr','mfe_short_atr'}
        feature_cols = [
            c for c in df.columns
            if not c.startswith('target_') and c != 'timestamp' and c not in forbidden
        ]

        X = df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], 0).fillna(0)

        # Drop high-NaN features
        nan_ratio = X.isna().mean()
        keep = nan_ratio[nan_ratio <= 0.3].index
        X = X[keep]
        self.feature_columns = list(keep)

        targets = {
            'strength': df['target_strength'],
            'duration_bins': df['target_duration_bins'],
            'duration_bars': df['target_duration_bars']
        }

        # Align indices by dropping rows with any NaN in targets
        target_df = pd.concat(targets, axis=1)
        valid_mask = ~target_df.isna().any(axis=1)
        X = X.loc[valid_mask]
        targets = {k: v.loc[valid_mask] for k, v in targets.items()}

        logger.info(f"Strength/Duration features prepared: {X.shape[1]} features, {len(X)} samples")
        return X, targets

    def _fit_classifier(self, X: pd.DataFrame, y: pd.Series) -> Any:
        tscv = TimeSeriesSplit(n_splits=self.config.get('cv_folds', 5))
        # Simple, stable params
        params = dict(
            objective='multiclass',
            num_class=int(y.max()) + 1 if y.notna().any() else 3,
            metric='multi_logloss',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', 4),
            verbose=-1,
        )

        # Scale within fit
        X_scaled = self.scaler.fit_transform(X)
        model = lgb.LGBMClassifier(**params)
        # Use last split for early stopping reference
        last_train_idx, last_val_idx = list(tscv.split(X_scaled))[-1]
        model.fit(
            X_scaled[last_train_idx], y.iloc[last_train_idx],
            eval_set=[(X_scaled[last_val_idx], y.iloc[last_val_idx])],
            callbacks=[lgb.early_stopping(self.config.get('early_stopping_rounds', 100), verbose=False)]
        )
        return model

    def _fit_regressor(self, X: pd.DataFrame, y: pd.Series) -> Any:
        tscv = TimeSeriesSplit(n_splits=self.config.get('cv_folds', 5))
        params = dict(
            objective='regression',
            metric='mae',
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', 4),
            verbose=-1,
        )
        X_scaled = self.scaler.fit_transform(X)
        model = lgb.LGBMRegressor(**params)
        last_train_idx, last_val_idx = list(tscv.split(X_scaled))[-1]
        model.fit(
            X_scaled[last_train_idx], y.iloc[last_train_idx],
            eval_set=[(X_scaled[last_val_idx], y.iloc[last_val_idx])],
            callbacks=[lgb.early_stopping(self.config.get('early_stopping_rounds', 100), verbose=False)]
        )
        return model

    def train(self, X_train: pd.DataFrame, targets: Dict[str, pd.Series],
              X_val: Optional[pd.DataFrame] = None, targets_val: Optional[Dict[str, pd.Series]] = None) -> Dict[str, Any]:
        """Train all three models; optionally calibrate classifiers on validation."""
        # Strength classifier
        y_strength = targets['strength'].astype(int)
        self.model_strength = self._fit_classifier(X_train, y_strength)

        # Duration bins classifier
        y_dur_bins = targets['duration_bins'].astype(int)
        self.model_duration_bins = self._fit_classifier(X_train, y_dur_bins)

        # Duration bars regressor
        y_dur_bars = targets['duration_bars'].astype(float)
        self.model_duration_bars = self._fit_regressor(X_train, y_dur_bars)

        # Optional calibration
        metrics = {}
        if X_val is not None and targets_val is not None:
            X_val_proc = X_val[self.feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
            X_val_scaled = self.scaler.transform(X_val_proc)

            if self.config.get('calibrate', True):
                # Strength calibration
                self.cal_strength = CalibratedClassifierCV(self.model_strength, method='isotonic', cv='prefit')
                self.cal_strength.fit(X_val_scaled, targets_val['strength'].astype(int))
                # Duration bins calibration
                self.cal_duration_bins = CalibratedClassifierCV(self.model_duration_bins, method='isotonic', cv='prefit')
                self.cal_duration_bins.fit(X_val_scaled, targets_val['duration_bins'].astype(int))

            # Metrics
            preds = self.predict(X_val)
            yv_s = targets_val['strength'].astype(int)
            metrics['strength_f1'] = f1_score(yv_s, preds['strength_pred'], average='macro')
            metrics['strength_bal_acc'] = balanced_accuracy_score(yv_s, preds['strength_pred'])

            yv_db = targets_val['duration_bins'].astype(int)
            metrics['duration_bins_f1'] = f1_score(yv_db, preds['duration_bins_pred'], average='macro')
            metrics['duration_bins_bal_acc'] = balanced_accuracy_score(yv_db, preds['duration_bins_pred'])

            yv_dr = targets_val['duration_bars'].astype(float)
            metrics['duration_bars_mae'] = mean_absolute_error(yv_dr, preds['duration_bars_pred'])

        # Label sanity checks
        if 'duration_bars' in targets:
            mae_train = mean_absolute_error(targets['duration_bars'].astype(float), self.model_duration_bars.predict(self.scaler.transform(X_train[self.feature_columns])))
            if mae_train > 5 or (np.isfinite(mae_train) and mae_train > 20):
                logger.error("label bug: wrong units")
                raise ValueError("label bug: wrong units")

        logger.info(f"Strength/Duration validation metrics: {metrics}")
        return metrics

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        if self.feature_columns is None:
            raise ValueError("Model not prepared. Train first.")
        # Ensure all expected columns exist; add missing with 0
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            for c in missing:
                X[c] = 0
        Xp = X[self.feature_columns].copy().replace([np.inf, -np.inf], 0).fillna(0)
        X_scaled = self.scaler.transform(Xp)

        # Strength probabilities
        if self.cal_strength is not None:
            strength_proba = self.cal_strength.predict_proba(X_scaled)
        else:
            strength_proba = self.model_strength.predict_proba(X_scaled)
        strength_pred = np.argmax(strength_proba, axis=1)

        # Duration bins probabilities
        if self.cal_duration_bins is not None:
            dur_bins_proba = self.cal_duration_bins.predict_proba(X_scaled)
        else:
            dur_bins_proba = self.model_duration_bins.predict_proba(X_scaled)
        duration_bins_pred = np.argmax(dur_bins_proba, axis=1)

        # Duration bars regression
        duration_bars_pred = self.model_duration_bars.predict(X_scaled)

        # Expected duration (use regression prediction)
        expected_duration = duration_bars_pred

        return {
            'strength_proba': strength_proba,
            'strength_pred': strength_pred,
            'duration_bins_proba': dur_bins_proba,
            'duration_bins_pred': duration_bins_pred,
            'duration_bars_pred': duration_bars_pred,
            'expected_duration': expected_duration
        }

    def save(self, output_dir: str, model_name_prefix: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Save models
        joblib.dump(self.model_strength, out / f"{model_name_prefix}_strength.joblib")
        joblib.dump(self.model_duration_bins, out / f"{model_name_prefix}_duration_bins.joblib")
        joblib.dump(self.model_duration_bars, out / f"{model_name_prefix}_duration_bars.joblib")
        # Save calibrators and scaler
        if self.cal_strength is not None:
            joblib.dump(self.cal_strength, out / f"{model_name_prefix}_strength_cal.joblib")
        if self.cal_duration_bins is not None:
            joblib.dump(self.cal_duration_bins, out / f"{model_name_prefix}_duration_bins_cal.joblib")
        joblib.dump(self.scaler, out / f"{model_name_prefix}_scaler.joblib")
        # Save metadata
        meta = {
            'feature_columns': self.feature_columns,
            'training_date': datetime.now().isoformat(),
            'config': self.config
        }
        with open(out / f"{model_name_prefix}_metadata.yaml", 'w') as f:
            yaml.dump(meta, f, default_flow_style=False)
        logger.info(f"Strength/Duration models saved to: {out}")


