#!/usr/bin/env python3
"""
Multiclass direction trainer for SOLUSDT 15m: UP / DOWN / SIDEWAYS.

Inputs:
  - features CSV with 15m timestamps and OHLCV-derived features

Outputs:
  - Saved models (LightGBM and RandomForest)
  - Evaluation report with Accuracy, F1 (macro), ROC-AUC (ovr)
  - Feature importance summary
"""

import argparse
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import yaml


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


LABELS = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}


def build_labels(df: pd.DataFrame, close_col: str = 'current_close') -> Tuple[pd.Series, Dict]:
    # Future 15m return
    future_close = df[close_col].shift(-1)
    ret = (future_close - df[close_col]) / df[close_col]
    # Use train-period percentiles to set strong move threshold later; placeholder for now
    return ret, {'raw_future_return': ret}


def split_time(df: pd.DataFrame, ts_col: str, train_end: str, val_end: str, test_end: str) -> Dict[str, pd.DataFrame]:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    train_end = pd.to_datetime(train_end, utc=True)
    val_end = pd.to_datetime(val_end, utc=True)
    test_end = pd.to_datetime(test_end, utc=True)
    train = df[df[ts_col] < train_end]
    val = df[(df[ts_col] >= train_end) & (df[ts_col] < val_end)]
    test = df[(df[ts_col] >= val_end) & (df[ts_col] <= test_end)]
    return {'train': train, 'val': val, 'test': test}


def label_with_thresholds(ret: pd.Series, strong_thr: float) -> pd.Series:
    labels = pd.Series(1, index=ret.index, dtype=int)  # 1 = SIDEWAYS
    labels[ret > strong_thr] = 2  # UP
    labels[ret < -strong_thr] = 0  # DOWN
    return labels


def prepare_xy(df: pd.DataFrame, ts_col: str, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c != ts_col and not c.startswith('target_') and c != label_col]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[label_col].astype(int)
    return X, y


def tune_lgbm(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
    param_grid = [
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': -1, 'num_leaves': 63},
        {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 8, 'num_leaves': 63},
        {'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 12, 'num_leaves': 127},
    ]
    best, best_acc = None, -1.0
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for p in param_grid:
        accs = []
        for tr, va in tscv.split(X):
            clf = lgb.LGBMClassifier(
                objective='multiclass', num_class=3, class_weight='balanced',
                n_jobs=4, random_state=42, **p
            )
            clf.fit(X.iloc[tr], y.iloc[tr])
            pred = clf.predict(X.iloc[va])
            accs.append(accuracy_score(y.iloc[va], pred))
        mean_acc = float(np.mean(accs))
        if mean_acc > best_acc:
            best, best_acc = p, mean_acc
    logger.info(f"Best LGBM CV accuracy: {best_acc:.4f} with {best}")
    return best


def train_and_eval(models: Dict[str, object], X_tr, y_tr, X_va, y_va, X_te, y_te, out_dir: Path) -> Dict:
    report = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_tr, y_tr)
        for split, Xs, ys in [( 'validation', X_va, y_va ), ('test', X_te, y_te)]:
            pred = model.predict(Xs)
            proba = model.predict_proba(Xs)
            acc = accuracy_score(ys, pred)
            f1m = f1_score(ys, pred, average='macro')
            try:
                roc = roc_auc_score(ys, proba, multi_class='ovr')
            except Exception:
                roc = float('nan')
            report[f'{name}_{split}'] = {'accuracy': acc, 'f1_macro': f1m, 'roc_auc_ovr': roc}
            logger.info(f"{name} {split}: acc={acc:.4f} f1m={f1m:.4f} roc-ovr={roc:.4f}")
        # Save model
        import joblib
        joblib.dump(model, out_dir / f"{name}.joblib")
    return report


def main():
    parser = argparse.ArgumentParser(description='Train multiclass direction model for SOLUSDT 15m')
    parser.add_argument('--features', default='data/processed/multi_timeframe_features_crypto_15m.csv')
    parser.add_argument('--timestamp_col', default='timestamp')
    parser.add_argument('--close_col', default='current_close')
    parser.add_argument('--train_end', default='2024-06-01T00:00:00Z')
    parser.add_argument('--val_end', default='2025-02-01T00:00:00Z')
    parser.add_argument('--test_end', default='2025-08-16T23:45:00Z')
    parser.add_argument('--out_dir', default='models/mc_direction')
    parser.add_argument('--strong_quantile', type=float, default=0.70, help='Quantile on |ret| to mark UP/DOWN')
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    # Labels from future return
    future_ret, aux = build_labels(df, args.close_col)
    df['future_ret'] = future_ret
    # Split first, then compute strong threshold from train
    splits = split_time(df, args.timestamp_col, args.train_end, args.val_end, args.test_end)
    strong_thr = splits['train']['future_ret'].abs().quantile(args.strong_quantile)
    logger.info(f"Strong move threshold (|ret|) from train P70: {strong_thr:.6f}")
    for k in splits:
        splits[k]['label'] = label_with_thresholds(splits[k]['future_ret'], strong_thr)
    # Drop last row in each split where future is NaN
    for k in splits:
        splits[k] = splits[k].dropna(subset=['label'])

    # Prepare X/y
    X_tr, y_tr = prepare_xy(splits['train'], args.timestamp_col, 'label')
    X_va, y_va = prepare_xy(splits['val'], args.timestamp_col, 'label')
    X_te, y_te = prepare_xy(splits['test'], args.timestamp_col, 'label')

    # Tune LGBM
    best_lgbm = tune_lgbm(X_tr, y_tr)
    lgbm = lgb.LGBMClassifier(objective='multiclass', num_class=3, class_weight='balanced', n_jobs=4, random_state=42, **best_lgbm)

    # RandomForest (simple, robust)
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, n_jobs=4, random_state=42, class_weight='balanced_subsample'
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = train_and_eval({'lgbm': lgbm, 'rf': rf}, X_tr, y_tr, X_va, y_va, X_te, y_te, out_dir)

    # Global feature importance (LGBM + RF)
    fi = {}
    try:
        fi['lgbm'] = dict(sorted(zip(X_tr.columns, lgbm.feature_importances_), key=lambda x: x[1], reverse=True)[:25])
    except Exception:
        fi['lgbm'] = {}
    try:
        fi['rf'] = dict(sorted(zip(X_tr.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)[:25])
    except Exception:
        fi['rf'] = {}

    # Per-class feature importance via lightweight One-vs-Rest LGBM
    fi_per_class = {}
    for cls, cls_name in LABELS.items():
        try:
            y_bin = (y_tr == cls).astype(int)
            clf_bin = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=63,
                class_weight='balanced',
                n_jobs=4,
                random_state=42
            )
            clf_bin.fit(X_tr, y_bin)
            imp = dict(sorted(zip(X_tr.columns, clf_bin.feature_importances_), key=lambda x: x[1], reverse=True)[:20])
            fi_per_class[cls_name] = imp
        except Exception:
            fi_per_class[cls_name] = {}

    # Persist report
    results_dir = Path('results') / f"mc_direction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'report.yaml', 'w') as f:
        yaml.dump({'threshold': float(strong_thr), 'strong_quantile': float(args.strong_quantile), 'metrics': report, 'feature_importance_global': fi, 'feature_importance_per_class': fi_per_class}, f)

    # Write Markdown summary
    def dict_to_md(d: Dict[str, float], k: int = 15) -> str:
        items = list(d.items())[:k]
        return "\n".join([f"- **{k}**: {v:.6f}" for k, v in items]) if items else "_n/a_"

    with open(results_dir / 'feature_importance.md', 'w') as md:
        md.write("# Multiclass Direction - Feature Importance\n\n")
        md.write(f"- Strong threshold: |ret| ≥ {strong_thr:.6f} (quantile={args.strong_quantile})\n\n")
        md.write("## Global (LightGBM)\n\n")
        md.write(dict_to_md(fi.get('lgbm', {})) + "\n\n")
        md.write("## Global (RandomForest)\n\n")
        md.write(dict_to_md(fi.get('rf', {})) + "\n\n")
        for cls_name in ['UP','SIDEWAYS','DOWN']:
            md.write(f"## Class: {cls_name}\n\n")
            md.write(dict_to_md(fi_per_class.get(cls_name, {})) + "\n\n")

    # Print acceptance check
    acc_ok = (report['lgbm_validation']['accuracy'] >= 0.97) and (report['lgbm_test']['accuracy'] >= 0.97)
    logger.info(f"Acceptance (accuracy ≥ 0.97 on val & test) -> {'PASS' if acc_ok else 'FAIL'}")


if __name__ == '__main__':
    main()


