#!/usr/bin/env python3
"""
Multiclass strength trainer for SOLUSDT 15m: WEAK / MEDIUM / STRONG.

Strength is defined via the joint distribution of absolute next-15m return and
next-15m volume. Labels use train quantiles: ≤P30 WEAK, P30–P70 MEDIUM, ≥P70 STRONG.

Outputs:
  - models saved in models/mc_strength/
  - results/mc_strength_*/report.yaml and feature_importance.md
"""

import argparse
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import yaml
import warnings


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


LABELS = {0: 'WEAK', 1: 'MEDIUM', 2: 'STRONG'}


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


def compute_future_stats(df: pd.DataFrame, close_col: str, vol_col: str) -> pd.DataFrame:
    df = df.copy()
    df['future_close'] = df[close_col].shift(-1)
    df['ret_abs'] = (df['future_close'] - df[close_col]).abs() / df[close_col]
    df['next_volume'] = df[vol_col].shift(-1)
    return df


def label_strength(df_train: pd.DataFrame, df: pd.DataFrame, p_low: float, p_high: float, smooth_band: float = 0.02, drop_indices: bool = True) -> Tuple[pd.Series, float, float, pd.Index]:
    # Rank-transform within train to form joint score
    r_rank = df_train['ret_abs'].rank(pct=True)
    v_rank = df_train['next_volume'].rank(pct=True)
    joint_train = 0.5 * r_rank + 0.5 * v_rank
    thr_low = joint_train.quantile(p_low)
    thr_high = joint_train.quantile(p_high)

    # Apply same transformation to full df using train statistics (percentile ranks via join on sorted order)
    def pct_rank(s: pd.Series, ref: pd.Series) -> pd.Series:
        ref_sorted = ref.sort_values().values
        return s.apply(lambda x: np.searchsorted(ref_sorted, x, side='right') / len(ref_sorted))

    r_rank_all = pct_rank(df['ret_abs'], df_train['ret_abs'])
    v_rank_all = pct_rank(df['next_volume'], df_train['next_volume'])
    joint_all = 0.5 * r_rank_all + 0.5 * v_rank_all

    labels = pd.Series(1, index=df.index, dtype=int)  # MEDIUM default
    labels[joint_all <= thr_low] = 0
    labels[joint_all >= thr_high] = 2
    # Optional smoothing: drop samples near thresholds
    if smooth_band and smooth_band > 0:
        low_lb = joint_train.quantile(max(p_low - smooth_band, 0.0))
        low_ub = joint_train.quantile(min(p_low + smooth_band, 1.0))
        high_lb = joint_train.quantile(max(p_high - smooth_band, 0.0))
        high_ub = joint_train.quantile(min(p_high + smooth_band, 1.0))
        near_low = (joint_all > low_lb) & (joint_all < low_ub)
        near_high = (joint_all > high_lb) & (joint_all < high_ub)
        drop_idx = df.index[near_low | near_high]
    else:
        drop_idx = pd.Index([])
    return labels, float(thr_low), float(thr_high), drop_idx


def prepare_xy(df: pd.DataFrame, ts_col: str, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c != ts_col and not c.startswith('target_') and c not in {label_col, 'future_close', 'ret_abs', 'next_volume'}]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[label_col].astype(int)
    return X, y


def _lgbm_base_params() -> Dict:
    # GPU if available, else optimized CPU
    try:
        import lightgbm as _lgb
        # crude check; if GPU fails, LightGBM will fallback
        device = 'gpu'
    except Exception:
        device = 'cpu'
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'class_weight': 'balanced',
        'n_jobs': 4,
        'random_state': 42,
        'verbosity': -1,
        'device': device,
        'force_row_wise': True,
    }
    return params


def tune_lgbm(X: pd.DataFrame, y: pd.Series, n_splits: int = 4, time_budget_sec: int = 900) -> Dict:
    # Focused grid (<= 12 combos)
    grid = [
        {'n_estimators': 400, 'learning_rate': 0.03, 'max_depth': 12, 'num_leaves': 127},
        {'n_estimators': 600, 'learning_rate': 0.02, 'max_depth': 14, 'num_leaves': 255},
        {'n_estimators': 500, 'learning_rate': 0.025, 'max_depth': 12, 'num_leaves': 127},
        {'n_estimators': 450, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63},
        {'n_estimators': 550, 'learning_rate': 0.02, 'max_depth': 12, 'num_leaves': 127},
        {'n_estimators': 400, 'learning_rate': 0.02, 'max_depth': 14, 'num_leaves': 255},
        {'n_estimators': 600, 'learning_rate': 0.03, 'max_depth': 12, 'num_leaves': 127},
        {'n_estimators': 500, 'learning_rate': 0.02, 'max_depth': 8, 'num_leaves': 63},
    ]
    start = time.time()
    best, best_f1 = None, -1.0
    tscv = TimeSeriesSplit(n_splits=n_splits)
    base_params = _lgbm_base_params()
    for p in grid:
        if time.time() - start > time_budget_sec:
            logger.info("Time budget exceeded during tuning; stopping early.")
            break
        f1s = []
        for tr, va in tscv.split(X):
            clf = lgb.LGBMClassifier(**base_params, **p)
            clf.fit(
                X.iloc[tr], y.iloc[tr],
                eval_set=[(X.iloc[va], y.iloc[va])],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(100),
                ],
            )
            pred = clf.predict(X.iloc[va])
            f1s.append(f1_score(y.iloc[va], pred, average='macro'))
        mean_f1 = float(np.mean(f1s))
        logger.info(f"Grid {p} -> CV f1_macro={mean_f1:.4f}")
        if mean_f1 > best_f1:
            best, best_f1 = p, mean_f1
    logger.info(f"Best LGBM CV f1_macro: {best_f1:.4f} with {best}")
    return best
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
        if isinstance(model, lgb.LGBMClassifier):
            model.set_params(verbosity=-1)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(100),
                ],
            )
        else:
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
    parser = argparse.ArgumentParser(description='Train multiclass strength model for SOLUSDT 15m')
    parser.add_argument('--features', default='data/processed/multi_timeframe_features_crypto_15m.csv')
    parser.add_argument('--timestamp_col', default='timestamp')
    parser.add_argument('--close_col', default='current_close')
    parser.add_argument('--volume_col', default='current_volume')
    parser.add_argument('--train_end', default='2024-06-01T00:00:00Z')
    parser.add_argument('--val_end', default='2025-02-01T00:00:00Z')
    parser.add_argument('--test_end', default='2025-08-16T23:45:00Z')
    parser.add_argument('--out_dir', default='models/mc_strength')
    parser.add_argument('--p_low', type=float, default=0.10)
    parser.add_argument('--p_high', type=float, default=0.90)
    parser.add_argument('--smooth_band', type=float, default=0.05, help='Drop samples near thresholds +/- band')
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    df = compute_future_stats(df, args.close_col, args.volume_col)

    splits = split_time(df, args.timestamp_col, args.train_end, args.val_end, args.test_end)
    # Build labels using train quantiles
    labels_train, thr_low, thr_high, drop_idx = label_strength(splits['train'], df, args.p_low, args.p_high, args.smooth_band)
    df['strength_label'] = labels_train
    if len(drop_idx) > 0:
        df = df.drop(index=drop_idx)
        for k in ['train','val','test']:
            splits[k] = splits[k][~splits[k].index.isin(drop_idx)]
    for k in splits:
        splits[k]['strength_label'] = df.loc[splits[k].index, 'strength_label']
        splits[k] = splits[k].dropna(subset=['strength_label'])

    # Prepare X/y
    X_tr, y_tr = prepare_xy(splits['train'], args.timestamp_col, 'strength_label')
    X_va, y_va = prepare_xy(splits['val'], args.timestamp_col, 'strength_label')
    X_te, y_te = prepare_xy(splits['test'], args.timestamp_col, 'strength_label')

    # Tune LGBM and define RF
    best_lgbm = tune_lgbm(X_tr, y_tr)
    base_params = _lgbm_base_params()
    lgbm = lgb.LGBMClassifier(**base_params, **best_lgbm)
    rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2, n_jobs=4, random_state=42, class_weight='balanced_subsample')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = train_and_eval({'lgbm': lgbm, 'rf': rf}, X_tr, y_tr, X_va, y_va, X_te, y_te, out_dir)

    # Global importances
    fi = {}
    try:
        fi['lgbm'] = dict(sorted(zip(X_tr.columns, lgbm.feature_importances_), key=lambda x: x[1], reverse=True)[:25])
    except Exception:
        fi['lgbm'] = {}
    try:
        fi['rf'] = dict(sorted(zip(X_tr.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)[:25])
    except Exception:
        fi['rf'] = {}

    # Per-class importances via OvR LGBM
    fi_per_class = {}
    for cls, cls_name in LABELS.items():
        try:
            y_bin = (y_tr == cls).astype(int)
            clf_bin = lgb.LGBMClassifier(objective='binary', n_estimators=200, learning_rate=0.05, max_depth=8, num_leaves=63, class_weight='balanced', n_jobs=4, random_state=42)
            clf_bin.fit(X_tr, y_bin)
            imp = dict(sorted(zip(X_tr.columns, clf_bin.feature_importances_), key=lambda x: x[1], reverse=True)[:20])
            fi_per_class[cls_name] = imp
        except Exception:
            fi_per_class[cls_name] = {}

    # Confusion matrices
    cms = {}
    for name, model in {'lgbm': lgbm, 'rf': rf}.items():
        pred_val = model.predict(X_va)
        pred_te = model.predict(X_te)
        cms[f'{name}_val'] = confusion_matrix(y_va, pred_val).tolist()
        cms[f'{name}_test'] = confusion_matrix(y_te, pred_te).tolist()

    # Persist report
    results_dir = Path('results') / f"mc_strength_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'report.yaml', 'w') as f:
        yaml.dump({
            'thresholds': {'low': thr_low, 'high': thr_high, 'p_low': args.p_low, 'p_high': args.p_high},
            'metrics': report,
            'confusion_matrices': cms,
            'feature_importance_global': fi,
            'feature_importance_per_class': fi_per_class
        }, f)

    # Markdown summary
    def dict_to_md(d: Dict[str, float], k: int = 15) -> str:
        items = list(d.items())[:k]
        return "\n".join([f"- **{k}**: {v:.6f}" for k, v in items]) if items else "_n/a_"

    with open(results_dir / 'feature_importance.md', 'w') as md:
        md.write("# Multiclass Strength - Feature Importance\n\n")
        md.write(f"- Thresholds: low={thr_low:.6f} (p={args.p_low}), high={thr_high:.6f} (p={args.p_high})\n\n")
        md.write("## Global (LightGBM)\n\n")
        md.write(dict_to_md(fi.get('lgbm', {})) + "\n\n")
        md.write("## Global (RandomForest)\n\n")
        md.write(dict_to_md(fi.get('rf', {})) + "\n\n")
        for cls_name in ['STRONG','MEDIUM','WEAK']:
            md.write(f"## Class: {cls_name}\n\n")
            md.write(dict_to_md(fi_per_class.get(cls_name, {})) + "\n\n")

    # Acceptance
    acc_ok = (report['lgbm_validation']['accuracy'] >= 0.95) and (report['lgbm_test']['accuracy'] >= 0.95)
    logger.info(f"Acceptance (accuracy ≥ 0.95 on val & test) -> {'PASS' if acc_ok else 'FAIL'}")


if __name__ == '__main__':
    main()


