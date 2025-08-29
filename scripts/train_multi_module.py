#!/usr/bin/env python3
"""
Multi-module training pipeline for SignaMentis AI Trading System.
Trains all modules and orchestrator for comprehensive trading decisions.
"""

import pandas as pd
import numpy as np
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_utils import DataUtils
from targets import TargetEngineer
from model_direction import DirectionalModel
from orchestrator import Orchestrator
from model_strength_duration import StrengthDurationModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiModuleTrainer:
    """Main training pipeline for all modules."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize MultiModuleTrainer with configuration."""
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # Initialize components
        self.data_utils = DataUtils(config_path)
        self.target_engineer = TargetEngineer(self.config.get('targets', {}))
        
        # Initialize models
        self.direction_models = {}
        self.orchestrator = Orchestrator(self.config.get('orchestrator', {}))
        self.strength_duration_model: Optional[StrengthDurationModel] = None
        
        # Training results
        self.training_results = {}
        self.canonical_features = None
        
    def _default_config(self) -> Dict:
        """Default configuration for multi-module training."""
        return {
            'general': {
                'seed': 42,
                'n_jobs': 4,
                'horizon_bars': 6
            },
            'data': {
                'features_csv': 'data/processed/multi_timeframe_features_3years_20250823_104329.csv',
                'timestamp_col': 'timestamp'
            },
            'splits': {
                'train_end': "2024-06-01T00:00:00Z",
                'val_end': "2025-02-01T00:00:00Z",
                'test_end': "2025-08-16T23:45:00Z"
            },
            'direction': {
                'targets': ['target_direction_1', 'target_direction_3', 'target_direction_5'],
                'min_precision_val': 0.97,
                'min_recall_val': 0.25
            },
            'orchestrator': {
                'algorithm': 'logistic',
                'min_precision_val': 0.97,
                'min_recall_val': 0.25,
                'use_only_model_outputs': True,
                'safe_feature_whitelist': [
                    'vol_regime_alignment','trend_strength_ratio','momentum_confluence','momentum_divergence',
                    'm5_range_expansion','m5_trend_consistency','m5_volatility_6','m15_trend_slope','m15_range_volatility',
                    'price_level_alignment','session_overlap','session_transition','current_spread','hour_sin','day_of_week_sin'
                ]
            },
            'output': {
                'models_dir': 'models',
                'results_dir': 'results',
                'run_name': None  # Will be auto-generated
            }
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
            return self._default_config()
    
    def run_training_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete multi-module training pipeline.
        
        Returns:
            Dictionary with training results and metadata
        """
        logger.info("🚀 Starting Multi-Module Training Pipeline")
        logger.info("=" * 60)
        
        # Set random seed
        np.random.seed(self.config['general']['seed'])
        
        # Generate run name
        if not self.config['output']['run_name']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config['output']['run_name'] = f"run_{timestamp}"
        
        run_name = self.config['output']['run_name']
        logger.info(f"Run name: {run_name}")
        
        try:
            # Step 1: Load and prepare data
            logger.info("\n📊 Step 1: Loading and preparing data...")
            data_splits = self._load_and_prepare_data()
            
            # Step 2: Create target variables
            logger.info("\n🎯 Step 2: Creating target variables...")
            data_with_targets = self._create_target_variables(data_splits)
            
            # Step 3: Train directional models
            logger.info("\n🧭 Step 3: Training directional models...")
            direction_results = self._train_directional_models(data_with_targets)
            
            # Step 3b: Train strength & duration models
            logger.info("\n💪 Step 3b: Training strength & duration models...")
            sd_results = self._train_strength_duration(data_with_targets)

            # Step 4: Train orchestrator
            logger.info("\n🎼 Step 4: Training orchestrator...")
            orchestrator_results = self._train_orchestrator(data_with_targets)
            
            # Step 5: Final evaluation
            logger.info("\n📈 Step 5: Final evaluation...")
            final_results = self._evaluate_all_modules(data_with_targets)
            
            # Step 6: Save models and results
            logger.info("\n💾 Step 6: Saving models and results...")
            self._save_all_artifacts(run_name)
            
            # Compile results
            self.training_results = {
                'run_name': run_name,
                'direction_models': direction_results,
                'strength_duration': sd_results,
                'orchestrator': orchestrator_results,
                'final_evaluation': final_results,
                'config': self.config,
                'training_date': datetime.now().isoformat()
            }
            
            logger.info("\n🎉 Multi-Module Training Pipeline Completed Successfully!")
            logger.info("=" * 60)
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            raise
    
    def _load_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Load features and split data by time periods."""
        # Load features
        features_path = self.config['data']['features_csv']
        df = self.data_utils.load_features(features_path)
        # Compute simple data hash for config metadata
        try:
            hash_input = (str(df.shape) + '|' + str(sorted(df.columns.tolist())[:10])).encode('utf-8')
            self.config['output']['data_hash'] = hashlib.md5(hash_input).hexdigest()
        except Exception:
            self.config['output']['data_hash'] = None
        
        # Set canonical features from training data
        self.data_utils.set_canonical_features(df)
        self.canonical_features = self.data_utils.canonical_features
        
        # Split data by time
        data_splits = self.data_utils.split_by_time(df)
        
        # Save canonical features
        output_dir = Path(self.config['output']['models_dir']) / self.config['output']['run_name']
        output_dir.mkdir(parents=True, exist_ok=True)
        self.data_utils.save_canonical_features(str(output_dir / 'canonical_features.yaml'))
        
        return data_splits
    
    def _create_target_variables(self, data_splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create target variables for all data splits."""
        data_with_targets = {}
        
        for split_name, split_data in data_splits.items():
            if len(split_data) > 0:
                logger.info(f"Creating targets for {split_name} split...")
                data_with_targets[split_name] = self.target_engineer.create_all_targets(split_data)
                
                # Get target summary
                summary = self.target_engineer.get_target_summary(data_with_targets[split_name])
                logger.info(f"{split_name.capitalize()} targets summary:")
                for target, info in summary.items():
                    if 'type' in info:
                        logger.info(f"  {target}: {info['type']} - {info.get('count', 'N/A')} samples")
        
        return data_with_targets
    
    def _train_directional_models(self, data_with_targets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train directional models for all targets."""
        logger.info("Training directional models...")
        
        direction_results = {}
        direction_targets = self.config['direction']['targets']
        
        for target_col in direction_targets:
            if target_col not in data_with_targets['train'].columns:
                logger.warning(f"Target {target_col} not found in training data. Skipping.")
                continue
            
            logger.info(f"\n🎯 Training for target: {target_col}")
            logger.info("-" * 40)
            
            try:
                # Initialize model
                model = DirectionalModel(self.config.get('direction', {}))
                
                # Prepare training data
                X_train, y_train = model.prepare_features(data_with_targets['train'], target_col)
                
                if len(X_train) == 0:
                    logger.warning(f"No training data for {target_col}. Skipping.")
                    continue
                
                # Optimize hyperparameters
                best_params = model.optimize_hyperparameters(X_train, y_train)
                
                # Train model
                model.train_model(X_train, y_train)
                
                # Prepare validation data
                if 'validation' in data_with_targets and len(data_with_targets['validation']) > 0:
                    X_val, y_val = model.prepare_features(data_with_targets['validation'], target_col)
                    
                    if len(X_val) > 0:
                        # Calibrate probabilities
                        model.calibrate_probabilities(X_val, y_val)
                        
                        # Evaluate on validation
                        val_metrics = model.evaluate(X_val, y_val, "validation")
                        
                        # Check quality gate
                        if self._check_direction_quality_gate(val_metrics):
                            logger.info(f"✅ Quality gate passed for {target_col}")
                        else:
                            logger.warning(f"⚠️ Quality gate not met for {target_col}")
                        
                        # Evaluate on test
                        if 'test' in data_with_targets and len(data_with_targets['test']) > 0:
                            X_test, y_test = model.prepare_features(data_with_targets['test'], target_col)
                            if len(X_test) > 0:
                                test_metrics = model.evaluate(X_test, y_test, "test")
                            else:
                                test_metrics = None
                        else:
                            test_metrics = None
                        
                        # Store results
                        direction_results[target_col] = {
                            'model': model,
                            'validation_metrics': val_metrics,
                            'test_metrics': test_metrics,
                            'best_params': best_params
                        }
                        
                        # Also store in self.direction_models for orchestrator access
                        self.direction_models[target_col] = direction_results[target_col]
                        
                        logger.info(f"✅ Training completed for {target_col}")
                    else:
                        logger.warning(f"No validation data for {target_col}")
                else:
                    logger.warning(f"No validation data available for {target_col}")
                
            except Exception as e:
                logger.error(f"❌ Training failed for {target_col}: {e}")
                continue
        
        logger.info(f"Directional models training completed: {len(direction_results)} models")
        return direction_results

    def _train_strength_duration(self, data_with_targets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train strength (3-class), duration bins (multi-class) and duration bars (regression)."""
        results: Dict[str, Any] = {}
        try:
            # Prepare features using canonical features
            X_train_full = self.data_utils.get_feature_matrix(data_with_targets['train'])
            X_val_full = self.data_utils.get_feature_matrix(data_with_targets.get('validation', data_with_targets['train'])) if 'validation' in data_with_targets else None

            # Prepare targets exist in the same dataframes
            df_train = data_with_targets['train']
            df_val = data_with_targets.get('validation', None)

            sd = StrengthDurationModel(self.config.get('strength_duration', {}))
            X_sd_train, targets_train = sd.prepare_features(df_train)
            if df_val is not None and len(df_val) > 0:
                X_sd_val, targets_val = sd.prepare_features(df_val)
            else:
                X_sd_val, targets_val = None, None

            metrics = sd.train(X_sd_train, targets_train, X_sd_val, targets_val)
            self.strength_duration_model = sd
            results['validation_metrics'] = metrics
            logger.info("✅ Strength/Duration training completed")
        except Exception as e:
            logger.error(f"❌ Strength/Duration training failed: {e}")
        return results
    
    def _check_direction_quality_gate(self, metrics: Dict[str, float]) -> bool:
        """Check if directional model meets quality gate requirements."""
        min_precision = self.config['direction']['min_precision_val']
        min_recall = self.config['direction']['min_recall_val']
        
        high_conf_precision = metrics.get('high_conf_precision', 0.0)
        high_conf_recall = metrics.get('high_conf_recall', 0.0)
        
        # Check if high-confidence predictions meet requirements
        if high_conf_precision >= min_precision and high_conf_recall >= min_recall:
            return True
        
        return False
    
    def _train_orchestrator(self, data_with_targets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train the orchestrator meta-learner."""
        logger.info("Training orchestrator meta-learner...")
        
        if not self.direction_models:
            raise ValueError("No directional models trained. Cannot train orchestrator.")
        
        try:
            # Create orchestrator training data using OOF calibrated probabilities
            X_orchestrator, y_orchestrator = self._create_orchestrator_training_data(data_with_targets)
            
            if len(X_orchestrator) == 0:
                logger.warning("No orchestrator training data created. Skipping orchestrator training.")
                return {}
            
            # Train orchestrator
            self.orchestrator.train_orchestrator(X_orchestrator, y_orchestrator)
            
            # Calibrate probability threshold tau on validation
            if 'validation' in data_with_targets and len(data_with_targets['validation']) > 0:
                # Build validation features/labels from trained models
                X_val_orch, y_val_orch = self._create_orchestrator_validation_data(data_with_targets['validation'])
                
                if len(X_val_orch) > 0:
                    self.orchestrator.calibrate_probabilities(X_val_orch, y_val_orch)
                    
                    # Evaluate on validation
                    val_metrics = self.orchestrator.evaluate(X_val_orch, y_val_orch, "validation")
                    
                    # Check quality gate
                    if self._check_orchestrator_quality_gate(val_metrics):
                        logger.info("✅ Orchestrator quality gate passed")
                    else:
                        logger.warning("⚠️ Orchestrator quality gate not met")
                    
                    # Evaluate on test
                    if 'test' in data_with_targets and len(data_with_targets['test']) > 0:
                        X_test_orch, y_test_orch = self._create_orchestrator_validation_data(data_with_targets['test'])
                        if len(X_test_orch) > 0:
                            test_metrics = self.orchestrator.evaluate(X_test_orch, y_test_orch, "test")
                        else:
                            test_metrics = None
                    else:
                        test_metrics = None
                    
                    orchestrator_results = {
                        'model': self.orchestrator,
                        'validation_metrics': val_metrics,
                        'test_metrics': test_metrics
                    }
                    
                    logger.info("✅ Orchestrator training completed")
                    return orchestrator_results
            
            logger.warning("Orchestrator training completed but no validation data available")
            return {}
            
        except Exception as e:
            logger.error(f"❌ Orchestrator training failed: {e}")
            return {}
    
    def _create_orchestrator_training_data(self, data_with_targets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Create orchestrator training data using OOF calibrated p1/p3/p5 and gating labels."""
        train_df = data_with_targets['train'].reset_index(drop=True)
        # Prepare base features (for safe whitelist)
        X_base = self.data_utils.get_feature_matrix(train_df)

        # Generate OOF calibrated probabilities for each directional target
        p_cols = {}
        y_truth = {}
        for tgt, info in self.direction_models.items():
            model: DirectionalModel = info['model']
            # Prepare features and target for this specific target column
            X_all, y_all = model.prepare_features(train_df, tgt)
            X_all = X_all.reset_index(drop=True)
            y_all = y_all.reset_index(drop=True)
            n = len(X_all)
            oof = np.full(n, np.nan, dtype=float)
            tscv = TimeSeriesSplit(n_splits=5)
            for train_idx, val_idx in tscv.split(X_all):
                X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
                X_va, y_va = X_all.iloc[val_idx], y_all.iloc[val_idx]
                # Scale per fold
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)
                # Train LGBM with best params
                params = model.best_params or {
                    'n_estimators': 300,'max_depth': 10,'learning_rate': 0.05,'num_leaves': 63,
                    'min_child_samples': 20,'subsample': 1.0,'colsample_bytree': 0.9,
                    'reg_alpha': 0,'reg_lambda': 0.1,'objective': 'binary','metric': 'auc',
                    'boosting_type': 'gbdt','class_weight': 'balanced','random_state': 42,'n_jobs': self.config['general']['n_jobs'],'verbose': -1
                }
                clf = lgb.LGBMClassifier(**params)
                clf.fit(X_tr_s, y_tr)
                # Calibrate on the fold's val
                cal = CalibratedClassifierCV(estimator=clf, cv='prefit', method='isotonic')
                cal.fit(X_va_s, y_va)
                oof[val_idx] = cal.predict_proba(X_va_s)[:, 1]
            key = tgt.replace('target_direction_', 'p')  # target_direction_1 -> p1
            p_cols[key] = pd.Series(oof)
            y_truth[key] = y_all.astype(int).reset_index(drop=True)

        # Build core orchestrator feature matrix (align by position)
        df_feats = pd.DataFrame(p_cols)
        df_feats = df_feats[['p1','p3','p5']]
        df_feats = df_feats.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        # Fill any missing OOF with neutral 0.5 to avoid NaN rows
        for col in ['p1','p3','p5']:
            if col in df_feats.columns:
                df_feats[col] = pd.Series(df_feats[col]).astype(float)
        df_feats[['p1','p3','p5']] = df_feats[['p1','p3','p5']].fillna(0.5)
        # Compute derivatives
        p1 = df_feats.get('p1', pd.Series(0.5, index=df_feats.index))
        p3 = df_feats.get('p3', pd.Series(0.5, index=df_feats.index))
        p5 = df_feats.get('p5', pd.Series(0.5, index=df_feats.index))
        arr = np.vstack([p1.values, p3.values, p5.values]).T
        arr = np.nan_to_num(arr, nan=0.5)
        pmax = np.max(arr, axis=1)
        # second best
        sorted_arr = np.sort(arr, axis=1)[:, ::-1]
        p2nd = sorted_arr[:, 1]
        argmax_h = np.argmax(arr, axis=1)
        agreement = (((p1 >= 0.5) == (p3 >= 0.5)) & ((p3 >= 0.5) == (p5 >= 0.5))).astype(int).values
        # entropy of normalized
        eps = 1e-9
        norm = (arr.sum(axis=1) + eps)[:, None]
        q = arr / norm
        entropy = (- (q * np.log(q + eps)).sum(axis=1))
        df_feats['p1'] = p1
        df_feats['p3'] = p3
        df_feats['p5'] = p5
        df_feats['pmax'] = pmax
        df_feats['argmax_h'] = argmax_h
        df_feats['p_spread'] = pmax - p2nd
        df_feats['agreement'] = agreement
        df_feats['entropy'] = entropy

        # Safe whitelist features from X_base
        wl = self.config['orchestrator'].get('safe_feature_whitelist', [])
        if not self.config['orchestrator'].get('use_only_model_outputs', True) and wl:
            existing = [c for c in wl if c in X_base.columns]
            for col in existing:
                df_feats[f'context_{col}'] = X_base[col].values
        # Clean NaNs
        df_feats = df_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
        logger.info(f"Orchestrator feature set after safe select: {df_feats.shape[1]} cols (no missing)")

        # Create multiclass gating labels: 0:use_h1,1:use_h3,2:use_h5,3:abstain
        # True labels aligned by position from prepared feature targets
        y1 = y_truth.get('p1').iloc[:len(df_feats)].reset_index(drop=True)
        y3 = y_truth.get('p3').iloc[:len(df_feats)].reset_index(drop=True)
        y5 = y_truth.get('p5').iloc[:len(df_feats)].reset_index(drop=True)
        labels = np.full(len(df_feats), 3, dtype=int)
        threshold = 0.55
        for i in range(len(df_feats)):
            if df_feats['pmax'].iloc[i] < threshold:
                labels[i] = 3
                continue
            k = df_feats['argmax_h'].iloc[i]
            if k == 0:
                pred = int(df_feats['p1'].iloc[i] >= 0.5)
                labels[i] = 0 if pred == int(y1.iloc[i]) else 3
            elif k == 1:
                pred = int(df_feats['p3'].iloc[i] >= 0.5)
                labels[i] = 1 if pred == int(y3.iloc[i]) else 3
            else:
                pred = int(df_feats['p5'].iloc[i] >= 0.5)
                labels[i] = 2 if pred == int(y5.iloc[i]) else 3

        return df_feats.reset_index(drop=True), pd.Series(labels).reset_index(drop=True)
    
    def _create_orchestrator_validation_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create validation/test orchestrator features; labels from gating rules for reporting."""
        data_reset = data.reset_index(drop=True)
        X_base = self.data_utils.get_feature_matrix(data_reset)
        feats: Dict[str, Any] = {}
        # directional calibrated probabilities
        for tgt, info in self.direction_models.items():
            model: DirectionalModel = info['model']
            _, probs = model.predict(X_base, return_confidence=True)
            key = tgt.replace('target_direction_', 'p')
            feats[key] = probs
        df_feats = pd.DataFrame(feats)
        # derivatives
        p1 = df_feats.get('p1', pd.Series(0.5, index=df_feats.index))
        p3 = df_feats.get('p3', pd.Series(0.5, index=df_feats.index))
        p5 = df_feats.get('p5', pd.Series(0.5, index=df_feats.index))
        arr = np.vstack([p1.values, p3.values, p5.values]).T
        pmax = np.nanmax(arr, axis=1)
        sorted_arr = np.sort(arr, axis=1)[:, ::-1]
        p2nd = sorted_arr[:, 1]
        argmax_h = np.nanargmax(arr, axis=1)
        agreement = (((p1 >= 0.5) == (p3 >= 0.5)) & ((p3 >= 0.5) == (p5 >= 0.5))).astype(int).values
        eps = 1e-9
        norm = (arr.sum(axis=1) + eps)[:, None]
        q = arr / norm
        entropy = (- (q * np.log(q + eps)).sum(axis=1))
        df_feats['p1'] = p1
        df_feats['p3'] = p3
        df_feats['p5'] = p5
        df_feats['pmax'] = pmax
        df_feats['argmax_h'] = argmax_h
        df_feats['p_spread'] = pmax - p2nd
        df_feats['agreement'] = agreement
        df_feats['entropy'] = entropy
        # Safe whitelist
        wl = self.config['orchestrator'].get('safe_feature_whitelist', [])
        if not self.config['orchestrator'].get('use_only_model_outputs', True) and wl:
            existing = [c for c in wl if c in X_base.columns]
            for col in existing:
                df_feats[f'context_{col}'] = X_base[col].values
        df_feats = df_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
        logger.info(f"Orchestrator feature set after safe select: {df_feats.shape[1]} cols (no missing)")

        # Labels (for eval reporting)
        if {'target_direction_1','target_direction_3','target_direction_5'}.issubset(set(data_reset.columns)):
            y1 = data_reset['target_direction_1'].iloc[:len(df_feats)]
            y3 = data_reset['target_direction_3'].iloc[:len(df_feats)]
            y5 = data_reset['target_direction_5'].iloc[:len(df_feats)]
            labels = np.full(len(df_feats), 3, dtype=int)
            threshold = 0.55
            for i in range(len(df_feats)):
                if df_feats['pmax'].iloc[i] < threshold:
                    labels[i] = 3
                    continue
                k = df_feats['argmax_h'].iloc[i]
                if k == 0:
                    pred = int(df_feats['p1'].iloc[i] >= 0.5)
                    labels[i] = 0 if pred == int(y1.iloc[i]) else 3
                elif k == 1:
                    pred = int(df_feats['p3'].iloc[i] >= 0.5)
                    labels[i] = 1 if pred == int(y3.iloc[i]) else 3
                else:
                    pred = int(df_feats['p5'].iloc[i] >= 0.5)
                    labels[i] = 2 if pred == int(y5.iloc[i]) else 3
            y = pd.Series(labels, index=df_feats.index)
        else:
            y = pd.Series(np.zeros(len(df_feats), dtype=int), index=df_feats.index)

        return df_feats, y
    
    def _check_orchestrator_quality_gate(self, metrics: Dict[str, float]) -> bool:
        """Check if orchestrator meets quality gate requirements."""
        min_precision = self.config['orchestrator']['min_precision_val']
        min_recall = self.config['orchestrator']['min_recall_val']
        
        high_conf_precision = metrics.get('high_conf_precision', 0.0)
        high_conf_recall = metrics.get('high_conf_recall', 0.0)
        
        # Check if high-confidence predictions meet requirements
        if high_conf_precision >= min_precision and high_conf_recall >= min_recall:
            return True
        
        return False
    
    def _evaluate_all_modules(self, data_with_targets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Evaluate all modules on test data."""
        logger.info("Evaluating all modules...")
        
        final_results = {}
        
        # Evaluate directional models
        for target_col, model_info in self.direction_models.items():
            if 'test_metrics' in model_info and model_info['test_metrics'] is not None:
                final_results[f'direction_{target_col}'] = model_info['test_metrics']
        
        # Evaluate orchestrator
        if hasattr(self.orchestrator, 'model') and self.orchestrator.model is not None:
            if 'test' in data_with_targets and len(data_with_targets['test']) > 0:
                # Create test data for orchestrator
                X_test_orch, y_test_orch = self._create_orchestrator_validation_data(data_with_targets['test'])
                if len(X_test_orch) > 0:
                    test_metrics = self.orchestrator.evaluate(X_test_orch, y_test_orch, "test")
                    final_results['orchestrator'] = test_metrics
        
        return final_results
    
    def _save_all_artifacts(self, run_name: str) -> None:
        """Save all models, results, and artifacts."""
        # Create output directories
        models_dir = Path(self.config['output']['models_dir']) / run_name
        results_dir = Path(self.config['output']['results_dir']) / run_name
        
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving artifacts to: {models_dir}")
        
        # Save directional models
        for target_col, model_info in self.direction_models.items():
            model_name = f"direction_{target_col.replace('target_', '')}"
            model_info['model'].save_model(str(models_dir), model_name)
        
        # Save orchestrator
        if hasattr(self.orchestrator, 'model') and self.orchestrator.model is not None:
            self.orchestrator.save_model(str(models_dir), "orchestrator")

        # Save strength/duration
        if self.strength_duration_model is not None:
            self.strength_duration_model.save(str(models_dir), "strength_duration")
        
        # Save training results
        results_file = results_dir / "training_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(self.training_results, f, default_flow_style=False)
        
        # Save configuration
        config_file = models_dir / "training_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Generate summary report
        self._generate_summary_report(results_dir)
        
        logger.info(f"All artifacts saved successfully")
    
    def _generate_summary_report(self, results_dir: Path) -> None:
        """Generate a summary report of training results."""
        report_file = results_dir / "summary.md"
        
        with open(report_file, 'w') as f:
            f.write("# SignaMentis AI Trading System - Training Summary\n\n")
            f.write(f"**Run Name:** {self.config['output']['run_name']}\n")
            f.write(f"**Training Date:** {datetime.now().isoformat()}\n\n")
            
            f.write("## Directional Models\n\n")
            for target_col, model_info in self.direction_models.items():
                f.write(f"### {target_col}\n")
                if 'validation_metrics' in model_info:
                    val_metrics = model_info['validation_metrics']
                    f.write(f"- **Validation:** Accuracy: {val_metrics.get('accuracy', 'N/A'):.4f}, "
                           f"High-Conf Precision: {val_metrics.get('high_conf_precision', 'N/A'):.4f}\n")
                
                if 'test_metrics' in model_info and model_info['test_metrics']:
                    test_metrics = model_info['test_metrics']
                    f.write(f"- **Test:** Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}, "
                           f"High-Conf Precision: {test_metrics.get('high_conf_precision', 'N/A'):.4f}\n")
                f.write("\n")
            
            f.write("## Orchestrator\n\n")
            if 'orchestrator' in self.training_results:
                orch_results = self.training_results['orchestrator']
                if 'validation_metrics' in orch_results:
                    val_metrics = orch_results['validation_metrics']
                    f.write(f"- **Validation:** Precision: {val_metrics.get('precision', 'N/A'):.4f}, Recall: {val_metrics.get('recall', 'N/A'):.4f}, Coverage: {val_metrics.get('high_conf_coverage', 'N/A'):.4f}\n")
                    if 'tau' in orch_results:
                        f.write(f"- **τ:** {orch_results['tau']:.4f}\n")
                    if 'label_distribution' in orch_results:
                        f.write(f"- **Label distribution:** {orch_results['label_distribution']}\n")
                if 'test_metrics' in orch_results and orch_results['test_metrics']:
                    test_metrics = orch_results['test_metrics']
                    f.write(f"- **Test:** Precision: {test_metrics.get('precision', 'N/A'):.4f}, Recall: {test_metrics.get('recall', 'N/A'):.4f}, Coverage: {test_metrics.get('high_conf_coverage', 'N/A'):.4f}\n")
            
            f.write("\n## Configuration\n\n")
            f.write(f"- **Seed:** {self.config['general']['seed']}\n")
            f.write(f"- **N Jobs:** {self.config['general']['n_jobs']}\n")
            f.write(f"- **Horizon Bars:** {self.config['general']['horizon_bars']}\n")
            # Strength/Duration metrics if available
            if 'strength_duration' in self.training_results and 'validation_metrics' in self.training_results['strength_duration']:
                sd = self.training_results['strength_duration']['validation_metrics']
                f.write("\n## Strength/Duration\n\n")
                f.write(f"- **MAE duration (bars):** {sd.get('duration_bars_mae','N/A')}\n")
                f.write(f"- **Macro-F1 strength:** {sd.get('strength_f1','N/A')}\n")
        
        logger.info(f"Summary report generated: {report_file}")


def main():
    """Main entry point for multi-module training."""
    parser = argparse.ArgumentParser(description="Train multi-module AI trading system")
    parser.add_argument('--features', type=str, 
                       default='data/processed/multi_timeframe_features_3years_20250823_104329.csv',
                       help='Path to features CSV file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--n_jobs', type=int, default=4,
                       help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.config:
        config_path = args.config
    else:
        config_path = None
    
    # Create trainer
    trainer = MultiModuleTrainer(config_path)
    
    # Update config with command line arguments
    if args.features:
        trainer.config['data']['features_csv'] = args.features
    if args.n_jobs:
        trainer.config['general']['n_jobs'] = args.n_jobs
    
    # Run training pipeline
    try:
        results = trainer.run_training_pipeline()
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Run Name: {results['run_name']}")
        print(f"Directional Models: {len(results['direction_models'])}")
        print(f"Orchestrator: {'Trained' if results['orchestrator'] else 'Not trained'}")
        print(f"Results saved to: {trainer.config['output']['results_dir']}/{results['run_name']}")
        print(f"Models saved to: {trainer.config['output']['models_dir']}/{results['run_name']}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
