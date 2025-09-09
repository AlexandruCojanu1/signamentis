#!/usr/bin/env python3
"""
Orchestrator for SignaMentis AI Trading System.
Meta-learner that combines predictions from all modules for final trade decisions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Orchestrator:
    """Multiclass gating orchestrator: {use_h1, use_h3, use_h5, abstain}."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Orchestrator with configuration."""
        self.config = config or self._default_config()
        self.model = None
        self.calibrator = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.high_confidence_threshold = None
        self.trade_thresholds = None
        
    def _default_config(self) -> Dict:
        """Default configuration for orchestrator."""
        return {
            'algorithm': 'logistic',  # multinomial logistic
            'min_precision_val': 0.97,
            'min_recall_val': 0.25,
            'n_jobs': 4,
            'random_state': 42,
            'use_only_model_outputs': True,
            'safe_feature_whitelist': [],
            'risk_management': {
                'max_position_size': 0.1,  # 10% of portfolio
                'stop_loss_atr': 2.0,      # 2 ATR stop loss
                'take_profit_atr': 3.0,    # 3 ATR take profit
                'max_drawdown': 0.05       # 5% max drawdown
            }
        }
    
    def prepare_orchestrator_features(self, predictions: Dict[str, Any], 
                                    context_features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the orchestrator meta-learner.
        
        Args:
            predictions: Dictionary containing predictions from all modules
            context_features: Context features (ATR, session flags, etc.)
            
        Returns:
            DataFrame with orchestrator features
        """
        logger.info("Preparing orchestrator features...")
        
        # Extract predictions from each module
        features = {}
        
        # Directional predictions (required): p1, p3, p5 calibrated probs
        p1 = predictions.get('p1', predictions.get('p_up_1', 0.5))
        p3 = predictions.get('p3', predictions.get('p_up_3', 0.5))
        p5 = predictions.get('p5', predictions.get('p_up_5', 0.5))
        features.update({'p1': p1, 'p3': p3, 'p5': p5})
        # Derivatives
        p_arr = np.array([p1, p3, p5], dtype=float)
        p_sorted = np.sort(p_arr)[::-1]
        pmax = float(p_sorted[0])
        p2nd = float(p_sorted[1]) if len(p_sorted) > 1 else 0.0
        argmax_h = int(np.argmax(p_arr))  # 0->h1,1->h3,2->h5
        p_spread = pmax - p2nd
        agreement = int(((p1 >= 0.5) == (p3 >= 0.5) == (p5 >= 0.5)))
        # entropy over {p1,p3,p5} after normalization
        eps = 1e-9
        norm = p_arr.sum() + eps
        q = p_arr / norm
        entropy = float(-(q * np.log(q + eps)).sum())
        features.update({'pmax': pmax, 'argmax_h': argmax_h, 'p_spread': p_spread, 'agreement': agreement, 'entropy': entropy})
        
        # Strength predictions
        if 'strength' in predictions:
            strength_preds = predictions['strength']
            features.update({
                'p_weak': strength_preds.get('p_weak', 0.33),
                'p_medium': strength_preds.get('p_medium', 0.33),
                'p_strong': strength_preds.get('p_strong', 0.33),
                'strength_confidence': strength_preds.get('confidence', 0.5)
            })
        
        # Duration predictions
        if 'duration' in predictions:
            duration_preds = predictions['duration']
            features.update({
                'p_duration_0_2': duration_preds.get('p_bin_0', 0.25),
                'p_duration_3_6': duration_preds.get('p_bin_1', 0.25),
                'p_duration_7_12': duration_preds.get('p_bin_2', 0.25),
                'p_duration_not_reached': duration_preds.get('p_bin_3', 0.25),
                'expected_duration': duration_preds.get('expected_duration', 6.0),
                'duration_confidence': duration_preds.get('confidence', 0.5)
            })
        
        # Price target predictions
        if 'price_targets' in predictions:
            price_preds = predictions['price_targets']
            features.update({
                'q10_return': price_preds.get('q10', 0.0),
                'q50_return': price_preds.get('q50', 0.0),
                'q90_return': price_preds.get('q90', 0.0),
                'p_hit_0_5_atr': price_preds.get('p_hit_0_5_atr', 0.5),
                'p_hit_1_0_atr': price_preds.get('p_hit_1_0_atr', 0.5),
                'p_hit_1_5_atr': price_preds.get('p_hit_1_5_atr', 0.5),
                'price_targets_confidence': price_preds.get('confidence', 0.5)
            })
        
        # Context features (safe whitelist only)
        if context_features is not None:
            safe_list = self.config.get('safe_feature_whitelist', [])
            if self.config.get('use_only_model_outputs', False):
                safe_list = []
            if safe_list:
                existing = [c for c in safe_list if c in context_features.columns]
                for col in existing:
                    features[f'context_{col}'] = context_features[col]
        
        # Create feature DataFrame
        feature_df = pd.DataFrame([features])
        
        # No additional derived features beyond core derivatives
        
        # Store feature columns
        self.feature_columns = list(feature_df.columns)
        
        logger.info(f"Orchestrator features prepared: {len(self.feature_columns)} features")
        
        return feature_df
    
    def _calculate_derived_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features for the orchestrator."""
        # Directional agreement
        if all(col in feature_df.columns for col in ['p_up_1', 'p_up_3', 'p_up_5']):
            feature_df['direction_agreement'] = (
                (feature_df['p_up_1'] > 0.5).astype(int) + 
                (feature_df['p_up_3'] > 0.5).astype(int) + 
                (feature_df['p_up_5'] > 0.5).astype(int)
            ) / 3
        
        # Overall confidence
        confidence_cols = [col for col in feature_df.columns if 'confidence' in col]
        if confidence_cols:
            feature_df['overall_confidence'] = feature_df[confidence_cols].mean(axis=1)
        
        # Risk-adjusted expected return
        if all(col in feature_df.columns for col in ['q50_return', 'atr_normalized']):
            feature_df['risk_adjusted_return'] = feature_df['q50_return'] / feature_df['atr_normalized']
        
        # Strength-duration interaction
        if all(col in feature_df.columns for col in ['p_strong', 'expected_duration']):
            feature_df['strength_duration_score'] = feature_df['p_strong'] * (1 / (1 + feature_df['expected_duration']))
        
        return feature_df
    
    def create_training_data(self, all_predictions: List[Dict], 
                            context_features: pd.DataFrame,
                            trade_outcomes: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training data for the orchestrator.
        
        Args:
            all_predictions: List of prediction dictionaries for each timestamp
            context_features: Context features for each timestamp
            trade_outcomes: Binary outcomes (1 = profitable trade, 0 = loss)
            
        Returns:
            Tuple of (features, targets)
        """
        logger.info("Creating orchestrator training data...")
        
        features_list = []
        targets = []
        
        for i, (pred_dict, outcome) in enumerate(zip(all_predictions, trade_outcomes)):
            try:
                # Get context features for this timestamp
                if context_features is not None and i < len(context_features):
                    context_row = context_features.iloc[i:i+1]
                else:
                    context_row = None
                
                # Prepare features
                features = self.prepare_orchestrator_features(pred_dict, context_row)
                features_list.append(features)
                targets.append(outcome)
                
            except Exception as e:
                logger.warning(f"Failed to process timestamp {i}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid features generated")
        
        # Combine all features
        X = pd.concat(features_list, ignore_index=True)
        y = pd.Series(targets)
        
        # Remove rows with NaN values
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Training data created: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def train_orchestrator(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the orchestrator meta-learner.
        
        Args:
            X: Feature matrix
            y: Target vector (trade outcomes)
        """
        logger.info("Training orchestrator meta-learner...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Choose algorithm (multiclass)
        if self.config['algorithm'] == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=4,
                metric='multi_logloss',
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=63,
                random_state=self.config.get('random_state', 42),
                n_jobs=self.config.get('n_jobs', 4),
                verbose=-1
            )
        else:
            # Multinomial Logistic Regression
            self.model = LogisticRegression(
                random_state=self.config.get('random_state', 42),
                max_iter=1000,
                n_jobs=self.config.get('n_jobs', 4),
                multi_class='multinomial',
                class_weight='balanced'
            )
        
        # Train model
        self.model.fit(X_scaled, y)
        
        logger.info("Orchestrator training completed")
    
    def calibrate_probabilities(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Calibrate orchestrator probabilities using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info("Calibrating orchestrator probabilities...")
        
        if self.model is None:
            raise ValueError("Must train orchestrator first")
        
        # Scale validation features
        X_val_scaled = self.scaler.transform(X_val)
        
        # For multiclass gating, we use the max non-abstain probability as confidence
        probs = self.model.predict_proba(X_val_scaled)
        # Class order assumed: 0:use_h1,1:use_h3,2:use_h5,3:abstain
        max_non_abstain = probs[:, :3].max(axis=1)
        self._find_high_confidence_threshold(max_non_abstain, (y_val != 3).astype(int))
        
        logger.info("Orchestrator probability calibration completed")
    
    def _find_high_confidence_threshold(self, calibrated_probs: np.ndarray, y_true: pd.Series) -> None:
        """Find threshold for high-confidence predictions (≥97% precision)."""
        from sklearn.metrics import precision_score, recall_score
        
        # Calculate precision at different thresholds
        thresholds = []
        precisions = []
        recalls = []
        coverages = []
        
        for threshold in np.arange(0.5, 0.99, 0.01):
            high_conf_mask = calibrated_probs >= threshold
            if high_conf_mask.sum() > 0:
                y_pred_high_conf = (calibrated_probs >= threshold).astype(int)
                y_true_high_conf = y_true[high_conf_mask]
                y_pred_high_conf = y_pred_high_conf[high_conf_mask]
                
                precision = precision_score(y_true_high_conf, y_pred_high_conf, zero_division=0)
                recall = recall_score(y_true_high_conf, y_pred_high_conf, zero_division=0)
                coverage = high_conf_mask.mean()
                
                thresholds.append(threshold)
                precisions.append(precision)
                recalls.append(recall)
                coverages.append(coverage)
        
        # Find threshold meeting precision requirement
        target_precision = self.config['min_precision_val']
        target_recall = self.config['min_recall_val']
        
        valid_thresholds = []
        for i, (thresh, prec, rec, cov) in enumerate(zip(thresholds, precisions, recalls, coverages)):
            if prec >= target_precision and rec >= target_recall:
                valid_thresholds.append((thresh, prec, rec, cov))
        
        if valid_thresholds:
            # Choose threshold with highest coverage
            best_threshold, best_prec, best_rec, best_cov = max(valid_thresholds, key=lambda x: x[3])
            
            self.high_confidence_threshold = best_threshold
            
            logger.info(f"High-confidence threshold: {best_threshold:.3f}")
            logger.info(f"Precision: {best_prec:.3f}, Recall: {best_rec:.3f}, Coverage: {best_cov:.3f}")
        else:
            logger.warning(f"Could not find threshold meeting requirements: Precision ≥ {target_precision}, Recall ≥ {target_recall}")
            # Use default threshold
            self.high_confidence_threshold = 0.8
    
    def _fallback_decision(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rules when gating target not achieved or features missing."""
        p1 = predictions.get('p1', predictions.get('p_up_1', 0.5))
        p3 = predictions.get('p3', predictions.get('p_up_3', 0.5))
        p5 = predictions.get('p5', predictions.get('p_up_5', 0.5))
        agree = ((p1 >= 0.5) == (p3 >= 0.5) == (p5 >= 0.5))
        pmax = max(p1, p3, p5)
        if agree and pmax >= 0.90:
            idx = int(np.argmax([p1, p3, p5]))
            action = ['use_h1','use_h3','use_h5'][idx]
            return {'decision': action, 'confidence_score': pmax, 'fallback': True}
        if p1 >= 0.95:
            return {'decision': 'use_h1', 'confidence_score': p1, 'fallback': True}
        return {'decision': 'abstain', 'confidence_score': pmax, 'fallback': True}

    def make_trade_decision(self, predictions: Dict[str, Any], 
                           context_features: pd.DataFrame) -> Dict[str, Any]:
        """Make trade decision using trained orchestrator or fallback rules."""
        # Prepare features
        features = self.prepare_orchestrator_features(predictions, context_features)
        
        # Predict class
        features_scaled = self.scaler.transform(features)
        probs = self.model.predict_proba(features_scaled)[0]
        max_non_abstain = float(probs[:3].max())
        chosen = int(np.argmax(probs[:3]))  # 0,1,2 correspond to h1,h3,h5
        
        # Thresholding - enforce strict precision requirement
        tau = self.high_confidence_threshold or 0.9
        if max_non_abstain >= tau:
            action = ['use_h1','use_h3','use_h5'][chosen]
            return {'decision': action, 'confidence_score': max_non_abstain, 'fallback': False}
        else:
            # Fallback rules when threshold not met
            return self._fallback_decision(predictions)
    
    def _calculate_expected_edge(self, predictions: Dict[str, Any], 
                                context_features: pd.DataFrame, 
                                direction: int) -> float:
        """Calculate expected edge for the trade."""
        expected_edge = 0.0
        
        # Get price target predictions
        if 'price_targets' in predictions:
            price_preds = predictions['price_targets']
            q50_return = price_preds.get('q50', 0.0)
            
            # Adjust for direction
            if direction == 1:  # Long
                expected_edge = q50_return
            elif direction == -1:  # Short
                expected_edge = -q50_return
        
        # Adjust for strength
        if 'strength' in predictions:
            strength_preds = predictions['strength']
            p_strong = strength_preds.get('p_strong', 0.33)
            expected_edge *= (0.5 + p_strong * 0.5)  # Boost for strong moves
        
        # Adjust for duration
        if 'duration' in predictions:
            duration_preds = predictions['duration']
            expected_duration = duration_preds.get('expected_duration', 6.0)
            # Prefer shorter duration trades
            duration_factor = 1.0 / (1.0 + expected_duration / 12.0)
            expected_edge *= duration_factor
        
        return expected_edge
    
    def _calculate_risk_metrics(self, predictions: Dict[str, Any], 
                               context_features: pd.DataFrame, 
                               direction: int) -> Dict[str, float]:
        """Calculate risk metrics for the trade."""
        risk_metrics = {}
        
        # Get ATR for position sizing
        if context_features is not None and 'm5_atr' in context_features.columns:
            atr = context_features['m5_atr'].iloc[0]
            close = context_features['current_close'].iloc[0]
            
            # Stop loss and take profit levels
            sl_atr = self.config['risk_management']['stop_loss_atr']
            tp_atr = self.config['risk_management']['take_profit_atr']
            
            if direction == 1:  # Long
                stop_loss = close - sl_atr * atr
                take_profit = close + tp_atr * atr
            elif direction == -1:  # Short
                stop_loss = close + sl_atr * atr
                take_profit = close - tp_atr * atr
            else:
                stop_loss = take_profit = close
            
            risk_metrics.update({
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': tp_atr / sl_atr,
                'atr': atr,
                'position_size_atr': self.config['risk_management']['max_position_size']
            })
        
        # Volatility adjustment
        if 'price_targets' in predictions:
            price_preds = predictions['price_targets']
            q10 = price_preds.get('q10', 0.0)
            q90 = price_preds.get('q90', 0.0)
            
            risk_metrics['volatility_range'] = abs(q90 - q10)
        
        return risk_metrics
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, split_name: str = "test") -> Dict[str, float]:
        """
        Evaluate orchestrator performance.
        
        Args:
            X: Feature matrix
            y: True targets
            split_name: Name of the data split
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating orchestrator on {split_name} set...")
        
        # Get predictions
        X_scaled = self.scaler.transform(X)
        if self.calibrator is not None:
            predictions = self.calibrator.predict(X_scaled)
            proba = self.calibrator.predict_proba(X_scaled)
        else:
            predictions = self.model.predict(X_scaled)
            proba = self.model.predict_proba(X_scaled)
        # Confidence: max non-abstain probability (classes 0..2)
        confidence_scores = proba[:, :3].max(axis=1)
        
        # Calculate metrics
        # Multiclass-friendly metrics
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='macro', zero_division=0),
            'recall': recall_score(y, predictions, average='macro', zero_division=0),
            'f1': f1_score(y, predictions, average='macro', zero_division=0)
        }
        # Multiclass ROC-AUC (one-vs-rest) if possible
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y, proba, multi_class='ovr')
        except Exception:
            pass
        
        # High-confidence metrics
        if self.high_confidence_threshold is not None:
            high_conf_mask = confidence_scores >= self.high_confidence_threshold
            
            if high_conf_mask.sum() > 0:
                y_high_conf = y[high_conf_mask]
                pred_high_conf = predictions[high_conf_mask]
                conf_high_conf = confidence_scores[high_conf_mask]
                
                high_conf_metrics = {
                    'high_conf_precision': precision_score(y_high_conf, pred_high_conf, average='macro', zero_division=0),
                    'high_conf_recall': recall_score(y_high_conf, pred_high_conf, average='macro', zero_division=0),
                    'high_conf_coverage': high_conf_mask.mean(),
                    'high_conf_count': int(high_conf_mask.sum())
                }
                metrics.update(high_conf_metrics)
        
        # Log metrics
        logger.info(f"{split_name.capitalize()} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, output_dir: str, model_name: str) -> None:
        """Save trained orchestrator and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.calibrator is not None:
            joblib.dump(self.calibrator, output_path / f"{model_name}_calibrated.joblib")
        else:
            joblib.dump(self.model, output_path / f"{model_name}.joblib")
        
        # Save scaler
        joblib.dump(self.scaler, output_path / f"{model_name}_scaler.joblib")
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'algorithm': self.config['algorithm'],
            'feature_columns': self.feature_columns,
            'high_confidence_threshold': self.high_confidence_threshold,
            'training_date': datetime.now().isoformat(),
            'config': self.config
        }
        
        with open(output_path / f"{model_name}_metadata.yaml", 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        logger.info(f"Orchestrator saved to: {output_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load trained orchestrator from file."""
        model_path = Path(model_path)
        
        # Load model
        if (model_path.parent / f"{model_path.stem}_calibrated.joblib").exists():
            self.calibrator = joblib.load(model_path.parent / f"{model_path.stem}_calibrated.joblib")
            self.model = self.calibrator.estimator_
        else:
            self.model = joblib.load(model_path)
        
        # Load scaler
        self.scaler = joblib.load(model_path.parent / f"{model_path.stem}_scaler.joblib")
        
        # Load metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.high_confidence_threshold = metadata['high_confidence_threshold']
        
        logger.info(f"Orchestrator loaded from: {model_path}")


if __name__ == "__main__":
    # Test the Orchestrator class
    orchestrator = Orchestrator()
    
    # Create sample predictions
    sample_predictions = {
        'direction': {
            'p_up_1': 0.8,
            'p_up_3': 0.7,
            'p_up_5': 0.6,
            'confidence_1': 0.9,
            'confidence_3': 0.8,
            'confidence_5': 0.7
        },
        'strength': {
            'p_weak': 0.1,
            'p_medium': 0.3,
            'p_strong': 0.6,
            'confidence': 0.8
        },
        'duration': {
            'p_bin_0': 0.2,
            'p_bin_1': 0.5,
            'p_bin_2': 0.2,
            'p_bin_3': 0.1,
            'expected_duration': 4.0,
            'confidence': 0.7
        },
        'price_targets': {
            'q10': -0.002,
            'q50': 0.005,
            'q90': 0.012,
            'p_hit_0_5_atr': 0.8,
            'p_hit_1_0_atr': 0.6,
            'p_hit_1_5_atr': 0.4,
            'confidence': 0.75
        }
    }
    
    # Create sample context features
    sample_context = pd.DataFrame({
        'current_close': [1800.0],
        'm5_atr': [2.5],
        'is_london_session': [1],
        'is_ny_session': [0]
    })
    
    print("Sample data created successfully")
    
    # Test feature preparation
    features = orchestrator.prepare_orchestrator_features(sample_predictions, sample_context)
    print(f"Features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    
    print("Orchestrator test completed successfully!")
