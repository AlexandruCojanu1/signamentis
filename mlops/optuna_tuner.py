#!/usr/bin/env python3
"""
SignaMentis - Optuna Hyperparameter Tuner

This module implements hyperparameter optimization using Optuna for all AI models
in the SignaMentis trading system.

Author: SignaMentis Team
Version: 2.0.0
"""

import optuna
import logging
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import mlflow
import mlflow.pytorch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Optimization objectives."""
    MINIMIZE_LOSS = "minimize_loss"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_PROFIT_FACTOR = "maximize_profit_factor"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_WIN_RATE = "maximize_win_rate"

@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space for a model."""
    model_name: str
    parameters: Dict[str, Any]
    categorical_choices: Optional[Dict[str, List[Any]]] = None
    constraints: Optional[List[Callable]] = None

@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    trial_number: int
    best_params: Dict[str, Any]
    best_value: float
    optimization_history: List[float]
    study_name: str
    optimization_time: float
    n_trials: int
    model_name: str

class OptunaTuner:
    """
    Optuna-based hyperparameter tuner for SignaMentis AI models.
    
    Supports:
    - Multiple optimization objectives
    - Custom parameter spaces
    - MLflow integration
    - Early stopping and pruning
    - Multi-objective optimization
    - Study persistence and loading
    """
    
    def __init__(
        self,
        study_name: str = "signa_mentis_optimization",
        storage: Optional[str] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        mlflow_tracking: bool = True
    ):
        """
        Initialize Optuna tuner.
        
        Args:
            study_name: Name of the optimization study
            storage: Optuna storage backend (SQLite, PostgreSQL, etc.)
            sampler: Optuna sampler (TPE, Random, etc.)
            pruner: Optuna pruner (MedianPruner, etc.)
            mlflow_tracking: Enable MLflow integration
        """
        self.study_name = study_name
        self.storage = storage
        self.sampler = sampler or optuna.samplers.TPESampler(seed=42)
        self.pruner = pruner or optuna.pruners.MedianPruner()
        self.mlflow_tracking = mlflow_tracking
        
        # Study object
        self.study = None
        
        # Optimization history
        self.optimization_history = []
        
        # Model configurations
        self.model_configs = {}
        
        # Initialize study
        self._create_study()
        
        logger.info(f"🚀 Optuna tuner initialized: {study_name}")
    
    def _create_study(self):
        """Create or load Optuna study."""
        try:
            if self.storage:
                self.study = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage,
                    sampler=self.sampler,
                    pruner=self.pruner,
                    load_if_exists=True
                )
            else:
                self.study = optuna.create_study(
                    study_name=self.study_name,
                    sampler=self.sampler,
                    pruner=self.pruner
                )
            
            logger.info(f"✅ Study '{self.study_name}' created/loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create study: {e}")
            raise
    
    def register_model_config(self, config: HyperparameterSpace):
        """Register a model configuration for optimization."""
        self.model_configs[config.model_name] = config
        logger.info(f"📝 Registered model config: {config.model_name}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            model_name: Name of the model
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not registered")
        
        config = self.model_configs[model_name]
        params = {}
        
        for param_name, param_config in config.parameters.items():
            if param_config['type'] == 'int':
                if 'log' in param_config and param_config['log']:
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=True
                    )
                else:
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
            
            elif param_config['type'] == 'float':
                if 'log' in param_config and param_config['log']:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
            
            elif param_config['type'] == 'categorical':
                if config.categorical_choices and param_name in config.categorical_choices:
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        config.categorical_choices[param_name]
                    )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
        
        return params
    
    def optimize(
        self,
        objective_func: Callable,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        model_name: str = "default"
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_func: Objective function to optimize
            n_trials: Number of trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            model_name: Name of the model being optimized
            
        Returns:
            Optimization result
        """
        try:
            logger.info(f"🎯 Starting optimization for {model_name}: {n_trials} trials")
            
            start_time = time.time()
            
            # Create objective wrapper
            def objective(trial):
                # Suggest hyperparameters
                params = self.suggest_hyperparameters(trial, model_name)
                
                # Log to MLflow if enabled
                if self.mlflow_tracking:
                    mlflow.log_params(params)
                    mlflow.set_tag("model_name", model_name)
                    mlflow.set_tag("trial_number", trial.number)
                
                # Run objective function
                result = objective_func(params)
                
                # Log metrics to MLflow
                if self.mlflow_tracking and isinstance(result, dict):
                    for metric_name, metric_value in result.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(metric_name, metric_value)
                
                # Return primary objective (first value)
                if isinstance(result, (int, float)):
                    return result
                elif isinstance(result, dict):
                    # Return first metric as primary objective
                    return list(result.values())[0]
                else:
                    return result
            
            # Run optimization
            self.study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs
            )
            
            optimization_time = time.time() - start_time
            
            # Create result
            result = OptimizationResult(
                trial_number=len(self.study.trials),
                best_params=self.study.best_params,
                best_value=self.study.best_value,
                optimization_history=[trial.value for trial in self.study.trials if trial.value is not None],
                study_name=self.study_name,
                optimization_time=optimization_time,
                n_trials=n_trials,
                model_name=model_name
            )
            
            self.optimization_history.append(result)
            
            logger.info(f"✅ Optimization completed: {result.best_value:.4f}")
            logger.info(f"   Best parameters: {result.best_params}")
            logger.info(f"   Time: {optimization_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            raise
    
    def optimize_model(
        self,
        model_class: type,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        model_name: str = "default",
        n_trials: int = 100,
        timeout: Optional[int] = None,
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_LOSS,
        cv_folds: int = 5
    ) -> OptimizationResult:
        """
        Optimize a specific model class.
        
        Args:
            model_class: Model class to optimize
            train_data: Training data (X, y)
            val_data: Validation data (X, y)
            model_name: Name of the model
            n_trials: Number of trials
            timeout: Timeout in seconds
            objective: Optimization objective
            cv_folds: Cross-validation folds
            
        Returns:
            Optimization result
        """
        def objective_func(params):
            try:
                # Create model with suggested parameters
                model = model_class(**params)
                
                # Train and evaluate
                if val_data is not None:
                    # Use validation data
                    model.fit(train_data[0], train_data[1])
                    y_pred = model.predict(val_data[0])
                    
                    if objective == OptimizationObjective.MINIMIZE_LOSS:
                        return mean_squared_error(val_data[1], y_pred)
                    elif objective == OptimizationObjective.MAXIMIZE_ACCURACY:
                        return -accuracy_score(val_data[1], y_pred)  # Negative for minimization
                    elif objective == OptimizationObjective.MAXIMIZE_F1:
                        return -f1_score(val_data[1], y_pred, average='weighted')
                else:
                    # Use cross-validation
                    scores = cross_val_score(
                        model, train_data[0], train_data[1], 
                        cv=cv_folds, scoring='neg_mean_squared_error'
                    )
                    return -np.mean(scores)  # Convert to positive for minimization
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')  # Return worst possible value
        
        return self.optimize(
            objective_func=objective_func,
            n_trials=n_trials,
            timeout=timeout,
            model_name=model_name
        )
    
    def optimize_trading_strategy(
        self,
        strategy_func: Callable,
        historical_data: pd.DataFrame,
        model_name: str = "trading_strategy",
        n_trials: int = 100,
        timeout: Optional[int] = None,
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE
    ) -> OptimizationResult:
        """
        Optimize trading strategy parameters.
        
        Args:
            strategy_func: Strategy function to optimize
            historical_data: Historical market data
            model_name: Name of the strategy
            n_trials: Number of trials
            timeout: Timeout in seconds
            objective: Optimization objective
            
        Returns:
            Optimization result
        """
        def objective_func(params):
            try:
                # Run strategy with parameters
                result = strategy_func(historical_data, params)
                
                # Extract objective metric
                if objective == OptimizationObjective.MAXIMIZE_SHARPE:
                    return -result.get('sharpe_ratio', -999)  # Negative for minimization
                elif objective == OptimizationObjective.MAXIMIZE_PROFIT_FACTOR:
                    return -result.get('profit_factor', -999)
                elif objective == OptimizationObjective.MINIMIZE_DRAWDOWN:
                    return result.get('max_drawdown', 999)
                elif objective == OptimizationObjective.MAXIMIZE_WIN_RATE:
                    return -result.get('win_rate', -999)
                else:
                    return -result.get('total_return', -999)
                
            except Exception as e:
                logger.warning(f"Strategy trial failed: {e}")
                return float('inf')
        
        return self.optimize(
            objective_func=objective_func,
            n_trials=n_trials,
            timeout=timeout,
            model_name=model_name
        )
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from study."""
        return self.study.best_params if self.study else {}
    
    def get_best_value(self) -> float:
        """Get best objective value from study."""
        return self.study.best_value if self.study else float('inf')
    
    def get_optimization_history(self) -> List[float]:
        """Get optimization history."""
        if not self.study:
            return []
        
        return [trial.value for trial in self.study.trials if trial.value is not None]
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            history = self.get_optimization_history()
            if not history:
                logger.warning("No optimization history to plot")
                return
            
            plt.figure(figsize=(12, 8))
            
            # Plot optimization history
            plt.subplot(2, 2, 1)
            plt.plot(history)
            plt.title('Optimization History')
            plt.xlabel('Trial')
            plt.ylabel('Objective Value')
            plt.grid(True)
            
            # Plot parameter importance
            if len(history) > 1:
                plt.subplot(2, 2, 2)
                importance = optuna.importance.get_param_importances(self.study)
                if importance:
                    plt.bar(range(len(importance)), list(importance.values()))
                    plt.xticks(range(len(importance)), list(importance.keys()), rotation=45)
                    plt.title('Parameter Importance')
                    plt.ylabel('Importance')
            
            # Plot parameter relationships
            if len(history) > 1:
                plt.subplot(2, 2, 3)
                optuna.visualization.matplotlib.plot_param_importances(self.study)
                plt.title('Parameter Importances')
            
            # Plot optimization surface
            if len(history) > 1:
                plt.subplot(2, 2, 4)
                optuna.visualization.matplotlib.plot_optimization_history(self.study)
                plt.title('Optimization History')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization plots saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting optimization history: {e}")
    
    def save_study(self, filepath: str):
        """Save study to file."""
        try:
            if self.study:
                with open(filepath, 'wb') as f:
                    pickle.dump(self.study, f)
                logger.info(f"✅ Study saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save study: {e}")
    
    def load_study(self, filepath: str):
        """Load study from file."""
        try:
            with open(filepath, 'rb') as f:
                self.study = pickle.load(f)
            logger.info(f"✅ Study loaded from {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to load study: {e}")
    
    def export_results(self, filepath: str):
        """Export optimization results to JSON."""
        try:
            results = {
                'study_name': self.study_name,
                'best_params': self.get_best_params(),
                'best_value': self.get_best_value(),
                'optimization_history': self.get_optimization_history(),
                'n_trials': len(self.study.trials) if self.study else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ Results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export results: {e}")

# Predefined hyperparameter spaces for SignaMentis models
def get_bilstm_hyperparameter_space() -> HyperparameterSpace:
    """Get BiLSTM hyperparameter space."""
    return HyperparameterSpace(
        model_name="bilstm",
        parameters={
            'hidden_size': {'type': 'int', 'low': 32, 'high': 256},
            'num_layers': {'type': 'int', 'low': 1, 'high': 4},
            'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
            'sequence_length': {'type': 'int', 'low': 10, 'high': 100}
        }
    )

def get_gru_hyperparameter_space() -> HyperparameterSpace:
    """Get GRU hyperparameter space."""
    return HyperparameterSpace(
        model_name="gru",
        parameters={
            'hidden_size': {'type': 'int', 'low': 32, 'high': 256},
            'num_layers': {'type': 'int', 'low': 1, 'high': 4},
            'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
            'sequence_length': {'type': 'int', 'low': 10, 'high': 100}
        }
    )

def get_transformer_hyperparameter_space() -> HyperparameterSpace:
    """Get Transformer hyperparameter space."""
    return HyperparameterSpace(
        model_name="transformer",
        parameters={
            'd_model': {'type': 'int', 'low': 64, 'high': 512},
            'nhead': {'type': 'int', 'low': 4, 'high': 16},
            'num_layers': {'type': 'int', 'low': 2, 'high': 8},
            'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]}
        }
    )

# Example usage
if __name__ == "__main__":
    def main():
        print("🚀 Optuna Hyperparameter Tuner Example")
        print("=" * 50)
        
        # Create tuner
        tuner = OptunaTuner(study_name="signa_mentis_example")
        
        # Register model configurations
        tuner.register_model_config(get_bilstm_hyperparameter_space())
        tuner.register_model_config(get_gru_hyperparameter_space())
        tuner.register_model_config(get_transformer_hyperparameter_space())
        
        print("✅ Model configurations registered")
        print("📊 Available models:")
        for model_name in tuner.model_configs.keys():
            print(f"   - {model_name}")
        
        print("\n🎯 Ready for hyperparameter optimization!")
        print("Use tuner.optimize() or tuner.optimize_model() to start optimization")
    
    # Run example
    main()
