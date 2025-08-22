"""
SignaMentis Ensemble Module

This module implements an ensemble of multiple AI models for XAU/USD prediction.
It combines predictions from different models using various aggregation methods:
- Weighted averaging
- Voting
- Stacking
- Dynamic weighting based on performance

Author: SignaMentis Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Container for model predictions."""
    model_name: str
    direction_probabilities: np.ndarray  # [P(down), P(up)]
    price_target: np.ndarray
    confidence: np.ndarray
    timestamp: datetime
    model_weight: float = 1.0
    performance_score: float = 0.0


@dataclass
class EnsemblePrediction:
    """Container for ensemble predictions."""
    direction_probabilities: np.ndarray  # [P(down), P(up)]
    predicted_direction: int  # 0 for down, 1 for up
    price_target: float
    confidence: float
    ensemble_confidence: float
    model_agreement: float
    timestamp: datetime
    individual_predictions: Dict[str, ModelPrediction]


class EnsembleAggregator:
    """
    Ensemble aggregator that combines predictions from multiple AI models.
    
    Supports multiple aggregation methods:
    - Weighted averaging
    - Voting
    - Stacking
    - Dynamic weighting based on recent performance
    """
    
    def __init__(self, 
                 method: str = "weighted_average",
                 dynamic_weighting: bool = True,
                 weight_update_frequency: str = "1h",
                 min_model_weight: float = 0.05,
                 max_model_weight: float = 0.4,
                 performance_lookback: int = 100):
        """
        Initialize the ensemble aggregator.
        
        Args:
            method: Aggregation method ("weighted_average", "voting", "stacking")
            dynamic_weighting: Whether to use dynamic weights based on performance
            weight_update_frequency: How often to update weights ("1h", "1d", "1w")
            min_model_weight: Minimum weight for any model
            max_model_weight: Maximum weight for any model
            performance_lookback: Number of predictions to consider for performance
        """
        self.method = method
        self.dynamic_weighting = dynamic_weighting
        self.weight_update_frequency = weight_update_frequency
        self.min_model_weight = min_model_weight
        self.max_model_weight = max_model_weight
        self.performance_lookback = performance_lookback
        
        # Model registry and performance tracking
        self.models = {}
        self.model_performance = {}
        self.model_weights = {}
        self.prediction_history = []
        self.last_weight_update = None
        
        # Performance metrics
        self.performance_metrics = {
            'direction_accuracy': {},
            'price_rmse': {},
            'confidence_calibration': {}
        }
        
        logger.info(f"Ensemble aggregator initialized with method: {method}")
    
    def register_model(self, 
                      model_name: str,
                      model_instance: object,
                      initial_weight: float = 1.0):
        """
        Register a model with the ensemble.
        
        Args:
            model_name: Unique name for the model
            model_instance: Model instance
            initial_weight: Initial weight for the model
        """
        self.models[model_name] = model_instance
        self.model_weights[model_name] = initial_weight
        self.model_performance[model_name] = {
            'predictions': [],
            'actual_directions': [],
            'actual_prices': [],
            'timestamps': []
        }
        
        logger.info(f"Model '{model_name}' registered with weight {initial_weight}")
    
    def unregister_model(self, model_name: str):
        """Remove a model from the ensemble."""
        if model_name in self.models:
            del self.models[model_name]
            del self.model_weights[model_name]
            del self.model_performance[model_name]
            logger.info(f"Model '{model_name}' unregistered")
    
    def add_prediction(self, 
                      model_name: str,
                      direction_probabilities: np.ndarray,
                      price_target: np.ndarray,
                      confidence: np.ndarray,
                      timestamp: Optional[datetime] = None):
        """
        Add a prediction from a specific model.
        
        Args:
            model_name: Name of the model
            direction_probabilities: Direction probabilities [P(down), P(up)]
            price_target: Predicted price target
            confidence: Model confidence score
            timestamp: Timestamp of prediction
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create prediction object
        prediction = ModelPrediction(
            model_name=model_name,
            direction_probabilities=direction_probabilities,
            price_target=price_target,
            confidence=confidence,
            timestamp=timestamp,
            model_weight=self.model_weights[model_name]
        )
        
        # Store prediction
        if model_name not in self.prediction_history:
            self.prediction_history[model_name] = []
        
        self.prediction_history[model_name].append(prediction)
        
        # Keep only recent predictions
        cutoff_time = timestamp - timedelta(hours=24)
        self.prediction_history[model_name] = [
            p for p in self.prediction_history[model_name] 
            if p.timestamp > cutoff_time
        ]
    
    def update_model_performance(self, 
                               model_name: str,
                               actual_direction: int,
                               actual_price: float,
                               timestamp: datetime):
        """
        Update model performance with actual results.
        
        Args:
            model_name: Name of the model
            actual_direction: Actual price direction (0 for down, 1 for up)
            actual_price: Actual price
            timestamp: Timestamp of the actual result
        """
        if model_name not in self.model_performance:
            return
        
        # Add to performance tracking
        self.model_performance[model_name]['actual_directions'].append(actual_direction)
        self.model_performance[model_name]['actual_prices'].append(actual_price)
        self.model_performance[model_name]['timestamps'].append(timestamp)
        
        # Keep only recent performance data
        cutoff_time = timestamp - timedelta(hours=24)
        valid_indices = [
            i for i, t in enumerate(self.model_performance[model_name]['timestamps'])
            if t > cutoff_time
        ]
        
        self.model_performance[model_name]['actual_directions'] = [
            self.model_performance[model_name]['actual_directions'][i] for i in valid_indices
        ]
        self.model_performance[model_name]['actual_prices'] = [
            self.model_performance[model_name]['actual_prices'][i] for i in valid_indices
        ]
        self.model_performance[model_name]['timestamps'] = [
            self.model_performance[model_name]['timestamps'][i] for i in valid_indices
        ]
        
        # Calculate performance metrics
        self._calculate_model_performance(model_name)
    
    def _calculate_model_performance(self, model_name: str):
        """Calculate performance metrics for a specific model."""
        if model_name not in self.model_performance:
            return
        
        perf = self.model_performance[model_name]
        
        if len(perf['actual_directions']) < 10:  # Need minimum data
            return
        
        # Direction accuracy
        if len(perf['predictions']) > 0:
            # Find matching predictions
            predictions = perf['predictions'][-len(perf['actual_directions']):]
            predicted_directions = [np.argmax(p.direction_probabilities) for p in predictions]
            
            accuracy = np.mean(np.array(predicted_directions) == np.array(perf['actual_directions']))
            self.performance_metrics['direction_accuracy'][model_name] = accuracy
        
        # Price RMSE
        if len(perf['predictions']) > 0:
            predictions = perf['predictions'][-len(perf['actual_prices']):]
            predicted_prices = [p.price_target[0] for p in predictions]
            
            rmse = np.sqrt(np.mean((np.array(predicted_prices) - np.array(perf['actual_prices'])) ** 2))
            self.performance_metrics['price_rmse'][model_name] = rmse
    
    def _should_update_weights(self) -> bool:
        """Check if weights should be updated based on frequency."""
        if not self.dynamic_weighting:
            return False
        
        if self.last_weight_update is None:
            return True
        
        current_time = datetime.now()
        
        if self.weight_update_frequency == "1h":
            return (current_time - self.last_weight_update).total_seconds() >= 3600
        elif self.weight_update_frequency == "1d":
            return (current_time - self.last_weight_update).total_seconds() >= 86400
        elif self.weight_update_frequency == "1w":
            return (current_time - self.last_weight_update).total_seconds() >= 604800
        
        return False
    
    def _update_model_weights(self):
        """Update model weights based on recent performance."""
        if not self.dynamic_weighting:
            return
        
        logger.info("Updating model weights based on performance...")
        
        # Calculate performance scores
        performance_scores = {}
        total_score = 0
        
        for model_name in self.models.keys():
            score = 0
            
            # Direction accuracy (40% weight)
            if model_name in self.performance_metrics['direction_accuracy']:
                acc = self.performance_metrics['direction_accuracy'][model_name]
                score += 0.4 * acc
            
            # Price RMSE (40% weight) - lower is better
            if model_name in self.performance_metrics['price_rmse']:
                rmse = self.performance_metrics['price_rmse'][model_name]
                # Normalize RMSE to 0-1 scale (assuming max RMSE of 100)
                normalized_rmse = max(0, 1 - (rmse / 100))
                score += 0.4 * normalized_rmse
            
            # Confidence calibration (20% weight)
            if model_name in self.performance_metrics['confidence_calibration']:
                cal = self.performance_metrics['confidence_calibration'][model_name]
                score += 0.2 * cal
            
            performance_scores[model_name] = max(0.1, score)  # Minimum score
            total_score += performance_scores[model_name]
        
        # Normalize weights
        if total_score > 0:
            for model_name in self.models.keys():
                raw_weight = performance_scores[model_name] / total_score
                
                # Apply min/max constraints
                constrained_weight = np.clip(raw_weight, self.min_model_weight, self.max_model_weight)
                self.model_weights[model_name] = constrained_weight
        
        # Renormalize to ensure weights sum to 1
        total_weight = sum(self.model_weights.values())
        for model_name in self.model_weights:
            self.model_weights[model_name] /= total_weight
        
        self.last_weight_update = datetime.now()
        
        logger.info(f"Updated model weights: {self.model_weights}")
    
    def aggregate_predictions(self, 
                            predictions: List[ModelPrediction],
                            timestamp: Optional[datetime] = None) -> EnsemblePrediction:
        """
        Aggregate predictions from multiple models.
        
        Args:
            predictions: List of model predictions
            timestamp: Timestamp for the ensemble prediction
            
        Returns:
            EnsemblePrediction object
        """
        if not predictions:
            raise ValueError("No predictions provided for aggregation")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update weights if needed
        if self._should_update_weights():
            self._update_model_weights()
        
        # Update model weights in predictions
        for pred in predictions:
            pred.model_weight = self.model_weights.get(pred.model_name, 1.0)
        
        # Aggregate based on method
        if self.method == "weighted_average":
            return self._weighted_average_aggregation(predictions, timestamp)
        elif self.method == "voting":
            return self._voting_aggregation(predictions, timestamp)
        elif self.method == "stacking":
            return self._stacking_aggregation(predictions, timestamp)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
    
    def _weighted_average_aggregation(self, 
                                    predictions: List[ModelPrediction],
                                    timestamp: datetime) -> EnsemblePrediction:
        """Aggregate predictions using weighted averaging."""
        # Normalize weights
        total_weight = sum(p.model_weight for p in predictions)
        normalized_weights = [p.model_weight / total_weight for p in predictions]
        
        # Weighted average of direction probabilities
        weighted_direction_probs = np.zeros(2)
        for pred, weight in zip(predictions, normalized_weights):
            weighted_direction_probs += weight * pred.direction_probabilities
        
        # Weighted average of price targets
        weighted_price_target = np.average(
            [p.price_target[0] for p in predictions],
            weights=normalized_weights
        )
        
        # Weighted average of confidence
        weighted_confidence = np.average(
            [p.confidence[0] for p in predictions],
            weights=normalized_weights
        )
        
        # Calculate ensemble confidence (agreement between models)
        direction_predictions = [np.argmax(p.direction_probabilities) for p in predictions]
        model_agreement = 1 - (np.std(direction_predictions) / 0.5)  # Normalize to 0-1
        
        # Overall ensemble confidence
        ensemble_confidence = (weighted_confidence + model_agreement) / 2
        
        return EnsemblePrediction(
            direction_probabilities=weighted_direction_probs,
            predicted_direction=np.argmax(weighted_direction_probs),
            price_target=weighted_price_target,
            confidence=weighted_confidence,
            ensemble_confidence=ensemble_confidence,
            model_agreement=model_agreement,
            timestamp=timestamp,
            individual_predictions={p.model_name: p for p in predictions}
        )
    
    def _voting_aggregation(self, 
                           predictions: List[ModelPrediction],
                           timestamp: datetime) -> EnsemblePrediction:
        """Aggregate predictions using majority voting."""
        # Count votes for each direction
        votes = [0, 0]  # [down, up]
        for pred in predictions:
            direction = np.argmax(pred.direction_probabilities)
            votes[direction] += pred.model_weight
        
        # Determine majority direction
        predicted_direction = 1 if votes[1] > votes[0] else 0
        
        # Calculate ensemble probabilities based on vote proportions
        total_votes = sum(votes)
        direction_probabilities = np.array([votes[0] / total_votes, votes[1] / total_votes])
        
        # Average price target and confidence
        price_target = np.mean([p.price_target[0] for p in predictions])
        confidence = np.mean([p.confidence[0] for p in predictions])
        
        # Model agreement (consensus strength)
        model_agreement = max(votes) / total_votes
        
        # Ensemble confidence
        ensemble_confidence = (confidence + model_agreement) / 2
        
        return EnsemblePrediction(
            direction_probabilities=direction_probabilities,
            predicted_direction=predicted_direction,
            price_target=price_target,
            confidence=confidence,
            ensemble_confidence=ensemble_confidence,
            model_agreement=model_agreement,
            timestamp=timestamp,
            individual_predictions={p.model_name: p for p in predictions}
        )
    
    def _stacking_aggregation(self, 
                             predictions: List[ModelPrediction],
                             timestamp: datetime) -> EnsemblePrediction:
        """Aggregate predictions using stacking (meta-learning)."""
        # For simplicity, use a weighted average approach
        # In a full implementation, this would use a meta-learner trained on validation data
        
        # Use model performance as weights for stacking
        performance_weights = []
        for pred in predictions:
            if pred.model_name in self.performance_metrics['direction_accuracy']:
                acc = self.performance_metrics['direction_accuracy'][pred.model_name]
                performance_weights.append(acc)
            else:
                performance_weights.append(0.5)  # Default weight
        
        # Normalize performance weights
        total_perf_weight = sum(performance_weights)
        if total_perf_weight > 0:
            normalized_perf_weights = [w / total_perf_weight for w in performance_weights]
        else:
            normalized_perf_weights = [1.0 / len(predictions)] * len(predictions)
        
        # Weighted aggregation using performance-based weights
        weighted_direction_probs = np.zeros(2)
        for pred, weight in zip(predictions, normalized_perf_weights):
            weighted_direction_probs += weight * pred.direction_probabilities
        
        # Other aggregations
        price_target = np.average(
            [p.price_target[0] for p in predictions],
            weights=normalized_perf_weights
        )
        
        confidence = np.average(
            [p.confidence[0] for p in predictions],
            weights=normalized_perf_weights
        )
        
        # Model agreement
        direction_predictions = [np.argmax(p.direction_probabilities) for p in predictions]
        model_agreement = 1 - (np.std(direction_predictions) / 0.5)
        
        ensemble_confidence = (confidence + model_agreement) / 2
        
        return EnsemblePrediction(
            direction_probabilities=weighted_direction_probs,
            predicted_direction=np.argmax(weighted_direction_probs),
            price_target=price_target,
            confidence=confidence,
            ensemble_confidence=ensemble_confidence,
            model_agreement=model_agreement,
            timestamp=timestamp,
            individual_predictions={p.model_name: p for p in predictions}
        )
    
    def get_ensemble_summary(self) -> Dict:
        """Get summary of ensemble performance and configuration."""
        return {
            'method': self.method,
            'dynamic_weighting': self.dynamic_weighting,
            'registered_models': list(self.models.keys()),
            'model_weights': self.model_weights.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'last_weight_update': self.last_weight_update,
            'total_predictions': sum(len(preds) for preds in self.prediction_history.values())
        }
    
    def reset_performance(self):
        """Reset all performance tracking data."""
        for model_name in self.model_performance:
            self.model_performance[model_name] = {
                'predictions': [],
                'actual_directions': [],
                'actual_prices': [],
                'timestamps': []
            }
        
        self.performance_metrics = {
            'direction_accuracy': {},
            'price_rmse': {},
            'confidence_calibration': {}
        }
        
        self.prediction_history = {}
        logger.info("Performance tracking reset")


class EnsembleManager:
    """
    High-level manager for the ensemble system.
    
    Handles model registration, prediction collection, and ensemble aggregation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ensemble manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Create aggregator
        aggregator_config = self.config.get('ensemble', {})
        self.aggregator = EnsembleAggregator(**aggregator_config)
        
        # Model instances
        self.models = {}
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_duration = timedelta(minutes=5)  # 5 minutes
        
        logger.info("Ensemble manager initialized")
    
    def register_model(self, 
                      model_name: str,
                      model_instance: object,
                      initial_weight: float = 1.0):
        """Register a model with the ensemble."""
        self.models[model_name] = model_instance
        self.aggregator.register_model(model_name, model_instance, initial_weight)
        logger.info(f"Model '{model_name}' registered with ensemble manager")
    
    def collect_predictions(self, 
                          input_data: np.ndarray,
                          timestamp: Optional[datetime] = None) -> Dict[str, ModelPrediction]:
        """
        Collect predictions from all registered models.
        
        Args:
            input_data: Input data for prediction
            timestamp: Timestamp for predictions
            
        Returns:
            Dictionary of model predictions
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Get prediction from model
                if hasattr(model, 'predict'):
                    pred_dict = model.predict(input_data)
                    
                    # Create ModelPrediction object
                    prediction = ModelPrediction(
                        model_name=model_name,
                        direction_probabilities=pred_dict['direction_probabilities'],
                        price_target=pred_dict['price_targets'],
                        confidence=pred_dict['confidence'],
                        timestamp=timestamp
                    )
                    
                    predictions[model_name] = prediction
                    
                    # Add to aggregator
                    self.aggregator.add_prediction(
                        model_name,
                        prediction.direction_probabilities,
                        prediction.price_target,
                        prediction.confidence,
                        timestamp
                    )
                    
                else:
                    logger.warning(f"Model '{model_name}' does not have predict method")
                    
            except Exception as e:
                logger.error(f"Error getting prediction from model '{model_name}': {e}")
        
        return predictions
    
    def get_ensemble_prediction(self, 
                              input_data: np.ndarray,
                              timestamp: Optional[datetime] = None) -> EnsemblePrediction:
        """
        Get ensemble prediction for input data.
        
        Args:
            input_data: Input data for prediction
            timestamp: Timestamp for prediction
            
        Returns:
            EnsemblePrediction object
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check cache first
        cache_key = hash(str(input_data.tobytes()) + str(timestamp))
        if cache_key in self.prediction_cache:
            cached_pred, cached_time = self.prediction_cache[cache_key]
            if timestamp - cached_time < self.cache_duration:
                return cached_pred
        
        # Collect predictions from all models
        predictions = self.collect_predictions(input_data, timestamp)
        
        if not predictions:
            raise ValueError("No predictions collected from any model")
        
        # Aggregate predictions
        ensemble_pred = self.aggregator.aggregate_predictions(
            list(predictions.values()), timestamp
        )
        
        # Cache result
        self.prediction_cache[cache_key] = (ensemble_pred, timestamp)
        
        return ensemble_pred
    
    def update_performance(self, 
                          actual_direction: int,
                          actual_price: float,
                          timestamp: datetime):
        """
        Update performance tracking with actual results.
        
        Args:
            actual_direction: Actual price direction
            actual_price: Actual price
            timestamp: Timestamp of actual result
        """
        self.aggregator.update_model_performance(
            'ensemble', actual_direction, actual_price, timestamp
        )
    
    def get_ensemble_status(self) -> Dict:
        """Get current status of the ensemble system."""
        return {
            'registered_models': list(self.models.keys()),
            'aggregator_summary': self.aggregator.get_ensemble_summary(),
            'cache_size': len(self.prediction_cache),
            'total_models': len(self.models)
        }


# Convenience function for quick ensemble creation
def create_ensemble(config: Optional[Dict] = None) -> EnsembleManager:
    """
    Create an ensemble manager with default or custom configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        EnsembleManager instance
    """
    default_config = {
        'ensemble': {
            'method': 'weighted_average',
            'dynamic_weighting': True,
            'weight_update_frequency': '1h',
            'min_model_weight': 0.05,
            'max_model_weight': 0.4,
            'performance_lookback': 100
        }
    }
    
    if config:
        default_config.update(config)
    
    return EnsembleManager(default_config)


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create ensemble manager
    ensemble = create_ensemble()
    
    # Mock model predictions
    mock_predictions = [
        ModelPrediction(
            model_name="model_1",
            direction_probabilities=np.array([0.3, 0.7]),
            price_target=np.array([2050.0]),
            confidence=np.array([0.8]),
            timestamp=datetime.now(),
            model_weight=0.4
        ),
        ModelPrediction(
            model_name="model_2",
            direction_probabilities=np.array([0.2, 0.8]),
            price_target=np.array([2055.0]),
            confidence=np.array([0.9]),
            timestamp=datetime.now(),
            model_weight=0.6
        )
    ]
    
    # Test aggregation
    ensemble_pred = ensemble.aggregator.aggregate_predictions(mock_predictions)
    
    print(f"Ensemble prediction:")
    print(f"Direction probabilities: {ensemble_pred.direction_probabilities}")
    print(f"Predicted direction: {ensemble_pred.predicted_direction}")
    print(f"Price target: {ensemble_pred.price_target}")
    print(f"Confidence: {ensemble_pred.confidence}")
    print(f"Model agreement: {ensemble_pred.model_agreement}")
    
    # Test performance update
    ensemble.aggregator.update_model_performance("model_1", 1, 2052.0, datetime.now())
    ensemble.aggregator.update_model_performance("model_2", 1, 2052.0, datetime.now())
    
    # Get status
    status = ensemble.get_ensemble_status()
    print(f"\nEnsemble status: {status}")
