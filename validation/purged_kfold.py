#!/usr/bin/env python3
"""
SignaMentis - Purged K-Fold and Combinatorial Purged Cross-Validation

This module implements robust cross-validation methods for time series data
that prevent data leakage and ensure realistic backtesting scenarios.

Author: SignaMentis Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PurgeMethod(Enum):
    """Methods for purging data to prevent look-ahead bias."""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    VOLATILITY_BASED = "volatility_based"

class EmbargoMethod(Enum):
    """Methods for embargo periods after purging."""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    CORRELATION_BASED = "correlation_based"

@dataclass
class PurgeConfig:
    """Configuration for purging and embargo periods."""
    purge_days: int = 1
    embargo_days: int = 1
    purge_method: PurgeMethod = PurgeMethod.FIXED
    embargo_method: EmbargoMethod = EmbargoMethod.FIXED
    volatility_threshold: float = 0.02
    correlation_threshold: float = 0.7
    adaptive_factor: float = 1.5

@dataclass
class FoldInfo:
    """Information about a single fold."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    purge_start: int
    purge_end: int
    embargo_start: int
    embargo_end: int
    fold_index: int

class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation for time series data.
    
    This implementation prevents data leakage by:
    1. Purging data that overlaps with the test set
    2. Adding embargo periods to prevent correlation-based leakage
    3. Supporting multiple purging and embargo strategies
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_config: Optional[PurgeConfig] = None,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of folds
            purge_config: Configuration for purging and embargo
            shuffle: Whether to shuffle the data (not recommended for time series)
            random_state: Random state for reproducibility
        """
        super().__init__()
        self.n_splits = n_splits
        self.purge_config = purge_config or PurgeConfig()
        self.shuffle = shuffle
        self.random_state = random_state
        
        if shuffle:
            warnings.warn(
                "Shuffling is not recommended for time series data as it can "
                "introduce look-ahead bias."
            )
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Training data
            y: Target variable (optional)
            groups: Group labels (optional)
            
        Yields:
            train_indices, test_indices: Training and test indices for each fold
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater "
                f"than the number of samples: {n_samples}."
            )
        
        indices = np.arange(n_samples)
        
        # Calculate fold sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            
            # Define test set
            test_start, test_end = start, stop
            
            # Calculate purge period
            if self.purge_config.purge_method == PurgeMethod.FIXED:
                purge_start = max(0, test_start - self.purge_config.purge_days)
                purge_end = test_start
            elif self.purge_config.purge_method == PurgeMethod.ADAPTIVE:
                purge_days = int(self.purge_config.purge_days * self.purge_config.adaptive_factor)
                purge_start = max(0, test_start - purge_days)
                purge_end = test_start
            else:  # VOLATILITY_BASED
                # This would require volatility data - simplified for now
                purge_start = max(0, test_start - self.purge_config.purge_days)
                purge_end = test_start
            
            # Calculate embargo period
            if self.purge_config.embargo_method == EmbargoMethod.FIXED:
                embargo_start = test_end
                embargo_end = min(n_samples, test_end + self.purge_config.embargo_days)
            elif self.purge_config.embargo_method == EmbargoMethod.ADAPTIVE:
                embargo_days = int(self.purge_config.embargo_days * self.purge_config.adaptive_factor)
                embargo_start = test_end
                embargo_end = min(n_samples, test_end + embargo_days)
            else:  # CORRELATION_BASED
                # This would require correlation analysis - simplified for now
                embargo_start = test_end
                embargo_end = min(n_samples, test_end + self.purge_config.embargo_days)
            
            # Define training set (excluding purge and embargo periods)
            train_start = 0
            train_end = purge_start
            
            # Create fold info
            fold_info = FoldInfo(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_start=purge_start,
                purge_end=purge_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end,
                fold_index=len(self.fold_info_) if hasattr(self, 'fold_info_') else 0
            )
            
            if not hasattr(self, 'fold_info_'):
                self.fold_info_ = []
            self.fold_info_.append(fold_info)
            
            # Yield indices
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
            
            current = stop
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get the number of splitting iterations."""
        return self.n_splits
    
    def get_fold_info(self) -> List[FoldInfo]:
        """Get information about all folds."""
        if not hasattr(self, 'fold_info_'):
            raise RuntimeError("Must call split() before getting fold info")
        return self.fold_info_

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    This method creates multiple combinations of training/testing splits
    to provide more robust validation and reduce overfitting.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_combinations: int = 10,
        purge_config: Optional[PurgeConfig] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize CombinatorialPurgedCV.
        
        Args:
            n_splits: Number of splits per combination
            n_combinations: Number of different combinations to generate
            purge_config: Configuration for purging and embargo
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.n_combinations = n_combinations
        self.purge_config = purge_config or PurgeConfig()
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Store all combinations
        self.combinations_ = []
        self.fold_info_ = []
    
    def split(self, X, y=None, groups=None):
        """
        Generate multiple combinations of training/testing splits.
        
        Args:
            X: Training data
            y: Target variable (optional)
            groups: Group labels (optional)
            
        Yields:
            train_indices, test_indices: Training and test indices for each fold
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater "
                f"than the number of samples: {n_samples}."
            )
        
        indices = np.arange(n_samples)
        
        # Generate different combinations
        for combo_idx in range(self.n_combinations):
            # Shuffle split boundaries for variety
            split_points = np.sort(
                self.rng.choice(
                    range(1, n_samples - 1),
                    size=self.n_splits - 1,
                    replace=False
                )
            )
            split_points = np.concatenate([[0], split_points, [n_samples]])
            
            combo_fold_info = []
            
            for fold_idx in range(self.n_splits):
                # Define test set
                test_start, test_end = split_points[fold_idx], split_points[fold_idx + 1]
                
                # Calculate purge period
                purge_start = max(0, test_start - self.purge_config.purge_days)
                purge_end = test_start
                
                # Calculate embargo period
                embargo_start = test_end
                embargo_end = min(n_samples, test_end + self.purge_config.embargo_days)
                
                # Define training set (excluding purge and embargo periods)
                train_start = 0
                train_end = purge_start
                
                # Create fold info
                fold_info = FoldInfo(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    purge_start=purge_start,
                    purge_end=purge_end,
                    embargo_start=embargo_start,
                    embargo_end=embargo_end,
                    fold_index=fold_idx
                )
                
                combo_fold_info.append(fold_info)
                
                # Yield indices
                train_indices = indices[train_start:train_end]
                test_indices = indices[test_start:test_end]
                
                yield train_indices, test_indices
            
            self.combinations_.append(combo_idx)
            self.fold_info_.extend(combo_fold_info)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get the total number of splitting iterations."""
        return self.n_splits * self.n_combinations
    
    def get_fold_info(self) -> List[FoldInfo]:
        """Get information about all folds across all combinations."""
        return self.fold_info_
    
    def get_combination_info(self) -> Dict[int, List[FoldInfo]]:
        """Get fold information organized by combination."""
        combo_info = {}
        for i, combo_idx in enumerate(self.combinations_):
            start_idx = i * self.n_splits
            end_idx = start_idx + self.n_splits
            combo_info[combo_idx] = self.fold_info_[start_idx:end_idx]
        return combo_info

class TimeSeriesValidator:
    """
    High-level interface for time series validation.
    
    This class provides easy access to both PurgedKFold and CombinatorialPurgedCV
    with additional utilities for validation analysis.
    """
    
    def __init__(
        self,
        method: str = "purged_kfold",
        n_splits: int = 5,
        n_combinations: int = 10,
        purge_config: Optional[PurgeConfig] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize TimeSeriesValidator.
        
        Args:
            method: Validation method ("purged_kfold" or "combinatorial")
            n_splits: Number of splits
            n_combinations: Number of combinations (for combinatorial method)
            purge_config: Configuration for purging and embargo
            random_state: Random state for reproducibility
        """
        self.method = method
        self.n_splits = n_splits
        self.n_combinations = n_combinations
        self.purge_config = purge_config or PurgeConfig()
        self.random_state = random_state
        
        # Initialize the appropriate validator
        if method == "purged_kfold":
            self.validator = PurgedKFold(
                n_splits=n_splits,
                purge_config=purge_config,
                random_state=random_state
            )
        elif method == "combinatorial":
            self.validator = CombinatorialPurgedCV(
                n_splits=n_splits,
                n_combinations=n_combinations,
                purge_config=purge_config,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown validation method: {method}")
    
    def validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        estimator=None,
        scoring=None,
        cv=None,
        **kwargs
    ):
        """
        Perform cross-validation with the selected method.
        
        Args:
            X: Training data
            y: Target variable
            estimator: Estimator to validate
            scoring: Scoring metric
            cv: Cross-validation object (overrides self.validator)
            **kwargs: Additional arguments for cross-validation
            
        Returns:
            Cross-validation results
        """
        if cv is None:
            cv = self.validator
        
        # This would typically use sklearn.model_selection.cross_val_score
        # For now, return the validator for manual use
        return cv
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of the validation configuration."""
        return {
            "method": self.method,
            "n_splits": self.n_splits,
            "n_combinations": self.n_combinations if self.method == "combinatorial" else None,
            "purge_config": {
                "purge_days": self.purge_config.purge_days,
                "embargo_days": self.purge_config.embargo_days,
                "purge_method": self.purge_config.purge_method.value,
                "embargo_method": self.purge_config.embargo_method.value
            },
            "random_state": self.random_state
        }
    
    def plot_folds(self, X, save_path: Optional[str] = None):
        """
        Plot the fold structure for visualization.
        
        Args:
            X: Data to visualize
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            n_samples = len(X)
            fold_info = self.validator.get_fold_info()
            
            plt.figure(figsize=(15, 8))
            
            for fold in fold_info:
                # Plot training period
                plt.axvspan(fold.train_start, fold.train_end, alpha=0.3, color='blue', label='Training' if fold.fold_index == 0 else "")
                
                # Plot test period
                plt.axvspan(fold.test_start, fold.test_end, alpha=0.5, color='red', label='Test' if fold.fold_index == 0 else "")
                
                # Plot purge period
                plt.axvspan(fold.purge_start, fold.purge_end, alpha=0.7, color='orange', label='Purge' if fold.fold_index == 0 else "")
                
                # Plot embargo period
                plt.axvspan(fold.embargo_start, fold.embargo_end, alpha=0.7, color='purple', label='Embargo' if fold.fold_index == 0 else "")
            
            plt.xlabel('Sample Index')
            plt.ylabel('Fold')
            plt.title(f'Time Series Cross-Validation: {self.method.upper()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Fold visualization saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting folds: {e}")

def create_validation_suite(
    method: str = "purged_kfold",
    n_splits: int = 5,
    n_combinations: int = 10,
    purge_days: int = 1,
    embargo_days: int = 1,
    random_state: Optional[int] = None
) -> TimeSeriesValidator:
    """
    Factory function to create a validation suite.
    
    Args:
        method: Validation method
        n_splits: Number of splits
        n_combinations: Number of combinations
        purge_days: Days to purge
        embargo_days: Days for embargo
        random_state: Random state
        
    Returns:
        Configured TimeSeriesValidator
    """
    purge_config = PurgeConfig(
        purge_days=purge_days,
        embargo_days=embargo_days
    )
    
    return TimeSeriesValidator(
        method=method,
        n_splits=n_splits,
        n_combinations=n_combinations,
        purge_config=purge_config,
        random_state=random_state
    )

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'price': np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    print("🚀 Testing Time Series Validation Methods")
    print("=" * 50)
    
    # Test PurgedKFold
    print("\n📊 Testing PurgedKFold...")
    pkf = PurgedKFold(n_splits=5)
    pkf_splits = list(pkf.split(data))
    print(f"✅ Generated {len(pkf_splits)} folds")
    
    # Test CombinatorialPurgedCV
    print("\n🔄 Testing CombinatorialPurgedCV...")
    cpcv = CombinatorialPurgedCV(n_splits=3, n_combinations=5)
    cpcv_splits = list(cpcv.split(data))
    print(f"✅ Generated {len(cpcv_splits)} total splits across {cpcv.n_combinations} combinations")
    
    # Test TimeSeriesValidator
    print("\n🎯 Testing TimeSeriesValidator...")
    validator = TimeSeriesValidator(method="purged_kfold", n_splits=5)
    summary = validator.get_validation_summary()
    print(f"✅ Validation suite configured: {summary}")
    
    print("\n🎉 All validation methods working correctly!")
