#!/usr/bin/env python3
"""
SignaMentis - Property-Based Tests with Hypothesis

This module implements property-based tests using Hypothesis for comprehensive
testing of the trading system components.

Author: SignaMentis Team
Version: 2.0.0
"""

import pytest
from hypothesis import given, strategies as st, settings, Verbosity
from hypothesis.extra.numpy import arrays
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import system components for testing
try:
    from scripts.feature_engineering import FeatureEngineer
    from scripts.risk_manager import RiskManager
    from scripts.strategy import SuperTrendStrategy
    from scripts.ensemble import EnsembleManager
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    logger.warning("System components not available for testing")

# Hypothesis strategies for trading data
@st.composite
def market_data_strategy(draw):
    """Generate realistic market data for testing."""
    n_samples = draw(st.integers(min_value=100, max_value=1000))
    
    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(minutes=i*5) for i in range(n_samples)]
    
    # Generate OHLCV data with realistic constraints
    base_price = draw(st.floats(min_value=1000, max_value=3000))
    volatility = draw(st.floats(min_value=0.001, max_value=0.1))
    
    prices = []
    for i in range(n_samples):
        if i == 0:
            price = base_price
        else:
            # Random walk with volatility
            change = np.random.normal(0, volatility)
            price = prices[-1] * (1 + change)
        
        prices.append(max(0.01, price))
    
    # Generate OHLC from prices
    data = []
    for i in range(n_samples):
        price = prices[i]
        spread = price * 0.0001  # 1 pip spread
        
        open_price = price
        high_price = price * (1 + abs(np.random.normal(0, 0.002)))
        low_price = price * (1 - abs(np.random.normal(0, 0.002)))
        close_price = price * (1 + np.random.normal(0, 0.001))
        volume = draw(st.floats(min_value=100, max_value=10000))
        
        # Ensure OHLC relationships
        high_price = max(open_price, high_price, close_price)
        low_price = min(open_price, low_price, close_price)
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)

@st.composite
def trading_signal_strategy(draw):
    """Generate trading signals for testing."""
    signal_type = draw(st.sampled_from(['buy', 'sell', 'hold']))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    price = draw(st.floats(min_value=1000, max_value=3000))
    
    return {
        'type': signal_type,
        'confidence': confidence,
        'price': price,
        'timestamp': datetime.now(),
        'symbol': 'XAUUSD'
    }

@st.composite
def risk_parameters_strategy(draw):
    """Generate risk management parameters for testing."""
    return {
        'max_position_size': draw(st.floats(min_value=0.01, max_value=10.0)),
        'max_daily_loss': draw(st.floats(min_value=100, max_value=10000)),
        'max_drawdown': draw(st.floats(min_value=0.01, max_value=0.5)),
        'stop_loss_pct': draw(st.floats(min_value=0.001, max_value=0.1)),
        'take_profit_pct': draw(st.floats(min_value=0.002, max_value=0.2))
    }

# Property-based tests
class TestFeatureEngineering:
    """Property-based tests for feature engineering."""
    
    @given(market_data_strategy())
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_feature_engineering_properties(self, market_data):
        """Test that feature engineering preserves data properties."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        # Create feature engineer
        engineer = FeatureEngineer()
        
        # Generate features
        features = engineer.create_features(market_data)
        
        # Property 1: Features should have same number of rows as input
        assert len(features) == len(market_data)
        
        # Property 2: Features should not contain infinite values
        assert not np.isinf(features.select_dtypes(include=[np.number])).any().any()
        
        # Property 3: Features should not contain NaN values in critical columns
        critical_features = ['sma_20', 'rsi', 'macd']
        for feature in critical_features:
            if feature in features.columns:
                assert not features[feature].isna().all()
        
        # Property 4: Technical indicators should be within expected ranges
        if 'rsi' in features.columns:
            rsi_values = features['rsi'].dropna()
            if len(rsi_values) > 0:
                assert rsi_values.min() >= 0
                assert rsi_values.max() <= 100
        
        # Property 5: Moving averages should be numeric
        if 'sma_20' in features.columns:
            sma_values = features['sma_20'].dropna()
            if len(sma_values) > 0:
                assert np.issubdtype(sma_values.dtype, np.number)
    
    @given(market_data_strategy())
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_feature_engineering_idempotency(self, market_data):
        """Test that feature engineering is idempotent."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        engineer = FeatureEngineer()
        
        # Generate features twice
        features1 = engineer.create_features(market_data)
        features2 = engineer.create_features(market_data)
        
        # Property: Results should be identical
        pd.testing.assert_frame_equal(features1, features2)

class TestRiskManager:
    """Property-based tests for risk management."""
    
    @given(
        st.floats(min_value=1000, max_value=100000),
        risk_parameters_strategy()
    )
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_position_sizing_properties(self, account_balance, risk_params):
        """Test position sizing properties."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        risk_manager = RiskManager(risk_params)
        
        # Calculate position size
        position_size = risk_manager.calculate_position_size(account_balance)
        
        # Property 1: Position size should be positive
        assert position_size > 0
        
        # Property 2: Position size should not exceed max position size
        max_size = risk_params['max_position_size']
        assert position_size <= max_size
        
        # Property 3: Position size should be proportional to account balance
        if account_balance > 0:
            ratio = position_size / account_balance
            assert ratio <= risk_params['max_position_size']
    
    @given(
        st.floats(min_value=1000, max_value=3000),
        st.floats(min_value=0.001, max_value=0.1),
        st.floats(min_value=0.002, max_value=0.2)
    )
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_stop_loss_take_profit_properties(self, entry_price, stop_loss_pct, take_profit_pct):
        """Test stop loss and take profit properties."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        risk_params = {
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        risk_manager = RiskManager(risk_params)
        
        # Calculate SL and TP
        stop_loss = risk_manager.calculate_stop_loss(entry_price, 'buy')
        take_profit = risk_manager.calculate_take_profit(entry_price, 'buy')
        
        # Property 1: Stop loss should be below entry price for buy
        assert stop_loss < entry_price
        
        # Property 2: Take profit should be above entry price for buy
        assert take_profit > entry_price
        
        # Property 3: Risk-reward ratio should be reasonable
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        if risk > 0:
            ratio = reward / risk
            assert 0.5 <= ratio <= 5.0  # Reasonable risk-reward range

class TestTradingStrategy:
    """Property-based tests for trading strategy."""
    
    @given(market_data_strategy())
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_strategy_signal_properties(self, market_data):
        """Test trading strategy signal properties."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        strategy = SuperTrendStrategy()
        
        # Generate signals
        signals = strategy.generate_signals(market_data)
        
        # Property 1: Signals should be valid types
        valid_signals = ['buy', 'sell', 'hold']
        for signal in signals:
            assert signal['type'] in valid_signals
        
        # Property 2: Signal confidence should be between 0 and 1
        for signal in signals:
            assert 0 <= signal['confidence'] <= 1
        
        # Property 3: Signals should have timestamps
        for signal in signals:
            assert 'timestamp' in signal
            assert isinstance(signal['timestamp'], datetime)
    
    @given(market_data_strategy())
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_strategy_consistency(self, market_data):
        """Test that strategy produces consistent results."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        strategy = SuperTrendStrategy()
        
        # Generate signals twice
        signals1 = strategy.generate_signals(market_data)
        signals2 = strategy.generate_signals(market_data)
        
        # Property: Results should be identical for same input
        assert len(signals1) == len(signals2)
        for s1, s2 in zip(signals1, signals2):
            assert s1['type'] == s2['type']
            assert s1['confidence'] == s2['confidence']

class TestEnsembleManager:
    """Property-based tests for ensemble management."""
    
    @given(
        st.lists(trading_signal_strategy(), min_size=2, max_size=5),
        st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=5)
    )
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_ensemble_aggregation_properties(self, signals, weights):
        """Test ensemble aggregation properties."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Create ensemble manager
        ensemble = EnsembleManager()
        
        # Aggregate signals
        aggregated = ensemble.aggregate_signals(signals, weights)
        
        # Property 1: Aggregated signal should have valid type
        valid_types = ['buy', 'sell', 'hold']
        assert aggregated['type'] in valid_types
        
        # Property 2: Aggregated confidence should be between 0 and 1
        assert 0 <= aggregated['confidence'] <= 1
        
        # Property 3: Aggregated confidence should be weighted average
        expected_confidence = sum(s['confidence'] * w for s, w in zip(signals, weights))
        assert abs(aggregated['confidence'] - expected_confidence) < 0.001
    
    @given(
        st.lists(trading_signal_strategy(), min_size=1, max_size=10),
        st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=10)
    )
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_ensemble_weight_constraints(self, signals, weights):
        """Test ensemble weight constraints."""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Property 1: Weights should sum to 1
        assert abs(weights.sum() - 1.0) < 0.001
        
        # Property 2: All weights should be non-negative
        assert (weights >= 0).all()
        
        # Property 3: Weights should be between 0 and 1
        assert (weights <= 1).all()).all()

class TestDataValidation:
    """Property-based tests for data validation."""
    
    @given(market_data_strategy())
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_market_data_integrity(self, market_data):
        """Test market data integrity properties."""
        # Property 1: OHLC relationships should be valid
        for _, row in market_data.iterrows():
            assert row['high'] >= row['low']
            assert row['high'] >= row['open']
            assert row['high'] >= row['close']
            assert row['low'] <= row['open']
            assert row['low'] <= row['close']
        
        # Property 2: Volume should be non-negative
        assert (market_data['volume'] >= 0).all()
        
        # Property 3: Prices should be positive
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            assert (market_data[col] > 0).all()
        
        # Property 4: Timestamps should be in order
        timestamps = market_data['timestamp'].tolist()
        assert timestamps == sorted(timestamps)
    
    @given(
        st.lists(st.floats(min_value=-1000, max_value=1000), min_size=10, max_size=100),
        st.lists(st.floats(min_value=0, max_value=1), min_size=10, max_size=100)
    )
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_numeric_validation(self, returns, probabilities):
        """Test numeric validation properties."""
        # Property 1: Returns should be finite
        assert all(np.isfinite(r) for r in returns)
        
        # Property 2: Probabilities should be between 0 and 1
        assert all(0 <= p <= 1 for p in probabilities)
        
        # Property 3: Sum of probabilities should be reasonable
        total_prob = sum(probabilities)
        assert 0 <= total_prob <= len(probabilities)

class TestPerformanceMetrics:
    """Property-based tests for performance metrics."""
    
    @given(
        st.lists(st.floats(min_value=-1000, max_value=1000), min_size=10, max_size=100)
    )
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_sharpe_ratio_properties(self, returns):
        """Test Sharpe ratio properties."""
        if len(returns) == 0:
            return
        
        returns = np.array(returns)
        
        # Calculate Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            sharpe = mean_return / std_return
            
            # Property 1: Sharpe ratio should be finite
            assert np.isfinite(sharpe)
            
            # Property 2: Sharpe ratio should be reasonable
            assert -10 <= sharpe <= 10
    
    @given(
        st.lists(st.floats(min_value=-1000, max_value=1000), min_size=10, max_size=100)
    )
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_drawdown_properties(self, returns):
        """Test drawdown properties."""
        if len(returns) == 0:
            return
        
        returns = np.array(returns)
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        # Property 1: Maximum drawdown should be non-positive
        max_dd = np.min(drawdown)
        assert max_dd <= 0
        
        # Property 2: Drawdown should be between -1 and 0
        assert -1 <= max_dd <= 0

# Test configuration
def pytest_configure(config):
    """Configure pytest for Hypothesis."""
    config.addinivalue_line(
        "markers", "hypothesis: marks tests as Hypothesis property-based tests"
    )

def pytest_collection_modifyitems(config, items):
    """Mark Hypothesis tests."""
    for item in items:
        if "Test" in item.name and "hypothesis" in item.name.lower():
            item.add_marker(pytest.mark.hypothesis)

# Example usage
if __name__ == "__main__":
    print("🚀 Hypothesis Property-Based Tests")
    print("=" * 50)
    print("This module provides comprehensive property-based testing")
    print("for the SignaMentis trading system.")
    print()
    print("Run with pytest:")
    print("  pytest tests/property_tests.py -v")
    print()
    print("Run specific test class:")
    print("  pytest tests/property_tests.py::TestFeatureEngineering -v")
    print()
    print("Run with Hypothesis settings:")
    print("  pytest tests/property_tests.py --hypothesis-verbosity=verbose")
