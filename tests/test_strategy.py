#!/usr/bin/env python3
"""
Unit tests for trading strategy and risk management modules.

This module tests the functionality of:
- SuperTrend Strategy
- Risk Manager
- Backtester
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add scripts directory to path
sys.path.append('scripts')

from strategy import SuperTrendStrategy
from risk_manager import RiskManager, create_risk_manager
from backtester import Backtester, PurgedKFold, create_backtester


class TestSuperTrendStrategy(unittest.TestCase):
    """Test cases for SuperTrend Strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = SuperTrendStrategy()
        
        # Create sample OHLCV data
        dates = pd.date_range('2023-08-11', periods=100, freq='5T')
        self.sample_data = pd.DataFrame({
            'open': np.random.normal(2000, 50, 100),
            'high': np.random.normal(2010, 50, 100),
            'low': np.random.normal(1990, 50, 100),
            'close': np.random.normal(2000, 50, 100),
            'volume': np.random.normal(100, 20, 100)
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        self.sample_data['high'] = np.maximum(self.sample_data['high'], self.sample_data['close'])
        self.sample_data['low'] = np.minimum(self.sample_data['low'], self.sample_data['close'])
        self.sample_data['open'] = np.clip(self.sample_data['open'], 
                                          self.sample_data['low'], 
                                          self.sample_data['high'])
    
    def test_strategy_initialization(self):
        """Test SuperTrend Strategy initialization."""
        self.assertIsNotNone(self.strategy)
        self.assertIsInstance(self.strategy, SuperTrendStrategy)
    
    def test_supertrend_calculation(self):
        """Test SuperTrend indicator calculation."""
        try:
            # Calculate SuperTrend
            supertrend_data = self.strategy._calculate_supertrend(self.sample_data)
            
            # Check that SuperTrend column was added
            self.assertIn('supertrend', supertrend_data.columns)
            self.assertIn('supertrend_direction', supertrend_data.columns)
            
            # Check data types
            self.assertTrue(pd.api.types.is_numeric_dtype(supertrend_data['supertrend']))
            self.assertTrue(pd.api.types.is_numeric_dtype(supertrend_data['supertrend_direction']))
            
            # Check that SuperTrend values are reasonable
            self.assertTrue(all(supertrend_data['supertrend'] > 0))
            
        except Exception as e:
            self.fail(f"SuperTrend calculation failed: {e}")
    
    def test_breakout_detection(self):
        """Test breakout detection functionality."""
        try:
            # Calculate SuperTrend first
            supertrend_data = self.strategy._calculate_supertrend(self.sample_data)
            
            # Detect breakouts
            breakouts = self.strategy._detect_breakouts(supertrend_data)
            
            # Check that breakouts DataFrame was created
            self.assertIsInstance(breakouts, pd.DataFrame)
            self.assertGreater(len(breakouts), 0)
            
            # Check required columns
            required_columns = ['timestamp', 'breakout_type', 'price', 'volume', 'supertrend_value']
            for col in required_columns:
                self.assertIn(col, breakouts.columns)
            
        except Exception as e:
            self.fail(f"Breakout detection failed: {e}")
    
    def test_signal_validation(self):
        """Test signal validation with AI confirmation."""
        try:
            # Create sample AI prediction
            ai_prediction = {
                'direction': 'BUY',
                'confidence': 0.85,
                'price_target': 2050.0
            }
            
            # Validate signal
            is_valid = self.strategy._validate_signal_with_ai(
                'BUY', 2000.0, ai_prediction
            )
            
            # Should be valid with high confidence
            self.assertTrue(is_valid)
            
            # Test with low confidence
            low_confidence_prediction = {
                'direction': 'BUY',
                'confidence': 0.45,  # Below threshold
                'price_target': 2050.0
            }
            
            is_valid_low = self.strategy._validate_signal_with_ai(
                'BUY', 2000.0, low_confidence_prediction
            )
            
            # Should be invalid with low confidence
            self.assertFalse(is_valid_low)
            
        except Exception as e:
            self.fail(f"Signal validation failed: {e}")
    
    def test_trade_plan_generation(self):
        """Test trade plan generation."""
        try:
            # Generate trade plan
            trade_plan = self.strategy.generate_trade_plan(
                symbol='XAUUSD',
                direction='BUY',
                entry_price=2000.0,
                stop_loss=1980.0,
                take_profit=2050.0,
                volume=0.1
            )
            
            # Check trade plan structure
            self.assertIsInstance(trade_plan, dict)
            required_keys = ['symbol', 'direction', 'entry_price', 'stop_loss', 
                           'take_profit', 'volume', 'risk_reward_ratio']
            
            for key in required_keys:
                self.assertIn(key, trade_plan)
            
            # Check risk-reward ratio calculation
            risk = trade_plan['entry_price'] - trade_plan['stop_loss']
            reward = trade_plan['take_profit'] - trade_plan['entry_price']
            expected_ratio = reward / risk if risk > 0 else 0
            
            self.assertAlmostEqual(trade_plan['risk_reward_ratio'], expected_ratio, places=2)
            
        except Exception as e:
            self.fail(f"Trade plan generation failed: {e}")


class TestRiskManager(unittest.TestCase):
    """Test cases for Risk Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'max_position_size': 0.5,
            'max_daily_loss': 1000.0,
            'max_drawdown': 0.1,
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_atr_multiplier': 3.0
        }
        
        self.risk_manager = create_risk_manager(self.config)
        
        # Create sample account data
        self.account_data = {
            'balance': 10000.0,
            'equity': 10200.0,
            'margin': 500.0,
            'free_margin': 9700.0
        }
        
        # Create sample position data
        self.position_data = {
            'symbol': 'XAUUSD',
            'type': 'BUY',
            'volume': 0.1,
            'price_open': 2000.0,
            'price_current': 2010.0,
            'stop_loss': 1980.0,
            'take_profit': 2050.0
        }
    
    def test_risk_manager_initialization(self):
        """Test Risk Manager initialization."""
        self.assertIsNotNone(self.risk_manager)
        self.assertIsInstance(self.risk_manager, RiskManager)
        self.assertEqual(self.risk_manager.max_position_size, 0.5)
        self.assertEqual(self.risk_manager.max_daily_loss, 1000.0)
    
    def test_position_size_calculation(self):
        """Test position size calculation."""
        try:
            # Calculate position size based on risk
            risk_amount = 500.0  # $500 risk
            stop_loss_pips = 20  # 20 pips stop loss
            pip_value = 1.0  # $1 per pip
            
            position_size = self.risk_manager.calculate_position_size(
                risk_amount, stop_loss_pips, pip_value
            )
            
            # Check that position size is reasonable
            self.assertGreater(position_size, 0)
            self.assertLessEqual(position_size, self.risk_manager.max_position_size)
            
            # Verify risk calculation
            calculated_risk = position_size * stop_loss_pips * pip_value
            self.assertAlmostEqual(calculated_risk, risk_amount, places=2)
            
        except Exception as e:
            self.fail(f"Position size calculation failed: {e}")
    
    def test_risk_limits_checking(self):
        """Test risk limits checking."""
        try:
            # Check daily loss limit
            daily_pnl = -500.0
            is_within_limits = self.risk_manager.check_daily_loss_limit(daily_pnl)
            self.assertTrue(is_within_limits)
            
            # Check daily loss limit exceeded
            daily_pnl_exceeded = -1500.0
            is_within_limits_exceeded = self.risk_manager.check_daily_loss_limit(daily_pnl_exceeded)
            self.assertFalse(is_within_limits_exceeded)
            
            # Check drawdown limit
            current_drawdown = 0.05  # 5%
            is_drawdown_ok = self.risk_manager.check_drawdown_limit(current_drawdown)
            self.assertTrue(is_drawdown_ok)
            
            # Check drawdown limit exceeded
            high_drawdown = 0.15  # 15%
            is_drawdown_ok_exceeded = self.risk_manager.check_drawdown_limit(high_drawdown)
            self.assertFalse(is_drawdown_ok_exceeded)
            
        except Exception as e:
            self.fail(f"Risk limits checking failed: {e}")
    
    def test_stop_loss_calculation(self):
        """Test stop loss calculation."""
        try:
            # Calculate stop loss based on ATR
            entry_price = 2000.0
            atr = 10.0
            direction = 'BUY'
            
            stop_loss = self.risk_manager.calculate_stop_loss(
                entry_price, atr, direction
            )
            
            # For BUY position, stop loss should be below entry price
            self.assertLess(stop_loss, entry_price)
            
            # Check ATR multiplier
            expected_stop_loss = entry_price - (atr * self.risk_manager.stop_loss_atr_multiplier)
            self.assertAlmostEqual(stop_loss, expected_stop_loss, places=2)
            
        except Exception as e:
            self.fail(f"Stop loss calculation failed: {e}")
    
    def test_take_profit_calculation(self):
        """Test take profit calculation."""
        try:
            # Calculate take profit based on ATR
            entry_price = 2000.0
            atr = 10.0
            direction = 'BUY'
            
            take_profit = self.risk_manager.calculate_take_profit(
                entry_price, atr, direction
            )
            
            # For BUY position, take profit should be above entry price
            self.assertGreater(take_profit, entry_price)
            
            # Check ATR multiplier
            expected_take_profit = entry_price + (atr * self.risk_manager.take_profit_atr_multiplier)
            self.assertAlmostEqual(take_profit, expected_take_profit, places=2)
            
        except Exception as e:
            self.fail(f"Take profit calculation failed: {e}")
    
    def test_risk_adjustment(self):
        """Test dynamic risk adjustment."""
        try:
            # Test risk adjustment based on market conditions
            base_risk = 0.02  # 2% base risk
            volatility_multiplier = 1.5
            confidence_multiplier = 0.8
            
            adjusted_risk = self.risk_manager.adjust_risk(
                base_risk, volatility_multiplier, confidence_multiplier
            )
            
            # Check that risk was adjusted
            expected_risk = base_risk * volatility_multiplier * confidence_multiplier
            self.assertAlmostEqual(adjusted_risk, expected_risk, places=4)
            
        except Exception as e:
            self.fail(f"Risk adjustment failed: {e}")


class TestBacktester(unittest.TestCase):
    """Test cases for Backtester."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backtester = create_backtester()
        
        # Create sample historical data
        dates = pd.date_range('2023-08-11', periods=1000, freq='5T')
        self.historical_data = pd.DataFrame({
            'open': np.random.normal(2000, 50, 1000),
            'high': np.random.normal(2010, 50, 1000),
            'low': np.random.normal(1990, 50, 1000),
            'close': np.random.normal(2000, 50, 1000),
            'volume': np.random.normal(100, 20, 1000)
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        self.historical_data['high'] = np.maximum(self.historical_data['high'], self.historical_data['close'])
        self.historical_data['low'] = np.minimum(self.historical_data['low'], self.historical_data['close'])
        self.historical_data['open'] = np.clip(self.historical_data['open'], 
                                              self.historical_data['low'], 
                                              self.historical_data['high'])
    
    def test_backtester_initialization(self):
        """Test Backtester initialization."""
        self.assertIsNotNone(self.backtester)
        self.assertIsInstance(self.backtester, Backtester)
    
    def test_purged_kfold_initialization(self):
        """Test PurgedKFold initialization."""
        try:
            purged_kfold = PurgedKFold(n_splits=5, purge=10, embargo=5)
            self.assertIsInstance(purged_kfold, PurgedKFold)
            self.assertEqual(purged_kfold.n_splits, 5)
            self.assertEqual(purged_kfold.purge, 10)
            self.assertEqual(purged_kfold.embargo, 5)
            
        except Exception as e:
            self.fail(f"PurgedKFold initialization failed: {e}")
    
    def test_data_preparation(self):
        """Test data preparation for backtesting."""
        try:
            # Prepare data for backtesting
            prepared_data = self.backtester._prepare_data(self.historical_data)
            
            # Check that data was prepared correctly
            self.assertIsInstance(prepared_data, pd.DataFrame)
            self.assertGreater(len(prepared_data), 0)
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                self.assertIn(col, prepared_data.columns)
            
            # Check that no NaN values exist
            self.assertFalse(prepared_data.isnull().any().any())
            
        except Exception as e:
            self.fail(f"Data preparation failed: {e}")
    
    def test_trade_simulation(self):
        """Test trade simulation functionality."""
        try:
            # Create sample trade
            trade = {
                'entry_time': self.historical_data.index[100],
                'exit_time': self.historical_data.index[200],
                'entry_price': 2000.0,
                'exit_price': 2010.0,
                'direction': 'BUY',
                'volume': 0.1
            }
            
            # Simulate trade
            trade_result = self.backtester._simulate_trade(trade, self.historical_data)
            
            # Check trade result
            self.assertIsInstance(trade_result, dict)
            required_keys = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 
                           'direction', 'volume', 'pnl', 'return_pct']
            
            for key in required_keys:
                self.assertIn(key, trade_result)
            
            # Check P&L calculation
            expected_pnl = (trade_result['exit_price'] - trade_result['entry_price']) * trade_result['volume']
            self.assertAlmostEqual(trade_result['pnl'], expected_pnl, places=2)
            
        except Exception as e:
            self.fail(f"Trade simulation failed: {e}")
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        try:
            # Create sample trade results
            trade_results = [
                {'pnl': 100.0, 'return_pct': 0.05, 'duration': timedelta(hours=2)},
                {'pnl': -50.0, 'return_pct': -0.025, 'duration': timedelta(hours=1)},
                {'pnl': 200.0, 'return_pct': 0.1, 'duration': timedelta(hours=3)}
            ]
            
            # Calculate performance metrics
            metrics = self.backtester._calculate_performance_metrics(trade_results)
            
            # Check metrics structure
            self.assertIsInstance(metrics, dict)
            required_metrics = ['total_trades', 'winning_trades', 'losing_trades', 
                              'win_rate', 'total_pnl', 'total_return', 'sharpe_ratio']
            
            for metric in required_metrics:
                self.assertIn(metric, metrics)
            
            # Check calculations
            self.assertEqual(metrics['total_trades'], 3)
            self.assertEqual(metrics['winning_trades'], 2)
            self.assertEqual(metrics['losing_trades'], 1)
            self.assertAlmostEqual(metrics['win_rate'], 2/3, places=2)
            self.assertEqual(metrics['total_pnl'], 250.0)
            
        except Exception as e:
            self.fail(f"Performance metrics calculation failed: {e}")
    
    def test_backtest_execution(self):
        """Test complete backtest execution."""
        try:
            # Run backtest
            results = self.backtester.run_backtest(
                start_date=datetime(2023, 8, 11),
                end_date=datetime(2023, 8, 12),
                symbol='XAUUSD',
                timeframe='M5',
                initial_balance=10000.0
            )
            
            # Check results structure
            self.assertIsInstance(results, dict)
            required_keys = ['trades', 'performance_metrics', 'equity_curve', 'summary']
            
            for key in required_keys:
                self.assertIn(key, results)
            
        except Exception as e:
            self.fail(f"Backtest execution failed: {e}")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSuperTrendStrategy))
    test_suite.addTest(unittest.makeSuite(TestRiskManager))
    test_suite.addTest(unittest.makeSuite(TestBacktester))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running SignaMentis Strategy & Risk Management Tests...")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    print(f"\nTest execution completed with {'success' if success else 'failure'}.")
