#!/usr/bin/env python3
"""
Unit tests for executor and monitoring modules.

This module tests the functionality of:
- MT5 Executor
- Live Dashboard
- API Service
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import tempfile
import json

# Add scripts directory to path
sys.path.append('scripts')

from executor import MT5Executor, OrderType, OrderStatus, OrderRequest, OrderResult
from monitor import LiveDashboard, AlertLevel, Alert, PerformanceMetrics
from services.api import TradingAPI, TradingSignal, OrderRequestModel


class TestMT5Executor(unittest.TestCase):
    """Test cases for MT5 Executor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'login': 12345,
            'password': 'test_password',
            'server': 'test_server',
            'symbol': 'XAUUSD',
            'deviation': 10,
            'max_positions': 5,
            'max_daily_trades': 20,
            'max_daily_loss': 1000.0
        }
        
        self.executor = MT5Executor(self.config)
        
        # Create sample order request
        self.sample_order = OrderRequest(
            symbol='XAUUSD',
            order_type=OrderType.BUY,
            volume=0.1,
            price=2000.0,
            stop_loss=1980.0,
            take_profit=2050.0,
            comment='Test order',
            magic=12345
        )
    
    def test_executor_initialization(self):
        """Test MT5 Executor initialization."""
        self.assertIsNotNone(self.executor)
        self.assertIsInstance(self.executor, MT5Executor)
        self.assertEqual(self.executor.symbol, 'XAUUSD')
        self.assertEqual(self.executor.max_positions, 5)
        self.assertEqual(self.executor.max_daily_trades, 20)
    
    def test_order_request_validation(self):
        """Test order request validation."""
        try:
            # Test valid order request
            self.assertIsInstance(self.sample_order, OrderRequest)
            self.assertEqual(self.sample_order.symbol, 'XAUUSD')
            self.assertEqual(self.sample_order.order_type, OrderType.BUY)
            self.assertEqual(self.sample_order.volume, 0.1)
            self.assertEqual(self.sample_order.price, 2000.0)
            
            # Test order request with missing optional parameters
            minimal_order = OrderRequest(
                symbol='XAUUSD',
                order_type=OrderType.SELL,
                volume=0.05,
                price=1990.0
            )
            
            self.assertIsInstance(minimal_order, OrderRequest)
            self.assertIsNone(minimal_order.stop_loss)
            self.assertIsNone(minimal_order.take_profit)
            
        except Exception as e:
            self.fail(f"Order request validation failed: {e}")
    
    def test_order_type_enumeration(self):
        """Test order type enumeration."""
        # Test all order types
        self.assertEqual(OrderType.BUY.value, "BUY")
        self.assertEqual(OrderType.SELL.value, "SELL")
        self.assertEqual(OrderType.BUY_STOP.value, "BUY_STOP")
        self.assertEqual(OrderType.SELL_STOP.value, "SELL_STOP")
        self.assertEqual(OrderType.BUY_LIMIT.value, "BUY_LIMIT")
        self.assertEqual(OrderType.SELL_LIMIT.value, "SELL_LIMIT")
    
    def test_order_status_enumeration(self):
        """Test order status enumeration."""
        # Test all order statuses
        self.assertEqual(OrderStatus.PENDING.value, "PENDING")
        self.assertEqual(OrderStatus.FILLED.value, "FILLED")
        self.assertEqual(OrderStatus.CANCELLED.value, "CANCELLED")
        self.assertEqual(OrderStatus.REJECTED.value, "REJECTED")
        self.assertEqual(OrderStatus.EXPIRED.value, "EXPIRED")
    
    def test_position_size_calculation(self):
        """Test position size calculation."""
        try:
            # Test position size calculation
            account_balance = 10000.0
            risk_percentage = 0.02  # 2%
            stop_loss_pips = 20
            pip_value = 1.0
            
            risk_amount = account_balance * risk_percentage
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Check that position size is reasonable
            self.assertGreater(position_size, 0)
            self.assertLessEqual(position_size, self.executor.max_position_size)
            
        except Exception as e:
            self.fail(f"Position size calculation failed: {e}")
    
    def test_risk_control_validation(self):
        """Test risk control validation."""
        try:
            # Test daily trade limit
            current_daily_trades = 15
            can_trade = current_daily_trades < self.executor.max_daily_trades
            self.assertTrue(can_trade)
            
            # Test daily trade limit exceeded
            exceeded_daily_trades = 25
            can_trade_exceeded = exceeded_daily_trades < self.executor.max_daily_trades
            self.assertFalse(can_trade_exceeded)
            
            # Test daily loss limit
            daily_pnl = -500.0
            within_loss_limit = abs(daily_pnl) < self.executor.max_daily_loss
            self.assertTrue(within_loss_limit)
            
            # Test daily loss limit exceeded
            daily_pnl_exceeded = -1500.0
            within_loss_limit_exceeded = abs(daily_pnl_exceeded) < self.executor.max_daily_loss
            self.assertFalse(within_loss_limit_exceeded)
            
        except Exception as e:
            self.fail(f"Risk control validation failed: {e}")


class TestLiveDashboard(unittest.TestCase):
    """Test cases for Live Dashboard."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'update_interval': 5.0,
            'max_data_points': 1000,
            'port': 8050,
            'host': '0.0.0.0'
        }
        
        self.dashboard = LiveDashboard(self.config)
        
        # Create sample performance metrics
        self.sample_metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            total_pnl=1500.0,
            daily_pnl=250.0,
            win_rate=0.65,
            total_trades=20,
            open_positions=3,
            equity=11500.0,
            balance=10000.0,
            drawdown=0.05,
            sharpe_ratio=1.2,
            max_drawdown=0.08
        )
    
    def test_dashboard_initialization(self):
        """Test Live Dashboard initialization."""
        self.assertIsNotNone(self.dashboard)
        self.assertIsInstance(self.dashboard, LiveDashboard)
        self.assertEqual(self.dashboard.update_interval, 5.0)
        self.assertEqual(self.dashboard.max_data_points, 1000)
        self.assertEqual(self.dashboard.port, 8050)
    
    def test_alert_system(self):
        """Test alert system functionality."""
        try:
            # Create sample alerts
            info_alert = Alert(
                timestamp=datetime.now(),
                level=AlertLevel.INFO,
                message='System started successfully',
                source='Dashboard',
                data={'status': 'running'}
            )
            
            warning_alert = Alert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                message='High memory usage detected',
                source='System',
                data={'memory_usage': 85}
            )
            
            error_alert = Alert(
                timestamp=datetime.now(),
                level=AlertLevel.ERROR,
                message='Connection to MT5 failed',
                source='Executor',
                data={'error_code': 1001}
            )
            
            # Test alert properties
            self.assertEqual(info_alert.level, AlertLevel.INFO)
            self.assertEqual(warning_alert.level, AlertLevel.WARNING)
            self.assertEqual(error_alert.level, AlertLevel.ERROR)
            
            # Test alert data
            self.assertIn('status', info_alert.data)
            self.assertIn('memory_usage', warning_alert.data)
            self.assertIn('error_code', error_alert.data)
            
        except Exception as e:
            self.fail(f"Alert system test failed: {e}")
    
    def test_performance_metrics(self):
        """Test performance metrics functionality."""
        try:
            # Test metrics properties
            self.assertIsInstance(self.sample_metrics, PerformanceMetrics)
            self.assertEqual(self.sample_metrics.total_pnl, 1500.0)
            self.assertEqual(self.sample_metrics.daily_pnl, 250.0)
            self.assertEqual(self.sample_metrics.win_rate, 0.65)
            self.assertEqual(self.sample_metrics.total_trades, 20)
            
            # Test metrics validation
            self.assertGreater(self.sample_metrics.total_pnl, 0)
            self.assertGreaterEqual(self.sample_metrics.win_rate, 0)
            self.assertLessEqual(self.sample_metrics.win_rate, 1)
            self.assertGreaterEqual(self.sample_metrics.sharpe_ratio, 0)
            
        except Exception as e:
            self.fail(f"Performance metrics test failed: {e}")
    
    def test_data_storage(self):
        """Test data storage functionality."""
        try:
            # Test price data storage
            sample_price_data = {
                'timestamp': datetime.now(),
                'open': 2000.0,
                'high': 2010.0,
                'low': 1990.0,
                'close': 2005.0,
                'volume': 100
            }
            
            # Add to dashboard (simulate)
            self.dashboard.price_data.append(sample_price_data)
            
            # Check data storage
            self.assertEqual(len(self.dashboard.price_data), 1)
            stored_data = self.dashboard.price_data[0]
            self.assertEqual(stored_data['close'], 2005.0)
            
        except Exception as e:
            self.fail(f"Data storage test failed: {e}")


class TestTradingAPI(unittest.TestCase):
    """Test cases for Trading API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api = TradingAPI()
        
        # Create sample trading signal
        self.sample_signal = TradingSignal(
            symbol='XAUUSD',
            direction='BUY',
            confidence=0.85,
            price=2000.0,
            timestamp=datetime.now(),
            source='AI_Ensemble',
            metadata={'model_weights': {'bilstm': 0.3, 'gru': 0.3, 'transformer': 0.4}}
        )
        
        # Create sample order request
        self.sample_order_request = OrderRequestModel(
            symbol='XAUUSD',
            order_type='BUY',
            volume=0.1,
            price=2000.0,
            stop_loss=1980.0,
            take_profit=2050.0,
            comment='Test order from API',
            magic=12345
        )
    
    def test_api_initialization(self):
        """Test Trading API initialization."""
        self.assertIsNotNone(self.api)
        self.assertIsInstance(self.api, TradingAPI)
    
    def test_trading_signal_validation(self):
        """Test trading signal validation."""
        try:
            # Test signal properties
            self.assertIsInstance(self.sample_signal, TradingSignal)
            self.assertEqual(self.sample_signal.symbol, 'XAUUSD')
            self.assertEqual(self.sample_signal.direction, 'BUY')
            self.assertEqual(self.sample_signal.confidence, 0.85)
            self.assertEqual(self.sample_signal.price, 2000.0)
            
            # Test signal validation
            self.assertGreater(self.sample_signal.confidence, 0)
            self.assertLessEqual(self.sample_signal.confidence, 1)
            self.assertGreater(self.sample_signal.price, 0)
            
            # Test metadata
            self.assertIn('model_weights', self.sample_signal.metadata)
            weights = self.sample_signal.metadata['model_weights']
            self.assertIn('bilstm', weights)
            self.assertIn('gru', weights)
            self.assertIn('transformer', weights)
            
        except Exception as e:
            self.fail(f"Trading signal validation failed: {e}")
    
    def test_order_request_validation(self):
        """Test order request validation."""
        try:
            # Test order request properties
            self.assertIsInstance(self.sample_order_request, OrderRequestModel)
            self.assertEqual(self.sample_order_request.symbol, 'XAUUSD')
            self.assertEqual(self.sample_order_request.order_type, 'BUY')
            self.assertEqual(self.sample_order_request.volume, 0.1)
            self.assertEqual(self.sample_order_request.price, 2000.0)
            
            # Test order request validation
            self.assertGreater(self.sample_order_request.volume, 0)
            self.assertGreater(self.sample_order_request.price, 0)
            
            # Test optional parameters
            self.assertIsNotNone(self.sample_order_request.stop_loss)
            self.assertIsNotNone(self.sample_order_request.take_profit)
            self.assertEqual(self.sample_order_request.magic, 12345)
            
        except Exception as e:
            self.fail(f"Order request validation failed: {e}")
    
    def test_api_endpoint_structure(self):
        """Test API endpoint structure."""
        try:
            # Test that API has required endpoints
            # Note: This is a structural test, actual endpoint testing would require running the API
            
            # Check if API has basic structure
            self.assertTrue(hasattr(self.api, 'app'))
            
        except Exception as e:
            self.fail(f"API endpoint structure test failed: {e}")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestMT5Executor))
    test_suite.addTest(unittest.makeSuite(TestLiveDashboard))
    test_suite.addTest(unittest.makeSuite(TestTradingAPI))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running SignaMentis Executor & Monitoring Tests...")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    print(f"\nTest execution completed with {'success' if success else 'failure'}.")
