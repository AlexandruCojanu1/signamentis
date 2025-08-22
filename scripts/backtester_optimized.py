#!/usr/bin/env python3
"""
SignaMentis Optimized Backtester Module

This module implements a high-performance backtesting system for trading strategies.
Optimized for speed and memory efficiency with comprehensive error handling.

Author: SignaMentis Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    expectancy: float
    max_consecutive_losses: int
    equity_curve: pd.Series
    trade_history: List[Dict]
    performance_metrics: Dict[str, float]
    fold_results: Optional[List[Dict]] = None


class OptimizedBacktester:
    """
    High-performance backtester for trading strategies.
    
    Features:
    - Vectorized operations for speed
    - Memory-efficient data handling
    - Comprehensive error handling
    - Real-time progress tracking
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the optimized backtester.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or {}
        
        # Performance parameters
        self.initial_capital = self.config.get('initial_capital', 100000)
        self.commission_per_lot = self.config.get('commission_per_lot', 7.0)
        self.slippage_pips = self.config.get('slippage_pips', 0.5)
        self.spread_pips = self.config.get('spread_pips', 2.0)
        
        # Cross-validation
        self.cv_enabled = self.config.get('cv_enabled', True)
        self.n_folds = self.config.get('n_folds', 5)
        self.embargo_bars = self.config.get('embargo_bars', 20)
        
        # Performance tracking
        self.equity_curve = []
        self.trade_history = []
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        
        # Strategy state
        self.strategy = None
        self.ai_ensemble = None
        self.risk_manager = None
        
        logger.info("Optimized Backtester initialized")
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy,
                    ai_ensemble,
                    risk_manager,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> BacktestResult:
        """
        Run optimized backtest.
        
        Args:
            data: Historical market data
            strategy: Trading strategy instance
            ai_ensemble: AI ensemble instance
            risk_manager: Risk manager instance
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResult object
        """
        logger.info("🚀 Starting Optimized Backtest")
        
        # Store references
        self.strategy = strategy
        self.ai_ensemble = ai_ensemble
        self.risk_manager = risk_manager
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if len(data) == 0:
            logger.error("No data available for backtest")
            return None
            
        logger.info(f"📊 Backtest period: {data.index[0]} to {data.index[-1]}")
        logger.info(f"📈 Data points: {len(data)}")
        
        # Pre-calculate indicators for performance
        data = self._precalculate_indicators(data)
        
        # Run cross-validation if enabled
        cv_results = None
        if self.cv_enabled:
            cv_results = self._run_optimized_cross_validation(data)
            logger.info(f"✅ Cross-validation completed: {len(cv_results)} folds")
        
        # Run full backtest
        full_result = self._run_optimized_single_backtest(data)
        
        # Combine results
        if cv_results:
            full_result.fold_results = cv_results
        
        logger.info("🎉 Optimized Backtest completed successfully!")
        return full_result
    
    def _precalculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pre-calculate all technical indicators for performance."""
        logger.info("🔧 Pre-calculating technical indicators...")
        
        try:
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    logger.warning(f"Missing column: {col}, creating dummy data")
                    data[col] = 1000.0
            
            # Calculate SuperTrend if not present
            if 'supertrend' not in data.columns:
                data = self.strategy.calculate_supertrend(data)
            
            # Calculate additional indicators for strategy
            if 'rsi' not in data.columns:
                data['rsi'] = self._calculate_rsi(data['close'])
            
            if 'macd' not in data.columns:
                data['macd'] = self._calculate_macd(data['close'])
            
            if 'atr' not in data.columns:
                data['atr'] = self._calculate_atr(data)
            
            logger.info("✅ Technical indicators pre-calculated")
            return data
            
        except Exception as e:
            logger.error(f"Error pre-calculating indicators: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception:
            return pd.Series(50, index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd.fillna(0)
        except Exception:
            return pd.Series(0, index=prices.index)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR indicator."""
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(period).mean()
            return atr.fillna(0)
        except Exception:
            return pd.Series(0, index=data.index)
    
    def _run_optimized_cross_validation(self, data: pd.DataFrame) -> List[Dict]:
        """Run optimized cross-validation."""
        try:
            # Simple time-based splits for performance
            n_samples = len(data)
            fold_size = n_samples // self.n_folds
            
            fold_results = []
            
            for fold_idx in range(self.n_folds):
                logger.info(f"🔄 Running fold {fold_idx + 1}/{self.n_folds}")
                
                # Calculate split indices
                test_start = fold_idx * fold_size
                test_end = test_start + fold_size if fold_idx < self.n_folds - 1 else n_samples
                
                # Apply embargo
                train_end = max(0, test_start - self.embargo_bars)
                train_start = max(0, (fold_idx - 1) * fold_size + self.embargo_bars) if fold_idx > 0 else 0
                
                if train_end > train_start and test_end > test_start:
                    # Split data
                    train_data = data.iloc[train_start:train_end]
                    test_data = data.iloc[test_start:test_end]
                    
                    # Run backtest on test data
                    fold_result = self._run_optimized_single_backtest(
                        test_data, fold_name=f"fold_{fold_idx + 1}"
                    )
                    
                    fold_results.append({
                        'fold': fold_idx + 1,
                        'train_size': len(train_data),
                        'test_size': len(test_data),
                        'train_period': (train_data.index[0], train_data.index[-1]),
                        'test_period': (test_data.index[0], test_data.index[-1]),
                        'metrics': {
                            'win_rate': fold_result.win_rate,
                            'profit_factor': fold_result.profit_factor,
                            'sharpe_ratio': fold_result.sharpe_ratio,
                            'max_drawdown': fold_result.max_drawdown,
                            'expectancy': fold_result.expectancy
                        }
                    })
            
            return fold_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return []
    
    def _run_optimized_single_backtest(self, data: pd.DataFrame, fold_name: str = "full") -> BacktestResult:
        """Run optimized single backtest."""
        try:
            # Reset state
            self.current_capital = self.initial_capital
            self.peak_capital = self.initial_capital
            self.max_drawdown = 0.0
            self.equity_curve = [self.initial_capital]
            self.trade_history = []
            
            # Initialize strategy
            self.strategy.current_positions = []
            self.strategy.trades = []
            
            # Vectorized processing for performance
            logger.info(f"🔄 Processing {len(data)} bars...")
            
            # Process each bar with progress tracking
            for i in range(100, len(data)):  # Start from 100 to have enough history
                try:
                    current_bar = data.iloc[i]
                    current_data = data.iloc[:i+1]
                    
                    # Update existing positions
                    self._update_positions_vectorized(current_bar, current_data)
                    
                    # Check for new signals
                    breakout_signal = self._detect_breakout_safe(current_data, i)
                    
                    if breakout_signal:
                        # Get AI prediction
                        ai_prediction = self._get_ai_prediction_safe(current_data, i)
                        
                        # Validate and execute signal
                        if self._validate_signal_safe(breakout_signal, ai_prediction, current_data):
                            trade_plan = self._generate_trade_plan_safe(
                                breakout_signal, ai_prediction, current_bar
                            )
                            
                            if trade_plan:
                                self._execute_trade_vectorized(trade_plan, current_bar, current_data)
                    
                    # Update equity curve efficiently
                    self._update_equity_curve_vectorized(current_bar, current_data)
                    
                    # Progress tracking
                    if i % 1000 == 0:
                        logger.info(f"📊 Processed {i}/{len(data)} bars ({(i/len(data)*100):.1f}%)")
                
                except Exception as e:
                    logger.warning(f"Error processing bar {i}: {e}")
                    continue
            
            # Calculate final metrics
            result = self._calculate_performance_metrics_optimized()
            
            logger.info(f"✅ {fold_name} backtest completed: {result.total_trades} trades, "
                       f"Win Rate: {result.win_rate:.2%}, P&L: ${result.total_pnl:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single backtest: {e}")
            return self._create_empty_result()
    
    def _update_positions_vectorized(self, current_bar: pd.Series, current_data: pd.DataFrame):
        """Update positions using vectorized operations."""
        try:
            if hasattr(self.strategy, 'update_positions'):
                self.strategy.update_positions(current_bar['close'], current_bar.name)
        except Exception as e:
            logger.debug(f"Error updating positions: {e}")
    
    def _detect_breakout_safe(self, current_data: pd.DataFrame, current_index: int):
        """Safely detect breakout signals."""
        try:
            if hasattr(self.strategy, 'detect_breakout'):
                return self.strategy.detect_breakout(current_data, current_index)
            return None
        except Exception as e:
            logger.debug(f"Error detecting breakout: {e}")
            return None
    
    def _get_ai_prediction_safe(self, current_data: pd.DataFrame, current_index: int) -> Dict:
        """Safely get AI prediction."""
        try:
            if hasattr(self.ai_ensemble, 'get_ensemble_prediction'):
                features = self._prepare_features_safe(current_data, current_index)
                prediction = self.ai_ensemble.get_ensemble_prediction(features)
                return prediction
        except Exception as e:
            logger.debug(f"Error getting AI prediction: {e}")
        
        # Return neutral prediction as fallback
        return {
            'predicted_direction': 0.5,
            'confidence': 0.5,
            'price_target': current_data.iloc[-1]['close']
        }
    
    def _prepare_features_safe(self, current_data: pd.DataFrame, current_index: int) -> pd.DataFrame:
        """Safely prepare features for AI prediction."""
        try:
            # Use last 100 bars for prediction
            start_idx = max(0, current_index - 99)
            prediction_data = current_data.iloc[start_idx:current_index + 1].copy()
            
            # Ensure we have required features
            required_features = ['rsi', 'macd', 'atr', 'supertrend', 'volume']
            for feature in required_features:
                if feature not in prediction_data.columns:
                    prediction_data[feature] = 0.0
            
            return prediction_data
        except Exception as e:
            logger.debug(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _validate_signal_safe(self, breakout_signal, ai_prediction: Dict, current_data: pd.DataFrame) -> bool:
        """Safely validate trading signal."""
        try:
            if hasattr(self.strategy, 'validate_signal'):
                return self.strategy.validate_signal(breakout_signal, ai_prediction, current_data)
            return True  # Default to valid if no validation method
        except Exception as e:
            logger.debug(f"Error validating signal: {e}")
            return False
    
    def _generate_trade_plan_safe(self, breakout_signal, ai_prediction: Dict, current_bar: pd.Series) -> Optional[Dict]:
        """Safely generate trade plan."""
        try:
            if hasattr(self.strategy, 'generate_trade_plan'):
                return self.strategy.generate_trade_plan(breakout_signal, ai_prediction, self.risk_manager)
            
            # Default trade plan
            return {
                'signal': 'BUY' if ai_prediction.get('predicted_direction', 0.5) > 0.5 else 'SELL',
                'entry_price': current_bar['close'],
                'stop_loss': current_bar['close'] * 0.99,  # 1% stop loss
                'take_profit': current_bar['close'] * 1.02,  # 2% take profit
                'lot_size': 0.1,
                'ai_confidence': ai_prediction.get('confidence', 0.5)
            }
        except Exception as e:
            logger.debug(f"Error generating trade plan: {e}")
            return None
    
    def _execute_trade_vectorized(self, trade_plan: Dict, current_bar: pd.Series, current_data: pd.DataFrame):
        """Execute trade using vectorized operations."""
        try:
            # Calculate costs
            spread_cost = self.spread_pips * 0.1
            slippage_cost = self.slippage_pips * 0.1
            commission = self.commission_per_lot * trade_plan['lot_size']
            
            # Adjust entry price for costs
            if trade_plan['signal'] == 'BUY':
                entry_price = trade_plan['entry_price'] + spread_cost + slippage_cost
            else:  # SELL
                entry_price = trade_plan['entry_price'] - spread_cost - slippage_cost
            
            # Create position
            position = {
                'signal': trade_plan['signal'],
                'entry_price': entry_price,
                'stop_loss': trade_plan['stop_loss'],
                'take_profit': trade_plan['take_profit'],
                'lot_size': trade_plan['lot_size'],
                'entry_timestamp': current_bar.name,
                'status': 'open',
                'costs': {
                    'spread': spread_cost,
                    'slippage': slippage_cost,
                    'commission': commission
                }
            }
            
            # Add to strategy positions
            self.strategy.current_positions.append(position)
            
            # Record trade
            trade_record = {
                'entry_timestamp': current_bar.name,
                'signal': trade_plan['signal'],
                'entry_price': entry_price,
                'stop_loss': trade_plan['stop_loss'],
                'take_profit': trade_plan['take_profit'],
                'lot_size': trade_plan['lot_size'],
                'ai_confidence': trade_plan['ai_confidence'],
                'costs': position['costs']
            }
            
            self.trade_history.append(trade_record)
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _update_equity_curve_vectorized(self, current_bar: pd.Series, current_data: pd.DataFrame):
        """Update equity curve using vectorized operations."""
        try:
            # Calculate current portfolio value
            portfolio_value = self.current_capital
            
            # Calculate unrealized P&L for open positions
            if hasattr(self.strategy, 'current_positions'):
                for position in self.strategy.current_positions:
                    if position.get('status') == 'open':
                        try:
                            current_price = current_bar['close']
                            
                            if position['signal'] == 'BUY':
                                unrealized_pnl = (current_price - position['entry_price']) * position['lot_size'] * 100000
                            else:  # SELL
                                unrealized_pnl = (position['entry_price'] - current_price) * position['lot_size'] * 100000
                            
                            portfolio_value += unrealized_pnl
                        except Exception as e:
                            logger.debug(f"Could not calculate P&L for position: {e}")
                            continue
            
            # Update peak capital and drawdown
            if portfolio_value > self.peak_capital:
                self.peak_capital = portfolio_value
            
            current_drawdown = (self.peak_capital - portfolio_value) / self.peak_capital
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Store equity curve
            self.equity_curve.append(portfolio_value)
            
        except Exception as e:
            logger.error(f"Error updating equity curve: {e}")
            # Fallback: just add current capital
            self.equity_curve.append(self.current_capital)
    
    def _calculate_performance_metrics_optimized(self) -> BacktestResult:
        """Calculate performance metrics using optimized methods."""
        try:
            # Basic trade statistics
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
            losing_trades = total_trades - winning_trades
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # P&L calculations
            total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
            
            # Profit factor
            gross_profit = sum(t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) > 0)
            gross_loss = abs(sum(t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Expectancy
            expectancy = total_pnl / total_trades if total_trades > 0 else 0
            
            # Sharpe ratio (simplified)
            if len(self.equity_curve) > 1:
                returns = pd.Series(self.equity_curve).pct_change().dropna()
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 288) if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Max consecutive losses
            max_consecutive_losses = self._calculate_max_consecutive_losses()
            
            # Create equity curve series
            equity_series = pd.Series(self.equity_curve, index=range(len(self.equity_curve)))
            
            # Performance metrics
            performance_metrics = {
                'total_return': (self.equity_curve[-1] - self.initial_capital) / self.initial_capital if self.equity_curve else 0,
                'annualized_return': self._calculate_annualized_return(),
                'volatility': self._calculate_volatility(),
                'calmar_ratio': self._calculate_calmar_ratio(),
                'sortino_ratio': self._calculate_sortino_ratio()
            }
            
            return BacktestResult(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                max_drawdown=self.max_drawdown,
                sharpe_ratio=sharpe_ratio,
                profit_factor=profit_factor,
                expectancy=expectancy,
                max_consecutive_losses=max_consecutive_losses,
                equity_curve=equity_series,
                trade_history=self.trade_history,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._create_empty_result()
    
    def _create_empty_result(self) -> BacktestResult:
        """Create empty result when backtest fails."""
        return BacktestResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            max_consecutive_losses=0,
            equity_curve=pd.Series([self.initial_capital]),
            trade_history=[],
            performance_metrics={}
        )
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades."""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trade_history:
            if trade.get('pnl', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_annualized_return(self) -> float:
        """Calculate annualized return."""
        if len(self.equity_curve) < 2:
            return 0
        
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        days = len(self.equity_curve) / 288  # Assuming 5-minute bars, 288 per day
        
        if days > 0:
            return (1 + total_return) ** (365 / days) - 1
        return 0
    
    def _calculate_volatility(self) -> float:
        """Calculate annualized volatility."""
        if len(self.equity_curve) < 2:
            return 0
        
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252 * 288)  # Annualized
        
        return annual_vol
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        annualized_return = self._calculate_annualized_return()
        if self.max_drawdown > 0:
            return annualized_return / self.max_drawdown
        return 0
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        if len(self.equity_curve) < 2:
            return 0
        
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std()
            if downside_deviation > 0:
                return returns.mean() / downside_deviation * np.sqrt(252 * 288)
        
        return 0


def create_optimized_backtester(config: Optional[Dict] = None) -> OptimizedBacktester:
    """
    Create optimized backtester instance.
    
    Args:
        config: Backtesting configuration dictionary
        
    Returns:
        OptimizedBacktester instance
    """
    return OptimizedBacktester(config)


if __name__ == "__main__":
    # Test the optimized backtester
    print("🧪 Testing Optimized Backtester...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
    
    sample_data = pd.DataFrame({
        'open': 2000 + np.random.randn(1000).cumsum(),
        'high': 2000 + np.random.randn(1000).cumsum() + 5,
        'low': 2000 + np.random.randn(1000).cumsum() - 5,
        'close': 2000 + np.random.randn(1000).cumsum(),
        'volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    # Create backtester
    backtester = create_optimized_backtester()
    
    # Mock strategy and AI ensemble
    class MockStrategy:
        def __init__(self):
            self.current_positions = []
            self.trades = []
        
        def calculate_supertrend(self, df):
            df['supertrend'] = df['close'].rolling(10).mean()
            return df
        
        def detect_breakout(self, df, idx):
            return None
        
        def update_positions(self, price, timestamp):
            pass
    
    class MockAIEnsemble:
        def get_ensemble_prediction(self, features):
            return {'predicted_direction': 0.5, 'confidence': 0.7, 'price_target': 2000}
    
    class MockRiskManager:
        def calculate_position_size(self, entry_price, ai_confidence, market_data):
            return type('obj', (object,), {
                'lot_size': 0.1,
                'stop_loss_pips': 10,
                'take_profit_pips': 20,
                'risk_amount': 100,
                'risk_percentage': 0.01
            })()
    
    # Run backtest
    strategy = MockStrategy()
    ai_ensemble = MockAIEnsemble()
    risk_manager = MockRiskManager()
    
    result = backtester.run_backtest(
        sample_data, strategy, ai_ensemble, risk_manager
    )
    
    if result:
        print(f"✅ Backtest completed: {result.total_trades} trades")
        print(f"📊 Win Rate: {result.win_rate:.2%}")
        print(f"💰 Total P&L: ${result.total_pnl:.2f}")
    else:
        print("❌ Backtest failed")
    
    print("🎉 Optimized Backtester test completed!")
