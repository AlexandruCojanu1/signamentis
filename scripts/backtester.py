"""
SignaMentis Backtester Module

This module implements comprehensive backtesting for trading strategies with AI ensemble predictions.
Includes Purged K-Fold Cross-Validation for time series data and realistic slippage/spread modeling.

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


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation for time series data.
    
    Implements embargo periods to prevent data leakage between train/test sets.
    """
    
    def __init__(self, n_splits: int = 5, embargo_bars: int = 20):
        """
        Initialize Purged K-Fold.
        
        Args:
            n_splits: Number of folds
            embargo_bars: Number of bars to embargo between train/test sets
        """
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        logger.info(f"Purged K-Fold initialized: {n_splits} splits, {embargo_bars} bars embargo")
    
    def split(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with embargo.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(data)
        fold_size = n_samples // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            # Test set indices
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits - 1 else n_samples
            
            # Apply embargo
            train_end = max(0, test_start - self.embargo_bars)
            train_start = max(0, (i - 1) * fold_size + self.embargo_bars) if i > 0 else 0
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        return splits


class Backtester:
    """
    Comprehensive backtester for trading strategies with AI ensemble predictions.
    
    Features:
    - Realistic slippage and spread modeling
    - Session-based cost adjustments
    - Purged K-Fold Cross-Validation
    - Comprehensive performance metrics
    - Equity curve analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the backtester.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or {}
        
        # Backtesting parameters
        self.initial_capital = self.config.get('initial_capital', 100000)
        self.commission_per_lot = self.config.get('commission_per_lot', 7.0)  # $7 per lot
        self.slippage_pips = self.config.get('slippage_pips', 0.5)
        self.spread_pips = self.config.get('spread_pips', 2.0)
        
        # Session-based adjustments
        self.session_costs = {
            'asia': {'spread_multiplier': 1.2, 'slippage_multiplier': 1.1},
            'london': {'spread_multiplier': 1.0, 'slippage_multiplier': 1.0},
            'new_york': {'spread_multiplier': 0.9, 'slippage_multiplier': 0.9}
        }
        
        # Performance tracking
        self.equity_curve = []
        self.trade_history = []
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        
        # Cross-validation
        self.cv_enabled = self.config.get('cv_enabled', True)
        self.n_folds = self.config.get('n_folds', 5)
        self.embargo_bars = self.config.get('embargo_bars', 20)
        
        logger.info("Backtester initialized")
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy,
                    ai_ensemble,
                    risk_manager,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> BacktestResult:
        """
        Run comprehensive backtest.
        
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
        logger.info("Starting comprehensive backtest...")
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if len(data) == 0:
            logger.error("No data available for backtest")
            return None
            
        logger.info(f"Backtest period: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Data points: {len(data)}")
        
        # Calculate SuperTrend if not present
        if 'supertrend' not in data.columns:
            data = strategy.calculate_supertrend(data)
        
        # Run cross-validation if enabled
        if self.cv_enabled:
            cv_results = self._run_cross_validation(data, strategy, ai_ensemble, risk_manager)
            logger.info(f"Cross-validation completed: {len(cv_results)} folds")
        else:
            cv_results = None
        
        # Run full backtest
        full_result = self._run_single_backtest(data, strategy, ai_ensemble, risk_manager)
        
        # Combine results
        if cv_results:
            full_result.fold_results = cv_results
        
        logger.info("Backtest completed successfully!")
        return full_result
    
    def _run_cross_validation(self, 
                             data: pd.DataFrame,
                             strategy,
                             ai_ensemble,
                             risk_manager) -> List[Dict]:
        """
        Run Purged K-Fold Cross-Validation.
        
        Args:
            data: Historical market data
            strategy: Trading strategy instance
            ai_ensemble: AI ensemble instance
            risk_manager: Risk manager instance
            
        Returns:
            List of fold results
        """
        cv = PurgedKFold(n_splits=self.n_folds, embargo_bars=self.embargo_bars)
        splits = cv.split(data)
        
        fold_results = []
        
        for fold_idx, (train_indices, test_indices) in enumerate(splits):
            logger.info(f"Running fold {fold_idx + 1}/{len(splits)}")
            
            # Split data
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
            
            # Train models on training data (if needed)
            # For now, we'll assume models are pre-trained
            
            # Run backtest on test data
            fold_result = self._run_single_backtest(
                test_data, strategy, ai_ensemble, risk_manager, 
                fold_name=f"fold_{fold_idx + 1}"
            )
            
            fold_results.append({
                'fold': fold_idx + 1,
                'train_size': len(train_indices),
                'test_size': len(test_indices),
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
    
    def _run_single_backtest(self, 
                            data: pd.DataFrame,
                            strategy,
                            ai_ensemble,
                            risk_manager,
                            fold_name: str = "full") -> BacktestResult:
        """
        Run single backtest on data.
        
        Args:
            data: Market data for backtest
            strategy: Trading strategy instance
            ai_ensemble: AI ensemble instance
            risk_manager: Risk manager instance
            fold_name: Name of the fold for identification
            
        Returns:
            BacktestResult object
        """
        # Reset state
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        self.equity_curve = [self.initial_capital]
        self.trade_history = []
        
        # Initialize strategy
        strategy.current_positions = []
        strategy.trades = []
        
        # Process each bar
        for i in range(100, len(data)):  # Start from 100 to have enough history
            current_bar = data.iloc[i]
            current_data = data.iloc[:i+1]
            
            # Update existing positions
            strategy.update_positions(current_bar['close'], current_bar.name)
            
            # Check for new signals
            breakout_signal = strategy.detect_breakout(current_data, i)
            
            if breakout_signal:
                # Get AI prediction
                ai_prediction = self._get_ai_prediction(ai_ensemble, current_data, i)
                
                # Validate signal
                if strategy.validate_signal(breakout_signal, ai_prediction, current_data):
                    # Generate trade plan
                    trade_plan = strategy.generate_trade_plan(
                        breakout_signal, ai_prediction, risk_manager
                    )
                    
                    if trade_plan:
                        # Execute trade
                        self._execute_trade(trade_plan, current_bar, current_data)
            
            # Update equity curve
            self._update_equity_curve(current_bar.name)
        
        # Calculate final metrics
        result = self._calculate_performance_metrics()
        
        logger.info(f"{fold_name} backtest completed: {result.total_trades} trades, "
                   f"Win Rate: {result.win_rate:.2%}, P&L: ${result.total_pnl:.2f}")
        
        return result
    
    def _get_ai_prediction(self, 
                          ai_ensemble,
                          data: pd.DataFrame,
                          current_index: int) -> Dict:
        """
        Get AI ensemble prediction for current data.
        
        Args:
            ai_ensemble: AI ensemble instance
            data: Market data up to current index
            current_index: Current bar index
            
        Returns:
            AI prediction dictionary
        """
        try:
            # Prepare features for AI prediction
            features = self._prepare_features_for_prediction(data, current_index)
            
            # Get ensemble prediction
            prediction = ai_ensemble.get_ensemble_prediction(features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error getting AI prediction: {e}")
            # Return neutral prediction
            return {
                'predicted_direction': 0.5,
                'confidence': 0.5,
                'price_target': data.iloc[current_index]['close']
            }
    
    def _prepare_features_for_prediction(self, 
                                       data: pd.DataFrame,
                                       current_index: int) -> pd.DataFrame:
        """
        Prepare features for AI prediction.
        
        Args:
            data: Market data
            current_index: Current bar index
            
        Returns:
            DataFrame with features for prediction
        """
        # Use last 100 bars for prediction
        start_idx = max(0, current_index - 99)
        prediction_data = data.iloc[start_idx:current_index + 1].copy()
        
        # Ensure we have all required features
        required_features = ['rsi', 'macd', 'atr', 'supertrend', 'volume']
        for feature in required_features:
            if feature not in prediction_data.columns:
                # Create dummy feature if missing
                prediction_data[feature] = 0.0
        
        return prediction_data
    
    def _execute_trade(self, 
                      trade_plan: Dict,
                      current_bar: pd.Series,
                      current_data: pd.DataFrame):
        """
        Execute trade with realistic costs.
        
        Args:
            trade_plan: Trade plan from strategy
            current_bar: Current market bar
            current_data: Current market data
        """
        try:
            # Calculate session-based costs
            session = self._get_current_session(current_bar.name)
            cost_multipliers = self.session_costs.get(session, self.session_costs['london'])
            
            # Apply costs
            spread_cost = self.spread_pips * cost_multipliers['spread_multiplier'] * 0.1
            slippage_cost = self.slippage_pips * cost_multipliers['slippage_multiplier'] * 0.1
            
            # Calculate commission
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
            strategy.current_positions.append(position)
            
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
            
            logger.info(f"Trade executed: {trade_plan['signal']} {trade_plan['lot_size']} lots "
                       f"at {entry_price:.2f}, SL: {trade_plan['stop_loss']:.2f}, "
                       f"TP: {trade_plan['take_profit']:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _get_current_session(self, timestamp: datetime) -> str:
        """
        Determine current trading session.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Session name
        """
        hour = timestamp.hour
        
        if 0 <= hour < 8:
            return 'asia'
        elif 8 <= hour < 16:
            return 'london'
        else:
            return 'new_york'
    
    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve with current positions."""
        try:
            # Calculate current portfolio value
            portfolio_value = self.current_capital
            
            for position in strategy.current_positions:
                if position['status'] == 'open':
                    # Calculate unrealized P&L
                    current_price = data.loc[timestamp, 'close']
                    
                    if position['signal'] == 'BUY':
                        unrealized_pnl = (current_price - position['entry_price']) * position['lot_size'] * 100000
                    else:  # SELL
                        unrealized_pnl = (position['entry_price'] - current_price) * position['lot_size'] * 100000
                    
                    portfolio_value += unrealized_pnl
            
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
    
    def _calculate_performance_metrics(self) -> BacktestResult:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            BacktestResult object
        """
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
            return None
    
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
    
    def generate_report(self, result: BacktestResult, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive backtest report.
        
        Args:
            result: Backtest result
            output_path: Path to save report
            
        Returns:
            Report content
        """
        try:
            report = []
            report.append("=" * 80)
            report.append("SIGNAMENTIS BACKTEST REPORT")
            report.append("=" * 80)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Summary statistics
            report.append("SUMMARY STATISTICS")
            report.append("-" * 40)
            report.append(f"Total Trades: {result.total_trades}")
            report.append(f"Winning Trades: {result.winning_trades}")
            report.append(f"Losing Trades: {result.losing_trades}")
            report.append(f"Win Rate: {result.win_rate:.2%}")
            report.append(f"Total P&L: ${result.total_pnl:,.2f}")
            report.append(f"Max Drawdown: {result.max_drawdown:.2%}")
            report.append(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            report.append(f"Profit Factor: {result.profit_factor:.2f}")
            report.append(f"Expectancy: ${result.expectancy:.2f}")
            report.append(f"Max Consecutive Losses: {result.max_consecutive_losses}")
            report.append("")
            
            # Performance metrics
            report.append("PERFORMANCE METRICS")
            report.append("-" * 40)
            for metric, value in result.performance_metrics.items():
                if isinstance(value, float):
                    report.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    report.append(f"{metric.replace('_', ' ').title()}: {value}")
            report.append("")
            
            # Cross-validation results
            if result.fold_results:
                report.append("CROSS-VALIDATION RESULTS")
                report.append("-" * 40)
                for fold in result.fold_results:
                    report.append(f"Fold {fold['fold']}: Win Rate: {fold['metrics']['win_rate']:.2%}, "
                               f"Profit Factor: {fold['metrics']['profit_factor']:.2f}, "
                               f"Sharpe: {fold['metrics']['sharpe_ratio']:.2f}")
                report.append("")
            
            # Trade analysis
            if result.trade_history:
                report.append("TRADE ANALYSIS")
                report.append("-" * 40)
                
                # P&L distribution
                pnls = [t.get('pnl', 0) for t in result.trade_history]
                report.append(f"Average Win: ${np.mean([p for p in pnls if p > 0]):.2f}")
                report.append(f"Average Loss: ${np.mean([p for p in pnls if p < 0]):.2f}")
                report.append(f"Largest Win: ${max(pnls):.2f}")
                report.append(f"Largest Loss: ${min(pnls):.2f}")
                report.append("")
            
            report_content = "\n".join(report)
            
            # Save report if path provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report_content)
                logger.info(f"Report saved to {output_path}")
            
            return report_content
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return "Error generating report"
    
    def plot_results(self, result: BacktestResult, output_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            result: Backtest result
            output_path: Path to save plots
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('SignaMentis Backtest Results', fontsize=16)
            
            # Equity curve
            axes[0, 0].plot(result.equity_curve.index, result.equity_curve.values, 'b-', linewidth=2)
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_xlabel('Trade Number')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Drawdown
            equity_series = pd.Series(result.equity_curve.values)
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max * 100
            axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            axes[0, 1].plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
            axes[0, 1].set_title('Drawdown (%)')
            axes[0, 1].set_xlabel('Trade Number')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Win/Loss distribution
            if result.trade_history:
                pnls = [t.get('pnl', 0) for t in result.trade_history]
                axes[1, 0].hist(pnls, bins=20, alpha=0.7, color='green', edgecolor='black')
                axes[1, 0].set_title('P&L Distribution')
                axes[1, 0].set_xlabel('P&L ($)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Performance metrics
            metrics = ['Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Max Drawdown']
            values = [result.win_rate, result.profit_factor, result.sharpe_ratio, result.max_drawdown]
            colors = ['green' if v > 0.5 else 'red' for v in values]
            
            bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
            axes[1, 1].set_title('Key Metrics')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plots saved to {output_path}")
            
            # plt.show()  # Commented out to avoid blocking in headless environments
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")


# Convenience function for creating backtester
def create_backtester(config: Optional[Dict] = None) -> Backtester:
    """
    Create a backtester instance with configuration.
    
    Args:
        config: Backtesting configuration dictionary
        
    Returns:
        Backtester instance
    """
    return Backtester(config)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
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
    backtester = create_backtester()
    
    # Mock strategy and AI ensemble (replace with actual instances)
    class MockStrategy:
        def __init__(self):
            self.current_positions = []
            self.trades = []
        
        def calculate_supertrend(self, df):
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
        sample_data, strategy, ai_ensemble, risk_manager,
        start_date='2024-01-01', end_date='2024-12-31'
    )
    
    # Generate report and plots
    if result:
        report = backtester.generate_report(result)
        print(report)
        
        # Save plots instead of showing them to avoid blocking
        try:
            backtester.plot_results(result, output_path="backtest_results.png")
            print("✅ Plots saved to backtest_results.png")
        except Exception as e:
            print(f"⚠️  Could not save plots: {e}")
    
    print("Backtester test completed successfully!")
