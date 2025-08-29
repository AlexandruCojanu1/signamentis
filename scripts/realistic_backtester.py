#!/usr/bin/env python3
"""
Realistic Backtester for SignaMentis

This module implements realistic backtesting with proper costs:
- Spread costs
- Commission 
- Slippage
- Market session impact
- Real market conditions
"""

import sys
import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealisticBacktester:
    """
    Realistic backtester with proper trading costs and market conditions.
    
    Features:
    - Real spread costs per session
    - Commission and slippage
    - Market session impact
    - SuperTrend + RF strategy
    - Comprehensive performance metrics
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize realistic backtester.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Trading costs
        self.spread_pips = self.config.get('spread_pips', 0.3)  # Average spread in pips
        self.commission_per_lot = self.config.get('commission_per_lot', 7.0)  # USD per lot
        self.slippage_pips = self.config.get('slippage_pips', 0.1)  # Slippage in pips
        
        # Position sizing
        self.initial_balance = self.config.get('initial_balance', 10000.0)  # USD
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)  # 2% per trade
        self.max_position_size = self.config.get('max_position_size', 1.0)  # Max 1 lot
        
        # Strategy parameters
        self.supertrend_period = self.config.get('supertrend_period', 10)
        self.supertrend_multiplier = self.config.get('supertrend_multiplier', 3.0)
        self.rf_confidence_threshold = self.config.get('rf_confidence_threshold', 0.6)
        
        # Session definitions (GMT/UTC times)
        self.sessions = {
            'asia': {'start': time(22, 0), 'end': time(8, 0), 'spread_multiplier': 1.2},
            'london': {'start': time(8, 0), 'end': time(16, 0), 'spread_multiplier': 0.8},
            'newyork': {'start': time(13, 0), 'end': time(22, 0), 'spread_multiplier': 0.9}
        }
        
        # Results storage
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
        
        logger.info("RealisticBacktester initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f).get('realistic_backtester', {})
        else:
            # Default configuration
            return {
                'spread_pips': 0.3,
                'commission_per_lot': 7.0,
                'slippage_pips': 0.1,
                'initial_balance': 10000.0,
                'risk_per_trade': 0.02,
                'max_position_size': 1.0,
                'supertrend_period': 10,
                'supertrend_multiplier': 3.0,
                'rf_confidence_threshold': 0.6
            }
    
    def get_session(self, timestamp: pd.Timestamp) -> str:
        """Get trading session for given timestamp."""
        hour = timestamp.time()
        
        # Check each session
        for session_name, session_info in self.sessions.items():
            start_time = session_info['start']
            end_time = session_info['end']
            
            if start_time > end_time:  # Overnight session (Asia)
                if hour >= start_time or hour < end_time:
                    return session_name
            else:  # Day session
                if start_time <= hour < end_time:
                    return session_name
        
        return 'asian'  # Default
    
    def calculate_position_size(self, balance: float, atr: float, risk_per_trade: float) -> float:
        """
        Calculate position size based on risk management.
        
        Args:
            balance: Current account balance
            atr: Average True Range for stop loss calculation
            risk_per_trade: Risk percentage per trade
            
        Returns:
            Position size in lots
        """
        # Risk amount in USD
        risk_amount = balance * risk_per_trade
        
        # Stop loss in pips (2 * ATR converted to pips)
        stop_loss_pips = atr * 10000 * 2  # ATR to pips * 2
        
        # Position size calculation
        pip_value = 10  # For XAU/USD, 1 pip = $10 per lot
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Cap at maximum position size
        position_size = min(position_size, self.max_position_size)
        
        return max(0.01, position_size)  # Minimum 0.01 lots
    
    def calculate_costs(self, position_size: float, session: str) -> Dict[str, float]:
        """
        Calculate all trading costs for a position.
        
        Args:
            position_size: Position size in lots
            session: Trading session
            
        Returns:
            Dictionary with cost breakdown
        """
        # Session-adjusted spread
        session_multiplier = self.sessions[session].get('spread_multiplier', 1.0)
        effective_spread = self.spread_pips * session_multiplier
        
        # Calculate costs
        spread_cost = effective_spread * 10 * position_size  # Spread cost in USD
        commission_cost = self.commission_per_lot * position_size  # Commission in USD
        slippage_cost = self.slippage_pips * 10 * position_size  # Slippage cost in USD
        
        total_cost = spread_cost + commission_cost + slippage_cost
        
        return {
            'spread_cost': spread_cost,
            'commission_cost': commission_cost,
            'slippage_cost': slippage_cost,
            'total_cost': total_cost,
            'session': session,
            'session_multiplier': session_multiplier
        }
    
    def generate_signals(self, df: pd.DataFrame, model_path: str) -> pd.DataFrame:
        """
        Generate trading signals using SuperTrend + Random Forest.
        
        Args:
            df: OHLCV data with features
            model_path: Path to trained Random Forest model
            
        Returns:
            DataFrame with signals
        """
        logger.info("Generating trading signals")
        
        # Load Random Forest model
        if Path(model_path).exists():
            model_data = joblib.load(model_path)
            rf_model = model_data['model']
            scaler = model_data['scaler']
        else:
            logger.warning(f"Model file {model_path} not found. Using dummy signals.")
            # Generate dummy signals for testing
            df['rf_signal'] = np.random.choice([0, 1], size=len(df))
            df['rf_confidence'] = np.random.uniform(0.4, 0.9, size=len(df))
            df['signal'] = df['rf_signal']
            return df
        
        # Prepare features for prediction
        feature_cols = [col for col in df.columns if not col.startswith('target_') and col != 'timestamp']
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        
        # Get Random Forest predictions
        rf_predictions = rf_model.predict(X_scaled)
        rf_probabilities = rf_model.predict_proba(X_scaled)
        rf_confidence = np.max(rf_probabilities, axis=1)
        
        # Add RF signals
        df['rf_signal'] = rf_predictions
        df['rf_confidence'] = rf_confidence
        
        # Generate combined signals (SuperTrend + RF)
        # Use SuperTrend as base signal, RF as confirmation
        supertrend_signal = (df['supertrend'] < df['close']).astype(int)
        
        # Combined signal: both SuperTrend and RF agree + high confidence
        df['signal'] = (
            (supertrend_signal == df['rf_signal']) & 
            (df['rf_confidence'] >= self.rf_confidence_threshold)
        ).astype(int)
        
        logger.info(f"Signals generated: {df['signal'].sum()} buy signals out of {len(df)} bars")
        return df
    
    def run_backtest(self, df: pd.DataFrame, model_path: str) -> Dict:
        """
        Run realistic backtest with proper costs.
        
        Args:
            df: OHLCV data with features
            model_path: Path to trained Random Forest model
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting realistic backtest")
        
        # Generate signals
        df_with_signals = self.generate_signals(df, model_path)
        
        # Initialize backtest variables
        balance = self.initial_balance
        position = 0  # Current position size
        entry_price = 0
        entry_time = None
        
        self.trades = []
        self.equity_curve = []
        
        # Backtest loop
        for i in range(1, len(df_with_signals)):
            current_bar = df_with_signals.iloc[i]
            prev_bar = df_with_signals.iloc[i-1]
            
            timestamp = current_bar['timestamp'] if 'timestamp' in current_bar else i
            session = self.get_session(pd.to_datetime(timestamp) if isinstance(timestamp, str) else pd.Timestamp.now())
            
            # Close existing position
            if position != 0:
                # Check for exit conditions
                current_signal = current_bar['signal']
                
                # Exit on signal change or stop loss
                should_exit = False
                exit_reason = ""
                
                if position > 0 and current_signal == 0:
                    should_exit = True
                    exit_reason = "signal_change"
                elif position < 0 and current_signal == 1:
                    should_exit = True
                    exit_reason = "signal_change"
                
                # Simple stop loss based on SuperTrend
                if position > 0 and current_bar['close'] < current_bar['supertrend']:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif position < 0 and current_bar['close'] > current_bar['supertrend']:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                if should_exit:
                    # Close position
                    exit_price = current_bar['close']
                    position_size = abs(position)
                    
                    # Calculate P&L
                    if position > 0:
                        pnl = (exit_price - entry_price) * position_size * 100  # Convert to USD
                    else:
                        pnl = (entry_price - exit_price) * position_size * 100
                    
                    # Calculate exit costs
                    exit_costs = self.calculate_costs(position_size, session)
                    net_pnl = pnl - exit_costs['total_cost']
                    
                    # Update balance
                    balance += net_pnl
                    
                    # Record trade
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': position_size,
                        'direction': 'long' if position > 0 else 'short',
                        'pnl': pnl,
                        'costs': exit_costs['total_cost'],
                        'net_pnl': net_pnl,
                        'exit_reason': exit_reason,
                        'session': session
                    }
                    self.trades.append(trade)
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    entry_time = None
            
            # Open new position
            if position == 0 and current_bar['signal'] == 1:
                # Calculate position size
                atr = current_bar.get('atr', 0.01)  # Default ATR if not available
                position_size = self.calculate_position_size(balance, atr, self.risk_per_trade)
                
                # Calculate entry costs
                entry_costs = self.calculate_costs(position_size, session)
                
                # Check if we have enough balance for costs
                if balance > entry_costs['total_cost']:
                    # Determine direction based on SuperTrend
                    if current_bar['close'] > current_bar['supertrend']:
                        position = position_size  # Long position
                    else:
                        position = -position_size  # Short position
                    
                    entry_price = current_bar['close']
                    entry_time = timestamp
                    
                    # Deduct entry costs
                    balance -= entry_costs['total_cost']
            
            # Record equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'balance': balance,
                'equity': balance + (position * (current_bar['close'] - entry_price) * 100 if position != 0 else 0)
            })
        
        # Close any remaining position
        if position != 0:
            final_bar = df_with_signals.iloc[-1]
            exit_price = final_bar['close']
            position_size = abs(position)
            
            if position > 0:
                pnl = (exit_price - entry_price) * position_size * 100
            else:
                pnl = (entry_price - exit_price) * position_size * 100
            
            final_session = self.get_session(pd.to_datetime(final_bar.get('timestamp', pd.Timestamp.now())))
            exit_costs = self.calculate_costs(position_size, final_session)
            net_pnl = pnl - exit_costs['total_cost']
            balance += net_pnl
            
            trade = {
                'entry_time': entry_time,
                'exit_time': final_bar.get('timestamp', 'final'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'direction': 'long' if position > 0 else 'short',
                'pnl': pnl,
                'costs': exit_costs['total_cost'],
                'net_pnl': net_pnl,
                'exit_reason': 'final_close',
                'session': final_session
            }
            self.trades.append(trade)
        
        # Calculate performance metrics
        self.performance_metrics = self.calculate_performance_metrics()
        
        logger.info(f"Backtest completed: {len(self.trades)} trades, Final balance: ${balance:.2f}")
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'performance_metrics': self.performance_metrics,
            'final_balance': balance
        }
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {}
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['net_pnl'].sum()
        gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average metrics
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Drawdown calculation
        equity_df['peak'] = equity_df['equity'].expanding().max()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Returns calculation
        initial_balance = self.initial_balance
        final_balance = equity_df['equity'].iloc[-1] if len(equity_df) > 0 else initial_balance
        total_return = (final_balance - initial_balance) / initial_balance
        
        # Sharpe ratio calculation (simplified - daily returns)
        if len(equity_df) > 1:
            equity_df['daily_return'] = equity_df['equity'].pct_change()
            sharpe_ratio = equity_df['daily_return'].mean() / equity_df['daily_return'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'initial_balance': initial_balance,
            'final_balance': final_balance
        }
    
    def generate_backtest_report(self, output_dir: str = "results") -> str:
        """
        Generate comprehensive backtest report.
        
        Args:
            output_dir: Output directory for report
            
        Returns:
            Path to generated report
        """
        logger.info("Generating backtest report")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report
        report = {
            'backtest_summary': {
                'strategy': 'SuperTrend + Random Forest',
                'generation_date': timestamp,
                'total_trades': len(self.trades),
                'data_period': 'Historical Data'
            },
            'performance_metrics': self.performance_metrics,
            'trading_costs': {
                'spread_pips': self.spread_pips,
                'commission_per_lot': self.commission_per_lot,
                'slippage_pips': self.slippage_pips
            },
            'strategy_parameters': {
                'supertrend_period': self.supertrend_period,
                'supertrend_multiplier': self.supertrend_multiplier,
                'rf_confidence_threshold': self.rf_confidence_threshold,
                'risk_per_trade': self.risk_per_trade,
                'max_position_size': self.max_position_size
            },
            'session_analysis': self._analyze_session_performance(),
            'acceptance_criteria_check': self._check_acceptance_criteria()
        }
        
        # Save report
        report_path = Path(output_dir) / f"backtest_report_{timestamp}.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        # Generate plots
        self._generate_plots(output_dir, timestamp)
        
        logger.info(f"Backtest report saved: {report_path}")
        return str(report_path)
    
    def _analyze_session_performance(self) -> Dict:
        """Analyze performance by trading session."""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        session_analysis = {}
        
        for session in ['asia', 'london', 'newyork']:
            session_trades = trades_df[trades_df['session'] == session]
            if len(session_trades) > 0:
                session_analysis[session] = {
                    'total_trades': len(session_trades),
                    'win_rate': len(session_trades[session_trades['net_pnl'] > 0]) / len(session_trades),
                    'total_pnl': session_trades['net_pnl'].sum(),
                    'avg_pnl': session_trades['net_pnl'].mean()
                }
        
        return session_analysis
    
    def _check_acceptance_criteria(self) -> Dict:
        """Check against acceptance criteria."""
        metrics = self.performance_metrics
        
        # Acceptance criteria from user requirements
        target_profit_factor = 1.2
        target_max_drawdown = 0.15  # 15%
        target_sharpe = 0.8
        
        return {
            'profit_factor': {
                'value': metrics.get('profit_factor', 0),
                'target': target_profit_factor,
                'pass': metrics.get('profit_factor', 0) >= target_profit_factor
            },
            'max_drawdown': {
                'value': abs(metrics.get('max_drawdown', 1)),
                'target': target_max_drawdown,
                'pass': abs(metrics.get('max_drawdown', 1)) <= target_max_drawdown
            },
            'sharpe_ratio': {
                'value': metrics.get('sharpe_ratio', 0),
                'target': target_sharpe,
                'pass': metrics.get('sharpe_ratio', 0) >= target_sharpe
            }
        }
    
    def _generate_plots(self, output_dir: str, timestamp: str):
        """Generate equity curve and performance plots."""
        if not self.equity_curve:
            return
        
        # Setup plot style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtest Performance Analysis', fontsize=16)
        
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Equity curve
        axes[0, 0].plot(equity_df['equity'])
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Account Balance (USD)')
        axes[0, 0].grid(True)
        
        # Drawdown
        if len(equity_df) > 0:
            peak = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - peak) / peak * 100
            axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
            axes[0, 1].set_title('Drawdown (%)')
            axes[0, 1].set_ylabel('Drawdown %')
            axes[0, 1].grid(True)
        
        # Trade P&L distribution
        if len(trades_df) > 0:
            axes[1, 0].hist(trades_df['net_pnl'], bins=20, alpha=0.7)
            axes[1, 0].axvline(0, color='red', linestyle='--')
            axes[1, 0].set_title('Trade P&L Distribution')
            axes[1, 0].set_xlabel('P&L (USD)')
            axes[1, 0].set_ylabel('Frequency')
        
        # Performance metrics
        metrics = self.performance_metrics
        metrics_text = f"""
        Total Trades: {metrics.get('total_trades', 0)}
        Win Rate: {metrics.get('win_rate', 0):.2%}
        Profit Factor: {metrics.get('profit_factor', 0):.2f}
        Max Drawdown: {abs(metrics.get('max_drawdown', 0)):.2%}
        Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        Total Return: {metrics.get('total_return', 0):.2%}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(output_dir) / f"backtest_charts_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Backtest charts saved: {plot_path}")


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    # Example usage
    backtester = RealisticBacktester()
    
    # Load test data (placeholder - replace with actual data loading)
    print("RealisticBacktester ready for testing!")
    print("Use backtester.run_backtest(df, model_path) to run backtest")