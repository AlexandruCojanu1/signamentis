"""
SignaMentis Risk Management Module

This module implements comprehensive risk management for XAU/USD trading including:
- Dynamic risk adjustment based on AI confidence
- Volatility-based position sizing
- Dynamic stop-loss and take-profit levels
- Drawdown management and circuit breakers
- Market condition filters

Author: SignaMentis Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Container for risk management parameters."""
    # Base risk settings
    base_risk_per_trade: float = 0.01  # 1% of account balance
    max_risk_per_trade: float = 0.03   # 3% maximum
    min_risk_per_trade: float = 0.005  # 0.5% minimum
    
    # Daily limits
    daily_risk_limit: float = 0.10     # 10% daily maximum
    max_daily_drawdown: float = 0.05   # 5% daily drawdown limit
    
    # Stop loss and take profit
    base_stop_loss_usd: float = 5.0    # Base stop loss in USD
    base_take_profit_usd: float = 15.0 # Base take profit in USD
    min_risk_reward_ratio: float = 2.0  # Minimum risk-reward ratio
    
    # Volatility adjustments
    atr_multiplier_sl: float = 2.0     # ATR multiplier for stop loss
    atr_multiplier_tp: float = 3.0     # ATR multiplier for take profit
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.85
    medium_confidence_threshold: float = 0.70
    low_confidence_threshold: float = 0.60
    
    # Risk multipliers
    high_confidence_multiplier: float = 1.2
    medium_confidence_multiplier: float = 1.0
    low_confidence_multiplier: float = 0.5


@dataclass
class PositionSize:
    """Container for position sizing information."""
    lot_size: float
    risk_amount: float
    risk_percentage: float
    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float
    confidence_adjustment: float
    volatility_adjustment: float


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    current_drawdown: float
    daily_risk_used: float
    consecutive_losses: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float


class RiskManager:
    """
    Comprehensive risk management system for XAU/USD trading.
    
    Features:
    - Dynamic risk adjustment based on AI confidence
    - Volatility-based position sizing
    - Dynamic stop-loss and take-profit levels
    - Drawdown management and circuit breakers
    - Market condition filters
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 config_file: Optional[str] = None):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary
            config_file: Path to configuration file
        """
        self.config = config or {}
        
        # Load configuration from file if provided
        if config_file:
            self._load_config(config_file)
        
        # Initialize risk parameters
        self.risk_params = RiskParameters(**self.config.get('risk', {}))
        
        # Account state
        self.account_balance = 10000.0  # Default starting balance
        self.current_positions = []
        self.trade_history = []
        self.daily_stats = {
            'date': datetime.now().date(),
            'risk_used': 0.0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'max_drawdown': 0.0
        }
        
        # Performance tracking
        self.performance_history = []
        self.drawdown_history = []
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = ""
        self.circuit_breaker_start = None
        
        # Market condition filters
        self.market_filters = self.config.get('market_filters', {})
        
        logger.info("Risk manager initialized")
    
    def _load_config(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as file:
                file_config = yaml.safe_load(file)
                self.config.update(file_config)
            logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
    
    def update_account_balance(self, balance: float):
        """Update current account balance."""
        self.account_balance = balance
        logger.info(f"Account balance updated: ${balance:,.2f}")
    
    def calculate_position_size(self, 
                              entry_price: float,
                              ai_confidence: float,
                              market_data: pd.DataFrame,
                              risk_percentage: Optional[float] = None) -> PositionSize:
        """
        Calculate optimal position size based on risk parameters and AI confidence.
        
        Args:
            entry_price: Entry price for the trade
            ai_confidence: AI model confidence (0-1)
            market_data: Market data for volatility calculation
            risk_percentage: Override risk percentage if specified
            
        Returns:
            PositionSize object with sizing information
        """
        # Determine risk percentage based on AI confidence
        if risk_percentage is None:
            risk_percentage = self._calculate_confidence_based_risk(ai_confidence)
        
        # Apply daily risk limits
        risk_percentage = self._apply_daily_limits(risk_percentage)
        
        # Calculate base risk amount
        risk_amount = self.account_balance * risk_percentage
        
        # Calculate volatility-adjusted stop loss and take profit
        atr = self._calculate_atr(market_data)
        stop_loss_pips = self._calculate_dynamic_stop_loss(atr, entry_price)
        take_profit_pips = self._calculate_dynamic_take_profit(atr, entry_price)
        
        # Ensure minimum risk-reward ratio
        if take_profit_pips / stop_loss_pips < self.risk_params.min_risk_reward_ratio:
            take_profit_pips = stop_loss_pips * self.risk_params.min_risk_reward_ratio
        
        # Calculate lot size based on risk amount and stop loss
        pip_value = 0.1  # For XAU/USD, 1 pip = $0.1 per 0.01 lot
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Apply lot size constraints
        lot_size = self._apply_lot_size_constraints(lot_size)
        
        # Recalculate actual risk amount
        actual_risk_amount = lot_size * stop_loss_pips * pip_value
        actual_risk_percentage = actual_risk_amount / self.account_balance
        
        # Calculate risk-reward ratio
        risk_reward_ratio = take_profit_pips / stop_loss_pips
        
        # Calculate adjustments
        confidence_adjustment = self._calculate_confidence_adjustment(ai_confidence)
        volatility_adjustment = self._calculate_volatility_adjustment(atr, market_data)
        
        return PositionSize(
            lot_size=lot_size,
            risk_amount=actual_risk_amount,
            risk_percentage=actual_risk_percentage,
            stop_loss_pips=stop_loss_pips,
            take_profit_pips=take_profit_pips,
            risk_reward_ratio=risk_reward_ratio,
            confidence_adjustment=confidence_adjustment,
            volatility_adjustment=volatility_adjustment
        )
    
    def _calculate_confidence_based_risk(self, ai_confidence: float) -> float:
        """Calculate risk percentage based on AI confidence."""
        if ai_confidence >= self.risk_params.high_confidence_threshold:
            multiplier = self.risk_params.high_confidence_multiplier
        elif ai_confidence >= self.risk_params.medium_confidence_threshold:
            multiplier = self.risk_params.medium_confidence_multiplier
        else:
            multiplier = self.risk_params.low_confidence_multiplier
        
        base_risk = self.risk_params.base_risk_per_trade
        adjusted_risk = base_risk * multiplier
        
        # Apply min/max constraints
        adjusted_risk = np.clip(
            adjusted_risk,
            self.risk_params.min_risk_per_trade,
            self.risk_params.max_risk_per_trade
        )
        
        return adjusted_risk
    
    def _apply_daily_limits(self, risk_percentage: float) -> float:
        """Apply daily risk limits."""
        # Check daily risk limit
        if self.daily_stats['risk_used'] + risk_percentage > self.risk_params.daily_risk_limit:
            remaining_risk = self.risk_params.daily_risk_limit - self.daily_stats['risk_used']
            risk_percentage = max(0, remaining_risk)
            logger.warning(f"Daily risk limit reached, adjusted risk to {risk_percentage:.3f}")
        
        # Check daily drawdown limit
        if self.daily_stats['max_drawdown'] >= self.risk_params.max_daily_drawdown:
            risk_percentage = 0
            logger.warning("Daily drawdown limit reached, no new trades allowed")
        
        return risk_percentage
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for volatility measurement."""
        if len(market_data) < period:
            return 1.0  # Default value
        
        high = market_data['high'].values
        low = market_data['low'].values
        close = market_data['close'].values
        
        # True Range calculation
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(true_range[1:period+1])  # Skip first NaN
        
        return atr
    
    def _calculate_dynamic_stop_loss(self, atr: float, entry_price: float) -> float:
        """Calculate dynamic stop loss based on ATR."""
        # Base stop loss in pips
        base_sl_pips = self.risk_params.base_stop_loss_usd / 0.1  # Convert USD to pips
        
        # ATR-based adjustment
        atr_pips = atr * 10  # Convert ATR to pips (approximate)
        dynamic_sl_pips = max(
            base_sl_pips,
            atr_pips * self.risk_params.atr_multiplier_sl
        )
        
        # Apply min/max constraints
        min_sl_pips = self.risk_params.base_stop_loss_usd / 0.1 * 0.4  # 40% of base
        max_sl_pips = self.risk_params.base_stop_loss_usd / 0.1 * 4.0   # 400% of base
        
        dynamic_sl_pips = np.clip(dynamic_sl_pips, min_sl_pips, max_sl_pips)
        
        return dynamic_sl_pips
    
    def _calculate_dynamic_take_profit(self, atr: float, entry_price: float) -> float:
        """Calculate dynamic take profit based on ATR."""
        # Base take profit in pips
        base_tp_pips = self.risk_params.base_take_profit_usd / 0.1
        
        # ATR-based adjustment
        atr_pips = atr * 10
        dynamic_tp_pips = max(
            base_tp_pips,
            atr_pips * self.risk_params.atr_multiplier_tp
        )
        
        # Apply min/max constraints
        min_tp_pips = self.risk_params.base_take_profit_usd / 0.1 * 0.8  # 80% of base
        max_tp_pips = self.risk_params.base_take_profit_usd / 0.1 * 5.0   # 500% of base
        
        dynamic_tp_pips = np.clip(dynamic_tp_pips, min_tp_pips, max_tp_pips)
        
        return dynamic_tp_pips
    
    def _apply_lot_size_constraints(self, lot_size: float) -> float:
        """Apply lot size constraints."""
        # Minimum lot size (0.01 for XAU/USD)
        min_lot = 0.01
        lot_size = max(lot_size, min_lot)
        
        # Maximum lot size (5% of account balance)
        max_lot = self.account_balance * 0.05 / 1000  # Approximate max lot size
        lot_size = min(lot_size, max_lot)
        
        # Round to nearest 0.01
        lot_size = round(lot_size, 2)
        
        return lot_size
    
    def _calculate_confidence_adjustment(self, ai_confidence: float) -> float:
        """Calculate confidence-based adjustment factor."""
        if ai_confidence >= self.risk_params.high_confidence_threshold:
            return self.risk_params.high_confidence_multiplier
        elif ai_confidence >= self.risk_params.medium_confidence_threshold:
            return self.risk_params.medium_confidence_multiplier
        else:
            return self.risk_params.low_confidence_multiplier
    
    def _calculate_volatility_adjustment(self, atr: float, market_data: pd.DataFrame) -> float:
        """Calculate volatility-based adjustment factor."""
        # Calculate historical volatility
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().dropna()
            historical_vol = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Volatility regime classification
            if historical_vol > 0.25:  # High volatility
                return 0.8  # Reduce position size
            elif historical_vol < 0.15:  # Low volatility
                return 1.1  # Slightly increase position size
            else:
                return 1.0  # Normal volatility
        
        return 1.0
    
    def check_market_conditions(self, 
                               market_data: pd.DataFrame,
                               current_time: datetime) -> Tuple[bool, str]:
        """
        Check if market conditions are suitable for trading.
        
        Args:
            market_data: Current market data
            current_time: Current timestamp
            
        Returns:
            Tuple of (can_trade, reason)
        """
        # Check circuit breaker
        if self.circuit_breaker_active:
            return False, f"Circuit breaker active: {self.circuit_breaker_reason}"
        
        # Check daily drawdown limit
        if self.daily_stats['max_drawdown'] >= self.risk_params.max_daily_drawdown:
            return False, "Daily drawdown limit exceeded"
        
        # Check daily risk limit
        if self.daily_stats['risk_used'] >= self.risk_params.daily_risk_limit:
            return False, "Daily risk limit reached"
        
        # Check market volatility
        atr = self._calculate_atr(market_data)
        if atr > 10:  # Very high volatility
            return False, "Market volatility too high"
        
        # Check trading hours (avoid low liquidity periods)
        hour = current_time.hour
        if hour < 2 or hour > 22:  # Avoid very early/late hours
            return False, "Outside optimal trading hours"
        
        # Check for major news events (simplified)
        if self._is_news_time(current_time):
            return False, "Major news event expected"
        
        return True, "Market conditions suitable"
    
    def _is_news_time(self, current_time: datetime) -> bool:
        """Check if current time is near major news events."""
        # Simplified news filter - in production, this would use an economic calendar
        # Major news times: 8:30 AM, 10:00 AM, 2:00 PM EST (approximate)
        hour = current_time.hour
        minute = current_time.minute
        
        # Check if within 1 hour of major news times
        news_hours = [8, 10, 14]  # 8 AM, 10 AM, 2 PM
        for news_hour in news_hours:
            if abs(hour - news_hour) <= 1:
                return True
        
        return False
    
    def record_trade(self, 
                    trade_data: Dict):
        """
        Record a completed trade for performance tracking.
        
        Args:
            trade_data: Dictionary containing trade information
        """
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now()
        
        # Add to trade history
        self.trade_history.append(trade_data)
        
        # Update daily statistics
        self._update_daily_stats(trade_data)
        
        # Update performance history
        self._update_performance_history()
        
        # Check for circuit breaker conditions
        self._check_circuit_breaker()
        
        logger.info(f"Trade recorded: {trade_data.get('symbol', 'Unknown')} - "
                   f"P&L: ${trade_data.get('pnl', 0):.2f}")
    
    def _update_daily_stats(self, trade_data: Dict):
        """Update daily trading statistics."""
        trade_date = trade_data['timestamp'].date()
        
        # Reset daily stats if it's a new day
        if trade_date != self.daily_stats['date']:
            self.daily_stats = {
                'date': trade_date,
                'risk_used': 0.0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0,
                'max_drawdown': 0.0
            }
        
        # Update statistics
        self.daily_stats['trades'] += 1
        self.daily_stats['pnl'] += trade_data.get('pnl', 0)
        
        if trade_data.get('pnl', 0) > 0:
            self.daily_stats['wins'] += 1
        else:
            self.daily_stats['losses'] += 1
        
        # Update risk used
        if 'risk_amount' in trade_data:
            self.daily_stats['risk_used'] += trade_data['risk_amount'] / self.account_balance
        
        # Update max drawdown
        current_drawdown = -self.daily_stats['pnl'] / self.account_balance
        self.daily_stats['max_drawdown'] = max(self.daily_stats['max_drawdown'], current_drawdown)
    
    def _update_performance_history(self):
        """Update performance tracking history."""
        if len(self.trade_history) < 2:
            return
        
        # Calculate performance metrics
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        returns = [t.get('pnl', 0) / self.account_balance for t in self.trade_history]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate VaR
        if len(returns) >= 20:
            var_95 = np.percentile(returns, 5)
        else:
            var_95 = 0
        
        # Update account balance
        self.account_balance += sum(t.get('pnl', 0) for t in self.trade_history)
        
        # Store performance metrics
        performance = {
            'timestamp': datetime.now(),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'account_balance': self.account_balance,
            'var_95': var_95
        }
        
        self.performance_history.append(performance)
    
    def _check_circuit_breaker(self):
        """Check if circuit breaker conditions are met."""
        # Check consecutive losses
        recent_trades = self.trade_history[-10:]  # Last 10 trades
        consecutive_losses = 0
        
        for trade in reversed(recent_trades):
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
            else:
                break
        
        # Circuit breaker on consecutive losses
        if consecutive_losses >= 5:
            self._activate_circuit_breaker("5 consecutive losses")
        
        # Circuit breaker on daily drawdown
        if self.daily_stats['max_drawdown'] >= self.risk_params.max_daily_drawdown:
            self._activate_circuit_breaker("Daily drawdown limit exceeded")
    
    def _activate_circuit_breaker(self, reason: str):
        """Activate circuit breaker."""
        self.circuit_breaker_active = True
        self.circuit_breaker_reason = reason
        self.circuit_breaker_start = datetime.now()
        
        logger.warning(f"Circuit breaker activated: {reason}")
    
    def deactivate_circuit_breaker(self):
        """Deactivate circuit breaker."""
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = ""
        self.circuit_breaker_start = None
        
        logger.info("Circuit breaker deactivated")
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        if not self.performance_history:
            return RiskMetrics(
                current_drawdown=0.0,
                daily_risk_used=self.daily_stats['risk_used'],
                consecutive_losses=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                var_95=0.0
            )
        
        latest_performance = self.performance_history[-1]
        
        # Calculate current drawdown
        current_drawdown = -self.daily_stats['pnl'] / self.account_balance
        
        # Calculate consecutive losses
        recent_trades = self.trade_history[-10:]
        consecutive_losses = 0
        for trade in reversed(recent_trades):
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
            else:
                break
        
        # Calculate max drawdown
        if self.performance_history:
            balances = [p['account_balance'] for p in self.performance_history]
            peak = max(balances)
            current = balances[-1]
            max_drawdown = (peak - current) / peak if peak > 0 else 0
        else:
            max_drawdown = 0
        
        return RiskMetrics(
            current_drawdown=current_drawdown,
            daily_risk_used=self.daily_stats['risk_used'],
            consecutive_losses=consecutive_losses,
            win_rate=latest_performance['win_rate'],
            profit_factor=latest_performance['profit_factor'],
            sharpe_ratio=latest_performance['sharpe_ratio'],
            max_drawdown=max_drawdown,
            var_95=latest_performance['var_95']
        )
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary."""
        risk_metrics = self.get_risk_metrics()
        
        return {
            'account_balance': self.account_balance,
            'daily_stats': self.daily_stats.copy(),
            'risk_metrics': {
                'current_drawdown': risk_metrics.current_drawdown,
                'daily_risk_used': risk_metrics.daily_risk_used,
                'consecutive_losses': risk_metrics.consecutive_losses,
                'win_rate': risk_metrics.win_rate,
                'profit_factor': risk_metrics.profit_factor,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'max_drawdown': risk_metrics.max_drawdown,
                'var_95': risk_metrics.var_95
            },
            'circuit_breaker': {
                'active': self.circuit_breaker_active,
                'reason': self.circuit_breaker_reason,
                'start_time': self.circuit_breaker_start
            },
            'risk_parameters': {
                'base_risk_per_trade': self.risk_params.base_risk_per_trade,
                'max_risk_per_trade': self.risk_params.max_risk_per_trade,
                'daily_risk_limit': self.risk_params.daily_risk_limit,
                'max_daily_drawdown': self.risk_params.max_daily_drawdown
            }
        }


# Convenience function for quick risk manager creation
def create_risk_manager(config: Optional[Dict] = None) -> RiskManager:
    """
    Create a risk manager with default or custom configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RiskManager instance
    """
    default_config = {
        'risk': {
            'base_risk_per_trade': 0.01,
            'max_risk_per_trade': 0.03,
            'min_risk_per_trade': 0.005,
            'daily_risk_limit': 0.10,
            'max_daily_drawdown': 0.05,
            'base_stop_loss_usd': 5.0,
            'base_take_profit_usd': 15.0,
            'min_risk_reward_ratio': 2.0,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0,
            'high_confidence_threshold': 0.85,
            'medium_confidence_threshold': 0.70,
            'low_confidence_threshold': 0.60,
            'high_confidence_multiplier': 1.2,
            'medium_confidence_multiplier': 1.0,
            'low_confidence_multiplier': 0.5
        }
    }
    
    if config:
        default_config.update(config)
    
    return RiskManager(default_config)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='5T')
    
    sample_data = pd.DataFrame({
        'open': 2000 + np.random.randn(100).cumsum(),
        'high': 2000 + np.random.randn(100).cumsum() + 5,
        'low': 2000 + np.random.randn(100).cumsum() - 5,
        'close': 2000 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    # Create risk manager
    risk_manager = create_risk_manager()
    
    # Update account balance
    risk_manager.update_account_balance(50000.0)
    
    # Check market conditions
    can_trade, reason = risk_manager.check_market_conditions(sample_data, datetime.now())
    print(f"Can trade: {can_trade}, Reason: {reason}")
    
    # Calculate position size
    entry_price = 2050.0
    ai_confidence = 0.85
    
    position_size = risk_manager.calculate_position_size(
        entry_price, ai_confidence, sample_data
    )
    
    print(f"\nPosition Size Calculation:")
    print(f"Lot size: {position_size.lot_size}")
    print(f"Risk amount: ${position_size.risk_amount:.2f}")
    print(f"Risk percentage: {position_size.risk_percentage:.3f}")
    print(f"Stop loss: {position_size.stop_loss_pips:.1f} pips")
    print(f"Take profit: {position_size.take_profit_pips:.1f} pips")
    print(f"Risk-reward ratio: {position_size.risk_reward_ratio:.2f}")
    
    # Record a sample trade
    sample_trade = {
        'symbol': 'XAUUSD',
        'entry_price': 2050.0,
        'exit_price': 2055.0,
        'lot_size': 0.1,
        'pnl': 50.0,
        'risk_amount': 25.0,
        'timestamp': datetime.now()
    }
    
    risk_manager.record_trade(sample_trade)
    
    # Get risk summary
    risk_summary = risk_manager.get_risk_summary()
    print(f"\nRisk Summary:")
    print(f"Account balance: ${risk_summary['account_balance']:,.2f}")
    print(f"Daily P&L: ${risk_summary['daily_stats']['pnl']:.2f}")
    print(f"Win rate: {risk_summary['risk_metrics']['win_rate']:.2f}")
    print(f"Circuit breaker active: {risk_summary['circuit_breaker']['active']}")
