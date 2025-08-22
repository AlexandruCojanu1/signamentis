"""
SignaMentis Trading Strategy Module

This module implements the SuperTrend breakout trading strategy with AI confirmation.
The strategy uses SuperTrend indicator for entry signals and AI ensemble predictions for confirmation.

Author: SignaMentis Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SuperTrendStrategy:
    """
    SuperTrend breakout trading strategy with AI confirmation.
    
    Strategy:
    1. Wait for SuperTrend breakout (price crosses above/below SuperTrend line)
    2. Confirm with AI ensemble prediction (confidence >= 70%)
    3. Place BUY STOP / SELL STOP orders at breakout level
    4. Set dynamic SL/TP based on risk management
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the SuperTrend strategy.
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config or {}
        
        # Strategy parameters
        self.supertrend_period = self.config.get('supertrend_period', 10)
        self.supertrend_multiplier = self.config.get('supertrend_multiplier', 3.0)
        self.min_ai_confidence = self.config.get('min_ai_confidence', 0.70)
        self.min_breakout_strength = self.config.get('min_breakout_strength', 0.5)
        self.max_spread_pips = self.config.get('max_spread_pips', 2.0)
        
        # Risk parameters
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.01)  # 1%
        self.min_risk_reward_ratio = self.config.get('min_risk_reward_ratio', 2.0)
        
        # Position management
        self.max_positions = self.config.get('max_positions', 3)
        self.position_sizing_method = self.config.get('position_sizing_method', 'fixed_fractional')
        
        # Market condition filters
        self.volatility_filter = self.config.get('volatility_filter', True)
        self.session_filter = self.config.get('session_filter', True)
        self.news_filter = self.config.get('news_filter', True)
        
        # Performance tracking
        self.trades = []
        self.current_positions = []
        self.strategy_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
        
        logger.info("SuperTrend strategy initialized")
    
    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SuperTrend indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with SuperTrend columns
        """
        try:
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(self.supertrend_period).mean()
            
            # Calculate SuperTrend
            hl2 = (df['high'] + df['low']) / 2
            basic_upperband = hl2 + (self.supertrend_multiplier * atr)
            basic_lowerband = hl2 - (self.supertrend_multiplier * atr)
            
            # Initialize SuperTrend
            final_upperband = basic_upperband.copy()
            final_lowerband = basic_lowerband.copy()
            supertrend = pd.Series(index=df.index, dtype=float)
            
            for i in range(1, len(df)):
                # Upper band
                if basic_upperband.iloc[i] < final_upperband.iloc[i-1] or df['close'].iloc[i-1] > final_upperband.iloc[i-1]:
                    final_upperband.iloc[i] = basic_upperband.iloc[i]
                else:
                    final_upperband.iloc[i] = final_upperband.iloc[i-1]
                
                # Lower band
                if basic_lowerband.iloc[i] > final_lowerband.iloc[i-1] or df['close'].iloc[i-1] < final_lowerband.iloc[i-1]:
                    final_lowerband.iloc[i] = basic_lowerband.iloc[i]
                else:
                    final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
                
                # SuperTrend
                if supertrend.iloc[i-1] == final_upperband.iloc[i-1] and df['close'].iloc[i] <= final_upperband.iloc[i]:
                    supertrend.iloc[i] = final_upperband.iloc[i]
                elif supertrend.iloc[i-1] == final_upperband.iloc[i-1] and df['close'].iloc[i] > final_upperband.iloc[i]:
                    supertrend.iloc[i] = final_lowerband.iloc[i]
                elif supertrend.iloc[i-1] == final_lowerband.iloc[i-1] and df['close'].iloc[i] >= final_lowerband.iloc[i]:
                    supertrend.iloc[i] = final_lowerband.iloc[i]
                elif supertrend.iloc[i-1] == final_lowerband.iloc[i-1] and df['close'].iloc[i] < final_lowerband.iloc[i]:
                    supertrend.iloc[i] = final_upperband.iloc[i]
            
            # Add SuperTrend columns
            df['supertrend'] = supertrend
            df['supertrend_upper'] = final_upperband
            df['supertrend_lower'] = final_lowerband
            df['supertrend_direction'] = np.where(df['close'] > supertrend, 1, -1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {e}")
            return df
    
    def detect_breakout(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        """
        Detect SuperTrend breakout.
        
        Args:
            df: DataFrame with SuperTrend data
            current_index: Current bar index
            
        Returns:
            Breakout signal dictionary or None
        """
        try:
            if current_index < 2:
                return None
            
            current_bar = df.iloc[current_index]
            previous_bar = df.iloc[current_index - 1]
            
            # Check for bullish breakout (price crosses above SuperTrend)
            bullish_breakout = (
                previous_bar['close'] <= previous_bar['supertrend'] and
                current_bar['close'] > current_bar['supertrend'] and
                current_bar['supertrend_direction'] == 1
            )
            
            # Check for bearish breakout (price crosses below SuperTrend)
            bearish_breakout = (
                previous_bar['close'] >= previous_bar['supertrend'] and
                current_bar['close'] < current_bar['supertrend'] and
                current_bar['supertrend_direction'] == -1
            )
            
            if bullish_breakout:
                return {
                    'signal': 'BUY',
                    'type': 'bullish_breakout',
                    'price': current_bar['close'],
                    'supertrend_level': current_bar['supertrend'],
                    'strength': self._calculate_breakout_strength(df, current_index, 'bullish'),
                    'timestamp': current_bar.name
                }
            
            elif bearish_breakout:
                return {
                    'signal': 'SELL',
                    'type': 'bearish_breakout',
                    'price': current_bar['close'],
                    'supertrend_level': current_bar['supertrend'],
                    'strength': self._calculate_breakout_strength(df, current_index, 'bearish'),
                    'timestamp': current_bar.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting breakout: {e}")
            return None
    
    def _calculate_breakout_strength(self, df: pd.DataFrame, current_index: int, direction: str) -> float:
        """
        Calculate breakout strength based on volume and price movement.
        
        Args:
            df: DataFrame with market data
            current_index: Current bar index
            direction: Breakout direction ('bullish' or 'bearish')
            
        Returns:
            Breakout strength (0-1)
        """
        try:
            if current_index < 5:
                return 0.5
            
            current_bar = df.iloc[current_index]
            
            # Volume analysis
            volume_avg = df['volume'].iloc[current_index-5:current_index].mean()
            volume_ratio = current_bar['volume'] / volume_avg if volume_avg > 0 else 1.0
            
            # Price movement analysis
            if direction == 'bullish':
                price_change = (current_bar['close'] - current_bar['open']) / current_bar['open']
            else:
                price_change = (current_bar['open'] - current_bar['close']) / current_bar['open']
            
            # Combine factors
            strength = (0.6 * min(volume_ratio / 2.0, 1.0) + 
                       0.4 * min(abs(price_change) * 100, 1.0))
            
            return min(strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating breakout strength: {e}")
            return 0.5
    
    def validate_signal(self, signal: Dict, ai_prediction: Dict, market_data: pd.DataFrame) -> bool:
        """
        Validate trading signal with AI prediction and market conditions.
        
        Args:
            signal: Breakout signal
            ai_prediction: AI ensemble prediction
            market_data: Current market data
            
        Returns:
            bool: True if signal is valid
        """
        try:
            # Check AI confidence
            ai_confidence = ai_prediction.get('confidence', 0.0)
            if ai_confidence < self.min_ai_confidence:
                logger.info(f"Signal rejected: AI confidence {ai_confidence:.2f} < {self.min_ai_confidence}")
                return False
            
            # Check AI direction alignment
            ai_direction = ai_prediction.get('predicted_direction', 0)
            if signal['signal'] == 'BUY' and ai_direction == 0:
                logger.info("Signal rejected: AI predicts DOWN for BUY signal")
                return False
            elif signal['signal'] == 'SELL' and ai_direction == 1:
                logger.info("Signal rejected: AI predicts UP for SELL signal")
                return False
            
            # Check breakout strength
            if signal['strength'] < self.min_breakout_strength:
                logger.info(f"Signal rejected: Breakout strength {signal['strength']:.2f} < {self.min_breakout_strength}")
                return False
            
            # Check market conditions
            if not self._check_market_conditions(market_data):
                logger.info("Signal rejected: Market conditions unfavorable")
                return False
            
            # Check position limits
            if len(self.current_positions) >= self.max_positions:
                logger.info(f"Signal rejected: Maximum positions {self.max_positions} reached")
                return False
            
            logger.info(f"Signal validated: {signal['signal']} {signal['type']}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def _check_market_conditions(self, market_data: pd.DataFrame) -> bool:
        """
        Check if market conditions are favorable for trading.
        
        Args:
            market_data: Current market data
            
        Returns:
            bool: True if conditions are favorable
        """
        try:
            current_bar = market_data.iloc[-1]
            
            # Volatility filter
            if self.volatility_filter:
                atr = self._calculate_atr(market_data)
                if atr < 0.5:  # Low volatility
                    return False
            
            # Session filter
            if self.session_filter:
                current_hour = current_bar.name.hour
                # Avoid trading during low liquidity hours
                if current_hour in [0, 1, 2, 3, 22, 23]:
                    return False
            
            # Spread filter
            if hasattr(current_bar, 'spread'):
                spread_pips = current_bar['spread']
                if spread_pips > self.max_spread_pips:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            return True  # Default to allowing trades
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high_low = market_data['high'] - market_data['low']
            high_close = np.abs(market_data['high'] - market_data['close'].shift())
            low_close = np.abs(market_data['low'] - market_data['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 1.0
    
    def generate_trade_plan(self, signal: Dict, ai_prediction: Dict, risk_manager) -> Optional[Dict]:
        """
        Generate complete trade plan with entry, SL, TP, and position size.
        
        Args:
            signal: Validated trading signal
            ai_prediction: AI ensemble prediction
            risk_manager: Risk management instance
            
        Returns:
            Trade plan dictionary or None
        """
        try:
            # Get position sizing from risk manager
            position_size = risk_manager.calculate_position_size(
                entry_price=signal['price'],
                ai_confidence=ai_prediction.get('confidence', 0.5),
                market_data=pd.DataFrame()  # Pass current market data
            )
            
            if not position_size:
                logger.error("Failed to calculate position size")
                return None
            
            # Calculate entry, SL, and TP levels
            if signal['signal'] == 'BUY':
                entry_price = signal['price'] + 0.1  # BUY STOP above current price
                stop_loss = entry_price - position_size.stop_loss_pips * 0.1
                take_profit = entry_price + position_size.take_profit_pips * 0.1
            else:  # SELL
                entry_price = signal['price'] - 0.1  # SELL STOP below current price
                stop_loss = entry_price + position_size.stop_loss_pips * 0.1
                take_profit = entry_price - position_size.take_profit_pips * 0.1
            
            trade_plan = {
                'signal': signal['signal'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'lot_size': position_size.lot_size,
                'risk_amount': position_size.risk_amount,
                'risk_percentage': position_size.risk_percentage,
                'ai_confidence': ai_prediction.get('confidence', 0.0),
                'ai_direction': ai_prediction.get('predicted_direction', 0),
                'price_target': ai_prediction.get('price_target', 0.0),
                'timestamp': signal['timestamp'],
                'status': 'pending'
            }
            
            logger.info(f"Trade plan generated: {trade_plan}")
            return trade_plan
            
        except Exception as e:
            logger.error(f"Error generating trade plan: {e}")
            return None
    
    def update_positions(self, current_price: float, timestamp: datetime):
        """
        Update current positions and check for exits.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
        """
        try:
            for position in self.current_positions[:]:  # Copy list for iteration
                if position['status'] != 'open':
                    continue
                
                # Check stop loss
                if (position['signal'] == 'BUY' and current_price <= position['stop_loss']) or \
                   (position['signal'] == 'SELL' and current_price >= position['stop_loss']):
                    self._close_position(position, 'stop_loss', current_price, timestamp)
                
                # Check take profit
                elif (position['signal'] == 'BUY' and current_price >= position['take_profit']) or \
                     (position['signal'] == 'SELL' and current_price <= position['take_profit']):
                    self._close_position(position, 'take_profit', current_price, timestamp)
                
                # Check trailing stop (if implemented)
                elif self._should_update_trailing_stop(position, current_price):
                    self._update_trailing_stop(position, current_price)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _close_position(self, position: Dict, exit_reason: str, exit_price: float, timestamp: datetime):
        """Close a position and record the trade."""
        try:
            # Calculate P&L
            if position['signal'] == 'BUY':
                pnl = (exit_price - position['entry_price']) * position['lot_size'] * 100000
            else:  # SELL
                pnl = (position['entry_price'] - exit_price) * position['lot_size'] * 100000
            
            # Update position
            position['status'] = 'closed'
            position['exit_price'] = exit_price
            position['exit_reason'] = exit_reason
            position['exit_timestamp'] = timestamp
            position['pnl'] = pnl
            
            # Record trade
            trade_record = {
                'entry_timestamp': position['timestamp'],
                'exit_timestamp': timestamp,
                'signal': position['signal'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'lot_size': position['lot_size'],
                'pnl': pnl,
                'exit_reason': exit_reason,
                'ai_confidence': position['ai_confidence']
            }
            
            self.trades.append(trade_record)
            
            # Update statistics
            self._update_strategy_stats(trade_record)
            
            # Remove from current positions
            self.current_positions.remove(position)
            
            logger.info(f"Position closed: {position['signal']} {exit_reason}, P&L: {pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _should_update_trailing_stop(self, position: Dict, current_price: float) -> bool:
        """Check if trailing stop should be updated."""
        # Implement trailing stop logic here
        return False
    
    def _update_trailing_stop(self, position: Dict, current_price: float):
        """Update trailing stop level."""
        # Implement trailing stop update logic here
        pass
    
    def _update_strategy_stats(self, trade_record: Dict):
        """Update strategy performance statistics."""
        try:
            self.strategy_stats['total_trades'] += 1
            self.strategy_stats['total_pnl'] += trade_record['pnl']
            
            if trade_record['pnl'] > 0:
                self.strategy_stats['winning_trades'] += 1
            else:
                self.strategy_stats['losing_trades'] += 1
            
            # Calculate win rate
            if self.strategy_stats['total_trades'] > 0:
                self.strategy_stats['win_rate'] = (
                    self.strategy_stats['winning_trades'] / self.strategy_stats['total_trades']
                )
            
            # Calculate profit factor
            if self.strategy_stats['losing_trades'] > 0:
                total_wins = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
                total_losses = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
                self.strategy_stats['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
            
            # Update max drawdown
            running_pnl = 0
            max_pnl = 0
            for trade in self.trades:
                running_pnl += trade['pnl']
                max_pnl = max(max_pnl, running_pnl)
                drawdown = max_pnl - running_pnl
                self.strategy_stats['max_drawdown'] = max(self.strategy_stats['max_drawdown'], drawdown)
                
        except Exception as e:
            logger.error(f"Error updating strategy stats: {e}")
    
    def get_strategy_summary(self) -> Dict:
        """Get current strategy performance summary."""
        return self.strategy_stats.copy()
    
    def get_current_positions(self) -> List[Dict]:
        """Get list of current open positions."""
        return [pos for pos in self.current_positions if pos['status'] == 'open']
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get list of recent trades."""
        return sorted(self.trades, key=lambda x: x['exit_timestamp'], reverse=True)[:limit]


# Convenience function for creating strategy instance
def create_supertrend_strategy(config: Optional[Dict] = None) -> SuperTrendStrategy:
    """
    Create a SuperTrend strategy instance with configuration.
    
    Args:
        config: Strategy configuration dictionary
        
    Returns:
        SuperTrendStrategy instance
    """
    return SuperTrendStrategy(config)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='5T')
    
    sample_data = pd.DataFrame({
        'open': 2000 + np.random.randn(100).cumsum(),
        'high': 2000 + np.random.randn(100).cumsum() + 5,
        'low': 2000 + np.random.randn(100).cumsum() - 5,
        'close': 2000 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    # Create strategy
    strategy = create_supertrend_strategy()
    
    # Calculate SuperTrend
    sample_data = strategy.calculate_supertrend(sample_data)
    
    # Detect breakouts
    for i in range(10, len(sample_data)):
        breakout = strategy.detect_breakout(sample_data, i)
        if breakout:
            print(f"Breakout detected: {breakout}")
    
    print(f"Strategy initialized successfully!")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"SuperTrend columns: {[col for col in sample_data.columns if 'supertrend' in col]}")
