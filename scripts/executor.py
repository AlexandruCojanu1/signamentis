"""
SignaMentis MT5 Executor Module

This module handles order placement and trade execution through MetaTrader 5.
Includes order management, position tracking, and risk controls.

Author: SignaMentis Team
Version: 1.0.0
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import time
import warnings
from dataclasses import dataclass
from enum import Enum
import json
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class OrderRequest:
    """Order request container."""
    symbol: str
    order_type: OrderType
    volume: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = ""
    magic: int = 0
    expiration: Optional[datetime] = None


@dataclass
class OrderResult:
    """Order result container."""
    order_ticket: int
    order_type: OrderType
    symbol: str
    volume: float
    price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    status: OrderStatus
    comment: str
    magic: int
    timestamp: datetime
    error_code: Optional[int] = None
    error_description: Optional[str] = None


class MT5Executor:
    """
    MetaTrader 5 executor for order placement and trade management.
    
    Features:
    - Order placement and management
    - Position tracking
    - Risk controls
    - Order reconciliation
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize MT5 executor.
        
        Args:
            config: Executor configuration
        """
        self.config = config or {}
        
        # MT5 connection parameters
        self.login = self.config.get('login', 0)
        self.password = self.config.get('password', "")
        self.server = self.config.get('server', "")
        self.timeout = self.config.get('timeout', 60000)
        
        # Trading parameters
        self.symbol = self.config.get('symbol', "XAUUSD")
        self.deviation = self.config.get('deviation', 10)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
        # Risk controls
        self.max_positions = self.config.get('max_positions', 5)
        self.max_daily_trades = self.config.get('max_daily_trades', 20)
        self.max_daily_loss = self.config.get('max_daily_loss', 1000.0)
        
        # State tracking
        self.connected = False
        self.account_info = None
        self.positions = {}
        self.orders = {}
        self.trade_history = []
        
        # Performance metrics
        self.execution_latency = []
        self.order_success_rate = 0.0
        self.total_orders = 0
        self.successful_orders = 0
        
        logger.info("MT5 Executor initialized")
    
    def connect(self) -> bool:
        """
        Connect to MetaTrader 5.
        
        Returns:
            bool: True if connection successful
        """
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login to account
            if not mt5.login(login=self.login, password=self.password, server=self.server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
            
            # Get account info
            self.account_info = mt5.account_info()
            if not self.account_info:
                logger.error("Failed to get account info")
                return False
            
            # Get symbol info
            symbol_info = mt5.symbol_info(self.symbol)
            if not symbol_info:
                logger.error(f"Symbol {self.symbol} not found")
                return False
            
            # Enable symbol for trading
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Failed to select symbol {self.symbol}")
                return False
            
            self.connected = True
            logger.info(f"Connected to MT5: {self.account_info.login} on {self.server}")
            logger.info(f"Account: {self.account_info.company}, Balance: ${self.account_info.balance:.2f}")
            logger.info(f"Symbol: {self.symbol}, Spread: {symbol_info.spread} points")
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MetaTrader 5."""
        try:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                logger.info("Disconnected from MT5")
        except Exception as e:
            logger.error(f"Error disconnecting from MT5: {e}")
    
    def place_order(self, order_request: OrderRequest) -> Optional[OrderResult]:
        """
        Place order in MT5.
        
        Args:
            order_request: Order request object
            
        Returns:
            OrderResult object or None if failed
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
        
        try:
            # Validate order request
            if not self._validate_order_request(order_request):
                return None
            
            # Check risk limits
            if not self._check_risk_limits(order_request):
                return None
            
            # Prepare order parameters
            order_params = self._prepare_order_params(order_request)
            
            # Place order with retries
            for attempt in range(self.max_retries):
                try:
                    start_time = time.time()
                    
                    # Send order
                    result = mt5.order_send(order_params)
                    
                    execution_time = time.time() - start_time
                    self.execution_latency.append(execution_time)
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        # Order successful
                        order_result = self._create_order_result(result, order_request, execution_time)
                        self._update_order_tracking(order_result)
                        
                        logger.info(f"Order placed successfully: {order_result.order_ticket} - "
                                  f"{order_result.order_type.value} {order_result.volume} lots "
                                  f"at {order_result.price}")
                        
                        return order_result
                    
                    else:
                        # Order failed
                        error_msg = f"Order failed (attempt {attempt + 1}): {result.retcode} - {result.comment}"
                        logger.warning(error_msg)
                        
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                            continue
                        else:
                            # Final attempt failed
                            order_result = self._create_failed_order_result(order_request, result.retcode, result.comment)
                            return order_result
                
                except Exception as e:
                    logger.error(f"Error placing order (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        order_result = self._create_failed_order_result(order_request, -1, str(e))
                        return order_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error in place_order: {e}")
            return None
    
    def _validate_order_request(self, order_request: OrderRequest) -> bool:
        """Validate order request parameters."""
        try:
            # Check symbol
            if order_request.symbol != self.symbol:
                logger.error(f"Symbol mismatch: {order_request.symbol} != {self.symbol}")
                return False
            
            # Check volume
            if order_request.volume <= 0:
                logger.error(f"Invalid volume: {order_request.volume}")
                return False
            
            # Check price
            if order_request.price <= 0:
                logger.error(f"Invalid price: {order_request.price}")
                return False
            
            # Check stop loss and take profit
            if order_request.stop_loss is not None and order_request.stop_loss <= 0:
                logger.error(f"Invalid stop loss: {order_request.stop_loss}")
                return False
            
            if order_request.take_profit is not None and order_request.take_profit <= 0:
                logger.error(f"Invalid take profit: {order_request.take_profit}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating order request: {e}")
            return False
    
    def _check_risk_limits(self, order_request: OrderRequest) -> bool:
        """Check risk management limits."""
        try:
            # Check max positions
            if len(self.positions) >= self.max_positions:
                logger.warning(f"Maximum positions reached: {len(self.positions)}")
                return False
            
            # Check daily trade limit
            today = datetime.now().date()
            today_trades = len([t for t in self.trade_history if t['timestamp'].date() == today])
            if today_trades >= self.max_daily_trades:
                logger.warning(f"Daily trade limit reached: {today_trades}")
                return False
            
            # Check daily loss limit
            today_pnl = sum(t.get('pnl', 0) for t in self.trade_history if t['timestamp'].date() == today)
            if today_pnl <= -self.max_daily_loss:
                logger.warning(f"Daily loss limit reached: ${today_pnl:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    def _prepare_order_params(self, order_request: OrderRequest) -> Dict:
        """Prepare order parameters for MT5."""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(self.symbol)
            
            # Determine order type
            if order_request.order_type == OrderType.BUY:
                order_type = mt5.ORDER_TYPE_BUY
            elif order_request.order_type == OrderType.SELL:
                order_type = mt5.ORDER_TYPE_SELL
            elif order_request.order_type == OrderType.BUY_STOP:
                order_type = mt5.ORDER_TYPE_BUY_STOP
            elif order_request.order_type == OrderType.SELL_STOP:
                order_type = mt5.ORDER_TYPE_SELL_STOP
            elif order_request.order_type == OrderType.BUY_LIMIT:
                order_type = mt5.ORDER_TYPE_BUY_LIMIT
            elif order_request.order_type == OrderType.SELL_LIMIT:
                order_type = mt5.ORDER_TYPE_SELL_LIMIT
            else:
                raise ValueError(f"Unsupported order type: {order_request.order_type}")
            
            # Prepare parameters
            params = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order_request.symbol,
                "volume": order_request.volume,
                "type": order_type,
                "price": order_request.price,
                "deviation": self.deviation,
                "magic": order_request.magic,
                "comment": order_request.comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit
            if order_request.stop_loss is not None:
                params["sl"] = order_request.stop_loss
            
            if order_request.take_profit is not None:
                params["tp"] = order_request.take_profit
            
            # Add expiration for pending orders
            if order_request.expiration is not None:
                params["type_time"] = mt5.ORDER_TIME_SPECIFIED
                params["expiration"] = order_request.expiration
            
            return params
            
        except Exception as e:
            logger.error(f"Error preparing order parameters: {e}")
            return {}
    
    def _create_order_result(self, 
                           mt5_result,
                           order_request: OrderRequest,
                           execution_time: float) -> OrderResult:
        """Create OrderResult from successful MT5 order."""
        return OrderResult(
            order_ticket=mt5_result.order,
            order_type=order_request.order_type,
            symbol=order_request.symbol,
            volume=order_request.volume,
            price=order_request.price,
            stop_loss=order_request.stop_loss,
            take_profit=order_request.take_profit,
            status=OrderStatus.FILLED,
            comment=order_request.comment,
            magic=order_request.magic,
            timestamp=datetime.now(),
            error_code=None,
            error_description=None
        )
    
    def _create_failed_order_result(self, 
                                  order_request: OrderRequest,
                                  error_code: int,
                                  error_description: str) -> OrderResult:
        """Create OrderResult from failed order."""
        return OrderResult(
            order_ticket=0,
            order_type=order_request.order_type,
            symbol=order_request.symbol,
            volume=order_request.volume,
            price=order_request.price,
            stop_loss=order_request.stop_loss,
            take_profit=order_request.take_profit,
            status=OrderStatus.REJECTED,
            comment=order_request.comment,
            magic=order_request.magic,
            timestamp=datetime.now(),
            error_code=error_code,
            error_description=error_description
        )
    
    def _update_order_tracking(self, order_result: OrderResult):
        """Update internal order tracking."""
        try:
            self.total_orders += 1
            
            if order_result.status == OrderStatus.FILLED:
                self.successful_orders += 1
            
            # Update success rate
            self.order_success_rate = self.successful_orders / self.total_orders
            
            # Store order
            self.orders[order_result.order_ticket] = order_result
            
            # Add to trade history
            self.trade_history.append({
                'timestamp': order_result.timestamp,
                'order_ticket': order_result.order_ticket,
                'order_type': order_result.order_type.value,
                'symbol': order_result.symbol,
                'volume': order_result.volume,
                'price': order_result.price,
                'status': order_result.status.value,
                'pnl': 0.0  # Will be updated when position is closed
            })
            
        except Exception as e:
            logger.error(f"Error updating order tracking: {e}")
    
    def get_positions(self) -> Dict[int, Dict]:
        """
        Get current open positions.
        
        Returns:
            Dictionary of position information
        """
        try:
            if not self.connected:
                return {}
            
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return {}
            
            self.positions = {}
            for pos in positions:
                self.positions[pos.ticket] = {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'stop_loss': pos.sl,
                    'take_profit': pos.tp,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'time': pos.time,
                    'magic': pos.magic,
                    'comment': pos.comment
                }
            
            return self.positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def close_position(self, ticket: int, volume: Optional[float] = None) -> bool:
        """
        Close position by ticket.
        
        Args:
            ticket: Position ticket
            volume: Volume to close (None for full position)
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.connected:
                logger.error("Not connected to MT5")
                return False
            
            # Get position
            position = self.positions.get(ticket)
            if not position:
                logger.error(f"Position {ticket} not found")
                return False
            
            # Determine volume to close
            close_volume = volume if volume is not None else position['volume']
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if position['type'] == 'BUY' else mt5.ORDER_TYPE_BUY
            
            close_params = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": close_volume,
                "type": close_type,
                "position": ticket,
                "price": mt5.symbol_info_tick(self.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(self.symbol).ask,
                "deviation": self.deviation,
                "magic": position['magic'],
                "comment": f"Close position {ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = mt5.order_send(close_params)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position {ticket} closed successfully")
                
                # Update tracking
                if volume is None or volume >= position['volume']:
                    # Full position closed
                    del self.positions[ticket]
                else:
                    # Partial close
                    self.positions[ticket]['volume'] -= volume
                
                return True
            else:
                logger.error(f"Failed to close position {ticket}: {result.retcode} - {result.comment}")
                return False
            
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def modify_position(self, ticket: int, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        """
        Modify position stop loss and take profit.
        
        Args:
            ticket: Position ticket
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.connected:
                logger.error("Not connected to MT5")
                return False
            
            # Get position
            position = self.positions.get(ticket)
            if not position:
                logger.error(f"Position {ticket} not found")
                return False
            
            # Prepare modify request
            modify_params = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": self.symbol,
                "position": ticket,
                "sl": stop_loss if stop_loss is not None else position['stop_loss'],
                "tp": take_profit if take_profit is not None else position['take_profit'],
            }
            
            # Send modify request
            result = mt5.order_send(modify_params)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position {ticket} modified successfully")
                
                # Update local tracking
                if stop_loss is not None:
                    self.positions[ticket]['stop_loss'] = stop_loss
                if take_profit is not None:
                    self.positions[ticket]['take_profit'] = take_profit
                
                return True
            else:
                logger.error(f"Failed to modify position {ticket}: {result.retcode} - {result.comment}")
                return False
            
        except Exception as e:
            logger.error(f"Error modifying position {ticket}: {e}")
            return False
    
    def get_account_info(self) -> Optional[Dict]:
        """Get current account information."""
        try:
            if not self.connected:
                return None
            
            account = mt5.account_info()
            if account:
                return {
                    'login': account.login,
                    'balance': account.balance,
                    'equity': account.equity,
                    'margin': account.margin,
                    'free_margin': account.margin_free,
                    'profit': account.profit,
                    'currency': account.currency,
                    'company': account.company,
                    'server': account.server
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_symbol_info(self) -> Optional[Dict]:
        """Get symbol information."""
        try:
            if not self.connected:
                return None
            
            symbol = mt5.symbol_info(self.symbol)
            if symbol:
                return {
                    'symbol': symbol.name,
                    'spread': symbol.spread,
                    'digits': symbol.digits,
                    'point': symbol.point,
                    'trade_mode': symbol.trade_mode,
                    'trade_stops_level': symbol.trade_stops_level,
                    'trade_freeze_level': symbol.trade_freeze_level,
                    'volume_min': symbol.volume_min,
                    'volume_max': symbol.volume_max,
                    'volume_step': symbol.volume_step
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict:
        """Get executor performance metrics."""
        try:
            avg_latency = np.mean(self.execution_latency) if self.execution_latency else 0
            max_latency = np.max(self.execution_latency) if self.execution_latency else 0
            min_latency = np.min(self.execution_latency) if self.execution_latency else 0
            
            return {
                'total_orders': self.total_orders,
                'successful_orders': self.successful_orders,
                'order_success_rate': self.order_success_rate,
                'average_execution_latency': avg_latency,
                'max_execution_latency': max_latency,
                'min_execution_latency': min_latency,
                'current_positions': len(self.positions),
                'connected': self.connected
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Convenience function for creating executor
def create_mt5_executor(config: Optional[Dict] = None) -> MT5Executor:
    """
    Create an MT5 executor instance with configuration.
    
    Args:
        config: Executor configuration dictionary
        
    Returns:
        MT5Executor instance
    """
    return MT5Executor(config)


if __name__ == "__main__":
    # Example usage
    config = {
        'login': 12345,
        'password': 'your_password',
        'server': 'your_broker_server',
        'symbol': 'XAUUSD',
        'max_positions': 5,
        'max_daily_trades': 20,
        'max_daily_loss': 1000.0
    }
    
    # Create executor
    executor = create_mt5_executor(config)
    
    try:
        # Connect to MT5
        if executor.connect():
            print("Connected to MT5 successfully!")
            
            # Get account info
            account_info = executor.get_account_info()
            if account_info:
                print(f"Account: {account_info['login']}, Balance: ${account_info['balance']:.2f}")
            
            # Get symbol info
            symbol_info = executor.get_symbol_info()
            if symbol_info:
                print(f"Symbol: {symbol_info['symbol']}, Spread: {symbol_info['spread']} points")
            
            # Get current positions
            positions = executor.get_positions()
            print(f"Current positions: {len(positions)}")
            
            # Get performance metrics
            metrics = executor.get_performance_metrics()
            print(f"Performance metrics: {metrics}")
            
        else:
            print("Failed to connect to MT5")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Disconnect
        executor.disconnect()
        print("Disconnected from MT5")
    
    print("MT5 Executor test completed!")
