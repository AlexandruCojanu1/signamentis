#!/usr/bin/env python3
"""
SignaMentis - MetaApi Executor

This module implements an alternative execution path using MetaApi cloud service
for trading operations when MT5 is not available.

Author: SignaMentis Team
Version: 2.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import time
from decimal import Decimal

# MetaApi imports
try:
    from metaapi_cloud_sdk import MetaApi
    from metaapi_cloud_sdk.meta_api.models import (
        MetatraderAccountInformation, MetatraderPosition, MetatraderOrder,
        MetatraderSymbolSpecification, MetatraderSymbolPrice
    )
    METAAPI_AVAILABLE = True
except ImportError:
    METAAPI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("MetaApi SDK not available. Install with: pip install metaapi-cloud-sdk")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """MetaApi order types."""
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"
    STOP_BUY = "STOP_BUY"
    STOP_SELL = "STOP_SELL"
    STOP_LIMIT_BUY = "STOP_LIMIT_BUY"
    STOP_LIMIT_SELL = "STOP_LIMIT_SELL"

class OrderStatus(Enum):
    """MetaApi order status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class PositionSide(Enum):
    """Position side."""
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class MetaApiOrderRequest:
    """Order request for MetaApi."""
    symbol: str
    order_type: OrderType
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: Optional[str] = None
    magic: Optional[int] = None
    client_order_id: Optional[str] = None

@dataclass
class MetaApiOrderResult:
    """Result of order operation."""
    success: bool
    order_id: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: Optional[datetime] = None
    fill_price: Optional[float] = None
    commission: Optional[float] = None
    swap: Optional[float] = None

@dataclass
class MetaApiPosition:
    """MetaApi position information."""
    position_id: str
    symbol: str
    side: PositionSide
    volume: float
    open_price: float
    current_price: float
    swap: float
    profit: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    magic: Optional[int] = None
    comment: Optional[str] = None
    time: datetime

@dataclass
class MetaApiAccountInfo:
    """MetaApi account information."""
    account_id: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    leverage: int
    currency: str
    broker: str
    server: str

class MetaApiExecutor:
    """
    MetaApi-based trading executor.
    
    Provides:
    - Account information
    - Position management
    - Order placement and management
    - Market data access
    - Risk controls
    """
    
    def __init__(
        self,
        token: str,
        account_id: str,
        domain: str = "agiliumtrade",
        risk_free_trading: bool = False,
        connection_timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize MetaApi executor.
        
        Args:
            token: MetaApi access token
            account_id: MetaTrader account ID
            domain: MetaApi domain
            risk_free_trading: Enable risk-free trading mode
            connection_timeout: Connection timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries
        """
        if not METAAPI_AVAILABLE:
            raise ImportError("MetaApi SDK not available. Install with: pip install metaapi-cloud-sdk")
        
        self.token = token
        self.account_id = account_id
        self.domain = domain
        self.risk_free_trading = risk_free_trading
        self.connection_timeout = connection_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # MetaApi client
        self.meta_api = MetaApi(token, domain)
        self.account = None
        self.connection = None
        
        # Connection state
        self.is_connected = False
        self.is_ready = False
        
        # Event handlers
        self.position_handlers: List[Callable] = []
        self.order_handlers: List[Callable] = []
        self.price_handlers: List[Callable] = []
        
        # Risk controls
        self.max_position_size = 1.0
        self.max_daily_loss = 1000.0
        self.max_drawdown = 0.1  # 10%
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        
        logger.info(f"🚀 MetaApi executor initialized for account {account_id}")
    
    async def connect(self) -> bool:
        """
        Connect to MetaApi and establish connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("🔌 Connecting to MetaApi...")
            
            # Get account
            self.account = await self.meta_api.metatrader_account_api.get_account(self.account_id)
            
            # Wait for deployment
            if not self.account.deployed:
                logger.info("📦 Deploying account...")
                await self.account.deploy()
                await self.account.wait_until_deployed()
            
            # Wait for connection
            if not self.account.connected:
                logger.info("🔗 Connecting account...")
                await self.account.wait_until_connected()
            
            # Get connection
            self.connection = await self.account.get_rpc_connection()
            
            # Wait for connection to be ready
            await self.connection.wait_synchronized()
            
            self.is_connected = True
            self.is_ready = True
            
            logger.info("✅ MetaApi connection established")
            
            # Start monitoring
            await self._start_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MetaApi: {e}")
            self.is_connected = False
            self.is_ready = False
            return False
    
    async def disconnect(self):
        """Disconnect from MetaApi."""
        try:
            if self.connection:
                await self.connection.close()
            
            if self.account:
                await self.account.undeploy()
            
            self.is_connected = False
            self.is_ready = False
            
            logger.info("🔌 MetaApi connection closed")
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting: {e}")
    
    async def _start_monitoring(self):
        """Start monitoring positions, orders, and prices."""
        try:
            # Monitor positions
            await self.connection.subscribe_to_positions()
            self.connection.on_position_update = self._on_position_update
            
            # Monitor orders
            await self.connection.subscribe_to_orders()
            self.connection.on_order_update = self._on_order_update
            
            # Monitor prices
            await self.connection.subscribe_to_symbol_prices()
            self.connection.on_symbol_price_update = self._on_symbol_price_update
            
            logger.info("📡 Started monitoring positions, orders, and prices")
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring: {e}")
    
    def _on_position_update(self, position: MetatraderPosition):
        """Handle position updates."""
        try:
            # Convert to our format
            meta_position = MetaApiPosition(
                position_id=str(position.id),
                symbol=position.symbol,
                side=PositionSide.BUY if position.type == 0 else PositionSide.SELL,
                volume=position.volume,
                open_price=position.open_price,
                current_price=position.current_price,
                swap=position.swap,
                profit=position.profit,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                magic=position.magic,
                comment=position.comment,
                time=datetime.fromtimestamp(position.time / 1000)
            )
            
            # Call handlers
            for handler in self.position_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(meta_position))
                    else:
                        handler(meta_position)
                except Exception as e:
                    logger.error(f"Error in position handler: {e}")
            
        except Exception as e:
            logger.error(f"Error processing position update: {e}")
    
    def _on_order_update(self, order: MetatraderOrder):
        """Handle order updates."""
        try:
            # Convert to our format
            meta_order = {
                'id': str(order.id),
                'symbol': order.symbol,
                'type': order.type,
                'volume': order.volume,
                'price': order.price,
                'stop_loss': order.stop_loss,
                'take_profit': order.take_profit,
                'comment': order.comment,
                'magic': order.magic,
                'status': order.status
            }
            
            # Call handlers
            for handler in self.order_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(meta_order))
                    else:
                        handler(meta_order)
                except Exception as e:
                    logger.error(f"Error in order handler: {e}")
            
        except Exception as e:
            logger.error(f"Error processing order update: {e}")
    
    def _on_symbol_price_update(self, price: MetatraderSymbolPrice):
        """Handle symbol price updates."""
        try:
            # Convert to our format
            meta_price = {
                'symbol': price.symbol,
                'bid': price.bid,
                'ask': price.ask,
                'time': datetime.fromtimestamp(price.time / 1000)
            }
            
            # Call handlers
            for handler in self.price_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(meta_price))
                    else:
                        handler(meta_price)
                except Exception as e:
                    logger.error(f"Error in price handler: {e}")
            
        except Exception as e:
            logger.error(f"Error processing price update: {e}")
    
    async def get_account_info(self) -> Optional[MetaApiAccountInfo]:
        """
        Get account information.
        
        Returns:
            Account information or None if failed
        """
        try:
            if not self.is_ready:
                raise RuntimeError("Connection not ready")
            
            account_info = await self.connection.get_account_information()
            
            return MetaApiAccountInfo(
                account_id=str(account_info.account_id),
                balance=account_info.balance,
                equity=account_info.equity,
                margin=account_info.margin,
                free_margin=account_info.free_margin,
                margin_level=account_info.margin_level,
                leverage=account_info.leverage,
                currency=account_info.currency,
                broker=account_info.broker,
                server=account_info.server
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get account info: {e}")
            return None
    
    async def get_positions(self) -> List[MetaApiPosition]:
        """
        Get all open positions.
        
        Returns:
            List of open positions
        """
        try:
            if not self.is_ready:
                raise RuntimeError("Connection not ready")
            
            positions = await self.connection.get_positions()
            
            meta_positions = []
            for position in positions:
                meta_position = MetaApiPosition(
                    position_id=str(position.id),
                    symbol=position.symbol,
                    side=PositionSide.BUY if position.type == 0 else PositionSide.SELL,
                    volume=position.volume,
                    open_price=position.open_price,
                    current_price=position.current_price,
                    swap=position.swap,
                    profit=position.profit,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    magic=position.magic,
                    comment=position.comment,
                    time=datetime.fromtimestamp(position.time / 1000)
                )
                meta_positions.append(meta_position)
            
            return meta_positions
            
        except Exception as e:
            logger.error(f"❌ Failed to get positions: {e}")
            return []
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get all pending orders.
        
        Returns:
            List of pending orders
        """
        try:
            if not self.is_ready:
                raise RuntimeError("Connection not ready")
            
            orders = await self.connection.get_orders()
            
            meta_orders = []
            for order in orders:
                meta_order = {
                    'id': str(order.id),
                    'symbol': order.symbol,
                    'type': order.type,
                    'volume': order.volume,
                    'price': order.price,
                    'stop_loss': order.stop_loss,
                    'take_profit': order.take_profit,
                    'comment': order.comment,
                    'magic': order.magic,
                    'status': order.status
                }
                meta_orders.append(meta_order)
            
            return meta_orders
            
        except Exception as e:
            logger.error(f"❌ Failed to get orders: {e}")
            return []
    
    async def place_order(self, request: MetaApiOrderRequest) -> MetaApiOrderResult:
        """
        Place a new order.
        
        Args:
            request: Order request
            
        Returns:
            Order result
        """
        try:
            if not self.is_ready:
                raise RuntimeError("Connection not ready")
            
            # Risk checks
            if not await self._check_risk_limits(request):
                return MetaApiOrderResult(
                    success=False,
                    error_message="Risk limits exceeded"
                )
            
            # Prepare order parameters
            order_params = {
                'symbol': request.symbol,
                'type': request.order_type.value,
                'volume': request.volume,
                'comment': request.comment or 'SignaMentis',
                'magic': request.magic or 123456
            }
            
            if request.price:
                order_params['price'] = request.price
            
            if request.stop_loss:
                order_params['stopLoss'] = request.stop_loss
            
            if request.take_profit:
                order_params['takeProfit'] = request.take_profit
            
            # Place order
            start_time = time.time()
            result = await self.connection.create_market_buy_order(
                request.symbol,
                request.volume,
                request.stop_loss,
                request.take_profit,
                request.comment or 'SignaMentis',
                request.magic or 123456
            )
            
            execution_time = time.time() - start_time
            
            if result:
                # Update statistics
                self.total_trades += 1
                self.successful_trades += 1
                
                return MetaApiOrderResult(
                    success=True,
                    order_id=str(result.id),
                    execution_time=execution_time,
                    fill_price=result.price if hasattr(result, 'price') else None
                )
            else:
                return MetaApiOrderResult(
                    success=False,
                    error_message="Order placement failed"
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to place order: {e}")
            return MetaApiOrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def close_position(self, position_id: str) -> bool:
        """
        Close a position.
        
        Args:
            position_id: Position ID to close
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_ready:
                raise RuntimeError("Connection not ready")
            
            # Get position
            positions = await self.get_positions()
            position = next((p for p in positions if p.position_id == position_id), None)
            
            if not position:
                logger.warning(f"Position {position_id} not found")
                return False
            
            # Close position
            result = await self.connection.close_position(position_id)
            
            if result:
                logger.info(f"✅ Position {position_id} closed successfully")
                return True
            else:
                logger.error(f"❌ Failed to close position {position_id}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error closing position {position_id}: {e}")
            return False
    
    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Modify position stop loss and take profit.
        
        Args:
            position_id: Position ID to modify
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_ready:
                raise RuntimeError("Connection not ready")
            
            # Modify position
            result = await self.connection.modify_position(
                position_id,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if result:
                logger.info(f"✅ Position {position_id} modified successfully")
                return True
            else:
                logger.error(f"❌ Failed to modify position {position_id}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error modifying position {position_id}: {e}")
            return False
    
    async def get_symbol_specification(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol specification.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Symbol specification or None if failed
        """
        try:
            if not self.is_ready:
                raise RuntimeError("Connection not ready")
            
            spec = await self.connection.get_symbol_specification(symbol)
            
            if spec:
                return {
                    'symbol': spec.symbol,
                    'tick_size': spec.tick_size,
                    'spread': spec.spread,
                    'digits': spec.digits,
                    'contract_size': spec.contract_size,
                    'margin_currency': spec.margin_currency,
                    'swap_long': spec.swap_long,
                    'swap_short': spec.swap_short
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get symbol specification for {symbol}: {e}")
            return None
    
    async def get_symbol_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current symbol price.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Current price or None if failed
        """
        try:
            if not self.is_ready:
                raise RuntimeError("Connection not ready")
            
            price = await self.connection.get_symbol_price(symbol)
            
            if price:
                return {
                    'symbol': price.symbol,
                    'bid': price.bid,
                    'ask': price.ask,
                    'time': datetime.fromtimestamp(price.time / 1000)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get price for {symbol}: {e}")
            return None
    
    async def _check_risk_limits(self, request: MetaApiOrderRequest) -> bool:
        """
        Check risk limits before placing order.
        
        Args:
            request: Order request
            
        Returns:
            True if risk limits are satisfied, False otherwise
        """
        try:
            # Check position size
            if request.volume > self.max_position_size:
                logger.warning(f"Position size {request.volume} exceeds limit {self.max_position_size}")
                return False
            
            # Check daily loss
            if self.total_pnl < -self.max_daily_loss:
                logger.warning(f"Daily loss {abs(self.total_pnl)} exceeds limit {self.max_daily_loss}")
                return False
            
            # Check drawdown
            account_info = await self.get_account_info()
            if account_info:
                drawdown = (account_info.balance - account_info.equity) / account_info.balance
                if drawdown > self.max_drawdown:
                    logger.warning(f"Drawdown {drawdown:.2%} exceeds limit {self.max_drawdown:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    def add_position_handler(self, handler: Callable):
        """Add position update handler."""
        self.position_handlers.append(handler)
    
    def add_order_handler(self, handler: Callable):
        """Add order update handler."""
        self.order_handlers.append(handler)
    
    def add_price_handler(self, handler: Callable):
        """Add price update handler."""
        self.price_handlers.append(handler)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'success_rate': self.successful_trades / self.total_trades if self.total_trades > 0 else 0,
            'total_pnl': self.total_pnl,
            'is_connected': self.is_connected,
            'is_ready': self.is_ready
        }
    
    def set_risk_limits(
        self,
        max_position_size: Optional[float] = None,
        max_daily_loss: Optional[float] = None,
        max_drawdown: Optional[float] = None
    ):
        """Set risk limits."""
        if max_position_size is not None:
            self.max_position_size = max_position_size
        
        if max_daily_loss is not None:
            self.max_daily_loss = max_daily_loss
        
        if max_drawdown is not None:
            self.max_drawdown = max_drawdown
        
        logger.info(f"🔒 Risk limits updated: position_size={self.max_position_size}, "
                   f"daily_loss={self.max_daily_loss}, drawdown={self.max_drawdown}")

# Example usage
if __name__ == "__main__":
    async def main():
        # This would require actual MetaApi credentials
        print("🚀 MetaApi Executor Example")
        print("=" * 40)
        print("This module provides MetaApi-based trading execution")
        print("To use, provide valid MetaApi credentials and account ID")
        print("Features:")
        print("- Account information and monitoring")
        print("- Position and order management")
        print("- Risk controls and limits")
        print("- Real-time market data")
        print("- Event-driven updates")
    
    # Run example
    asyncio.run(main())
