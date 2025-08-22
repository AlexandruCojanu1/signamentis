"""
SignaMentis FastAPI Service Module

This module provides REST API endpoints for trading operations, monitoring, and system management.
Includes authentication, rate limiting, and comprehensive trading APIs.

Author: SignaMentis Team
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
import hashlib
import hmac
import time
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import trading system components
import sys
sys.path.append('..')
from scripts.executor import MT5Executor, OrderRequest, OrderType
from scripts.monitor import LiveDashboard, PerformanceMetrics, AlertLevel
from scripts.ensemble import EnsembleManager
from scripts.risk_manager import RiskManager
from scripts.strategy import SuperTrendStrategy
from scripts.backtester import Backtester


# Pydantic models for API requests/responses
class TradingSignal(BaseModel):
    """Trading signal model."""
    symbol: str = Field(..., description="Trading symbol")
    direction: str = Field(..., description="Signal direction (BUY/SELL)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence")
    price: float = Field(..., gt=0, description="Signal price")
    timestamp: datetime = Field(default_factory=datetime.now, description="Signal timestamp")
    source: str = Field(..., description="Signal source")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata")


class OrderRequestModel(BaseModel):
    """Order request model."""
    symbol: str = Field(..., description="Trading symbol")
    order_type: str = Field(..., description="Order type")
    volume: float = Field(..., gt=0, description="Order volume")
    price: float = Field(..., gt=0, description="Order price")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[float] = Field(None, gt=0, description="Take profit price")
    comment: Optional[str] = Field("", description="Order comment")
    magic: Optional[int] = Field(0, description="Magic number")


class OrderResponse(BaseModel):
    """Order response model."""
    order_ticket: int = Field(..., description="Order ticket")
    status: str = Field(..., description="Order status")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(..., description="Response timestamp")
    error_code: Optional[int] = Field(None, description="Error code if failed")
    error_description: Optional[str] = Field(None, description="Error description if failed")


class PositionInfo(BaseModel):
    """Position information model."""
    ticket: int = Field(..., description="Position ticket")
    symbol: str = Field(..., description="Trading symbol")
    type: str = Field(..., description="Position type")
    volume: float = Field(..., description="Position volume")
    price_open: float = Field(..., description="Open price")
    price_current: float = Field(..., description="Current price")
    profit: float = Field(..., description="Current profit/loss")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    swap: float = Field(..., description="Swap charges")
    time: datetime = Field(..., description="Position open time")


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response model."""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    total_pnl: float = Field(..., description="Total profit/loss")
    daily_pnl: float = Field(..., description="Daily profit/loss")
    win_rate: float = Field(..., description="Win rate percentage")
    total_trades: int = Field(..., description="Total number of trades")
    open_positions: int = Field(..., description="Number of open positions")
    equity: float = Field(..., description="Current equity")
    balance: float = Field(..., description="Account balance")
    drawdown: float = Field(..., description="Current drawdown percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")


class SystemStatus(BaseModel):
    """System status model."""
    status: str = Field(..., description="System status")
    uptime: str = Field(..., description="System uptime")
    version: str = Field(..., description="System version")
    last_update: datetime = Field(..., description="Last update timestamp")
    components: Dict[str, str] = Field(..., description="Component statuses")
    alerts: List[str] = Field(..., description="Recent alerts")


class BacktestRequest(BaseModel):
    """Backtest request model."""
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe (M1, M5, M15, M30, H1, H4, D1)")
    initial_balance: float = Field(10000.0, gt=0, description="Initial balance")
    commission: float = Field(0.0, ge=0, description="Commission per trade")
    spread: float = Field(0.0, ge=0, description="Spread in points")


class BacktestResponse(BaseModel):
    """Backtest response model."""
    backtest_id: str = Field(..., description="Unique backtest identifier")
    status: str = Field(..., description="Backtest status")
    results: Optional[Dict] = Field(None, description="Backtest results")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(..., description="Response timestamp")


# Authentication and security
class SecurityConfig:
    """Security configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    API_KEY_HEADER = "X-API-Key"
    API_KEY = os.getenv("API_KEY", "your-api-key-here")


security = HTTPBearer()
security_config = SecurityConfig()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication."""
    if credentials.credentials != security_config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting SignaMentis API service...")
    
    # Initialize components
    try:
        # Initialize trading components
        app.state.executor = None  # Will be initialized when needed
        app.state.dashboard = None  # Will be initialized when needed
        app.state.ensemble = None  # Will be initialized when needed
        app.state.risk_manager = None  # Will be initialized when needed
        app.state.strategy = None  # Will be initialized when needed
        app.state.backtester = None  # Will be initialized when needed
        
        logger.info("API service components initialized")
        
    except Exception as e:
        logger.error(f"Error initializing API service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down SignaMentis API service...")
    
    # Cleanup components
    try:
        if app.state.executor:
            app.state.executor.disconnect()
        if app.state.dashboard:
            app.state.dashboard.stop()
        
        logger.info("API service components cleaned up")
        
    except Exception as e:
        logger.error(f"Error cleaning up API service: {e}")


# Create FastAPI app
app = FastAPI(
    title="SignaMentis Trading API",
    description="REST API for SignaMentis AI-powered trading system",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Utility functions
def get_executor() -> MT5Executor:
    """Get or create MT5 executor instance."""
    if not app.state.executor:
        # Initialize executor with default config
        config = {
            'symbol': 'XAUUSD',
            'max_positions': 5,
            'max_daily_trades': 20,
            'max_daily_loss': 1000.0
        }
        app.state.executor = MT5Executor(config)
    
    return app.state.executor


def get_dashboard() -> LiveDashboard:
    """Get or create live dashboard instance."""
    if not app.state.dashboard:
        config = {
            'update_interval': 5.0,
            'max_data_points': 1000,
            'port': 8050,
            'host': '0.0.0.0'
        }
        app.state.dashboard = LiveDashboard(config)
    
    return app.state.dashboard


def get_ensemble() -> EnsembleManager:
    """Get or create ensemble manager instance."""
    if not app.state.ensemble:
        app.state.ensemble = EnsembleManager()
    
    return app.state.ensemble


def get_risk_manager() -> RiskManager:
    """Get or create risk manager instance."""
    if not app.state.risk_manager:
        app.state.risk_manager = RiskManager()
    
    return app.state.risk_manager


def get_strategy() -> SuperTrendStrategy:
    """Get or create strategy instance."""
    if not app.state.strategy:
        app.state.strategy = SuperTrendStrategy()
    
    return app.state.strategy


def get_backtester() -> Backtester:
    """Get or create backtester instance."""
    if not app.state.backtester:
        app.state.backtester = Backtester()
    
    return app.state.backtester


# API endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to SignaMentis Trading API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    try:
        # Check component statuses
        components = {
            "executor": "unknown",
            "dashboard": "unknown",
            "ensemble": "unknown",
            "risk_manager": "unknown",
            "strategy": "unknown",
            "backtester": "unknown"
        }
        
        # Check executor
        try:
            executor = get_executor()
            components["executor"] = "connected" if executor.connected else "disconnected"
        except Exception:
            components["executor"] = "error"
        
        # Check dashboard
        try:
            dashboard = get_dashboard()
            components["dashboard"] = "running" if dashboard.running else "stopped"
        except Exception:
            components["dashboard"] = "error"
        
        # Check other components
        try:
            get_ensemble()
            components["ensemble"] = "ready"
        except Exception:
            components["ensemble"] = "error"
        
        try:
            get_risk_manager()
            components["risk_manager"] = "ready"
        except Exception:
            components["risk_manager"] = "error"
        
        try:
            get_strategy()
            components["strategy"] = "ready"
        except Exception:
            components["strategy"] = "error"
        
        try:
            get_backtester()
            components["backtester"] = "ready"
        except Exception:
            components["backtester"] = "error"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": components
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@app.get("/status", response_model=SystemStatus)
async def get_system_status(api_key: str = Depends(verify_api_key)):
    """Get system status."""
    try:
        # Calculate uptime
        start_time = getattr(app.state, 'start_time', datetime.now())
        uptime = datetime.now() - start_time
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        # Get component statuses
        components = {}
        
        # Executor status
        try:
            executor = get_executor()
            components["executor"] = "connected" if executor.connected else "disconnected"
        except Exception:
            components["executor"] = "error"
        
        # Dashboard status
        try:
            dashboard = get_dashboard()
            components["dashboard"] = "running" if dashboard.running else "stopped"
        except Exception:
            components["dashboard"] = "error"
        
        # Other components
        for component_name in ["ensemble", "risk_manager", "strategy", "backtester"]:
            try:
                getattr(app.state, component_name)
                components[component_name] = "ready"
            except Exception:
                components[component_name] = "error"
        
        # Get recent alerts
        try:
            dashboard = get_dashboard()
            alerts = [alert.message for alert in list(dashboard.alerts)[-5:]]
        except Exception:
            alerts = []
        
        return SystemStatus(
            status="operational",
            uptime=uptime_str,
            version="1.0.0",
            last_update=datetime.now(),
            components=components,
            alerts=alerts
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system status: {str(e)}"
        )


@app.post("/connect", response_model=Dict[str, Any])
async def connect_to_mt5(api_key: str = Depends(verify_api_key)):
    """Connect to MetaTrader 5."""
    try:
        executor = get_executor()
        
        if executor.connected:
            return {
                "status": "already_connected",
                "message": "Already connected to MT5",
                "timestamp": datetime.now().isoformat()
            }
        
        # Attempt connection
        if executor.connect():
            return {
                "status": "connected",
                "message": "Successfully connected to MT5",
                "timestamp": datetime.now().isoformat(),
                "account_info": executor.get_account_info(),
                "symbol_info": executor.get_symbol_info()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to connect to MT5"
            )
            
    except Exception as e:
        logger.error(f"Error connecting to MT5: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error connecting to MT5: {str(e)}"
        )


@app.post("/disconnect", response_model=Dict[str, Any])
async def disconnect_from_mt5(api_key: str = Depends(verify_api_key)):
    """Disconnect from MetaTrader 5."""
    try:
        executor = get_executor()
        
        if not executor.connected:
            return {
                "status": "already_disconnected",
                "message": "Already disconnected from MT5",
                "timestamp": datetime.now().isoformat()
            }
        
        executor.disconnect()
        
        return {
            "status": "disconnected",
            "message": "Successfully disconnected from MT5",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error disconnecting from MT5: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error disconnecting from MT5: {str(e)}"
        )


@app.get("/account", response_model=Dict[str, Any])
async def get_account_info(api_key: str = Depends(verify_api_key)):
    """Get account information."""
    try:
        executor = get_executor()
        
        if not executor.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not connected to MT5"
            )
        
        account_info = executor.get_account_info()
        if not account_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get account info"
            )
        
        return {
            "status": "success",
            "data": account_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting account info: {str(e)}"
        )


@app.get("/positions", response_model=List[PositionInfo])
async def get_positions(api_key: str = Depends(verify_api_key)):
    """Get open positions."""
    try:
        executor = get_executor()
        
        if not executor.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not connected to MT5"
            )
        
        positions = executor.get_positions()
        
        # Convert to Pydantic models
        position_list = []
        for ticket, pos in positions.items():
            position_list.append(PositionInfo(
                ticket=ticket,
                symbol=pos['symbol'],
                type=pos['type'],
                volume=pos['volume'],
                price_open=pos['price_open'],
                price_current=pos['price_current'],
                profit=pos['profit'],
                stop_loss=pos['stop_loss'],
                take_profit=pos['take_profit'],
                swap=pos['swap'],
                time=pos['time']
            ))
        
        return position_list
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting positions: {str(e)}"
        )


@app.post("/order", response_model=OrderResponse)
async def place_order(order_request: OrderRequestModel, api_key: str = Depends(verify_api_key)):
    """Place a new order."""
    try:
        executor = get_executor()
        
        if not executor.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not connected to MT5"
            )
        
        # Convert to OrderRequest
        order_req = OrderRequest(
            symbol=order_request.symbol,
            order_type=OrderType(order_request.order_type),
            volume=order_request.volume,
            price=order_request.price,
            stop_loss=order_request.stop_loss,
            take_profit=order_request.take_profit,
            comment=order_request.comment,
            magic=order_request.magic
        )
        
        # Place order
        result = executor.place_order(order_req)
        
        if result:
            return OrderResponse(
                order_ticket=result.order_ticket,
                status=result.status.value,
                message="Order placed successfully",
                timestamp=result.timestamp,
                error_code=result.error_code,
                error_description=result.error_description
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to place order"
            )
            
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error placing order: {str(e)}"
        )


@app.delete("/position/{ticket}")
async def close_position(ticket: int, api_key: str = Depends(verify_api_key)):
    """Close a position."""
    try:
        executor = get_executor()
        
        if not executor.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not connected to MT5"
            )
        
        if executor.close_position(ticket):
            return {
                "status": "success",
                "message": f"Position {ticket} closed successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to close position {ticket}"
            )
            
    except Exception as e:
        logger.error(f"Error closing position {ticket}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error closing position: {str(e)}"
        )


@app.get("/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(api_key: str = Depends(verify_api_key)):
    """Get performance metrics."""
    try:
        dashboard = get_dashboard()
        
        if not dashboard.performance_data:
            # Return default metrics
            return PerformanceMetricsResponse(
                timestamp=datetime.now(),
                total_pnl=0.0,
                daily_pnl=0.0,
                win_rate=0.0,
                total_trades=0,
                open_positions=0,
                equity=0.0,
                balance=0.0,
                drawdown=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0
            )
        
        # Get latest metrics
        latest = dashboard.performance_data[-1]
        
        return PerformanceMetricsResponse(
            timestamp=latest.timestamp,
            total_pnl=latest.total_pnl,
            daily_pnl=latest.daily_pnl,
            win_rate=latest.win_rate,
            total_trades=latest.total_trades,
            open_positions=latest.open_positions,
            equity=latest.equity,
            balance=latest.balance,
            drawdown=latest.drawdown,
            sharpe_ratio=latest.sharpe_ratio,
            max_drawdown=latest.max_drawdown
        )
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting performance metrics: {str(e)}"
        )


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(backtest_request: BacktestRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    """Run backtest."""
    try:
        backtester = get_backtester()
        
        # Generate unique backtest ID
        backtest_id = f"bt_{int(time.time())}_{hash(backtest_request.json()) % 10000:04d}"
        
        # Start backtest in background
        background_tasks.add_task(
            run_backtest_task,
            backtest_id,
            backtest_request,
            backtester
        )
        
        return BacktestResponse(
            backtest_id=backtest_id,
            status="started",
            results=None,
            message="Backtest started successfully",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error starting backtest: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting backtest: {str(e)}"
        )


async def run_backtest_task(backtest_id: str, request: BacktestRequest, backtester: Backtester):
    """Background task to run backtest."""
    try:
        logger.info(f"Starting backtest {backtest_id}")
        
        # Parse dates
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        # Run backtest
        results = backtester.run_backtest(
            start_date=start_date,
            end_date=end_date,
            symbol=request.symbol,
            timeframe=request.timeframe,
            initial_balance=request.initial_balance,
            commission=request.commission,
            spread=request.spread
        )
        
        logger.info(f"Backtest {backtest_id} completed successfully")
        
        # Store results (in production, save to database)
        # For now, just log the results
        logger.info(f"Backtest {backtest_id} results: {results}")
        
    except Exception as e:
        logger.error(f"Error in backtest {backtest_id}: {e}")


@app.get("/backtest/{backtest_id}")
async def get_backtest_results(backtest_id: str, api_key: str = Depends(verify_api_key)):
    """Get backtest results."""
    try:
        # In production, retrieve from database
        # For now, return placeholder
        return {
            "backtest_id": backtest_id,
            "status": "completed",
            "message": "Backtest results retrieved",
            "timestamp": datetime.now().isoformat(),
            "results": {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting backtest results: {str(e)}"
        )


@app.get("/dashboard/start")
async def start_dashboard(api_key: str = Depends(verify_api_key)):
    """Start live dashboard."""
    try:
        dashboard = get_dashboard()
        
        if dashboard.running:
            return {
                "status": "already_running",
                "message": "Dashboard already running",
                "url": dashboard.get_dashboard_url(),
                "timestamp": datetime.now().isoformat()
            }
        
        # Start dashboard in background
        import threading
        dashboard_thread = threading.Thread(target=dashboard.start, daemon=True)
        dashboard_thread.start()
        
        # Wait a bit for dashboard to start
        await asyncio.sleep(2)
        
        return {
            "status": "started",
            "message": "Dashboard started successfully",
            "url": dashboard.get_dashboard_url(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting dashboard: {str(e)}"
        )


@app.get("/dashboard/stop")
async def stop_dashboard(api_key: str = Depends(verify_api_key)):
    """Stop live dashboard."""
    try:
        dashboard = get_dashboard()
        
        if not dashboard.running:
            return {
                "status": "already_stopped",
                "message": "Dashboard already stopped",
                "timestamp": datetime.now().isoformat()
            }
        
        dashboard.stop()
        
        return {
            "status": "stopped",
            "message": "Dashboard stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error stopping dashboard: {str(e)}"
        )


@app.get("/dashboard/url")
async def get_dashboard_url(api_key: str = Depends(verify_api_key)):
    """Get dashboard URL."""
    try:
        dashboard = get_dashboard()
        return {
            "url": dashboard.get_dashboard_url(),
            "running": dashboard.running,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting dashboard URL: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


if __name__ == "__main__":
    # Run the API service
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
