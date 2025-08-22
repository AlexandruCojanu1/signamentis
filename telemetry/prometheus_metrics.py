#!/usr/bin/env python3
"""
SignaMentis - Prometheus Metrics System

This module implements a comprehensive metrics collection system using Prometheus
for monitoring trading system performance, health, and operational metrics.

Author: SignaMentis Team
Version: 2.0.0
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info, Enum as PromEnum,
    generate_latest, CONTENT_TYPE_LATEST, start_http_server,
    CollectorRegistry, multiprocess
)
import psutil
import asyncio
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"
    ENUM = "enum"

class TradingMetricCategory(Enum):
    """Categories of trading metrics."""
    TRADE_EXECUTION = "trade_execution"
    RISK_MANAGEMENT = "risk_management"
    MODEL_PERFORMANCE = "model_performance"
    SYSTEM_HEALTH = "system_health"
    MARKET_DATA = "market_data"
    NEWS_SENTIMENT = "news_sentiment"
    PERFORMANCE = "performance"
    LATENCY = "latency"

@dataclass
class MetricConfig:
    """Configuration for a metric."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    quantiles: Optional[List[float]] = None
    namespace: str = "signa_mentis"
    subsystem: str = "trading_system"

class PrometheusMetricsManager:
    """
    Centralized metrics management system for SignaMentis.
    
    Provides:
    - Trading-specific metrics
    - System health monitoring
    - Performance tracking
    - Custom metric registration
    - HTTP metrics endpoint
    """
    
    def __init__(
        self,
        port: int = 8000,
        addr: str = '0.0.0.0',
        registry: Optional[CollectorRegistry] = None,
        multiprocess_mode: bool = False
    ):
        """
        Initialize metrics manager.
        
        Args:
            port: Port for HTTP metrics server
            addr: Address for HTTP metrics server
            registry: Custom Prometheus registry
            multiprocess_mode: Enable multiprocess mode
        """
        self.port = port
        self.addr = addr
        self.multiprocess_mode = multiprocess_mode
        
        # Initialize registry
        if multiprocess_mode:
            self.registry = multiprocess.MultiProcessCollector(CollectorRegistry())
        else:
            self.registry = registry or CollectorRegistry()
        
        # Metrics storage
        self.metrics: Dict[str, Any] = {}
        self.custom_metrics: Dict[str, Any] = {}
        
        # HTTP server
        self.http_server = None
        self.server_thread = None
        
        # System monitoring
        self.system_monitor = None
        self.monitoring_active = False
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        # Start HTTP server
        self._start_http_server()
    
    def _initialize_default_metrics(self):
        """Initialize default trading system metrics."""
        
        # Trade Execution Metrics
        self._register_metric(
            MetricConfig(
                name="trades_total",
                description="Total number of trades executed",
                metric_type=MetricType.COUNTER,
                labels=["symbol", "side", "status", "strategy"],
                subsystem="execution"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="trade_volume_total",
                description="Total volume of trades executed",
                metric_type=MetricType.COUNTER,
                labels=["symbol", "side", "status"],
                subsystem="execution"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="trade_pnl_total",
                description="Total P&L from trades",
                metric_type=MetricType.GAUGE,
                labels=["symbol", "strategy"],
                subsystem="execution"
            )
        )
        
        # Risk Management Metrics
        self._register_metric(
            MetricConfig(
                name="risk_level_current",
                description="Current risk level",
                metric_type=MetricType.GAUGE,
                labels=["risk_type", "symbol"],
                subsystem="risk"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="position_size_current",
                description="Current position size",
                metric_type=MetricType.GAUGE,
                labels=["symbol", "side"],
                subsystem="risk"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="drawdown_current",
                description="Current drawdown percentage",
                metric_type=MetricType.GAUGE,
                labels=["symbol", "strategy"],
                subsystem="risk"
            )
        )
        
        # Model Performance Metrics
        self._register_metric(
            MetricConfig(
                name="model_predictions_total",
                description="Total number of model predictions",
                metric_type=MetricType.COUNTER,
                labels=["model_name", "symbol"],
                subsystem="models"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="model_accuracy",
                description="Model prediction accuracy",
                metric_type=MetricType.GAUGE,
                labels=["model_name", "symbol", "timeframe"],
                subsystem="models"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="model_inference_time",
                description="Model inference time in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=["model_name", "symbol"],
                buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                subsystem="models"
            )
        )
        
        # System Health Metrics
        self._register_metric(
            MetricConfig(
                name="system_uptime_seconds",
                description="System uptime in seconds",
                metric_type=MetricType.GAUGE,
                subsystem="system"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="system_memory_usage_bytes",
                description="System memory usage in bytes",
                metric_type=MetricType.GAUGE,
                subsystem="system"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="system_cpu_usage_percent",
                description="System CPU usage percentage",
                metric_type=MetricType.GAUGE,
                subsystem="system"
            )
        )
        
        # Market Data Metrics
        self._register_metric(
            MetricConfig(
                name="market_data_updates_total",
                description="Total market data updates received",
                metric_type=MetricType.COUNTER,
                labels=["symbol", "data_type"],
                subsystem="market_data"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="market_data_latency_seconds",
                description="Market data latency in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=["symbol", "data_type"],
                buckets=[0.001, 0.01, 0.1, 0.5, 1.0],
                subsystem="market_data"
            )
        )
        
        # News Sentiment Metrics
        self._register_metric(
            MetricConfig(
                name="news_articles_processed_total",
                description="Total news articles processed",
                metric_type=MetricType.COUNTER,
                labels=["source", "category"],
                subsystem="news"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="sentiment_score_current",
                description="Current sentiment score",
                metric_type=MetricType.GAUGE,
                labels=["source", "category", "symbol"],
                subsystem="news"
            )
        )
        
        # Performance Metrics
        self._register_metric(
            MetricConfig(
                name="sharpe_ratio_current",
                description="Current Sharpe ratio",
                metric_type=MetricType.GAUGE,
                labels=["symbol", "strategy", "timeframe"],
                subsystem="performance"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="win_rate_current",
                description="Current win rate percentage",
                metric_type=MetricType.GAUGE,
                labels=["symbol", "strategy"],
                subsystem="performance"
            )
        )
        
        # Latency Metrics
        self._register_metric(
            MetricConfig(
                name="api_request_duration_seconds",
                description="API request duration in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=["endpoint", "method", "status_code"],
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                subsystem="api"
            )
        )
        
        self._register_metric(
            MetricConfig(
                name="database_query_duration_seconds",
                description="Database query duration in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=["operation", "table"],
                buckets=[0.001, 0.01, 0.1, 0.5, 1.0],
                subsystem="database"
            )
        )
    
    def _register_metric(self, config: MetricConfig):
        """Register a metric with Prometheus."""
        try:
            metric_name = f"{config.namespace}_{config.subsystem}_{config.name}"
            
            if config.metric_type == MetricType.COUNTER:
                metric = Counter(
                    metric_name,
                    config.description,
                    config.labels,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    metric_name,
                    config.description,
                    config.labels,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.HISTOGRAM:
                metric = Histogram(
                    metric_name,
                    config.description,
                    config.labels,
                    buckets=config.buckets or [0.1, 0.5, 1.0, 2.0, 5.0],
                    registry=self.registry
                )
            elif config.metric_type == MetricType.SUMMARY:
                metric = Summary(
                    metric_name,
                    config.description,
                    config.labels,
                    quantiles=config.quantiles or [0.5, 0.9, 0.99],
                    registry=self.registry
                )
            elif config.metric_type == MetricType.INFO:
                metric = Info(
                    metric_name,
                    config.description,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.ENUM:
                metric = PromEnum(
                    metric_name,
                    config.description,
                    config.labels,
                    registry=self.registry
                )
            else:
                raise ValueError(f"Unsupported metric type: {config.metric_type}")
            
            self.metrics[metric_name] = metric
            logger.debug(f"✅ Registered metric: {metric_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register metric {config.name}: {e}")
    
    def _start_http_server(self):
        """Start HTTP server for metrics endpoint."""
        try:
            def start_server():
                start_http_server(self.port, self.addr, registry=self.registry)
            
            self.server_thread = threading.Thread(target=start_server, daemon=True)
            self.server_thread.start()
            
            # Wait a bit for server to start
            time.sleep(1)
            
            logger.info(f"🌐 Prometheus metrics server started on {self.addr}:{self.port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start metrics server: {e}")
    
    def get_metric(self, name: str) -> Optional[Any]:
        """Get a registered metric by name."""
        return self.metrics.get(name)
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        try:
            metric = self.get_metric(name)
            if metric and hasattr(metric, 'inc'):
                if labels:
                    metric.labels(**labels).inc(value)
                else:
                    metric.inc(value)
        except Exception as e:
            logger.warning(f"Failed to increment counter {name}: {e}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        try:
            metric = self.get_metric(name)
            if metric and hasattr(metric, 'set'):
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
        except Exception as e:
            logger.warning(f"Failed to set gauge {name}: {e}")
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value in a histogram metric."""
        try:
            metric = self.get_metric(name)
            if metric and hasattr(metric, 'observe'):
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
        except Exception as e:
            logger.warning(f"Failed to observe histogram {name}: {e}")
    
    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value in a summary metric."""
        try:
            metric = self.get_metric(name)
            if metric and hasattr(metric, 'observe'):
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
        except Exception as e:
            logger.warning(f"Failed to observe summary {name}: {e}")
    
    def set_info(self, name: str, info: Dict[str, str]):
        """Set info metric values."""
        try:
            metric = self.get_metric(name)
            if metric and hasattr(metric, 'info'):
                metric.info(info)
        except Exception as e:
            logger.warning(f"Failed to set info {name}: {e}")
    
    def set_enum(self, name: str, value: str, labels: Optional[Dict[str, str]] = None):
        """Set enum metric value."""
        try:
            metric = self.get_metric(name)
            if metric and hasattr(metric, 'state'):
                if labels:
                    metric.labels(**labels).state(value)
                else:
                    metric.state(value)
        except Exception as e:
            logger.warning(f"Failed to set enum {name}: {e}")
    
    def start_system_monitoring(self, interval: int = 30):
        """Start system monitoring with periodic updates."""
        if self.monitoring_active:
            logger.warning("System monitoring already active")
            return
        
        self.monitoring_active = True
        self.system_monitor = threading.Thread(target=self._system_monitor_loop, args=(interval,), daemon=True)
        self.system_monitor.start()
        logger.info("🔍 System monitoring started")
    
    def stop_system_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.system_monitor:
            self.system_monitor.join(timeout=5)
        logger.info("🔍 System monitoring stopped")
    
    def _system_monitor_loop(self, interval: int):
        """System monitoring loop."""
        start_time = time.time()
        
        while self.monitoring_active:
            try:
                # Update system metrics
                self._update_system_metrics(start_time)
                
                # Sleep for interval
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(interval)
    
    def _update_system_metrics(self, start_time: float):
        """Update system health metrics."""
        try:
            # Uptime
            uptime = time.time() - start_time
            self.set_gauge("signa_mentis_system_system_uptime_seconds", uptime)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.set_gauge("signa_mentis_system_system_memory_usage_bytes", memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("signa_mentis_system_system_cpu_usage_percent", cpu_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            self.set_gauge("signa_mentis_system_system_disk_usage_percent", disk_usage)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.set_gauge("signa_mentis_system_system_network_bytes_sent", net_io.bytes_sent)
            self.set_gauge("signa_mentis_system_system_network_bytes_recv", net_io.bytes_recv)
            
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return ""
    
    def register_custom_metric(self, config: MetricConfig) -> bool:
        """Register a custom metric."""
        try:
            self._register_metric(config)
            return True
        except Exception as e:
            logger.error(f"Failed to register custom metric: {e}")
            return False
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all registered metrics."""
        summary = {
            "total_metrics": len(self.metrics),
            "metric_types": {},
            "subsystems": {},
            "custom_metrics": len(self.custom_metrics)
        }
        
        for metric_name, metric in self.metrics.items():
            # Count by type
            metric_type = type(metric).__name__
            summary["metric_types"][metric_type] = summary["metric_types"].get(metric_type, 0) + 1
            
            # Count by subsystem
            parts = metric_name.split('_')
            if len(parts) >= 3:
                subsystem = parts[2]
                summary["subsystems"][subsystem] = summary["subsystems"].get(subsystem, 0) + 1
        
        return summary
    
    def close(self):
        """Cleanup and close metrics manager."""
        try:
            self.stop_system_monitoring()
            logger.info("🔌 Prometheus metrics manager closed")
        except Exception as e:
            logger.error(f"Error closing metrics manager: {e}")

class MetricsDecorator:
    """Decorator for automatically measuring function performance."""
    
    def __init__(self, metrics_manager: PrometheusMetricsManager, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.metrics_manager = metrics_manager
        self.metric_name = metric_name
        self.labels = labels or {}
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                # Record duration
                if "duration" in self.metric_name:
                    self.metrics_manager.observe_histogram(self.metric_name, duration, self.labels)
                
                # Record success/failure
                if "total" in self.metric_name:
                    self.metrics_manager.increment_counter(self.metric_name, 1, self.labels)
            
            return result
        
        return wrapper

@contextmanager
def measure_time(metrics_manager: PrometheusMetricsManager, metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Context manager for measuring execution time."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        metrics_manager.observe_histogram(metric_name, duration, labels)

# Example usage
if __name__ == "__main__":
    def main():
        # Create metrics manager
        metrics = PrometheusMetricsManager(port=8000)
        
        # Start system monitoring
        metrics.start_system_monitoring(interval=10)
        
        # Example: Record some trading metrics
        metrics.increment_counter(
            "signa_mentis_execution_trades_total",
            value=1,
            labels={"symbol": "XAUUSD", "side": "buy", "status": "executed", "strategy": "supertrend"}
        )
        
        metrics.set_gauge(
            "signa_mentis_execution_trade_pnl_total",
            value=150.50,
            labels={"symbol": "XAUUSD", "strategy": "supertrend"}
        )
        
        # Example: Measure function execution time
        @MetricsDecorator(metrics, "signa_mentis_models_model_inference_time", {"model_name": "bilstm", "symbol": "XAUUSD"})
        def example_model_inference():
            time.sleep(0.1)  # Simulate model inference
            return "prediction"
        
        # Run example
        result = example_model_inference()
        print(f"Model inference result: {result}")
        
        # Example: Using context manager
        with measure_time(metrics, "signa_mentis_api_api_request_duration_seconds", {"endpoint": "/trade", "method": "POST", "status_code": "200"}):
            time.sleep(0.05)  # Simulate API call
        
        # Get metrics summary
        summary = metrics.get_metrics_summary()
        print(f"Metrics summary: {summary}")
        
        # Keep running for a bit to see metrics
        print("🚀 Metrics server running on http://localhost:8000")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
        finally:
            metrics.close()
    
    # Run example
    main()
