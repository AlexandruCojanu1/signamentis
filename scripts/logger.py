"""
SignaMentis Logging Module

This module provides comprehensive logging functionality for the trading system.
Includes structured logging, log rotation, and integration with monitoring systems.

Author: SignaMentis Team
Version: 1.0.0
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json
import traceback
from pathlib import Path
import threading
from queue import Queue
import time
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Configure basic logging
logging.basicConfig(level=logging.INFO)


class LogLevel(Enum):
    """Log levels enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Log categories enumeration."""
    SYSTEM = "SYSTEM"
    TRADING = "TRADING"
    AI_MODELS = "AI_MODELS"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    EXECUTION = "EXECUTION"
    PERFORMANCE = "PERFORMANCE"
    ALERTS = "ALERTS"
    API = "API"
    DATABASE = "DATABASE"
    NETWORK = "NETWORK"


@dataclass
class LogEntry:
    """Log entry container."""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    source: str
    thread_id: int
    process_id: int
    data: Optional[Dict] = None
    exception: Optional[str] = None
    stack_trace: Optional[str] = None


class StructuredFormatter(logging.Formatter):
    """Structured log formatter."""
    
    def __init__(self, include_timestamp: bool = True, include_thread: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_thread = include_thread
    
    def format(self, record):
        """Format log record."""
        # Base format
        if self.include_timestamp:
            timestamp = datetime.fromtimestamp(record.created).isoformat()
            base_format = f"[{timestamp}] "
        else:
            base_format = ""
        
        # Add thread info
        if self.include_thread and hasattr(record, 'threadName'):
            base_format += f"[{record.threadName}] "
        
        # Add level and category
        level = getattr(record, 'levelname', 'INFO')
        category = getattr(record, 'category', 'SYSTEM')
        base_format += f"[{level}] [{category}] "
        
        # Add source
        source = getattr(record, 'source', record.name)
        base_format += f"[{source}] "
        
        # Add message
        base_format += record.getMessage()
        
        # Add exception info
        if record.exc_info:
            base_format += f"\nException: {self.formatException(record.exc_info)}"
        
        return base_format


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': getattr(record, 'levelname', 'INFO'),
            'category': getattr(record, 'category', 'SYSTEM'),
            'source': getattr(record, 'source', record.name),
            'message': record.getMessage(),
            'thread_id': record.thread,
            'process_id': record.process,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in log_entry and not key.startswith('_'):
                log_entry[key] = value
        
        return json.dumps(log_entry)


class LogManager:
    """
    Centralized log manager for the SignaMentis trading system.
    
    Features:
    - Multiple log handlers (file, console, network)
    - Log rotation and archival
    - Structured logging
    - Performance monitoring
    - Alert system integration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize log manager.
        
        Args:
            config: Logging configuration
        """
        self.config = config or {}
        
        # Logging configuration
        self.log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        self.log_dir = self.config.get('log_dir', 'logs')
        self.max_file_size = self.config.get('max_file_size', 10 * 1024 * 1024)  # 10MB
        self.backup_count = self.config.get('backup_count', 5)
        self.enable_console = self.config.get('enable_console', True)
        self.enable_file = self.config.get('enable_file', True)
        self.enable_json = self.config.get('enable_json', False)
        self.enable_network = self.config.get('enable_network', False)
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize loggers
        self.loggers = {}
        self.handlers = {}
        
        # Performance tracking
        self.log_counts = {level.value: 0 for level in LogLevel}
        self.start_time = datetime.now()
        self.last_cleanup = datetime.now()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize system logger
        self._setup_system_logger()
        
        logger.info("Log Manager initialized")
    
    def _setup_system_logger(self):
        """Setup the main system logger."""
        # Create system logger
        system_logger = logging.getLogger('SignaMentis')
        system_logger.setLevel(self.log_level)
        
        # Remove existing handlers
        for handler in system_logger.handlers[:]:
            system_logger.removeHandler(handler)
        
        # Add console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            if self.enable_json:
                console_formatter = JSONFormatter()
            else:
                console_formatter = StructuredFormatter()
            
            console_handler.setFormatter(console_formatter)
            system_logger.addHandler(console_handler)
            self.handlers['console'] = console_handler
        
        # Add file handler
        if self.enable_file:
            # Main log file
            main_log_file = os.path.join(self.log_dir, 'signa_mentis.log')
            main_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            main_handler.setLevel(self.log_level)
            
            if self.enable_json:
                main_formatter = JSONFormatter()
            else:
                main_formatter = StructuredFormatter()
            
            main_handler.setFormatter(main_formatter)
            system_logger.addHandler(main_handler)
            self.handlers['main_file'] = main_handler
            
            # Error log file
            error_log_file = os.path.join(self.log_dir, 'errors.log')
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(main_formatter)
            system_logger.addHandler(error_handler)
            self.handlers['error_file'] = error_handler
        
        # Store system logger
        self.loggers['system'] = system_logger
        
        # Set as default logger
        logging.getLogger().handlers = system_logger.handlers
        logging.getLogger().setLevel(self.log_level)
    
    def get_logger(self, name: str, category: LogCategory = LogCategory.SYSTEM) -> logging.Logger:
        """
        Get or create a logger with specified name and category.
        
        Args:
            name: Logger name
            category: Log category
            
        Returns:
            Logger instance
        """
        with self.lock:
            if name not in self.loggers:
                # Create new logger
                logger = logging.getLogger(name)
                logger.setLevel(self.log_level)
                
                # Add category attribute
                logger.category = category.value
                logger.source = name
                
                # Add handlers from system logger
                for handler in self.loggers['system'].handlers:
                    logger.addHandler(handler)
                
                # Prevent propagation to avoid duplicate logs
                logger.propagate = False
                
                self.loggers[name] = logger
            
            return self.loggers[name]
    
    def log(self, level: LogLevel, category: LogCategory, message: str, 
            source: str, data: Optional[Dict] = None, exception: Optional[Exception] = None):
        """
        Log a message with structured information.
        
        Args:
            level: Log level
            category: Log category
            message: Log message
            source: Source of the log
            data: Additional data to log
            exception: Exception to log
        """
        try:
            # Get logger
            logger = self.get_logger(source, category)
            
            # Prepare extra fields
            extra = {
                'category': category.value,
                'source': source
            }
            
            if data:
                extra['data'] = data
            
            # Log with appropriate level
            if level == LogLevel.DEBUG:
                logger.debug(message, extra=extra, exc_info=exception)
            elif level == LogLevel.INFO:
                logger.info(message, extra=extra, exc_info=exception)
            elif level == LogLevel.WARNING:
                logger.warning(message, extra=extra, exc_info=exception)
            elif level == LogLevel.ERROR:
                logger.error(message, extra=extra, exc_info=exception)
            elif level == LogLevel.CRITICAL:
                logger.critical(message, extra=extra, exc_info=exception)
            
            # Update counters
            with self.lock:
                self.log_counts[level.value] += 1
            
        except Exception as e:
            # Fallback to basic logging if structured logging fails
            print(f"Logging error: {e}")
            print(f"Original message: [{level.value}] [{category.value}] [{source}] {message}")
    
    def debug(self, message: str, source: str, category: LogCategory = LogCategory.SYSTEM, 
              data: Optional[Dict] = None):
        """Log debug message."""
        self.log(LogLevel.DEBUG, category, message, source, data)
    
    def info(self, message: str, source: str, category: LogCategory = LogCategory.SYSTEM, 
             data: Optional[Dict] = None):
        """Log info message."""
        self.log(LogLevel.INFO, category, message, source, data)
    
    def warning(self, message: str, source: str, category: LogCategory = LogCategory.SYSTEM, 
                data: Optional[Dict] = None):
        """Log warning message."""
        self.log(LogLevel.WARNING, category, message, source, data)
    
    def error(self, message: str, source: str, category: LogCategory = LogCategory.SYSTEM, 
              data: Optional[Dict] = None, exception: Optional[Exception] = None):
        """Log error message."""
        self.log(LogLevel.ERROR, category, message, source, data, exception)
    
    def critical(self, message: str, source: str, category: LogCategory = LogCategory.SYSTEM, 
                 data: Optional[Dict] = None, exception: Optional[Exception] = None):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, category, message, source, data, exception)
    
    def log_trade(self, trade_data: Dict, source: str):
        """Log trade information."""
        self.info(
            f"Trade executed: {trade_data.get('type', 'UNKNOWN')} {trade_data.get('volume', 0)} lots "
            f"at {trade_data.get('price', 0)}",
            source,
            LogCategory.TRADING,
            trade_data
        )
    
    def log_signal(self, signal_data: Dict, source: str):
        """Log trading signal."""
        self.info(
            f"Signal generated: {signal_data.get('direction', 'UNKNOWN')} "
            f"confidence: {signal_data.get('confidence', 0):.2f}",
            source,
            LogCategory.AI_MODELS,
            signal_data
        )
    
    def log_risk_event(self, risk_data: Dict, source: str):
        """Log risk management event."""
        self.warning(
            f"Risk event: {risk_data.get('type', 'UNKNOWN')} - {risk_data.get('message', '')}",
            source,
            LogCategory.RISK_MANAGEMENT,
            risk_data
        )
    
    def log_performance(self, performance_data: Dict, source: str):
        """Log performance metrics."""
        self.info(
            f"Performance update: P&L: ${performance_data.get('total_pnl', 0):.2f}, "
            f"Win Rate: {performance_data.get('win_rate', 0):.1f}%",
            source,
            LogCategory.PERFORMANCE,
            performance_data
        )
    
    def log_api_request(self, request_data: Dict, source: str):
        """Log API request."""
        self.debug(
            f"API request: {request_data.get('method', 'UNKNOWN')} {request_data.get('endpoint', '')}",
            source,
            LogCategory.API,
            request_data
        )
    
    def log_api_response(self, response_data: Dict, source: str):
        """Log API response."""
        self.debug(
            f"API response: {response_data.get('status_code', 0)} - {response_data.get('message', '')}",
            source,
            LogCategory.API,
            response_data
        )
    
    def log_exception(self, message: str, source: str, exception: Exception, 
                      category: LogCategory = LogCategory.SYSTEM, data: Optional[Dict] = None):
        """Log exception with full details."""
        self.error(
            f"{message}: {str(exception)}",
            source,
            category,
            data,
            exception
        )
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self.lock:
            uptime = datetime.now() - self.start_time
            total_logs = sum(self.log_counts.values())
            
            return {
                'start_time': self.start_time.isoformat(),
                'uptime_seconds': uptime.total_seconds(),
                'total_logs': total_logs,
                'logs_by_level': self.log_counts.copy(),
                'active_loggers': len(self.loggers),
                'active_handlers': len(self.handlers),
                'log_directory': self.log_dir,
                'last_cleanup': self.last_cleanup.isoformat()
            }
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            log_dir = Path(self.log_dir)
            
            for log_file in log_dir.glob('*.log.*'):
                try:
                    # Try to get file creation time
                    stat = log_file.stat()
                    if stat.st_ctime < cutoff_date.timestamp():
                        log_file.unlink()
                        self.info(f"Deleted old log file: {log_file}", "LogManager", LogCategory.SYSTEM)
                except Exception as e:
                    self.warning(f"Could not delete old log file {log_file}: {e}", "LogManager", LogCategory.SYSTEM)
            
            self.last_cleanup = datetime.now()
            
        except Exception as e:
            self.error(f"Error during log cleanup: {e}", "LogManager", LogCategory.SYSTEM)
    
    def export_logs(self, start_time: datetime, end_time: datetime, 
                    categories: Optional[List[LogCategory]] = None,
                    levels: Optional[List[LogLevel]] = None) -> List[LogEntry]:
        """Export logs for specified time range and filters."""
        # This is a simplified export - in production, you'd query a database
        # For now, return empty list
        return []
    
    def set_log_level(self, level: str):
        """Set log level for all loggers."""
        try:
            new_level = getattr(logging, level.upper())
            
            with self.lock:
                for logger in self.loggers.values():
                    logger.setLevel(new_level)
                
                for handler in self.handlers.values():
                    handler.setLevel(new_level)
            
            self.log_level = new_level
            self.info(f"Log level changed to {level}", "LogManager", LogCategory.SYSTEM)
            
        except Exception as e:
            self.error(f"Error setting log level: {e}", "LogManager", LogCategory.SYSTEM)
    
    def flush_logs(self):
        """Flush all log handlers."""
        try:
            with self.lock:
                for handler in self.handlers.values():
                    handler.flush()
            
            self.info("All log handlers flushed", "LogManager", LogCategory.SYSTEM)
            
        except Exception as e:
            self.error(f"Error flushing logs: {e}", "LogManager", LogCategory.SYSTEM)


# Global log manager instance
_log_manager = None


def get_log_manager(config: Optional[Dict] = None) -> LogManager:
    """
    Get or create global log manager instance.
    
    Args:
        config: Logging configuration
        
    Returns:
        LogManager instance
    """
    global _log_manager
    
    if _log_manager is None:
        _log_manager = LogManager(config)
    
    return _log_manager


def get_logger(name: str, category: LogCategory = LogCategory.SYSTEM) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        category: Log category
        
    Returns:
        Logger instance
    """
    log_manager = get_log_manager()
    return log_manager.get_logger(name, category)


# Convenience functions for quick logging
def log_debug(message: str, source: str, category: LogCategory = LogCategory.SYSTEM, 
              data: Optional[Dict] = None):
    """Quick debug log."""
    log_manager = get_log_manager()
    log_manager.debug(message, source, category, data)


def log_info(message: str, source: str, category: LogCategory = LogCategory.SYSTEM, 
             data: Optional[Dict] = None):
    """Quick info log."""
    log_manager = get_log_manager()
    log_manager.info(message, source, category, data)


def log_warning(message: str, source: str, category: LogCategory = LogCategory.SYSTEM, 
                data: Optional[Dict] = None):
    """Quick warning log."""
    log_manager = get_log_manager()
    log_manager.warning(message, source, category, data)


def log_error(message: str, source: str, category: LogCategory = LogCategory.SYSTEM, 
              data: Optional[Dict] = None, exception: Optional[Exception] = None):
    """Quick error log."""
    log_manager = get_log_manager()
    log_manager.error(message, source, category, data, exception)


def log_critical(message: str, source: str, category: LogCategory = LogCategory.SYSTEM, 
                 data: Optional[Dict] = None, exception: Optional[Exception] = None):
    """Quick critical log."""
    log_manager = get_log_manager()
    log_manager.critical(message, source, category, data, exception)


if __name__ == "__main__":
    # Example usage
    config = {
        'log_level': 'DEBUG',
        'log_dir': 'logs',
        'enable_console': True,
        'enable_file': True,
        'enable_json': False
    }
    
    # Initialize log manager
    log_manager = get_log_manager(config)
    
    # Test different log levels and categories
    log_manager.info("System started", "Main", LogCategory.SYSTEM)
    log_manager.debug("Debug message", "Test", LogCategory.SYSTEM)
    log_manager.warning("Warning message", "Test", LogCategory.SYSTEM)
    log_manager.error("Error message", "Test", LogCategory.SYSTEM)
    
    # Test category-specific logging
    log_manager.log_trade({
        'type': 'BUY',
        'volume': 0.1,
        'price': 2000.0,
        'symbol': 'XAUUSD'
    }, "TradingEngine")
    
    log_manager.log_signal({
        'direction': 'BUY',
        'confidence': 0.85,
        'price': 2000.0
    }, "AIEnsemble")
    
    log_manager.log_risk_event({
        'type': 'POSITION_LIMIT',
        'message': 'Maximum positions reached'
    }, "RiskManager")
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        log_manager.log_exception("Test error occurred", "Test", e, LogCategory.SYSTEM)
    
    # Get statistics
    stats = log_manager.get_log_statistics()
    print(f"Log statistics: {stats}")
    
    print("Logger test completed!")
