#!/usr/bin/env python3
"""
SignaMentis Database Manager

This module provides comprehensive database management using SQLite
for storing backtesting results, trading data, and system metadata.

Author: SignaMentis Team
Version: 1.0.0
"""

import sqlite3
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class BacktestRecord:
    """Container for backtest database record."""
    id: Optional[int]
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    win_rate: float
    total_trades: int
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    execution_time: float
    parameters: Dict
    results_summary: Dict
    created_at: datetime


@dataclass
class TradeRecord:
    """Container for trade database record."""
    id: Optional[int]
    backtest_id: int
    timestamp: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    lot_size: float
    pnl: float
    stop_loss: float
    take_profit: float
    ai_confidence: float
    sentiment_score: float
    metadata: Dict


class DatabaseManager:
    """
    Comprehensive database manager for SignaMentis.
    
    Manages SQLite databases for:
    - Backtesting results
    - Trade history
    - Performance metrics
    - System configuration
    - Sentiment data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the database manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.db_path = Path(self.config.get('db_path', 'data/signamentis.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Database Manager initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with all required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create backtests table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS backtests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        start_date TIMESTAMP NOT NULL,
                        end_date TIMESTAMP NOT NULL,
                        initial_capital REAL NOT NULL,
                        final_capital REAL NOT NULL,
                        total_return REAL NOT NULL,
                        win_rate REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        max_drawdown REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        profit_factor REAL NOT NULL,
                        execution_time REAL NOT NULL,
                        parameters TEXT NOT NULL,
                        results_summary TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        backtest_id INTEGER NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        lot_size REAL NOT NULL,
                        pnl REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        ai_confidence REAL,
                        sentiment_score REAL,
                        metadata TEXT,
                        FOREIGN KEY (backtest_id) REFERENCES backtests (id)
                    )
                ''')
                
                # Create performance_metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        backtest_id INTEGER NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (backtest_id) REFERENCES backtests (id)
                    )
                ''')
                
                # Create equity_curves table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS equity_curves (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        backtest_id INTEGER NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        equity_value REAL NOT NULL,
                        drawdown REAL NOT NULL,
                        FOREIGN KEY (backtest_id) REFERENCES backtests (id)
                    )
                ''')
                
                # Create sentiment_data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sentiment_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT NOT NULL,
                        content TEXT,
                        sentiment_score REAL NOT NULL,
                        sentiment_polarity TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        keywords TEXT,
                        metadata TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create system_config table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_config (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        config_key TEXT UNIQUE NOT NULL,
                        config_value TEXT NOT NULL,
                        description TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create model_performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtests_strategy ON backtests (strategy_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtests_date ON backtests (start_date, end_date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_backtest ON trades (backtest_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp ON sentiment_data (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_source ON sentiment_data (source)')
                
                conn.commit()
                logger.info("Database initialized successfully with all tables")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def store_backtest_result(self, backtest_data: Dict) -> int:
        """
        Store backtest result in database.
        
        Args:
            backtest_data: Dictionary containing backtest results
            
        Returns:
            int: Backtest ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert backtest record
                cursor.execute('''
                    INSERT INTO backtests 
                    (strategy_name, start_date, end_date, initial_capital, final_capital,
                     total_return, win_rate, total_trades, max_drawdown, sharpe_ratio,
                     profit_factor, execution_time, parameters, results_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    backtest_data.get('strategy_name', 'Unknown'),
                    backtest_data.get('start_date'),
                    backtest_data.get('end_date'),
                    backtest_data.get('initial_capital', 0.0),
                    backtest_data.get('final_capital', 0.0),
                    backtest_data.get('total_return', 0.0),
                    backtest_data.get('win_rate', 0.0),
                    backtest_data.get('total_trades', 0),
                    backtest_data.get('max_drawdown', 0.0),
                    backtest_data.get('sharpe_ratio', 0.0),
                    backtest_data.get('profit_factor', 0.0),
                    backtest_data.get('execution_time', 0.0),
                    json.dumps(backtest_data.get('parameters', {})),
                    json.dumps(backtest_data.get('results_summary', {}))
                ))
                
                backtest_id = cursor.lastrowid
                
                # Store trades if available
                if 'trades' in backtest_data:
                    self._store_trades(backtest_id, backtest_data['trades'])
                
                # Store equity curve if available
                if 'equity_curve' in backtest_data:
                    self._store_equity_curve(backtest_id, backtest_data['equity_curve'])
                
                # Store performance metrics if available
                if 'performance_metrics' in backtest_data:
                    self._store_performance_metrics(backtest_id, backtest_data['performance_metrics'])
                
                conn.commit()
                logger.info(f"Backtest result stored with ID: {backtest_id}")
                return backtest_id
                
        except Exception as e:
            logger.error(f"Error storing backtest result: {e}")
            raise
    
    def _store_trades(self, backtest_id: int, trades: List[Dict]):
        """Store trades for a backtest."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for trade in trades:
                    cursor.execute('''
                        INSERT INTO trades 
                        (backtest_id, timestamp, symbol, direction, entry_price, exit_price,
                         lot_size, pnl, stop_loss, take_profit, ai_confidence, sentiment_score, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        backtest_id,
                        trade.get('timestamp'),
                        trade.get('symbol', 'XAUUSD'),
                        trade.get('direction', 'UNKNOWN'),
                        trade.get('entry_price', 0.0),
                        trade.get('exit_price', 0.0),
                        trade.get('lot_size', 0.0),
                        trade.get('pnl', 0.0),
                        trade.get('stop_loss'),
                        trade.get('take_profit'),
                        trade.get('ai_confidence'),
                        trade.get('sentiment_score'),
                        json.dumps(trade.get('metadata', {}))
                    ))
                
                logger.info(f"Stored {len(trades)} trades for backtest {backtest_id}")
                
        except Exception as e:
            logger.error(f"Error storing trades: {e}")
    
    def _store_equity_curve(self, backtest_id: int, equity_curve: Union[List, pd.Series]):
        """Store equity curve for a backtest."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert to list if pandas Series
                if isinstance(equity_curve, pd.Series):
                    equity_data = [(backtest_id, idx, value, 0.0) for idx, value in equity_curve.items()]
                else:
                    equity_data = [(backtest_id, i, value, 0.0) for i, value in enumerate(equity_curve)]
                
                cursor.executemany('''
                    INSERT INTO equity_curves (backtest_id, timestamp, equity_value, drawdown)
                    VALUES (?, ?, ?, ?)
                ''', equity_data)
                
                logger.info(f"Stored equity curve with {len(equity_data)} points for backtest {backtest_id}")
                
        except Exception as e:
            logger.error(f"Error storing equity curve: {e}")
    
    def _store_performance_metrics(self, backtest_id: int, metrics: Dict):
        """Store performance metrics for a backtest."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for metric_name, metric_value in metrics.items():
                    cursor.execute('''
                        INSERT INTO performance_metrics (backtest_id, metric_name, metric_value)
                        VALUES (?, ?, ?)
                    ''', (backtest_id, metric_name, metric_value))
                
                logger.info(f"Stored {len(metrics)} performance metrics for backtest {backtest_id}")
                
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
    
    def get_backtest_results(self, 
                           strategy_name: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           limit: int = 100) -> List[BacktestRecord]:
        """
        Retrieve backtest results from database.
        
        Args:
            strategy_name: Filter by strategy name
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results
            
        Returns:
            List of BacktestRecord objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = 'SELECT * FROM backtests WHERE 1=1'
                params = []
                
                if strategy_name:
                    query += ' AND strategy_name = ?'
                    params.append(strategy_name)
                
                if start_date:
                    query += ' AND start_date >= ?'
                    params.append(start_date)
                
                if end_date:
                    query += ' AND end_date <= ?'
                    params.append(end_date)
                
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to BacktestRecord objects
                results = []
                for row in rows:
                    try:
                        record = BacktestRecord(
                            id=row[0],
                            strategy_name=row[1],
                            start_date=datetime.fromisoformat(row[2]),
                            end_date=datetime.fromisoformat(row[3]),
                            initial_capital=row[4],
                            final_capital=row[5],
                            total_return=row[6],
                            win_rate=row[7],
                            total_trades=row[8],
                            max_drawdown=row[9],
                            sharpe_ratio=row[10],
                            profit_factor=row[11],
                            execution_time=row[12],
                            parameters=json.loads(row[13]),
                            results_summary=json.loads(row[14]),
                            created_at=datetime.fromisoformat(row[15])
                        )
                        results.append(record)
                    except Exception as e:
                        logger.warning(f"Error parsing backtest record: {e}")
                        continue
                
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving backtest results: {e}")
            return []
    
    def get_trades_for_backtest(self, backtest_id: int) -> List[TradeRecord]:
        """Get all trades for a specific backtest."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM trades WHERE backtest_id = ? ORDER BY timestamp
                ''', (backtest_id,))
                
                rows = cursor.fetchall()
                
                # Convert to TradeRecord objects
                trades = []
                for row in rows:
                    try:
                        trade = TradeRecord(
                            id=row[0],
                            backtest_id=row[1],
                            timestamp=datetime.fromisoformat(row[2]),
                            symbol=row[3],
                            direction=row[4],
                            entry_price=row[5],
                            exit_price=row[6],
                            lot_size=row[7],
                            pnl=row[8],
                            stop_loss=row[9],
                            take_profit=row[10],
                            ai_confidence=row[11],
                            sentiment_score=row[12],
                            metadata=json.loads(row[13]) if row[13] else {}
                        )
                        trades.append(trade)
                    except Exception as e:
                        logger.warning(f"Error parsing trade record: {e}")
                        continue
                
                return trades
                
        except Exception as e:
            logger.error(f"Error retrieving trades: {e}")
            return []
    
    def get_equity_curve_for_backtest(self, backtest_id: int) -> pd.Series:
        """Get equity curve for a specific backtest."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, equity_value FROM equity_curves 
                    WHERE backtest_id = ? ORDER BY timestamp
                ''', (backtest_id,))
                
                rows = cursor.fetchall()
                
                if not rows:
                    return pd.Series()
                
                # Convert to pandas Series
                timestamps = [datetime.fromisoformat(row[0]) for row in rows]
                values = [row[1] for row in rows]
                
                return pd.Series(values, index=timestamps)
                
        except Exception as e:
            logger.error(f"Error retrieving equity curve: {e}")
            return pd.Series()
    
    def store_sentiment_data(self, sentiment_data: Dict) -> int:
        """Store sentiment data in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO sentiment_data 
                    (source, content, sentiment_score, sentiment_polarity, confidence, keywords, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sentiment_data.get('source', 'unknown'),
                    sentiment_data.get('content', ''),
                    sentiment_data.get('sentiment_score', 50.0),
                    sentiment_data.get('sentiment_polarity', 'neutral'),
                    sentiment_data.get('confidence', 0.5),
                    json.dumps(sentiment_data.get('keywords', [])),
                    json.dumps(sentiment_data.get('metadata', {}))
                ))
                
                sentiment_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Sentiment data stored with ID: {sentiment_id}")
                return sentiment_id
                
        except Exception as e:
            logger.error(f"Error storing sentiment data: {e}")
            raise
    
    def get_sentiment_data(self, 
                          source: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          limit: int = 1000) -> List[Dict]:
        """Retrieve sentiment data from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = 'SELECT * FROM sentiment_data WHERE 1=1'
                params = []
                
                if source:
                    query += ' AND source = ?'
                    params.append(source)
                
                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)
                
                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)
                
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                results = []
                for row in rows:
                    try:
                        result = {
                            'id': row[0],
                            'source': row[1],
                            'content': row[2],
                            'sentiment_score': row[3],
                            'sentiment_polarity': row[4],
                            'confidence': row[5],
                            'keywords': json.loads(row[6]) if row[6] else [],
                            'metadata': json.loads(row[7]) if row[7] else {},
                            'timestamp': datetime.fromisoformat(row[8])
                        }
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Error parsing sentiment data row: {e}")
                        continue
                
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving sentiment data: {e}")
            return []
    
    def store_system_config(self, config_key: str, config_value: Any, description: str = ""):
        """Store system configuration in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO system_config (config_key, config_value, description)
                    VALUES (?, ?, ?)
                ''', (config_key, json.dumps(config_value), description))
                
                conn.commit()
                logger.info(f"System config stored: {config_key}")
                
        except Exception as e:
            logger.error(f"Error storing system config: {e}")
    
    def get_system_config(self, config_key: str) -> Optional[Any]:
        """Retrieve system configuration from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT config_value FROM system_config WHERE config_key = ?', (config_key,))
                row = cursor.fetchone()
                
                if row:
                    return json.loads(row[0])
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving system config: {e}")
            return None
    
    def store_model_performance(self, model_name: str, metric_name: str, metric_value: float):
        """Store model performance metric."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO model_performance (model_name, metric_name, metric_value)
                    VALUES (?, ?, ?)
                ''', (model_name, metric_name, metric_value))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing model performance: {e}")
    
    def get_model_performance(self, model_name: str, metric_name: Optional[str] = None) -> List[Dict]:
        """Retrieve model performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if metric_name:
                    cursor.execute('''
                        SELECT * FROM model_performance 
                        WHERE model_name = ? AND metric_name = ? 
                        ORDER BY timestamp DESC
                    ''', (model_name, metric_name))
                else:
                    cursor.execute('''
                        SELECT * FROM model_performance 
                        WHERE model_name = ? 
                        ORDER BY timestamp DESC
                    ''', (model_name,))
                
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                results = []
                for row in rows:
                    result = {
                        'id': row[0],
                        'model_name': row[1],
                        'metric_name': row[2],
                        'metric_value': row[3],
                        'timestamp': datetime.fromisoformat(row[4])
                    }
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving model performance: {e}")
            return []
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                tables = ['backtests', 'trades', 'sentiment_data', 'performance_metrics', 'equity_curves']
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # Database size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                stats['database_size_bytes'] = cursor.fetchone()[0]
                
                # Oldest and newest records
                cursor.execute('SELECT MIN(created_at), MAX(created_at) FROM backtests')
                date_range = cursor.fetchone()
                if date_range[0]:
                    stats['oldest_backtest'] = date_range[0]
                    stats['newest_backtest'] = date_range[1]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data from database."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old backtests and related data
                cursor.execute('DELETE FROM backtests WHERE created_at < ?', (cutoff_date,))
                deleted_backtests = cursor.rowcount
                
                # Delete orphaned trades
                cursor.execute('''
                    DELETE FROM trades WHERE backtest_id NOT IN 
                    (SELECT id FROM backtests)
                ''')
                deleted_trades = cursor.rowcount
                
                # Delete old sentiment data
                cursor.execute('DELETE FROM sentiment_data WHERE timestamp < ?', (cutoff_date,))
                deleted_sentiment = cursor.rowcount
                
                # Delete old model performance data
                cursor.execute('DELETE FROM model_performance WHERE timestamp < ?', (cutoff_date,))
                deleted_performance = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleanup completed: {deleted_backtests} backtests, {deleted_trades} trades, "
                           f"{deleted_sentiment} sentiment records, {deleted_performance} performance records")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def export_to_csv(self, table_name: str, output_path: str):
        """Export table data to CSV file."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
                df.to_csv(output_path, index=False)
                logger.info(f"Exported {table_name} to {output_path}")
                
        except Exception as e:
            logger.error(f"Error exporting {table_name} to CSV: {e}")
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database."""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error backing up database: {e}")


def create_database_manager(config: Optional[Dict] = None) -> DatabaseManager:
    """
    Create database manager instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(config)


if __name__ == "__main__":
    # Test the database manager
    manager = create_database_manager()
    
    # Test storing backtest result
    test_backtest = {
        'strategy_name': 'SuperTrend',
        'start_date': datetime.now() - timedelta(days=30),
        'end_date': datetime.now(),
        'initial_capital': 10000.0,
        'final_capital': 10500.0,
        'total_return': 0.05,
        'win_rate': 0.65,
        'total_trades': 20,
        'max_drawdown': 0.02,
        'sharpe_ratio': 1.2,
        'profit_factor': 1.5,
        'execution_time': 15.5,
        'parameters': {'period': 10, 'multiplier': 3.0},
        'results_summary': {'total_pnl': 500.0, 'avg_trade': 25.0},
        'trades': [
            {
                'timestamp': datetime.now() - timedelta(hours=1),
                'symbol': 'XAUUSD',
                'direction': 'BUY',
                'entry_price': 2000.0,
                'exit_price': 2010.0,
                'lot_size': 0.1,
                'pnl': 100.0,
                'stop_loss': 1990.0,
                'take_profit': 2020.0,
                'ai_confidence': 0.8,
                'sentiment_score': 0.7
            }
        ]
    }
    
    # Store backtest
    backtest_id = manager.store_backtest_result(test_backtest)
    print(f"Stored backtest with ID: {backtest_id}")
    
    # Retrieve backtest
    results = manager.get_backtest_results(limit=5)
    print(f"Retrieved {len(results)} backtest results")
    
    # Get database stats
    stats = manager.get_database_stats()
    print(f"Database stats: {stats}")
    
    print("Database manager test completed successfully!")

