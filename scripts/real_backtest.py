#!/usr/bin/env python3
"""
SignaMentis Real Data Backtesting Script
Uses actual XAUUSD data for comprehensive backtesting with 15-minute timeframe
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.strategy import SuperTrendStrategy
from scripts.ensemble import EnsembleManager
from scripts.risk_manager import RiskManager
from scripts.backtester import Backtester

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_xauusd_data(file_path: str, timeframe: str = "15") -> pd.DataFrame:
    """
    Load and preprocess XAUUSD data.
    
    Args:
        file_path: Path to CSV file
        timeframe: Timeframe in minutes ("5" or "15")
        
    Returns:
        Preprocessed DataFrame with OHLCV data
    """
    logger.info(f"Loading XAUUSD data from {file_path}")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Display basic info
        logger.info(f"Raw data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"First few rows:\n{df.head()}")
        
        # Standardize column names
        column_mapping = {
            'Local time': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Convert to UTC and remove timezone info for consistency
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            df = df.set_index('timestamp')
        
        # Data already has OHLC structure, just resample if needed
        if 'close' in df.columns:
            # If we need to resample to different timeframe
            if timeframe != "15":  # Default is 15 minutes
                resampled = df['close'].resample(f'{timeframe}T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum' if 'volume' in df.columns else 'count'
                })
            else:
                # Use data as is for 15-minute timeframe
                resampled = df.copy()
            
            # Forward fill missing values
            resampled = resampled.fillna(method='ffill')
            
            # Remove rows with missing data
            resampled = resampled.dropna()
            
            logger.info(f"Resampled data shape: {resampled.shape}")
            logger.info(f"Date range: {resampled.index.min()} to {resampled.index.max()}")
            
            return resampled
        
        else:
            logger.error("Could not find 'close' column in data")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def create_real_strategy() -> SuperTrendStrategy:
    """Create a real SuperTrend strategy instance."""
    config = {
        'supertrend_period': 10,
        'supertrend_multiplier': 3.0,
        'min_ai_confidence': 0.70,
        'min_breakout_strength': 0.5,
        'max_spread_pips': 2.0,
        'max_risk_per_trade': 0.02,  # 2%
        'min_risk_reward_ratio': 2.0,
        'max_positions': 3,
        'position_sizing_method': 'fixed_fractional',
        'volatility_filter': True,
        'session_filter': True,
        'news_filter': True
    }
    
    return SuperTrendStrategy(config)


def create_real_ai_ensemble() -> EnsembleManager:
    """Create a real AI ensemble instance."""
    config = {
        'models': ['bilstm', 'gru', 'transformer'],
        'ensemble_method': 'weighted_average',
        'confidence_threshold': 0.70,
        'prediction_horizon': 15,  # 15 minutes
        'feature_window': 100,
        'retrain_frequency': 'daily'
    }
    
    return EnsembleManager(config)


def create_real_risk_manager() -> RiskManager:
    """Create a real risk manager instance."""
    config = {
        'max_risk_per_trade': 0.02,  # 2%
        'max_portfolio_risk': 0.06,  # 6%
        'max_daily_loss': 0.05,  # 5%
        'position_sizing_method': 'fixed_fractional',
        'stop_loss_method': 'atr_based',
        'take_profit_method': 'risk_reward_ratio',
        'min_risk_reward_ratio': 2.0,
        'max_positions': 3,
        'correlation_threshold': 0.7
    }
    
    return RiskManager(config)


def run_real_backtest(data_file: str, timeframe: str = "15", 
                     start_date: str = None, end_date: str = None) -> dict:
    """
    Run comprehensive backtest with real data.
    
    Args:
        data_file: Path to CSV data file
        timeframe: Timeframe in minutes
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        
    Returns:
        Backtest results dictionary
    """
    logger.info("🚀 Starting Real Data Backtest")
    logger.info("=" * 60)
    
    # Load data
    df = load_xauusd_data(data_file, timeframe)
    if df.empty:
        logger.error("Failed to load data")
        return {}
    
    # Filter by date range if specified
    if start_date:
        start_dt = pd.to_datetime(start_date)
        # Filter by date only, not time
        start_dt = start_dt.normalize()
        df = df[df.index.date >= start_dt.date()]
        logger.info(f"Filtered data from {start_date}")
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        # Filter by date only, not time
        end_dt = end_dt.normalize()
        df = df[df.index.date <= end_dt.date()]
        logger.info(f"Filtered data to {end_date}")
    
    logger.info(f"Final data shape: {df.shape}")
    if df.shape[0] > 0:
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    else:
        logger.warning("No data after filtering! Check date range.")
        return {}
    
    # Create strategy components
    strategy = create_real_strategy()
    ai_ensemble = create_real_ai_ensemble()
    risk_manager = create_real_risk_manager()
    
    # Create backtester
    backtester_config = {
        'initial_capital': 10000,
        'commission': 0.0001,  # 1 pip for XAUUSD
        'slippage': 0.0001,    # 1 pip slippage
        'k_folds': 5,
        'embargo_bars': 20,
        'min_trades_per_fold': 5
    }
    
    backtester = Backtester(backtester_config)
    
    # Run backtest
    logger.info("Running comprehensive backtest...")
    result = backtester.run_backtest(
        df, strategy, ai_ensemble, risk_manager
        # start_date and end_date already filtered above
    )
    
    if result:
        # Generate report
        report = backtester.generate_report(result)
        logger.info("📊 Backtest Report Generated")
        
        # Save plots
        try:
            plot_file = f"real_backtest_results_{timeframe}min.png"
            backtester.plot_results(result, output_path=plot_file)
            logger.info(f"📈 Plots saved to {plot_file}")
        except Exception as e:
            logger.warning(f"Could not save plots: {e}")
        
        return result
    else:
        logger.error("Backtest failed to produce results")
        return {}


def analyze_predictions_accuracy(results: dict) -> dict:
    """
    Analyze prediction accuracy for 15-minute timeframe.
    
    Args:
        results: Backtest results dictionary
        
    Returns:
        Accuracy analysis dictionary
    """
    if not results or 'trades' not in results:
        return {}
    
    trades = results['trades']
    if not trades:
        logger.warning("No trades to analyze")
        return {}
    
    logger.info("🔍 Analyzing Prediction Accuracy")
    logger.info("=" * 40)
    
    # Calculate accuracy metrics
    total_predictions = len(trades)
    correct_predictions = 0
    prediction_confidence = []
    
    for trade in trades:
        if 'ai_prediction' in trade and 'actual_direction' in trade:
            ai_pred = trade['ai_prediction']
            actual = trade['actual_direction']
            
            if ai_pred == actual:
                correct_predictions += 1
            
            if 'confidence' in trade:
                prediction_confidence.append(trade['confidence'])
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_confidence = np.mean(prediction_confidence) if prediction_confidence else 0
    
    # Time-based accuracy (15-minute predictions)
    time_accuracy = {
        '15min': accuracy,
        '30min': 0,  # Could be calculated if we have longer-term data
        '1hour': 0
    }
    
    analysis = {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'overall_accuracy': accuracy,
        'average_confidence': avg_confidence,
        'time_based_accuracy': time_accuracy,
        'win_rate': results.get('win_rate', 0),
        'profit_factor': results.get('profit_factor', 0),
        'sharpe_ratio': results.get('sharpe_ratio', 0)
    }
    
    # Log results
    logger.info(f"Total Predictions: {total_predictions}")
    logger.info(f"Correct Predictions: {correct_predictions}")
    logger.info(f"Overall Accuracy: {accuracy:.2%}")
    logger.info(f"Average Confidence: {avg_confidence:.2%}")
    logger.info(f"15-Minute Accuracy: {time_accuracy['15min']:.2%}")
    logger.info(f"Win Rate: {analysis['win_rate']:.2%}")
    logger.info(f"Profit Factor: {analysis['profit_factor']:.2f}")
    logger.info(f"Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
    
    return analysis


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SignaMentis Real Data Backtester")
    parser.add_argument("--data-file", default="../XAUUSD_Tickbar_15_BID_11.08.2023-11.08.2023.csv",
                       help="Path to XAUUSD CSV data file")
    parser.add_argument("--timeframe", default="15", choices=["5", "15"],
                       help="Timeframe in minutes")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze existing results")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Load existing results if available
        results_file = f"backtest_results_{args.timeframe}min.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                import json
                results = json.load(f)
            analyze_predictions_accuracy(results)
        else:
            logger.error(f"No existing results found: {results_file}")
        return
    
    # Run backtest
    logger.info("🎯 SignaMentis Real Data Backtesting")
    logger.info("=" * 60)
    
    results = run_real_backtest(
        args.data_file,
        args.timeframe,
        args.start_date,
        args.end_date
    )
    
    if results:
        # Analyze prediction accuracy
        accuracy_analysis = analyze_predictions_accuracy(results)
        
        # Save results
        results_file = f"backtest_results_{args.timeframe}min.json"
        try:
            with open(results_file, 'w') as f:
                import json
                json.dump(results, f, indent=2, default=str)
            logger.info(f"💾 Results saved to {results_file}")
        except Exception as e:
            logger.warning(f"Could not save results: {e}")
        
        logger.info("✅ Real Data Backtest Completed Successfully!")
        
        # Summary
        logger.info("\n📋 SUMMARY")
        logger.info("=" * 30)
        logger.info(f"Data: {args.data_file}")
        logger.info(f"Timeframe: {args.timeframe} minutes")
        logger.info(f"Period: {args.start_date or 'Start'} to {args.end_date or 'End'}")
        logger.info(f"Total Trades: {len(results.get('trades', []))}")
        logger.info(f"Accuracy: {accuracy_analysis.get('overall_accuracy', 0):.2%}")
        logger.info(f"Win Rate: {accuracy_analysis.get('win_rate', 0):.2%}")
        
    else:
        logger.error("❌ Backtest failed")


if __name__ == "__main__":
    main()
