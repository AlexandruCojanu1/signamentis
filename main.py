#!/usr/bin/env python3
"""
SignaMentis Main Entry Point

This is the main entry point for the SignaMentis trading system.
It integrates all components:
- Data loading and feature engineering
- AI model training and ensemble
- Risk management
- Trading strategy execution
- Live monitoring and API

Author: SignaMentis Team
Version: 1.0.0
"""

import os
import sys
import logging
import argparse
import yaml
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

# Import SignaMentis components
from scripts.data_loader import DataLoader, DataConfig
from scripts.feature_engineering import FeatureEngineer
from scripts.model_bilstm import BiLSTMModel, BiLSTMTrainer, create_bilstm_model
from scripts.model_gru import GRUModel, GRUTrainer, create_gru_model
from scripts.model_transformer import TransformerModel, TransformerTrainer, create_transformer_model
from scripts.model_lnn import LNNModel, LNNTrainer, create_lnn_model
from scripts.model_ltn import LTNModel, LTNTrainer, create_ltn_model
from scripts.ensemble import EnsembleManager, create_ensemble
from scripts.risk_manager import RiskManager, create_risk_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
            logging.FileHandler('signa_mentis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SignaMentis:
    """
    Main SignaMentis trading system class.
    
    Integrates all components and provides high-level control.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the SignaMentis system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/settings.yaml"
        self.config = self._load_config()
        
        # Initialize components
        self.data_loader = None
        self.feature_engineer = None
        self.models = {}
        self.ensemble = None
        self.risk_manager = None
        
        # System state
        self.is_running = False
        self.trading_enabled = False
        
        logger.info("SignaMentis system initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing system components...")
        
        try:
            # Initialize data loader
            data_config = DataConfig(
                symbol=self.config.get('trading_strategy', {}).get('symbol', 'XAUUSD'),
                timeframe=self.config.get('trading_strategy', {}).get('timeframe', 'M5'),
                data_source=self.config.get('data', {}).get('sources', {}).get('primary', 'csv'),
                raw_data_path=self.config.get('data', {}).get('storage', {}).get('raw_data_path', 'data/raw'),
                processed_data_path=self.config.get('data', {}).get('storage', {}).get('processed_data_path', 'data/processed')
            )
            self.data_loader = DataLoader(data_config)
            
            # Initialize feature engineer
            self.feature_engineer = FeatureEngineer()
            
            # Initialize AI models
            self._initialize_models()
            
            # Initialize ensemble
            self.ensemble = create_ensemble(self.config.get('models', {}))
            
            # Initialize risk manager
            self.risk_manager = create_risk_manager(self.config.get('risk', {}))
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _initialize_models(self):
        """Initialize AI models."""
        logger.info("Initializing AI models...")
        
        # Load model configuration
        model_config_path = "config/model_config.yaml"
        try:
            with open(model_config_path, 'r') as file:
                model_config = yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading model config: {e}")
            return
        
        # Initialize BiLSTM model
        try:
            bilstm_config = model_config.get('bilstm', {})
            input_size = bilstm_config.get('input_size', 50)
            
            bilstm_model = create_bilstm_model(input_size, bilstm_config)
            self.models['bilstm'] = bilstm_model
            
            # Register with ensemble
            if self.ensemble:
                self.ensemble.register_model('bilstm', bilstm_model, initial_weight=0.3)
            
            logger.info("BiLSTM model initialized")
            
        except Exception as e:
            logger.error(f"Error initializing BiLSTM model: {e}")
        
        # Initialize GRU model
        try:
            gru_config = model_config.get('gru', {})
            input_size = gru_config.get('input_size', 50)
            
            gru_model = create_gru_model(input_size, gru_config)
            self.models['gru'] = gru_model
            
            # Register with ensemble
            if self.ensemble:
                self.ensemble.register_model('gru', gru_model, initial_weight=0.2)
            
            logger.info("GRU model initialized")
            
        except Exception as e:
            logger.error(f"Error initializing GRU model: {e}")
        
        # Initialize Transformer model
        try:
            transformer_config = model_config.get('transformer', {})
            input_size = transformer_config.get('input_size', 50)
            
            transformer_model = create_transformer_model(input_size, transformer_config)
            self.models['transformer'] = transformer_model
            
            # Register with ensemble
            if self.ensemble:
                self.ensemble.register_model('transformer', transformer_model, initial_weight=0.2)
            
            logger.info("Transformer model initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Transformer model: {e}")
        
        # Initialize LNN model
        try:
            lnn_config = model_config.get('lnn', {})
            input_size = lnn_config.get('input_size', 50)
            
            lnn_model = create_lnn_model(input_size, lnn_config)
            self.models['lnn'] = lnn_model
            
            # Register with ensemble
            if self.ensemble:
                self.ensemble.register_model('lnn', lnn_model, initial_weight=0.15)
            
            logger.info("LNN model initialized")
            
        except Exception as e:
            logger.error(f"Error initializing LNN model: {e}")
        
        # Initialize LTN model
        try:
            ltn_config = model_config.get('ltn', {})
            input_size = ltn_config.get('input_size', 50)
            
            ltn_model = create_ltn_model(input_size, ltn_config)
            self.models['ltn'] = ltn_model
            
            # Register with ensemble
            if self.ensemble:
                self.ensemble.register_model('ltn', ltn_model, initial_weight=0.15)
            
            logger.info("LTN model initialized")
            
        except Exception as e:
            logger.error(f"Error initializing LTN model: {e}")
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def load_and_prepare_data(self, 
                             start_date: str = "2024-01-01",
                             end_date: str = "2025-01-01") -> bool:
        """
        Load and prepare data for training and trading.
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Loading and preparing data...")
            
            # Load raw data
            df = self.data_loader.get_data(
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                logger.error("No data loaded")
                return False
            
            logger.info(f"Loaded {len(df)} bars of data")
            
            # Create features
            df_with_features = self.feature_engineer.create_all_features(df)
            
            # Save processed data
            processed_file = self.data_loader.save_data(
                df_with_features, "XAUUSD", "M5", "parquet"
            )
            
            logger.info(f"Data prepared and saved to {processed_file}")
            
            # Store data for later use
            self.market_data = df_with_features
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return False
    
    def train_models(self, 
                    sequence_length: int = 100,
                    feature_columns: Optional[List[str]] = None) -> bool:
        """
        Train all AI models.
        
        Args:
            sequence_length: Length of input sequences
            feature_columns: List of feature columns to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Training AI models...")
            
            if not hasattr(self, 'market_data'):
                logger.error("No market data available. Load data first.")
                return False
            
            # Select feature columns
            if feature_columns is None:
                # Use technical indicators and basic features
                feature_columns = [
                    'rsi_14', 'macd_line', 'bb_percent_b_20', 'atr_14',
                    'supertrend_direction_10', 'stoch_k_14', 'williams_r_14',
                    'cci_20', 'adx_14', 'vol_regime_encoded'
                ]
            
            # Filter data to include only selected features
            available_features = [col for col in feature_columns if col in self.market_data.columns]
            if not available_features:
                logger.error("No valid feature columns found")
                return False
            
            # Prepare data for training
            training_data = self.market_data[['open', 'high', 'low', 'close', 'volume'] + available_features].copy()
            
            # Train BiLSTM model
            if 'bilstm' in self.models:
                try:
                    logger.info("Training BiLSTM model...")
                    
                    # Create trainer
                    trainer = BiLSTMTrainer(self.models['bilstm'])
                    
                    # Prepare data
                    data_tensors = trainer.prepare_data(
                        training_data,
                        sequence_length=sequence_length,
                        feature_columns=available_features
                    )
                    
                    # Train model
                    history = trainer.train(*data_tensors[:8])  # First 8 tensors for training
                    
                    # Save trained model
                    model_save_path = "models/saved/bilstm_model.pth"
                    trainer.save_model(model_save_path)
                    
                    logger.info(f"BiLSTM model trained and saved to {model_save_path}")
                    
                except Exception as e:
                    logger.error(f"Error training BiLSTM model: {e}")
            
            # TODO: Train other models
            
            logger.info("Model training completed")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def get_prediction(self, 
                      market_data: pd.DataFrame,
                      sequence_length: int = 100) -> Optional[Dict]:
        """
        Get prediction from the ensemble.
        
        Args:
            market_data: Current market data
            sequence_length: Length of input sequences
            
        Returns:
            Dictionary with ensemble prediction or None
        """
        try:
            if not self.ensemble:
                logger.error("Ensemble not initialized")
                return None
            
            # Prepare features
            features_data = self.feature_engineer.create_all_features(market_data)
            
            # Select features for prediction
            feature_columns = [
                'rsi_14', 'macd_line', 'bb_percent_b_20', 'atr_14',
                'supertrend_direction_10', 'stoch_k_14', 'williams_r_14',
                'cci_20', 'adx_14', 'vol_regime_encoded'
            ]
            
            available_features = [col for col in feature_columns if col in features_data.columns]
            if not available_features:
                logger.error("No valid features for prediction")
                return None
            
            # Create input sequence
            if len(features_data) < sequence_length:
                logger.warning(f"Insufficient data for prediction. Need {sequence_length}, got {len(features_data)}")
                return None
            
            # Get latest sequence
            latest_sequence = features_data[available_features].tail(sequence_length).values
            
            # Get ensemble prediction
            ensemble_pred = self.ensemble.get_ensemble_prediction(latest_sequence)
            
            return {
                'direction': ensemble_pred.predicted_direction,
                'direction_probabilities': ensemble_pred.direction_probabilities,
                'price_target': ensemble_pred.price_target,
                'confidence': ensemble_pred.confidence,
                'ensemble_confidence': ensemble_pred.ensemble_confidence,
                'model_agreement': ensemble_pred.model_agreement,
                'timestamp': ensemble_pred.timestamp
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return None
    
    def execute_trade_signal(self, 
                           prediction: Dict,
                           current_price: float,
                           market_data: pd.DataFrame) -> bool:
        """
        Execute trade based on AI prediction.
        
        Args:
            prediction: AI prediction dictionary
            current_price: Current market price
            market_data: Current market data
            
        Returns:
            bool: True if trade executed, False otherwise
        """
        try:
            # Check if trading is enabled
            if not self.trading_enabled:
                logger.info("Trading is disabled")
                return False
            
            # Check market conditions
            can_trade, reason = self.risk_manager.check_market_conditions(
                market_data, datetime.now()
            )
            
            if not can_trade:
                logger.info(f"Cannot trade: {reason}")
                return False
            
            # Check confidence threshold
            min_confidence = self.config.get('risk', {}).get('min_confidence_threshold', 0.70)
            if prediction['ensemble_confidence'] < min_confidence:
                logger.info(f"Confidence too low: {prediction['ensemble_confidence']:.3f}")
                return False
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                current_price,
                prediction['ensemble_confidence'],
                market_data
            )
            
            # Execute trade (simulated for now)
            trade_data = {
                'symbol': 'XAUUSD',
                'direction': 'BUY' if prediction['direction'] == 1 else 'SELL',
                'entry_price': current_price,
                'lot_size': position_size.lot_size,
                'stop_loss': current_price - (position_size.stop_loss_pips * 0.1),
                'take_profit': current_price + (position_size.take_profit_pips * 0.1),
                'confidence': prediction['ensemble_confidence'],
                'timestamp': datetime.now()
            }
            
            logger.info(f"Trade executed: {trade_data}")
            
            # Record trade in risk manager
            self.risk_manager.record_trade(trade_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def run_backtest(self, 
                    start_date: str = "2024-01-01",
                    end_date: str = "2024-12-31") -> Dict:
        """
        Run backtest of the trading system.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        try:
            logger.info("Running backtest...")
            
            # Load data for backtest period
            if not self.load_and_prepare_data(start_date, end_date):
                return {}
            
            # Train models
            if not self.train_models():
                return {}
            
            # Simulate trading
            results = self._simulate_trading()
            
            logger.info("Backtest completed")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    def _simulate_trading(self) -> Dict:
        """Simulate trading on historical data."""
        try:
            # Initialize backtest variables
            initial_balance = 10000.0
            current_balance = initial_balance
            trades = []
            equity_curve = []
            
            # Get predictions for each data point
            sequence_length = 100
            for i in range(sequence_length, len(self.market_data)):
                # Get data slice
                data_slice = self.market_data.iloc[i-sequence_length:i+1]
                
                # Get prediction
                prediction = self.get_prediction(data_slice, sequence_length)
                
                if prediction and prediction['ensemble_confidence'] > 0.7:
                    # Simulate trade execution
                    entry_price = data_slice.iloc[-1]['close']
                    
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        entry_price,
                        prediction['ensemble_confidence'],
                        data_slice
                    )
                    
                    # Simulate trade outcome (simplified)
                    # In a real backtest, you'd track stop loss and take profit
                    if prediction['direction'] == 1:  # Long
                        exit_price = entry_price + (position_size.take_profit_pips * 0.1)
                        pnl = (exit_price - entry_price) * position_size.lot_size * 100
                    else:  # Short
                        exit_price = entry_price - (position_size.take_profit_pips * 0.1)
                        pnl = (entry_price - exit_price) * position_size.lot_size * 100
                    
                    # Record trade
                    trade = {
                        'timestamp': data_slice.index[-1],
                        'direction': prediction['direction'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'lot_size': position_size.lot_size,
                        'pnl': pnl,
                        'confidence': prediction['ensemble_confidence']
                    }
                    trades.append(trade)
                    
                    # Update balance
                    current_balance += pnl
                    equity_curve.append(current_balance)
                
                # Record equity curve
                if not equity_curve:
                    equity_curve.append(initial_balance)
            
            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(t['pnl'] for t in trades)
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            
            return {
                'initial_balance': initial_balance,
                'final_balance': current_balance,
                'total_return': (current_balance - initial_balance) / initial_balance,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'max_drawdown': max_drawdown,
                'trades': trades,
                'equity_curve': equity_curve
            }
            
        except Exception as e:
            logger.error(f"Error simulating trading: {e}")
            return {}
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    async def start_live_trading(self):
        """Start live trading mode."""
        logger.info("Starting live trading mode...")
        
        # Enable trading
        self.trading_enabled = True
        
        # Start monitoring loop
        self.is_running = True
        
        try:
            while self.is_running:
                # Get latest market data
                latest_data = self.data_loader.get_latest_data(bars=200)
                
                if not latest_data.empty:
                    # Get prediction
                    prediction = self.get_prediction(latest_data)
                    
                    if prediction:
                        current_price = latest_data.iloc[-1]['close']
                        
                        # Execute trade if conditions are met
                        self.execute_trade_signal(prediction, current_price, latest_data)
                
                # Wait for next update
                await asyncio.sleep(300)  # 5 minutes
                
        except KeyboardInterrupt:
            logger.info("Live trading stopped by user")
        except Exception as e:
            logger.error(f"Error in live trading: {e}")
        finally:
            self.trading_enabled = False
            self.is_running = False
    
    def stop_live_trading(self):
        """Stop live trading mode."""
        logger.info("Stopping live trading...")
        self.is_running = False
        self.trading_enabled = False
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            'is_running': self.is_running,
            'trading_enabled': self.trading_enabled,
            'components_initialized': all([
                self.data_loader is not None,
                self.feature_engineer is not None,
                len(self.models) > 0,
                self.ensemble is not None,
                self.risk_manager is not None
            ]),
            'models_loaded': list(self.models.keys()),
            'risk_summary': self.risk_manager.get_risk_summary() if self.risk_manager else None,
            'ensemble_status': self.ensemble.get_ensemble_status() if self.ensemble else None
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='SignaMentis Trading System')
    parser.add_argument('--mode', choices=['backtest', 'live', 'train'], 
                       default='backtest', help='Operation mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--start-date', type=str, default='2024-01-01', 
                       help='Start date for backtest/training')
    parser.add_argument('--end-date', type=str, default='2024-12-31', 
                       help='End date for backtest/training')
    parser.add_argument('--sequence-length', type=int, default=100, 
                       help='Sequence length for models')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        logger.info("Starting SignaMentis Trading System...")
        
        signa = SignaMentis(args.config)
        signa.initialize_components()
        
        if args.mode == 'backtest':
            # Run backtest
            results = signa.run_backtest(args.start_date, args.end_date)
            
            if results:
                print("\n=== Backtest Results ===")
                print(f"Initial Balance: ${results['initial_balance']:,.2f}")
                print(f"Final Balance: ${results['final_balance']:,.2f}")
                print(f"Total Return: {results['total_return']:.2%}")
                print(f"Total Trades: {results['total_trades']}")
                print(f"Win Rate: {results['win_rate']:.2%}")
                print(f"Total P&L: ${results['total_pnl']:,.2f}")
                print(f"Max Drawdown: {results['max_drawdown']:.2%}")
            else:
                print("Backtest failed")
        
        elif args.mode == 'train':
            # Load data and train models
            if signa.load_and_prepare_data(args.start_date, args.end_date):
                signa.train_models(args.sequence_length)
                print("Model training completed")
            else:
                print("Model training failed")
        
        elif args.mode == 'live':
            # Start live trading
            print("Starting live trading mode...")
            print("Press Ctrl+C to stop")
            
            # Run live trading
            asyncio.run(signa.start_live_trading())
        
        # Get system status
        status = signa.get_system_status()
        print(f"\nSystem Status: {status}")
        
        # Save results if backtest mode
        if args.mode == 'backtest' and results:
            results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(results_file, 'w') as f:
                    import json
                    json.dump(results, f, indent=2, default=str)
                print(f"💾 Results saved to {results_file}")
            except Exception as e:
                print(f"⚠️  Could not save results: {e}")
        
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
