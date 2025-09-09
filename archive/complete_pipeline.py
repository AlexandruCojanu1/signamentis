"""
SignaMentis Complete Pipeline

This script orchestrates the entire pipeline:
1. Data normalization (UTC, M15 derivation, validation)
2. Feature engineering (ATR, SuperTrend, SMA/EMA, RSI)
3. Data splitting (train/val/test for last 3 years)
4. Random Forest training and calibration
5. Realistic backtesting with costs

Author: SignaMentis Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution."""
    logger.info("🚀 Starting SignaMentis Complete Pipeline")
    
    try:
        # Configuration
        config = {
            'data': {
                'm5_path': 'data/external/XAUUSD_Tickbar_5_BID_11.08.2023-11.08.2023.csv',
                'm15_path': 'data/external/XAUUSD_Tickbar_15_BID_11.08.2023-11.08.2023.csv'
            },
            'normalization': {
                'max_spread_pips': 5.0
            },
            'features': {
                'atr_period': 14,
                'supertrend_period': 10,
                'supertrend_multiplier': 3.0,
                'sma_periods': [20, 50, 200],
                'ema_periods': [12, 26],
                'rsi_period': 14,
                'align_m15_on_m5': True
            },
            'splitting': {
                'train_years': 2,
                'val_months': 6,
                'test_months': 6,
                'session_aware': True
            },
            'training': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'calibrate_probabilities': True,
                'feature_selection': True
            },
            'backtesting': {
                'base_spread': 0.5,
                'commission': 7.0,
                'slippage': 0.2,
                'rf_confidence_threshold': 0.6,
                'position_size': 0.1
            }
        }
        
        logger.info("📋 Configuration loaded")
        
        # Step 1: Data Normalization
        logger.info("🔄 Step 1: Data Normalization")
        from scripts.data_normalizer import DataNormalizer
        
        normalizer = DataNormalizer(config['normalization'])
        m5_clean, m15_clean, norm_report = normalizer.normalize_data(
            m5_path=config['data']['m5_path'],
            m15_path=config['data']['m15_path']
        )
        
        # Save normalized data
        clean_paths = normalizer.save_normalized_data(m5_clean, m15_clean)
        logger.info(f"✅ Data normalized and saved: {clean_paths}")
        
        # Step 2: Feature Engineering
        logger.info("🔧 Step 2: Feature Engineering")
        from scripts.essential_features import EssentialFeatureEngineer
        
        feature_engineer = EssentialFeatureEngineer(config['features'])
        features_df = feature_engineer.create_essential_features(m5_clean, m15_clean)
        
        # Save features
        features_path = feature_engineer.save_features(features_df)
        logger.info(f"✅ Features created and saved: {features_path}")
        
        # Step 3: Data Splitting
        logger.info("✂️ Step 3: Data Splitting")
        from scripts.data_splitter import DataSplitter
        
        splitter = DataSplitter(config['splitting'])
        split_paths = splitter.split_data(features_df)
        logger.info(f"✅ Data split and saved: {split_paths}")
        
        # Step 4: Random Forest Training
        logger.info("🎯 Step 4: Random Forest Training")
        from scripts.rf_baseline_trainer import RandomForestBaselineTrainer
        
        trainer = RandomForestBaselineTrainer(config['training'])
        training_results = trainer.train_baseline(
            splits_dir=split_paths['output_dir'],
            output_dir="results"
        )
        logger.info(f"✅ Model trained and saved: {training_results['model_path']}")
        
        # Step 5: Realistic Backtesting
        logger.info("📊 Step 5: Realistic Backtesting")
        from scripts.realistic_backtester import RealisticBacktester
        
        backtester = RealisticBacktester(config['backtesting'])
        backtest_results = backtester.run_backtest(
            test_data_path=split_paths['test_path'],
            model_path=training_results['model_path'],
            output_dir="results"
        )
        logger.info(f"✅ Backtest completed: {backtest_results['report_path']}")
        
        # Step 6: Generate Final Report
        logger.info("📝 Step 6: Generating Final Report")
        generate_final_report(
            norm_report, features_df, training_results, backtest_results
        )
        
        logger.info("🎉 Pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 SIGNAMENTIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Data Normalization: {norm_report['total_m5_records']} M5, {norm_report['total_m15_records']} M15 records")
        print(f"🔧 Features Created: {len(features_df.columns)} total columns")
        print(f"🎯 Model Performance: {training_results['training_summary']['test_accuracy']:.2%} test accuracy")
        print(f"💰 Backtest Results: ${backtest_results['backtest_summary']['total_pnl']:.2f} P&L")
        print(f"📈 Win Rate: {backtest_results['backtest_summary']['win_rate']:.2%}")
        print(f"📉 Max Drawdown: {backtest_results['backtest_summary']['max_drawdown']:.2%}")
        print(f"⚡ Sharpe Ratio: {backtest_results['backtest_summary']['sharpe_ratio']:.2f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


def generate_final_report(norm_report, features_df, training_results, backtest_results):
    """Generate comprehensive final report."""
    
    report = {
        'pipeline_summary': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'status': 'completed_successfully'
        },
        'data_quality': {
            'm5_records': norm_report['total_m5_records'],
            'm15_records': norm_report['total_m15_records'],
            'validation_status': norm_report['normalization_status']
        },
        'feature_engineering': {
            'total_features': len(features_df.columns),
            'feature_columns': list(features_df.columns)
        },
        'model_training': {
            'model_path': training_results['model_path'],
            'test_accuracy': training_results['training_summary']['test_accuracy'],
            'selected_features': len(training_results['selected_features'])
        },
        'backtesting': {
            'total_trades': backtest_results['backtest_summary']['total_trades'],
            'total_pnl': backtest_results['backtest_summary']['total_pnl'],
            'win_rate': backtest_results['backtest_summary']['win_rate'],
            'profit_factor': backtest_results['backtest_summary']['profit_factor'],
            'max_drawdown': backtest_results['backtest_summary']['max_drawdown'],
            'sharpe_ratio': backtest_results['backtest_summary']['sharpe_ratio']
        }
    }
    
    # Save final report
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / f"final_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False)
    
    logger.info(f"Final report saved: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
