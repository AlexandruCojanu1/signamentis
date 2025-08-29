#!/usr/bin/env python3
"""
Complete Pipeline for SignaMentis Random Forest Baseline

This script orchestrates the entire data processing, feature engineering,
model training, and backtesting pipeline as requested by the user.

Pipeline Steps:
1. Normalize data (UTC, standard schema, derive M15 from M5)
2. Clean data (duplicates, gaps, OHLC validation, spread filtering)  
3. Validate quality (generate 1-page report)
4. Build essential features (ATR, SuperTrend, SMA/EMA, RSI, time features)
5. Define splits (Train/Val/Test from last 3 years)
6. Train baseline (Random Forest + probability calibration)
7. Backtest with real costs (spread, commission, slippage)

Deliverables:
- data/clean/ (M5 & M15 clean data)
- data/processed/ (features)
- data/splits/ (train/val/test manifest)
- Saved RF model + importance report
- Backtest report (equity curve, metrics)
- Data quality report (1 page)
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import pipeline components
from scripts.data_normalizer import DataNormalizer
from scripts.essential_features import EssentialFeaturesEngine
from scripts.data_splitter import DataSplitter
from scripts.rf_baseline_trainer import RandomForestBaselineTrainer
from scripts.realistic_backtester import RealisticBacktester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompletePipeline:
    """
    Complete pipeline orchestrator for Random Forest baseline.
    
    Implements the user's detailed requirements for:
    - Clean M5 & M15 XAU/USD dataset
    - Solid Random Forest baseline with real costs backtest
    - Comprehensive reports and validation
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.data_normalizer = DataNormalizer()
        self.features_engine = EssentialFeaturesEngine()
        self.data_splitter = DataSplitter()
        self.rf_trainer = RandomForestBaselineTrainer()
        self.backtester = RealisticBacktester()
        
        # Setup paths
        self.setup_directories()
        
        logger.info("CompletePipeline initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}
    
    def setup_directories(self):
        """Setup required directories."""
        directories = [
            "data/raw",
            "data/clean", 
            "data/processed",
            "data/splits",
            "models",
            "results",
            "reports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Directories setup completed")
    
    def step1_normalize_data(self, input_file: str) -> str:
        """
        Step 1: Normalize data to UTC with standard schema.
        
        Args:
            input_file: Path to input CSV file
            
        Returns:
            Path to normalized M5 data file
        """
        logger.info("=== STEP 1: NORMALIZING DATA ===")
        
        # Load and normalize M5 data
        m5_clean_path = self.data_normalizer.normalize_m5_data(
            input_file=input_file,
            output_dir="data/clean"
        )
        
        # Derive M15 from M5
        m15_clean_path = self.data_normalizer.derive_m15_from_m5(
            m5_file=m5_clean_path,
            output_dir="data/clean"
        )
        
        logger.info(f"Data normalization completed:")
        logger.info(f"  M5 clean: {m5_clean_path}")
        logger.info(f"  M15 clean: {m15_clean_path}")
        
        return m5_clean_path, m15_clean_path
    
    def step2_clean_data(self, m5_file: str, m15_file: str) -> tuple:
        """
        Step 2: Clean data (duplicates, gaps, OHLC validation).
        
        Args:
            m5_file: Path to M5 data file
            m15_file: Path to M15 data file
            
        Returns:
            Tuple of (cleaned_m5_path, cleaned_m15_path)
        """
        logger.info("=== STEP 2: CLEANING DATA ===")
        
        # Clean M5 data
        m5_cleaned_path = self.data_normalizer.clean_data(
            input_file=m5_file,
            output_dir="data/clean"
        )
        
        # Clean M15 data  
        m15_cleaned_path = self.data_normalizer.clean_data(
            input_file=m15_file,
            output_dir="data/clean"
        )
        
        logger.info(f"Data cleaning completed:")
        logger.info(f"  M5 cleaned: {m5_cleaned_path}")
        logger.info(f"  M15 cleaned: {m15_cleaned_path}")
        
        return m5_cleaned_path, m15_cleaned_path
    
    def step3_validate_quality(self, m5_file: str, m15_file: str) -> str:
        """
        Step 3: Validate data quality and generate report.
        
        Args:
            m5_file: Path to cleaned M5 data
            m15_file: Path to cleaned M15 data
            
        Returns:
            Path to quality report
        """
        logger.info("=== STEP 3: VALIDATING DATA QUALITY ===")
        
        # Generate quality report
        report_path = self.data_normalizer.generate_quality_report(
            m5_file=m5_file,
            m15_file=m15_file,
            output_dir="reports"
        )
        
        logger.info(f"Data quality validation completed: {report_path}")
        return report_path
    
    def step4_build_features(self, m5_file: str, timeframe: str = "M15") -> str:
        """
        Step 4: Build essential features.
        
        Args:
            m5_file: Path to cleaned M5 data
            timeframe: Target timeframe for features
            
        Returns:
            Path to features file
        """
        logger.info("=== STEP 4: BUILDING ESSENTIAL FEATURES ===")
        
        # Create essential features
        features_path = self.features_engine.create_essential_features(
            input_file=m5_file,
            timeframe=timeframe,
            output_dir="data/processed"
        )
        
        logger.info(f"Feature engineering completed: {features_path}")
        return features_path
    
    def step5_define_splits(self, features_file: str) -> str:
        """
        Step 5: Define train/validation/test splits.
        
        Args:
            features_file: Path to features file
            
        Returns:
            Path to splits directory
        """
        logger.info("=== STEP 5: DEFINING DATA SPLITS ===")
        
        # Create splits
        splits_manifest = self.data_splitter.create_splits(
            input_file=features_file,
            output_dir="data/splits",
            split_method="time_based"
        )
        
        logger.info(f"Data splits completed: {splits_manifest}")
        return "data/splits"
    
    def step6_train_baseline(self, splits_dir: str) -> dict:
        """
        Step 6: Train Random Forest baseline.
        
        Args:
            splits_dir: Directory containing data splits
            
        Returns:
            Training results dictionary
        """
        logger.info("=== STEP 6: TRAINING RANDOM FOREST BASELINE ===")
        
        # Train baseline model
        training_results = self.rf_trainer.train_baseline(
            splits_dir=splits_dir,
            output_dir="results"
        )
        
        logger.info("Random Forest baseline training completed")
        return training_results
    
    def step7_backtest_realistic(self, features_file: str, model_path: str) -> dict:
        """
        Step 7: Run realistic backtest with costs.
        
        Args:
            features_file: Path to features file
            model_path: Path to trained model
            
        Returns:
            Backtest results dictionary
        """
        logger.info("=== STEP 7: RUNNING REALISTIC BACKTEST ===")
        
        # Load features data for backtesting
        df = pd.read_csv(features_file)
        
        # Run backtest
        backtest_results = self.backtester.run_backtest(
            df=df,
            model_path=model_path
        )
        
        # Generate backtest report
        report_path = self.backtester.generate_backtest_report(
            output_dir="results"
        )
        
        logger.info(f"Realistic backtest completed: {report_path}")
        return backtest_results
    
    def run_complete_pipeline(self, input_file: str) -> dict:
        """
        Run the complete pipeline from raw data to backtest results.
        
        Args:
            input_file: Path to raw data CSV file
            
        Returns:
            Dictionary with all results and file paths
        """
        logger.info("🚀 STARTING COMPLETE PIPELINE EXECUTION")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        results = {}
        
        try:
            # Step 1: Normalize data
            m5_clean, m15_clean = self.step1_normalize_data(input_file)
            results['step1'] = {'m5_clean': m5_clean, 'm15_clean': m15_clean}
            
            # Step 2: Clean data
            m5_cleaned, m15_cleaned = self.step2_clean_data(m5_clean, m15_clean)
            results['step2'] = {'m5_cleaned': m5_cleaned, 'm15_cleaned': m15_cleaned}
            
            # Step 3: Validate quality
            quality_report = self.step3_validate_quality(m5_cleaned, m15_cleaned)
            results['step3'] = {'quality_report': quality_report}
            
            # Step 4: Build features
            features_file = self.step4_build_features(m5_cleaned, timeframe="M15")
            results['step4'] = {'features_file': features_file}
            
            # Step 5: Define splits
            splits_dir = self.step5_define_splits(features_file)
            results['step5'] = {'splits_dir': splits_dir}
            
            # Step 6: Train baseline
            training_results = self.step6_train_baseline(splits_dir)
            results['step6'] = training_results
            
            # Step 7: Backtest with realistic costs
            model_path = training_results['model_path']
            backtest_results = self.step7_backtest_realistic(features_file, model_path)
            results['step7'] = backtest_results
            
            # Generate final summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            # Final summary
            summary = {
                'pipeline_execution': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration.total_seconds(),
                    'status': 'completed'
                },
                'deliverables': {
                    'clean_data': {
                        'm5': results['step2']['m5_cleaned'],
                        'm15': results['step2']['m15_cleaned']
                    },
                    'processed_features': results['step4']['features_file'],
                    'data_splits': results['step5']['splits_dir'],
                    'trained_model': results['step6']['model_path'],
                    'feature_importance_report': results['step6']['feature_report_path'],
                    'backtest_report': 'Generated in results/',
                    'quality_report': results['step3']['quality_report']
                },
                'performance_summary': {
                    'model_accuracy': {
                        'validation': results['step6']['training_summary']['validation_accuracy'],
                        'test': results['step6']['training_summary']['test_accuracy']
                    },
                    'backtest_metrics': results['step7']['performance_metrics'],
                    'acceptance_criteria': self._check_final_acceptance_criteria(results)
                }
            }
            
            results['final_summary'] = summary
            
            # Save final summary
            summary_path = f"results/pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            with open(summary_path, 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
            
            logger.info("=" * 60)
            logger.info("✅ COMPLETE PIPELINE EXECUTION FINISHED")
            logger.info(f"📊 Total Duration: {duration}")
            logger.info(f"📋 Summary Report: {summary_path}")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            logger.error("Pipeline stopped due to error")
            raise
    
    def _check_final_acceptance_criteria(self, results: dict) -> dict:
        """Check final acceptance criteria against user requirements."""
        try:
            backtest_metrics = results['step7']['performance_metrics']
            
            # User's acceptance criteria
            criteria = {
                'profit_factor_target': 1.2,
                'max_drawdown_target': 0.15,  # 15%
                'sharpe_ratio_target': 0.8
            }
            
            return {
                'profit_factor': {
                    'value': backtest_metrics.get('profit_factor', 0),
                    'target': criteria['profit_factor_target'],
                    'pass': backtest_metrics.get('profit_factor', 0) >= criteria['profit_factor_target']
                },
                'max_drawdown': {
                    'value': abs(backtest_metrics.get('max_drawdown', 1)),
                    'target': criteria['max_drawdown_target'],
                    'pass': abs(backtest_metrics.get('max_drawdown', 1)) <= criteria['max_drawdown_target']
                },
                'sharpe_ratio': {
                    'value': backtest_metrics.get('sharpe_ratio', 0),
                    'target': criteria['sharpe_ratio_target'],
                    'pass': backtest_metrics.get('sharpe_ratio', 0) >= criteria['sharpe_ratio_target']
                },
                'overall_pass': all([
                    backtest_metrics.get('profit_factor', 0) >= criteria['profit_factor_target'],
                    abs(backtest_metrics.get('max_drawdown', 1)) <= criteria['max_drawdown_target'],
                    backtest_metrics.get('sharpe_ratio', 0) >= criteria['sharpe_ratio_target']
                ])
            }
        except Exception as e:
            logger.warning(f"Could not check acceptance criteria: {e}")
            return {'error': str(e)}


def main():
    """Main execution function."""
    logger.info("SignaMentis Complete Pipeline")
    logger.info("Random Forest Baseline with Realistic Backtesting")
    
    # Initialize pipeline
    pipeline = CompletePipeline()
    
    # Check for input data
    possible_data_files = [
        "data/raw/XAUUSD_M5.csv",
        "data/raw/xauusd_m5.csv", 
        "data/raw/tickbar_M5.csv",
        "data/raw/GOLD_M5.csv",
        "data/XAUUSD_M5.csv"
    ]
    
    input_file = None
    for file_path in possible_data_files:
        if Path(file_path).exists():
            input_file = file_path
            break
    
    if input_file is None:
        logger.error("❌ No input data file found!")
        logger.error("Expected files:")
        for file_path in possible_data_files:
            logger.error(f"  - {file_path}")
        logger.error("Please place your XAU/USD M5 data in one of these locations.")
        return
    
    logger.info(f"📂 Using input file: {input_file}")
    
    # Run complete pipeline
    try:
        results = pipeline.run_complete_pipeline(input_file)
        
        # Print final summary
        summary = results['final_summary']
        print("\n" + "=" * 60)
        print("🎯 FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"⏱️  Duration: {summary['pipeline_execution']['duration_seconds']:.1f} seconds")
        print(f"📊 Model Validation Accuracy: {summary['performance_summary']['model_accuracy']['validation']:.3f}")
        print(f"📊 Model Test Accuracy: {summary['performance_summary']['model_accuracy']['test']:.3f}")
        
        backtest = summary['performance_summary']['backtest_metrics']
        print(f"💰 Profit Factor: {backtest.get('profit_factor', 0):.2f}")
        print(f"📉 Max Drawdown: {abs(backtest.get('max_drawdown', 0)):.2%}")
        print(f"📈 Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.2f}")
        print(f"🎲 Win Rate: {backtest.get('win_rate', 0):.2%}")
        print(f"🏆 Total Return: {backtest.get('total_return', 0):.2%}")
        
        # Acceptance criteria
        criteria = summary['performance_summary']['acceptance_criteria']
        if criteria.get('overall_pass', False):
            print("✅ ALL ACCEPTANCE CRITERIA PASSED!")
        else:
            print("⚠️  Some acceptance criteria not met:")
            for criterion, data in criteria.items():
                if criterion != 'overall_pass' and isinstance(data, dict):
                    status = "✅" if data['pass'] else "❌"
                    print(f"   {status} {criterion}: {data['value']:.3f} (target: {data['target']:.3f})")
        
        print("\n📁 DELIVERABLES:")
        deliverables = summary['deliverables']
        for key, value in deliverables.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for subkey, subvalue in value.items():
                    print(f"     - {subkey}: {subvalue}")
            else:
                print(f"   - {key}: {value}")
        
        print("=" * 60)
        print("🚀 Pipeline execution completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
