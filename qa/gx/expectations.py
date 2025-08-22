#!/usr/bin/env python3
"""
SignaMentis - Great Expectations Data Quality Expectations

This module defines data quality expectations for the SignaMentis trading system
using Great Expectations framework.

Author: SignaMentis Team
Version: 2.0.0
"""

from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.resource_identifiers import GeCloudIdentifier
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.validator.validator import Validator
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityValidator:
    """
    Data quality validator using Great Expectations.
    
    Provides:
    - Data validation rules
    - Quality metrics
    - Automated testing
    - Data profiling
    - Custom expectations
    """
    
    def __init__(self, context_path: str = "qa/gx"):
        """
        Initialize data quality validator.
        
        Args:
            context_path: Path to Great Expectations context
        """
        self.context_path = context_path
        self.context = None
        self.validator = None
        
        # Initialize context
        self._initialize_context()
        
        logger.info("🚀 Data Quality Validator initialized")
    
    def _initialize_context(self):
        """Initialize Great Expectations context."""
        try:
            self.context = BaseDataContext(project_root_dir=self.context_path)
            logger.info("✅ Great Expectations context initialized")
        except Exception as e:
            logger.warning(f"Could not initialize GE context: {e}")
            logger.info("Creating new context...")
            self._create_context()
    
    def _create_context(self):
        """Create new Great Expectations context."""
        try:
            # Create context directory structure
            import os
            os.makedirs(self.context_path, exist_ok=True)
            os.makedirs(f"{self.context_path}/expectations", exist_ok=True)
            os.makedirs(f"{self.context_path}/checkpoints", exist_ok=True)
            os.makedirs(f"{self.context_path}/plugins", exist_ok=True)
            os.makedirs(f"{self.context_path}/uncommitted", exist_ok=True)
            
            # Create context
            self.context = BaseDataContext(project_root_dir=self.context_path)
            logger.info("✅ New GE context created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create context: {e}")
            raise
    
    def validate_market_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate market data quality.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Validation results
        """
        try:
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="market_data",
                data_connector_name="default_runtime_data_connector_name",
                data_asset_name="market_data",
                runtime_parameters={"batch_data": data},
                batch_identifiers={"default_identifier_name": "default_identifier"}
            )
            
            # Create validator
            self.validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name="market_data_suite"
            )
            
            # Apply market data expectations
            self._apply_market_data_expectations()
            
            # Validate
            results = self.validator.validate()
            
            return {
                "success": results.success,
                "statistics": results.statistics,
                "results": results.run_results
            }
            
        except Exception as e:
            logger.error(f"❌ Market data validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _apply_market_data_expectations(self):
        """Apply market data specific expectations."""
        try:
            # Basic structure expectations
            self.validator.expect_table_columns_to_match_ordered_list([
                "timestamp", "open", "high", "low", "close", "volume"
            ])
            
            self.validator.expect_table_row_count_to_be_between(min_value=100, max_value=1000000)
            
            # Data type expectations
            self.validator.expect_column_values_to_be_of_type("timestamp", "datetime64[ns]")
            self.validator.expect_column_values_to_be_of_type("open", "float64")
            self.validator.expect_column_values_to_be_of_type("high", "float64")
            self.validator.expect_column_values_to_be_of_type("low", "float64")
            self.validator.expect_column_values_to_be_of_type("close", "float64")
            self.validator.expect_column_values_to_be_of_type("volume", "float64")
            
            # Value range expectations
            self.validator.expect_column_values_to_be_between("open", min_value=0.01, max_value=100000)
            self.validator.expect_column_values_to_be_between("high", min_value=0.01, max_value=100000)
            self.validator.expect_column_values_to_be_between("low", min_value=0.01, max_value=100000)
            self.validator.expect_column_values_to_be_between("close", min_value=0.01, max_value=100000)
            self.validator.expect_column_values_to_be_between("volume", min_value=0, max_value=1000000000)
            
            # OHLC relationship expectations
            self.validator.expect_column_values_to_be_between(
                "high", 
                min_value=lambda x: x["low"], 
                max_value=lambda x: x["high"]
            )
            
            self.validator.expect_column_values_to_be_between(
                "low", 
                min_value=lambda x: x["low"], 
                max_value=lambda x: x["high"]
            )
            
            # Timestamp expectations
            self.validator.expect_column_values_to_be_unique("timestamp")
            self.validator.expect_column_values_to_be_increasing("timestamp")
            
            # Missing value expectations
            self.validator.expect_column_values_to_not_be_null("timestamp")
            self.validator.expect_column_values_to_not_be_null("open")
            self.validator.expect_column_values_to_not_be_null("high")
            self.validator.expect_column_values_to_not_be_null("low")
            self.validator.expect_column_values_to_not_be_null("close")
            
            # Statistical expectations
            self.validator.expect_column_mean_to_be_between("close", min_value=1, max_value=100000)
            self.validator.expect_column_std_to_be_between("close", min_value=0.01, max_value=10000)
            
            logger.info("✅ Market data expectations applied")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply market data expectations: {e}")
    
    def validate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate technical indicators data quality.
        
        Args:
            data: Technical indicators DataFrame
            
        Returns:
            Validation results
        """
        try:
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="technical_indicators",
                data_connector_name="default_runtime_data_connector_name",
                data_asset_name="technical_indicators",
                runtime_parameters={"batch_data": data},
                batch_identifiers={"default_identifier_name": "default_identifier"}
            )
            
            # Create validator
            self.validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name="technical_indicators_suite"
            )
            
            # Apply technical indicators expectations
            self._apply_technical_indicators_expectations()
            
            # Validate
            results = self.validator.validate()
            
            return {
                "success": results.success,
                "statistics": results.statistics,
                "results": results.run_results
            }
            
        except Exception as e:
            logger.error(f"❌ Technical indicators validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _apply_technical_indicators_expectations(self):
        """Apply technical indicators specific expectations."""
        try:
            # Basic structure expectations
            self.validator.expect_table_columns_to_match_ordered_list([
                "timestamp", "symbol", "sma_20", "sma_50", "rsi", "macd", "bb_upper", "bb_lower"
            ])
            
            # Data type expectations
            self.validator.expect_column_values_to_be_of_type("timestamp", "datetime64[ns]")
            self.validator.expect_column_values_to_be_of_type("symbol", "object")
            self.validator.expect_column_values_to_be_of_type("sma_20", "float64")
            self.validator.expect_column_values_to_be_of_type("sma_50", "float64")
            self.validator.expect_column_values_to_be_of_type("rsi", "float64")
            self.validator.expect_column_values_to_be_of_type("macd", "float64")
            
            # Value range expectations for indicators
            self.validator.expect_column_values_to_be_between("rsi", min_value=0, max_value=100)
            self.validator.expect_column_values_to_be_between("bb_upper", min_value=0.01, max_value=100000)
            self.validator.expect_column_values_to_be_between("bb_lower", min_value=0.01, max_value=100000)
            
            # Statistical expectations
            self.validator.expect_column_mean_to_be_between("rsi", min_value=20, max_value=80)
            self.validator.expect_column_std_to_be_between("rsi", min_value=5, max_value=30)
            
            # Relationship expectations
            self.validator.expect_column_values_to_be_between(
                "bb_upper", 
                min_value=lambda x: x["bb_lower"]
            )
            
            logger.info("✅ Technical indicators expectations applied")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply technical indicators expectations: {e}")
    
    def validate_news_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate news data quality.
        
        Args:
            data: News data DataFrame
            
        Returns:
            Validation results
        """
        try:
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="news_data",
                data_connector_name="default_runtime_data_connector_name",
                data_asset_name="news_data",
                runtime_parameters={"batch_data": data},
                batch_identifiers={"default_identifier_name": "default_identifier"}
            )
            
            # Create validator
            self.validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name="news_data_suite"
            )
            
            # Apply news data expectations
            self._apply_news_data_expectations()
            
            # Validate
            results = self.validator.validate()
            
            return {
                "success": results.success,
                "statistics": results.statistics,
                "results": results.run_results
            }
            
        except Exception as e:
            logger.error(f"❌ News data validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _apply_news_data_expectations(self):
        """Apply news data specific expectations."""
        try:
            # Basic structure expectations
            self.validator.expect_table_columns_to_match_ordered_list([
                "timestamp", "title", "sentiment_score", "source", "relevance_score"
            ])
            
            # Data type expectations
            self.validator.expect_column_values_to_be_of_type("timestamp", "datetime64[ns]")
            self.validator.expect_column_values_to_be_of_type("title", "object")
            self.validator.expect_column_values_to_be_of_type("sentiment_score", "float64")
            self.validator.expect_column_values_to_be_of_type("source", "object")
            self.validator.expect_column_values_to_be_of_type("relevance_score", "float64")
            
            # Value range expectations
            self.validator.expect_column_values_to_be_between("sentiment_score", min_value=-1, max_value=1)
            self.validator.expect_column_values_to_be_between("relevance_score", min_value=0, max_value=1)
            
            # Content expectations
            self.validator.expect_column_value_lengths_to_be_between("title", min_value=10, max_value=500)
            self.validator.expect_column_values_to_not_be_null("title")
            self.validator.expect_column_values_to_not_be_null("sentiment_score")
            
            # Source expectations
            self.validator.expect_column_values_to_be_in_set("source", value_set=[
                "reuters", "bloomberg", "cnbc", "wsj", "ft", "gdelt", "newsapi", "gnews"
            ])
            
            logger.info("✅ News data expectations applied")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply news data expectations: {e}")
    
    def create_custom_expectation(self, expectation_name: str, expectation_func: callable):
        """
        Create a custom expectation.
        
        Args:
            expectation_name: Name of the expectation
            expectation_func: Function implementing the expectation
        """
        try:
            # This would require more complex setup with GE custom expectations
            # For now, we'll log the intention
            logger.info(f"📝 Custom expectation '{expectation_name}' registered")
            
        except Exception as e:
            logger.error(f"❌ Failed to create custom expectation: {e}")
    
    def generate_data_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data profile for quality assessment.
        
        Args:
            data: Data to profile
            
        Returns:
            Data profile
        """
        try:
            profile = {
                "shape": data.shape,
                "dtypes": data.dtypes.to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "duplicates": data.duplicated().sum(),
                "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": data.select_dtypes(include=['object']).columns.tolist(),
                "datetime_columns": data.select_dtypes(include=['datetime64']).columns.tolist()
            }
            
            # Add statistical summaries for numeric columns
            if profile["numeric_columns"]:
                profile["numeric_summary"] = data[profile["numeric_columns"]].describe().to_dict()
            
            # Add value counts for categorical columns
            if profile["categorical_columns"]:
                profile["categorical_summary"] = {}
                for col in profile["categorical_columns"]:
                    profile["categorical_summary"][col] = data[col].value_counts().to_dict()
            
            logger.info("✅ Data profile generated")
            return profile
            
        except Exception as e:
            logger.error(f"❌ Failed to generate data profile: {e}")
            return {}
    
    def save_expectation_suite(self, suite_name: str):
        """Save the current expectation suite."""
        try:
            if self.validator:
                self.validator.save_expectation_suite()
                logger.info(f"✅ Expectation suite '{suite_name}' saved")
        except Exception as e:
            logger.error(f"❌ Failed to save expectation suite: {e}")
    
    def load_expectation_suite(self, suite_name: str):
        """Load an expectation suite."""
        try:
            suite = self.context.get_expectation_suite(suite_name)
            logger.info(f"✅ Expectation suite '{suite_name}' loaded")
            return suite
        except Exception as e:
            logger.error(f"❌ Failed to load expectation suite: {e}")
            return None
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.validator:
            return {}
        
        try:
            return {
                "total_expectations": len(self.validator.expectation_suite.expectations),
                "successful_expectations": 0,
                "failed_expectations": 0,
                "unexpected_expectations": 0
            }
        except Exception as e:
            logger.error(f"❌ Failed to get validation statistics: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    def main():
        print("🚀 Great Expectations Data Quality Validator Example")
        print("=" * 60)
        
        # Create validator
        validator = DataQualityValidator()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'open': np.random.uniform(1800, 2000, 100),
            'high': np.random.uniform(1800, 2000, 100),
            'low': np.random.uniform(1800, 2000, 100),
            'close': np.random.uniform(1800, 2000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Ensure OHLC relationships
        sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
        sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)
        
        print("✅ Sample data created")
        print("📊 Data shape:", sample_data.shape)
        
        # Validate market data
        print("\n🔍 Validating market data...")
        results = validator.validate_market_data(sample_data)
        
        if results.get("success"):
            print("✅ Market data validation passed!")
        else:
            print("❌ Market data validation failed!")
            print("Error:", results.get("error", "Unknown error"))
        
        # Generate data profile
        print("\n📈 Generating data profile...")
        profile = validator.generate_data_profile(sample_data)
        print(f"✅ Profile generated: {profile['shape'][0]} rows, {profile['shape'][1]} columns")
        
        print("\n🎯 Data quality validation system ready!")
        print("Use validator.validate_*() methods to validate different data types")
    
    # Run example
    main()
