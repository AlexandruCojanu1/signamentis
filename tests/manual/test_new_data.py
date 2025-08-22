#!/usr/bin/env python3
"""
Test Script for New Historical Data

This script tests the loading and processing of the new 5-minute and 15-minute historical data.
"""

import sys
import os
sys.path.append('scripts')

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from data_loader import DataLoader, DataConfig
from data_cleaner import DataCleaner, create_data_cleaner
from feature_engineering import FeatureEngineer

def test_data_loading():
    """Test loading of new historical data."""
    print("🔍 Testing new historical data loading...")
    
    # Initialize data loader
    data_config = DataConfig(
        symbol="XAUUSD",
        timeframe="M5",
        data_source="csv",
        raw_data_path="data/raw",
        processed_data_path="data/processed"
    )
    
    data_loader = DataLoader(data_config)
    
    # Test 5-minute data
    print("\n📊 Testing 5-minute data...")
    try:
        data_5min = data_loader.load_csv_data("data/raw/XAUUSD_5min.csv")
        print(f"✅ 5-minute data loaded successfully: {len(data_5min)} rows")
        print(f"   Date range: {data_5min.index.min()} to {data_5min.index.max()}")
        print(f"   Columns: {list(data_5min.columns)}")
        print(f"   Sample data:")
        print(data_5min.head())
        
        # Check for missing values
        missing_values = data_5min.isnull().sum()
        print(f"   Missing values: {missing_values.to_dict()}")
        
    except Exception as e:
        print(f"❌ Error loading 5-minute data: {e}")
        return False
    
    # Test 15-minute data
    print("\n📊 Testing 15-minute data...")
    try:
        data_15min = data_loader.load_csv_data("data/raw/XAUUSD_15min.csv")
        print(f"✅ 15-minute data loaded successfully: {len(data_15min)} rows")
        print(f"   Date range: {data_15min.index.min()} to {data_15min.index.max()}")
        print(f"   Columns: {list(data_15min.columns)}")
        print(f"   Sample data:")
        print(data_15min.head())
        
        # Check for missing values
        missing_values = data_15min.isnull().sum()
        print(f"   Missing values: {missing_values.to_dict()}")
        
    except Exception as e:
        print(f"❌ Error loading 15-minute data: {e}")
        return False
    
    return True

def test_data_cleaning():
    """Test data cleaning functionality."""
    print("\n🧹 Testing data cleaning...")
    
    try:
        # Initialize data cleaner
        cleaner_config = {
            'outlier_method': 'IQR',
            'outlier_threshold': 3.0,
            'iqr_multiplier': 1.5,
            'min_data_quality': 0.7
        }
        
        cleaner = create_data_cleaner(cleaner_config)
        
        # Load and clean 5-minute data
        data_5min = pd.read_csv("data/raw/XAUUSD_5min.csv")
        
        # Clean the data
        cleaned_data = cleaner.clean_data(data_5min, "XAUUSD")
        
        print(f"✅ Data cleaning completed successfully")
        print(f"   Original shape: {data_5min.shape}")
        print(f"   Cleaned shape: {cleaned_data.shape}")
        
        # Generate quality report
        quality_report = cleaner.generate_quality_report(cleaned_data)
        print(f"\n📋 Data Quality Report:")
        print(quality_report)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data cleaning: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality."""
    print("\n⚙️ Testing feature engineering...")
    
    try:
        # Load cleaned data
        data_loader = DataLoader()
        data_5min = data_loader.load_csv_data("data/raw/XAUUSD_5min.csv")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Create features
        features_df = feature_engineer.create_features(data_5min)
        
        print(f"✅ Feature engineering completed successfully")
        print(f"   Original columns: {list(data_5min.columns)}")
        print(f"   Feature columns: {list(features_df.columns)}")
        print(f"   Total features created: {len(features_df.columns)}")
        
        # Show sample features
        print(f"\n📊 Sample features:")
        feature_sample = features_df.iloc[-5:, -10:]  # Last 5 rows, last 10 columns
        print(feature_sample)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return False

def create_data_visualization():
    """Create visualizations of the new data."""
    print("\n📈 Creating data visualizations...")
    
    try:
        # Load data
        data_loader = DataLoader()
        data_5min = data_loader.load_csv_data("data/raw/XAUUSD_5min.csv")
        data_15min = data_loader.load_csv_data("data/raw/XAUUSD_15min.csv")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SignaMentis - New Historical Data Analysis', fontsize=16)
        
        # 5-minute price chart
        axes[0, 0].plot(data_5min.index, data_5min['close'], linewidth=0.8, alpha=0.8)
        axes[0, 0].set_title('XAUUSD 5-Minute Close Prices')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 15-minute price chart
        axes[0, 1].plot(data_15min.index, data_15min['close'], linewidth=0.8, alpha=0.8, color='orange')
        axes[0, 1].set_title('XAUUSD 15-Minute Close Prices')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Price (USD)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Volume comparison
        axes[1, 0].bar(data_5min.index, data_5min['volume'], alpha=0.7, width=0.0001)
        axes[1, 0].set_title('5-Minute Volume')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Volume')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Price distribution
        axes[1, 1].hist(data_5min['close'], bins=50, alpha=0.7, color='green')
        axes[1, 1].set_title('5-Minute Close Price Distribution')
        axes[1, 1].set_xlabel('Price (USD)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = "data/new_data_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to {output_file}")
        
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 SignaMentis - New Historical Data Test Suite")
    print("=" * 60)
    
    # Test data loading
    if not test_data_loading():
        print("❌ Data loading tests failed")
        return
    
    # Test data cleaning
    if not test_data_cleaning():
        print("❌ Data cleaning tests failed")
        return
    
    # Test feature engineering
    if not test_feature_engineering():
        print("❌ Feature engineering tests failed")
        return
    
    # Create visualizations
    if not create_data_visualization():
        print("❌ Visualization creation failed")
        return
    
    print("\n🎉 All tests completed successfully!")
    print("✅ New historical data is ready for use in SignaMentis")

if __name__ == "__main__":
    main()
