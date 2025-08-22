#!/usr/bin/env python3
"""
Simplified Test Script for New Historical Data

This script tests the loading and processing of the new 5-minute and 15-minute historical data
without requiring external dependencies.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def test_csv_loading():
    """Test basic CSV loading functionality."""
    print("🔍 Testing CSV loading functionality...")
    
    # Test 5-minute data
    print("\n📊 Testing 5-minute data...")
    try:
        data_5min = pd.read_csv("data/raw/XAUUSD_5min.csv")
        print(f"✅ 5-minute data loaded successfully: {len(data_5min)} rows")
        print(f"   Columns: {list(data_5min.columns)}")
        print(f"   Sample data:")
        print(data_5min.head())
        
        # Check for missing values
        missing_values = data_5min.isnull().sum()
        print(f"   Missing values: {missing_values.to_dict()}")
        
        # Basic statistics
        print(f"   Price range: {data_5min['Close'].min():.2f} - {data_5min['Close'].max():.2f}")
        print(f"   Volume range: {data_5min['Volume'].min()} - {data_5min['Volume'].max()}")
        
    except Exception as e:
        print(f"❌ Error loading 5-minute data: {e}")
        return False
    
    # Test 15-minute data
    print("\n📊 Testing 15-minute data...")
    try:
        data_15min = pd.read_csv("data/raw/XAUUSD_15min.csv")
        print(f"✅ 15-minute data loaded successfully: {len(data_15min)} rows")
        print(f"   Columns: {list(data_15min.columns)}")
        print(f"   Sample data:")
        print(data_15min.head())
        
        # Check for missing values
        missing_values = data_15min.isnull().sum()
        print(f"   Missing values: {missing_values.to_dict()}")
        
        # Basic statistics
        print(f"   Price range: {data_15min['Close'].min():.2f} - {data_15min['Close'].max():.2f}")
        print(f"   Volume range: {data_15min['Volume'].min()} - {data_15min['Volume'].max()}")
        
    except Exception as e:
        print(f"❌ Error loading 15-minute data: {e}")
        return False
    
    return True

def test_data_processing():
    """Test basic data processing."""
    print("\n⚙️ Testing data processing...")
    
    try:
        # Load 5-minute data
        data_5min = pd.read_csv("data/raw/XAUUSD_5min.csv")
        
        # Convert timestamp
        data_5min['timestamp'] = pd.to_datetime(data_5min['Local time'], 
                                               format='%d.%m.%Y %H:%M:%S.%f GMT%z', 
                                               errors='coerce')
        
        # If timezone parsing fails, try without timezone
        if data_5min['timestamp'].isnull().all():
            data_5min['timestamp'] = pd.to_datetime(data_5min['Local time'].str.replace(' GMT+0300', ''), 
                                                   format='%d.%m.%Y %H:%M:%S.%f', 
                                                   errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            data_5min[col] = pd.to_numeric(data_5min[col], errors='coerce')
        
        # Remove rows with NaN values
        original_rows = len(data_5min)
        data_5min = data_5min.dropna()
        cleaned_rows = len(data_5min)
        
        print(f"✅ Data processing completed successfully")
        print(f"   Original rows: {original_rows}")
        print(f"   Cleaned rows: {cleaned_rows}")
        print(f"   Rows removed: {original_rows - cleaned_rows}")
        
        # Set timestamp as index
        data_5min.set_index('timestamp', inplace=True)
        data_5min.sort_index(inplace=True)
        
        print(f"   Date range: {data_5min.index.min()} to {data_5min.index.max()}")
        
        return data_5min
        
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return None

def test_feature_creation():
    """Test basic feature creation."""
    print("\n🔧 Testing feature creation...")
    
    try:
        # Load and process data
        data = test_data_processing()
        if data is None:
            return False
        
        # Create basic technical indicators
        features = data.copy()
        
        # Price changes
        features['price_change'] = features['Close'].pct_change()
        features['price_change_abs'] = features['price_change'].abs()
        
        # Moving averages
        features['sma_20'] = features['Close'].rolling(window=20).mean()
        features['ema_20'] = features['Close'].ewm(span=20).mean()
        
        # Volatility
        features['volatility'] = features['price_change'].rolling(window=20).std()
        
        # RSI
        delta = features['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Remove NaN values from features
        features = features.dropna()
        
        print(f"✅ Feature creation completed successfully")
        print(f"   Original features: {len(data.columns)}")
        print(f"   New features: {len(features.columns)}")
        print(f"   Total features: {len(features.columns)}")
        
        # Show feature statistics
        print(f"\n📊 Feature statistics:")
        print(features.describe())
        
        return features
        
    except Exception as e:
        print(f"❌ Error in feature creation: {e}")
        return None

def create_visualizations():
    """Create data visualizations."""
    print("\n📈 Creating visualizations...")
    
    try:
        # Load and process data
        data = test_data_processing()
        if data is None:
            return False
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SignaMentis - New Historical Data Analysis', fontsize=16)
        
        # Price chart
        axes[0, 0].plot(data.index, data['Close'], linewidth=0.8, alpha=0.8)
        axes[0, 0].set_title('XAUUSD 5-Minute Close Prices')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volume chart
        axes[0, 1].bar(data.index, data['Volume'], alpha=0.7, width=0.0001)
        axes[0, 1].set_title('5-Minute Volume')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Price distribution
        axes[1, 0].hist(data['Close'], bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Close Price Distribution')
        axes[1, 0].set_xlabel('Price (USD)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Price changes
        axes[1, 1].hist(data['Close'].pct_change().dropna(), bins=50, alpha=0.7, color='orange')
        axes[1, 1].set_title('Price Change Distribution')
        axes[1, 1].set_xlabel('Price Change (%)')
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

def save_processed_data():
    """Save processed data for later use."""
    print("\n💾 Saving processed data...")
    
    try:
        # Process data
        data = test_data_processing()
        if data is None:
            return False
        
        # Save to CSV
        output_file = "data/processed/XAUUSD_5min_processed.csv"
        data.to_csv(output_file)
        print(f"✅ Processed data saved to {output_file}")
        
        # Save to pickle for faster loading
        pickle_file = "data/processed/XAUUSD_5min_processed.pkl"
        data.to_pickle(pickle_file)
        print(f"✅ Processed data saved to {pickle_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving processed data: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 SignaMentis - New Historical Data Test Suite (Simplified)")
    print("=" * 70)
    
    # Test CSV loading
    if not test_csv_loading():
        print("❌ CSV loading tests failed")
        return
    
    # Test data processing
    data_processed = test_data_processing()
    if data_processed is None:
        print("❌ Data processing tests failed")
        return
    
    # Test feature creation
    features_created = test_feature_creation()
    if features_created is None:
        print("❌ Feature creation tests failed")
        return
    
    # Create visualizations
    if not create_visualizations():
        print("❌ Visualization creation failed")
        return
    
    # Save processed data
    if not save_processed_data():
        print("❌ Data saving failed")
        return
    
    print("\n🎉 All tests completed successfully!")
    print("✅ New historical data is ready for use in SignaMentis")
    print("📁 Processed data saved in data/processed/ directory")

if __name__ == "__main__":
    main()
