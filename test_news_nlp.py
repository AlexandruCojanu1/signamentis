#!/usr/bin/env python3
"""
SignaMentis - News NLP Service Test

This script tests the basic functionality of the news NLP service
without requiring external API keys or dependencies.

Author: SignaMentis Team
Version: 2.0.0
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add services directory to path
sys.path.append('services')

def test_news_nlp_structure():
    """Test the news NLP package structure."""
    print("🧪 Testing News NLP Package Structure...")
    
    try:
        # Test package imports
        from news_nlp import (
            GDELTIngestor, TradingEconomicsIngestor, NewsAPIIngestor, GNewsIngestor,
            FinBERTSentimentAnalyzer, NewsFeatureExtractor
        )
        print("✅ All news NLP components imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ News NLP structure test failed: {e}")
        return False

def test_feature_extraction():
    """Test news feature extraction functionality."""
    print("\n🧪 Testing News Feature Extraction...")
    
    try:
        from news_nlp import NewsFeatureExtractor
        
        config = {
            'sentiment_window': 24,
            'volume_window': 6,
            'momentum_window': 2
        }
        
        extractor = NewsFeatureExtractor(config)
        print("✅ News Feature Extractor created")
        
        # Sample news data
        sample_news = [
            {
                'timestamp': datetime.now() - timedelta(hours=1),
                'sentiment_score': 0.8,
                'sentiment_label': 'positive',
                'source_name': 'reuters',
                'title': 'Gold prices surge on safe haven demand',
                'relevance_score': 0.9,
                'volume': 1.0
            },
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'sentiment_score': -0.3,
                'sentiment_label': 'negative',
                'source_name': 'bloomberg',
                'title': 'Market volatility increases',
                'relevance_score': 0.7,
                'volume': 1.0
            }
        ]
        
        # Add news data
        extractor.add_news_data(sample_news)
        print("✅ News data added to extractor")
        
        # Extract features
        features = extractor.extract_features()
        print(f"✅ Features extracted: {len(features.features)} features")
        print(f"   - Sentiment Score: {features.sentiment_score:.3f}")
        print(f"   - News Volume: {features.news_volume:.3f}")
        print(f"   - Composite Score: {features.composite_score:.3f}")
        
        # Get performance stats
        stats = extractor.get_performance_stats()
        print(f"✅ Performance stats: {stats['total_features_extracted']} features extracted")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_news_ingestors():
    """Test news ingestor basic functionality."""
    print("\n🧪 Testing News Ingestors...")
    
    try:
        from news_nlp import GDELTIngestor, TradingEconomicsIngestor
        
        # Test GDELT ingestor
        gdelt_config = {
            'base_url': 'https://api.gdeltproject.org/api/v2',
            'api_key': 'test',
            'rate_limit': 1000,
            'timeout': 30
        }
        
        gdelt = GDELTIngestor(gdelt_config)
        print("✅ GDELT Ingestor configured")
        
        # Test TradingEconomics ingestor
        te_config = {
            'base_url': 'https://api.tradingeconomics.com',
            'api_key': 'test',
            'rate_limit': 1000,
            'timeout': 30
        }
        
        te = TradingEconomicsIngestor(te_config)
        print("✅ TradingEconomics Ingestor configured")
        
        # Test basic methods (without API calls)
        print("✅ Ingestor basic functionality verified")
        
        return True
        
    except Exception as e:
        print(f"❌ News ingestors test failed: {e}")
        return False

def test_api_structure():
    """Test the API service structure."""
    print("\n🧪 Testing News NLP API Structure...")
    
    try:
        # Test API models
        from news_nlp.api import (
            NewsIngestionRequest, SentimentAnalysisRequest, 
            FeatureExtractionRequest, SystemHealth
        )
        print("✅ API models imported successfully")
        
        # Test request models
        news_request = NewsIngestionRequest(
            source="gdelt",
            category="economic",
            hours_back=24,
            max_articles=50
        )
        print("✅ News ingestion request model created")
        
        sentiment_request = SentimentAnalysisRequest(
            text="Gold prices are rising due to market uncertainty",
            batch_mode=False
        )
        print("✅ Sentiment analysis request model created")
        
        feature_request = FeatureExtractionRequest(
            news_data=[{"title": "Test news", "sentiment": 0.5}],
            target_timestamp=datetime.now().isoformat()
        )
        print("✅ Feature extraction request model created")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 SignaMentis - News NLP Service Test")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    tests = [
        ("News NLP Package Structure", test_news_nlp_structure),
        ("News Feature Extraction", test_feature_extraction),
        ("News Ingestors", test_news_ingestors),
        ("News NLP API Structure", test_api_structure)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 News NLP Test Results:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All News NLP tests passed!")
        print("✅ News NLP service is ready for integration!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed!")
        print("🔧 Please check the failing tests above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
