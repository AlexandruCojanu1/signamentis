#!/usr/bin/env python3
"""
SignaMentis - News Features Module

This module extracts and processes news-based features for trading signals.
It combines news sentiment, volume, and timing to create actionable features.

Author: SignaMentis Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import defaultdict, deque
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsFeatureType(Enum):
    """News feature types enumeration."""
    SENTIMENT = "sentiment"
    VOLUME = "volume"
    TIMING = "timing"
    SOURCE = "source"
    COMPOSITE = "composite"


@dataclass
class NewsFeature:
    """News feature container."""
    feature_id: str
    feature_type: NewsFeatureType
    value: float
    timestamp: datetime
    source: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NewsFeatureSet:
    """Complete set of news features for a time period."""
    timestamp: datetime
    sentiment_score: float
    sentiment_volume: float
    sentiment_momentum: float
    news_volume: float
    news_urgency: float
    source_diversity: float
    composite_score: float
    features: List[NewsFeature]
    metadata: Optional[Dict[str, Any]] = None


class NewsFeatureExtractor:
    """
    News feature extractor for trading signals.
    
    Features:
    - Sentiment-based features
    - Volume-based features
    - Timing-based features
    - Source diversity features
    - Composite scoring
    - Rolling window analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize news feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Feature configuration
        self.sentiment_window = self.config.get('sentiment_window', 24)  # hours
        self.volume_window = self.config.get('volume_window', 6)  # hours
        self.momentum_window = self.config.get('momentum_window', 2)  # hours
        self.urgency_threshold = self.config.get('urgency_threshold', 0.7)
        
        # Weighting configuration
        self.sentiment_weight = self.config.get('sentiment_weight', 0.4)
        self.volume_weight = self.config.get('volume_weight', 0.3)
        self.timing_weight = self.config.get('timing_weight', 0.2)
        self.source_weight = self.config.get('source_weight', 0.1)
        
        # Data storage
        self.news_cache = deque(maxlen=10000)
        self.feature_history = deque(maxlen=1000)
        
        # Performance tracking
        self.total_features_extracted = 0
        self.start_time = datetime.now()
        
        logger.info("News Feature Extractor initialized")
    
    def add_news_data(self, news_items: List[Dict[str, Any]]):
        """
        Add news data for feature extraction.
        
        Args:
            news_items: List of news items with sentiment and metadata
        """
        try:
            for item in news_items:
                # Extract timestamp
                if isinstance(item.get('timestamp'), str):
                    timestamp = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                elif isinstance(item.get('timestamp'), datetime):
                    timestamp = item['timestamp']
                else:
                    timestamp = datetime.now()
                
                # Create news cache entry
                news_entry = {
                    'timestamp': timestamp,
                    'sentiment': item.get('sentiment_score', 0.0),
                    'sentiment_label': item.get('sentiment_label', 'neutral'),
                    'source': item.get('source_name', 'unknown'),
                    'title': item.get('title', ''),
                    'relevance': item.get('relevance_score', 0.5),
                    'volume': item.get('volume', 1.0),
                    'raw_data': item
                }
                
                self.news_cache.append(news_entry)
            
            logger.info(f"Added {len(news_items)} news items to cache")
            
        except Exception as e:
            logger.error(f"Failed to add news data: {e}")
    
    def extract_features(self, target_timestamp: Optional[datetime] = None) -> NewsFeatureSet:
        """
        Extract news features for a specific timestamp.
        
        Args:
            target_timestamp: Target timestamp for feature extraction
            
        Returns:
            Complete set of news features
        """
        try:
            if target_timestamp is None:
                target_timestamp = datetime.now()
            
            # Filter news data for the target time window
            window_start = target_timestamp - timedelta(hours=self.sentiment_window)
            relevant_news = [
                news for news in self.news_cache
                if window_start <= news['timestamp'] <= target_timestamp
            ]
            
            if not relevant_news:
                # Return neutral features if no news
                return self._create_neutral_features(target_timestamp)
            
            # Extract individual features
            features = []
            
            # Sentiment features
            sentiment_features = self._extract_sentiment_features(relevant_news, target_timestamp)
            features.extend(sentiment_features)
            
            # Volume features
            volume_features = self._extract_volume_features(relevant_news, target_timestamp)
            features.extend(volume_features)
            
            # Timing features
            timing_features = self._extract_timing_features(relevant_news, target_timestamp)
            features.extend(timing_features)
            
            # Source features
            source_features = self._extract_source_features(relevant_news, target_timestamp)
            features.extend(source_features)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(features)
            
            # Create feature set
            feature_set = NewsFeatureSet(
                timestamp=target_timestamp,
                sentiment_score=self._get_feature_value(sentiment_features, 'sentiment_score'),
                sentiment_volume=self._get_feature_value(sentiment_features, 'sentiment_volume'),
                sentiment_momentum=self._get_feature_value(sentiment_features, 'sentiment_momentum'),
                news_volume=self._get_feature_value(volume_features, 'news_volume'),
                news_urgency=self._get_feature_value(timing_features, 'news_urgency'),
                source_diversity=self._get_feature_value(source_features, 'source_diversity'),
                composite_score=composite_score,
                features=features,
                metadata={'total_news_items': len(relevant_news)}
            )
            
            # Store in history
            self.feature_history.append(feature_set)
            self.total_features_extracted += 1
            
            logger.info(f"Extracted {len(features)} features for {target_timestamp}")
            return feature_set
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return self._create_neutral_features(target_timestamp)
    
    def _extract_sentiment_features(self, news_items: List[Dict], target_timestamp: datetime) -> List[NewsFeature]:
        """Extract sentiment-based features."""
        try:
            features = []
            
            # Calculate sentiment scores
            sentiment_scores = [news['sentiment'] for news in news_items]
            sentiment_weights = [news['relevance'] for news in news_items]
            
            # Weighted average sentiment
            if sentiment_weights and sum(sentiment_weights) > 0:
                weighted_sentiment = np.average(sentiment_scores, weights=sentiment_weights)
            else:
                weighted_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            
            # Sentiment score feature
            sentiment_score_feature = NewsFeature(
                feature_id=f"sentiment_score_{target_timestamp.strftime('%Y%m%d_%H%M')}",
                feature_type=NewsFeatureType.SENTIMENT,
                value=weighted_sentiment,
                timestamp=target_timestamp,
                source="sentiment_analysis",
                confidence=0.8,
                metadata={'method': 'weighted_average', 'count': len(sentiment_scores)}
            )
            features.append(sentiment_score_feature)
            
            # Sentiment volume (number of news items)
            sentiment_volume = len(news_items)
            sentiment_volume_feature = NewsFeature(
                feature_id=f"sentiment_volume_{target_timestamp.strftime('%Y%m%d_%H%M')}",
                feature_type=NewsFeatureType.SENTIMENT,
                value=float(sentiment_volume),
                timestamp=target_timestamp,
                source="news_count",
                confidence=1.0,
                metadata={'method': 'count', 'count': sentiment_volume}
            )
            features.append(sentiment_volume_feature)
            
            # Sentiment momentum (change over time)
            if len(news_items) >= 2:
                # Sort by timestamp
                sorted_news = sorted(news_items, key=lambda x: x['timestamp'])
                mid_point = len(sorted_news) // 2
                
                early_sentiment = np.mean([news['sentiment'] for news in sorted_news[:mid_point]])
                late_sentiment = np.mean([news['sentiment'] for news in sorted_news[mid_point:]])
                
                sentiment_momentum = late_sentiment - early_sentiment
            else:
                sentiment_momentum = 0.0
            
            sentiment_momentum_feature = NewsFeature(
                feature_id=f"sentiment_momentum_{target_timestamp.strftime('%Y%m%d_%H%M')}",
                feature_type=NewsFeatureType.SENTIMENT,
                value=sentiment_momentum,
                timestamp=target_timestamp,
                source="momentum_calculation",
                confidence=0.7,
                metadata={'method': 'time_split', 'early_count': len(news_items) // 2}
            )
            features.append(sentiment_momentum_feature)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract sentiment features: {e}")
            return []
    
    def _extract_volume_features(self, news_items: List[Dict], target_timestamp: datetime) -> List[NewsFeature]:
        """Extract volume-based features."""
        try:
            features = []
            
            # News volume (total volume of news items)
            total_volume = sum(news.get('volume', 1.0) for news in news_items)
            
            news_volume_feature = NewsFeature(
                feature_id=f"news_volume_{target_timestamp.strftime('%Y%m%d_%H%M')}",
                feature_type=NewsFeatureType.VOLUME,
                value=total_volume,
                timestamp=target_timestamp,
                source="volume_sum",
                confidence=1.0,
                metadata={'method': 'sum', 'count': len(news_items)}
            )
            features.append(news_volume_feature)
            
            # Volume intensity (volume per hour)
            time_span = max(1, (target_timestamp - min(news['timestamp'] for news in news_items)).total_seconds() / 3600)
            volume_intensity = total_volume / time_span if time_span > 0 else 0.0
            
            volume_intensity_feature = NewsFeature(
                feature_id=f"volume_intensity_{target_timestamp.strftime('%Y%m%d_%H%M')}",
                feature_type=NewsFeatureType.VOLUME,
                value=volume_intensity,
                timestamp=target_timestamp,
                source="intensity_calculation",
                confidence=0.8,
                metadata={'method': 'per_hour', 'time_span_hours': time_span}
            )
            features.append(volume_intensity_feature)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract volume features: {e}")
            return []
    
    def _extract_timing_features(self, news_items: List[Dict], target_timestamp: datetime) -> List[NewsFeature]:
        """Extract timing-based features."""
        try:
            features = []
            
            # News urgency (how recent the news is)
            if news_items:
                # Calculate average time difference
                time_diffs = []
                for news in news_items:
                    time_diff = (target_timestamp - news['timestamp']).total_seconds() / 3600  # hours
                    time_diffs.append(time_diff)
                
                avg_time_diff = np.mean(time_diffs)
                # Convert to urgency score (0 = old, 1 = very recent)
                news_urgency = max(0, 1 - (avg_time_diff / self.sentiment_window))
            else:
                news_urgency = 0.0
            
            news_urgency_feature = NewsFeature(
                feature_id=f"news_urgency_{target_timestamp.strftime('%Y%m%d_%H%M')}",
                feature_type=NewsFeatureType.TIMING,
                value=news_urgency,
                timestamp=target_timestamp,
                source="timing_analysis",
                confidence=0.9,
                metadata={'method': 'time_difference', 'avg_hours_ago': avg_time_diff if news_items else 0}
            )
            features.append(news_urgency_feature)
            
            # Breaking news indicator
            recent_news = [news for news in news_items 
                          if (target_timestamp - news['timestamp']).total_seconds() <= 3600]  # Last hour
            
            breaking_news_score = len(recent_news) / max(1, len(news_items))
            
            breaking_news_feature = NewsFeature(
                feature_id=f"breaking_news_{target_timestamp.strftime('%Y%m%d_%H%M')}",
                feature_type=NewsFeatureType.TIMING,
                value=breaking_news_score,
                timestamp=target_timestamp,
                source="recent_news_ratio",
                confidence=0.8,
                metadata={'method': 'hourly_ratio', 'recent_count': len(recent_news)}
            )
            features.append(breaking_news_feature)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract timing features: {e}")
            return []
    
    def _extract_source_features(self, news_items: List[Dict], target_timestamp: datetime) -> List[NewsFeature]:
        """Extract source-based features."""
        try:
            features = []
            
            # Source diversity (number of unique sources)
            unique_sources = set(news['source'] for news in news_items)
            source_diversity = len(unique_sources) / max(1, len(news_items))
            
            source_diversity_feature = NewsFeature(
                feature_id=f"source_diversity_{target_timestamp.strftime('%Y%m%d_%H%M')}",
                feature_type=NewsFeatureType.SOURCE,
                value=source_diversity,
                timestamp=target_timestamp,
                source="source_analysis",
                confidence=1.0,
                metadata={'method': 'unique_ratio', 'unique_sources': len(unique_sources)}
            )
            features.append(source_diversity_feature)
            
            # Source reliability (weighted by source reputation)
            source_reliability = self._calculate_source_reliability(news_items)
            
            source_reliability_feature = NewsFeature(
                feature_id=f"source_reliability_{target_timestamp.strftime('%Y%m%d_%H%M')}",
                feature_type=NewsFeatureType.SOURCE,
                value=source_reliability,
                timestamp=target_timestamp,
                source="reliability_scoring",
                confidence=0.7,
                metadata={'method': 'weighted_reputation', 'sources': list(unique_sources)}
            )
            features.append(source_reliability_feature)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract source features: {e}")
            return []
    
    def _calculate_source_reliability(self, news_items: List[Dict]) -> float:
        """Calculate source reliability score."""
        try:
            # Simple source reliability mapping (in production, use a proper source database)
            source_reputation = {
                'reuters': 0.9,
                'bloomberg': 0.9,
                'cnbc': 0.8,
                'marketwatch': 0.8,
                'yahoo_finance': 0.7,
                'seeking_alpha': 0.7,
                'unknown': 0.5
            }
            
            total_reliability = 0.0
            total_weight = 0.0
            
            for news in news_items:
                source = news['source'].lower()
                reputation = source_reputation.get(source, source_reputation['unknown'])
                weight = news.get('relevance', 0.5)
                
                total_reliability += reputation * weight
                total_weight += weight
            
            return total_reliability / max(1, total_weight)
            
        except Exception as e:
            logger.error(f"Failed to calculate source reliability: {e}")
            return 0.5
    
    def _calculate_composite_score(self, features: List[NewsFeature]) -> float:
        """Calculate composite news score."""
        try:
            # Extract feature values
            sentiment_score = self._get_feature_value(features, 'sentiment_score')
            sentiment_volume = self._get_feature_value(features, 'sentiment_volume')
            news_volume = self._get_feature_value(features, 'news_volume')
            news_urgency = self._get_feature_value(features, 'news_urgency')
            source_diversity = self._get_feature_value(features, 'source_diversity')
            
            # Normalize values to 0-1 range
            normalized_sentiment = (sentiment_score + 1) / 2  # Convert from [-1,1] to [0,1]
            normalized_volume = min(1.0, news_volume / 100)  # Cap at 100
            normalized_urgency = news_urgency  # Already 0-1
            normalized_diversity = source_diversity  # Already 0-1
            
            # Calculate weighted composite score
            composite_score = (
                self.sentiment_weight * normalized_sentiment +
                self.volume_weight * normalized_volume +
                self.timing_weight * normalized_urgency +
                self.source_weight * normalized_diversity
            )
            
            return max(0.0, min(1.0, composite_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate composite score: {e}")
            return 0.5
    
    def _get_feature_value(self, features: List[NewsFeature], feature_name: str) -> float:
        """Get value of a specific feature by name."""
        try:
            for feature in features:
                if feature_name in feature.feature_id:
                    return feature.value
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get feature value: {e}")
            return 0.0
    
    def _create_neutral_features(self, timestamp: datetime) -> NewsFeatureSet:
        """Create neutral feature set when no news is available."""
        try:
            neutral_features = [
                NewsFeature(
                    feature_id=f"neutral_sentiment_{timestamp.strftime('%Y%m%d_%H%M')}",
                    feature_type=NewsFeatureType.SENTIMENT,
                    value=0.0,
                    timestamp=timestamp,
                    source="neutral_default",
                    confidence=1.0,
                    metadata={'method': 'default'}
                )
            ]
            
            return NewsFeatureSet(
                timestamp=timestamp,
                sentiment_score=0.0,
                sentiment_volume=0.0,
                sentiment_momentum=0.0,
                news_volume=0.0,
                news_urgency=0.0,
                source_diversity=0.0,
                composite_score=0.5,
                features=neutral_features,
                metadata={'total_news_items': 0, 'method': 'neutral_default'}
            )
            
        except Exception as e:
            logger.error(f"Failed to create neutral features: {e}")
            # Return minimal feature set
            return NewsFeatureSet(
                timestamp=timestamp,
                sentiment_score=0.0,
                sentiment_volume=0.0,
                sentiment_momentum=0.0,
                news_volume=0.0,
                news_urgency=0.0,
                source_diversity=0.0,
                composite_score=0.5,
                features=[],
                metadata={'error': str(e)}
            )
    
    def get_feature_history(self, hours_back: int = 24) -> List[NewsFeatureSet]:
        """Get feature history for the specified time period."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            return [
                feature_set for feature_set in self.feature_history
                if feature_set.timestamp >= cutoff_time
            ]
        except Exception as e:
            logger.error(f"Failed to get feature history: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return {
                'total_features_extracted': self.total_features_extracted,
                'news_cache_size': len(self.news_cache),
                'feature_history_size': len(self.feature_history),
                'uptime_seconds': uptime,
                'features_per_hour': self.total_features_extracted / (uptime / 3600) if uptime > 0 else 0,
                'start_time': self.start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {}
    
    def clear_cache(self):
        """Clear news cache and feature history."""
        try:
            self.news_cache.clear()
            self.feature_history.clear()
            logger.info("News feature cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


# Factory function
def create_news_feature_extractor(config: Optional[Dict] = None) -> NewsFeatureExtractor:
    """
    Create news feature extractor instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        News feature extractor instance
    """
    return NewsFeatureExtractor(config)


# Example usage
async def main():
    """Example usage of news feature extractor."""
    config = {
        'sentiment_window': 24,
        'volume_window': 6,
        'momentum_window': 2,
        'urgency_threshold': 0.7,
        'sentiment_weight': 0.4,
        'volume_weight': 0.3,
        'timing_weight': 0.2,
        'source_weight': 0.1
    }
    
    extractor = create_news_feature_extractor(config)
    
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
    
    # Extract features
    features = extractor.extract_features()
    
    print(f"Composite Score: {features.composite_score:.3f}")
    print(f"Sentiment Score: {features.sentiment_score:.3f}")
    print(f"News Volume: {features.news_volume:.3f}")
    
    # Get performance stats
    stats = extractor.get_performance_stats()
    print(f"Performance: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
