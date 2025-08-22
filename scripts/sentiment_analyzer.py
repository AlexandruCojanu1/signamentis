#!/usr/bin/env python3
"""
SignaMentis Sentiment Analyzer for XAUUSD

This module analyzes sentiment from various sources:
- Social media (Twitter, Reddit)
- News articles
- Economic indicators
- Market sentiment indices

Author: SignaMentis Team
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import re
import requests
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class SentimentData:
    """Container for sentiment analysis data."""
    source: str
    content: str
    sentiment_score: float
    sentiment_polarity: str
    confidence: float
    timestamp: datetime
    keywords: List[str]
    metadata: Optional[Dict] = None


@dataclass
class SentimentSummary:
    """Container for sentiment summary."""
    overall_sentiment: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    confidence: float
    source_breakdown: Dict[str, float]
    keyword_analysis: Dict[str, float]
    timestamp: datetime


class SentimentAnalyzer:
    """
    Comprehensive sentiment analyzer for XAUUSD trading.
    
    Analyzes sentiment from multiple sources and provides
    actionable insights for trading decisions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sources = self.config.get('sources', ['news', 'social', 'economic'])
        self.update_frequency = self.config.get('update_frequency', 300)  # 5 minutes
        self.sentiment_history = []
        self.keywords = self.config.get('keywords', [
            'gold', 'XAUUSD', 'precious metals', 'inflation', 'federal reserve',
            'dollar', 'USD', 'safe haven', 'economic crisis', 'geopolitical',
            'central banks', 'interest rates', 'quantitative easing'
        ])
        
        # Database for storing sentiment data
        self.db_path = Path("data/sentiment_analyzer.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Initialize sentiment models
        self._init_sentiment_models()
        
        logger.info("Sentiment Analyzer initialized")
    
    def _init_database(self):
        """Initialize SQLite database for storing sentiment data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create sentiment_data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sentiment_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT NOT NULL,
                        content TEXT,
                        sentiment_score REAL,
                        sentiment_polarity TEXT,
                        confidence REAL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        keywords TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Create sentiment_summaries table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sentiment_summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        overall_sentiment REAL,
                        positive_ratio REAL,
                        negative_ratio REAL,
                        neutral_ratio REAL,
                        confidence REAL,
                        source_breakdown TEXT,
                        keyword_analysis TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create keywords table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS keywords (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        keyword TEXT UNIQUE NOT NULL,
                        frequency INTEGER DEFAULT 0,
                        avg_sentiment REAL DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Sentiment database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing sentiment database: {e}")
    
    def _init_sentiment_models(self):
        """Initialize sentiment analysis models."""
        try:
            # Check if TextBlob is available
            import textblob
            self.textblob_available = True
            logger.info("Sentiment models initialized (TextBlob)")
        except ImportError:
            self.textblob_available = False
            logger.warning("TextBlob not available, using fallback sentiment analysis")
        except Exception as e:
            self.textblob_available = False
            logger.warning(f"Could not initialize sentiment models: {e}")
    
    async def analyze_news_sentiment(self, news_data: List[Dict]) -> List[SentimentData]:
        """
        Analyze sentiment from news articles.
        
        Args:
            news_data: List of news articles
            
        Returns:
            List of SentimentData objects
        """
        sentiment_results = []
        
        for article in news_data:
            try:
                # Extract text content
                content = article.get('title', '') + ' ' + article.get('description', '')
                
                # Analyze sentiment
                sentiment_score, polarity, confidence = self._analyze_text_sentiment(content)
                
                # Extract keywords
                extracted_keywords = self._extract_keywords(content)
                
                # Create sentiment data
                sentiment_data = SentimentData(
                    source='news',
                    content=content[:200] + '...' if len(content) > 200 else content,
                    sentiment_score=sentiment_score,
                    sentiment_polarity=polarity,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    keywords=extracted_keywords,
                    metadata={
                        'url': article.get('url'),
                        'published_at': article.get('published_at'),
                        'source_name': article.get('source_name')
                    }
                )
                
                sentiment_results.append(sentiment_data)
                
            except Exception as e:
                logger.error(f"Error analyzing news sentiment: {e}")
                continue
        
        return sentiment_results
    
    async def analyze_social_sentiment(self, social_data: List[Dict]) -> List[SentimentData]:
        """
        Analyze sentiment from social media posts.
        
        Args:
            social_data: List of social media posts
            
        Returns:
            List of SentimentData objects
        """
        sentiment_results = []
        
        for post in social_data:
            try:
                # Extract text content
                content = post.get('text', '') + ' ' + post.get('hashtags', '')
                
                # Analyze sentiment
                sentiment_score, polarity, confidence = self._analyze_text_sentiment(content)
                
                # Extract keywords
                extracted_keywords = self._extract_keywords(content)
                
                # Create sentiment data
                sentiment_data = SentimentData(
                    source='social',
                    content=content[:200] + '...' if len(content) > 200 else content,
                    sentiment_score=sentiment_score,
                    sentiment_polarity=polarity,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    keywords=extracted_keywords,
                    metadata={
                        'platform': post.get('platform'),
                        'user_id': post.get('user_id'),
                        'post_id': post.get('post_id'),
                        'engagement': post.get('engagement', {})
                    }
                )
                
                sentiment_results.append(sentiment_data)
                
            except Exception as e:
                logger.error(f"Error analyzing social sentiment: {e}")
                continue
        
        return sentiment_results
    
    async def analyze_economic_sentiment(self, economic_data: List[Dict]) -> List[SentimentData]:
        """
        Analyze sentiment from economic indicators and reports.
        
        Args:
            economic_data: List of economic data points
            
        Returns:
            List of SentimentData objects
        """
        sentiment_results = []
        
        for data_point in economic_data:
            try:
                # Extract text content
                content = data_point.get('description', '') + ' ' + str(data_point.get('value', ''))
                
                # Analyze sentiment
                sentiment_score, polarity, confidence = self._analyze_text_sentiment(content)
                
                # Extract keywords
                extracted_keywords = self._extract_keywords(content)
                
                # Create sentiment data
                sentiment_data = SentimentData(
                    source='economic',
                    content=content[:200] + '...' if len(content) > 200 else content,
                    sentiment_score=sentiment_score,
                    sentiment_polarity=polarity,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    keywords=extracted_keywords,
                    metadata={
                        'indicator': data_point.get('indicator'),
                        'value': data_point.get('value'),
                        'previous': data_point.get('previous'),
                        'forecast': data_point.get('forecast'),
                        'currency': data_point.get('currency')
                    }
                )
                
                sentiment_results.append(sentiment_data)
                
            except Exception as e:
                logger.error(f"Error analyzing economic sentiment: {e}")
                continue
        
        return sentiment_results
    
    def _analyze_text_sentiment(self, text: str) -> Tuple[float, str, float]:
        """
        Analyze sentiment of text using TextBlob or fallback method.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment_score, polarity, confidence)
        """
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            if self.textblob_available:
                # Use TextBlob for sentiment analysis
                import textblob
                blob = textblob.TextBlob(cleaned_text)
                
                # Get polarity (-1 to 1)
                polarity = blob.sentiment.polarity
                
                # Get subjectivity (0 to 1)
                subjectivity = blob.sentiment.subjectivity
                
                # Convert polarity to sentiment score (0 to 100)
                sentiment_score = (polarity + 1) * 50
                
                # Determine sentiment category
                if polarity > 0.1:
                    sentiment_polarity = 'positive'
                elif polarity < -0.1:
                    sentiment_polarity = 'negative'
                else:
                    sentiment_polarity = 'neutral'
                
                # Calculate confidence based on subjectivity
                confidence = 1.0 - subjectivity  # Higher subjectivity = lower confidence
                
                return sentiment_score, sentiment_polarity, confidence
            else:
                # Fallback sentiment analysis using simple keyword matching
                return self._fallback_sentiment_analysis(cleaned_text)
            
        except Exception as e:
            logger.error(f"Error in text sentiment analysis: {e}")
            return 50.0, 'neutral', 0.5  # Default neutral values
    
    def _fallback_sentiment_analysis(self, text: str) -> Tuple[float, str, float]:
        """Fallback sentiment analysis using keyword matching."""
        try:
            # Simple keyword-based sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'positive', 'bullish', 'up', 'rise', 'gain', 'profit', 'strong']
            negative_words = ['bad', 'terrible', 'negative', 'bearish', 'down', 'fall', 'loss', 'weak', 'crash', 'decline']
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            # Calculate simple polarity
            if positive_count > negative_count:
                polarity = 0.3
                sentiment_polarity = 'positive'
            elif negative_count > positive_count:
                polarity = -0.3
                sentiment_polarity = 'negative'
            else:
                polarity = 0.0
                sentiment_polarity = 'neutral'
            
            # Convert to sentiment score
            sentiment_score = (polarity + 1) * 50
            
            # Lower confidence for fallback method
            confidence = 0.6
            
            return sentiment_score, sentiment_polarity, confidence
            
        except Exception as e:
            logger.error(f"Error in fallback sentiment analysis: {e}")
            return 50.0, 'neutral', 0.5
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis."""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove special characters but keep spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Split into words
            words = cleaned_text.split()
            
            # Filter for relevant keywords
            relevant_keywords = []
            for word in words:
                if word.lower() in [kw.lower() for kw in self.keywords]:
                    relevant_keywords.append(word.lower())
            
            # Remove duplicates
            relevant_keywords = list(set(relevant_keywords))
            
            return relevant_keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    async def get_sentiment_summary(self, 
                                  time_window: str = "1h",
                                  sources: Optional[List[str]] = None) -> SentimentSummary:
        """
        Get sentiment summary for a time window.
        
        Args:
            time_window: Time window (e.g., "1h", "4h", "1d")
            sources: List of sources to include
            
        Returns:
            SentimentSummary object
        """
        try:
            # Calculate time range
            end_time = datetime.now()
            if time_window == "1h":
                start_time = end_time - timedelta(hours=1)
            elif time_window == "4h":
                start_time = end_time - timedelta(hours=4)
            elif time_window == "1d":
                start_time = end_time - timedelta(days=1)
            else:
                start_time = end_time - timedelta(hours=1)
            
            # Get sentiment data from database
            sentiment_data = await self._get_sentiment_data_from_db(start_time, end_time, sources)
            
            if not sentiment_data:
                return self._create_empty_sentiment_summary()
            
            # Calculate summary statistics
            overall_sentiment = np.mean([d.sentiment_score for d in sentiment_data])
            
            # Calculate ratios
            total_count = len(sentiment_data)
            positive_count = sum(1 for d in sentiment_data if d.sentiment_polarity == 'positive')
            negative_count = sum(1 for d in sentiment_data if d.sentiment_polarity == 'negative')
            neutral_count = total_count - positive_count - negative_count
            
            positive_ratio = positive_count / total_count if total_count > 0 else 0
            negative_ratio = negative_count / total_count if total_count > 0 else 0
            neutral_ratio = neutral_count / total_count if total_count > 0 else 0
            
            # Calculate confidence
            confidence = np.mean([d.confidence for d in sentiment_data])
            
            # Source breakdown
            source_breakdown = {}
            for data in sentiment_data:
                source = data.source
                if source not in source_breakdown:
                    source_breakdown[source] = []
                source_breakdown[source].append(data.sentiment_score)
            
            for source in source_breakdown:
                source_breakdown[source] = np.mean(source_breakdown[source])
            
            # Keyword analysis
            keyword_analysis = {}
            for data in sentiment_data:
                for keyword in data.keywords:
                    if keyword not in keyword_analysis:
                        keyword_analysis[keyword] = []
                    keyword_analysis[keyword].append(data.sentiment_score)
            
            for keyword in keyword_analysis:
                keyword_analysis[keyword] = np.mean(keyword_analysis[keyword])
            
            # Create summary
            summary = SentimentSummary(
                overall_sentiment=overall_sentiment,
                positive_ratio=positive_ratio,
                negative_ratio=negative_ratio,
                neutral_ratio=neutral_ratio,
                confidence=confidence,
                source_breakdown=source_breakdown,
                keyword_analysis=keyword_analysis,
                timestamp=datetime.now()
            )
            
            # Store summary in database
            await self._store_sentiment_summary(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return self._create_empty_sentiment_summary()
    
    def _create_empty_sentiment_summary(self) -> SentimentSummary:
        """Create empty sentiment summary."""
        return SentimentSummary(
            overall_sentiment=50.0,
            positive_ratio=0.0,
            negative_ratio=0.0,
            neutral_ratio=1.0,
            confidence=0.0,
            source_breakdown={},
            keyword_analysis={},
            timestamp=datetime.now()
        )
    
    async def _get_sentiment_data_from_db(self, 
                                        start_time: datetime,
                                        end_time: datetime,
                                        sources: Optional[List[str]] = None) -> List[SentimentData]:
        """Get sentiment data from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = '''
                    SELECT source, content, sentiment_score, sentiment_polarity, 
                           confidence, timestamp, keywords, metadata
                    FROM sentiment_data
                    WHERE timestamp BETWEEN ? AND ?
                '''
                params = [start_time, end_time]
                
                if sources:
                    placeholders = ','.join(['?' for _ in sources])
                    query += f' AND source IN ({placeholders})'
                    params.extend(sources)
                
                query += ' ORDER BY timestamp DESC'
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to SentimentData objects
                sentiment_data = []
                for row in rows:
                    try:
                        data = SentimentData(
                            source=row[0],
                            content=row[1],
                            sentiment_score=row[2],
                            sentiment_polarity=row[3],
                            confidence=row[4],
                            timestamp=datetime.fromisoformat(row[5]),
                            keywords=json.loads(row[6]) if row[6] else [],
                            metadata=json.loads(row[7]) if row[7] else {}
                        )
                        sentiment_data.append(data)
                    except Exception as e:
                        logger.warning(f"Error parsing sentiment data row: {e}")
                        continue
                
                return sentiment_data
                
        except Exception as e:
            logger.error(f"Error getting sentiment data from database: {e}")
            return []
    
    async def _store_sentiment_summary(self, summary: SentimentSummary):
        """Store sentiment summary in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sentiment_summaries 
                    (overall_sentiment, positive_ratio, negative_ratio, neutral_ratio,
                     confidence, source_breakdown, keyword_analysis, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    summary.overall_sentiment,
                    summary.positive_ratio,
                    summary.negative_ratio,
                    summary.neutral_ratio,
                    summary.confidence,
                    json.dumps(summary.source_breakdown),
                    json.dumps(summary.keyword_analysis),
                    summary.timestamp
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing sentiment summary: {e}")
    
    async def store_sentiment_data(self, sentiment_data: List[SentimentData]):
        """Store sentiment data in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for data in sentiment_data:
                    cursor.execute('''
                        INSERT INTO sentiment_data 
                        (source, content, sentiment_score, sentiment_polarity,
                         confidence, timestamp, keywords, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data.source,
                        data.content,
                        data.sentiment_score,
                        data.sentiment_polarity,
                        data.confidence,
                        data.timestamp,
                        json.dumps(data.keywords),
                        json.dumps(data.metadata) if data.metadata else None
                    ))
                
                conn.commit()
                logger.info(f"Stored {len(sentiment_data)} sentiment data points")
                
        except Exception as e:
            logger.error(f"Error storing sentiment data: {e}")
    
    def get_sentiment_trading_signal(self, sentiment_summary: SentimentSummary) -> Dict:
        """
        Generate trading signal based on sentiment analysis.
        
        Args:
            sentiment_summary: Sentiment summary
            
        Returns:
            Trading signal dictionary
        """
        try:
            signal = {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasoning': [],
                'sentiment_score': sentiment_summary.overall_sentiment,
                'timestamp': datetime.now()
            }
            
            # Analyze sentiment for trading decisions
            if sentiment_summary.overall_sentiment > 70:
                signal['action'] = 'BUY'
                signal['confidence'] = min(sentiment_summary.confidence * 0.8, 0.9)
                signal['reasoning'].append('Strong positive sentiment')
                
            elif sentiment_summary.overall_sentiment < 30:
                signal['action'] = 'SELL'
                signal['confidence'] = min(sentiment_summary.confidence * 0.8, 0.9)
                signal['reasoning'].append('Strong negative sentiment')
                
            else:
                signal['action'] = 'HOLD'
                signal['confidence'] = sentiment_summary.confidence * 0.5
                signal['reasoning'].append('Neutral sentiment')
            
            # Consider source breakdown
            if sentiment_summary.source_breakdown:
                best_source = max(sentiment_summary.source_breakdown.items(), key=lambda x: x[1])
                if best_source[1] > 70:
                    signal['reasoning'].append(f'Strong {best_source[0]} sentiment')
                elif best_source[1] < 30:
                    signal['reasoning'].append(f'Weak {best_source[0]} sentiment')
            
            # Consider keyword analysis
            if sentiment_summary.keyword_analysis:
                positive_keywords = [kw for kw, score in sentiment_summary.keyword_analysis.items() if score > 60]
                negative_keywords = [kw for kw, score in sentiment_summary.keyword_analysis.items() if score < 40]
                
                if positive_keywords:
                    signal['reasoning'].append(f'Positive keywords: {", ".join(positive_keywords[:3])}')
                if negative_keywords:
                    signal['reasoning'].append(f'Negative keywords: {", ".join(negative_keywords[:3])}')
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasoning': ['Error in sentiment analysis'],
                'sentiment_score': 50.0,
                'timestamp': datetime.now()
            }


def create_sentiment_analyzer(config: Optional[Dict] = None) -> SentimentAnalyzer:
    """
    Create sentiment analyzer instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SentimentAnalyzer instance
    """
    return SentimentAnalyzer(config)


if __name__ == "__main__":
    # Test the sentiment analyzer
    async def test_sentiment_analyzer():
        analyzer = create_sentiment_analyzer()
        
        # Test data
        test_news = [
            {
                'title': 'Gold prices surge on inflation concerns',
                'description': 'Gold prices reached new highs as investors seek safe haven assets',
                'url': 'https://example.com/news1',
                'published_at': '2024-01-01T10:00:00Z',
                'source_name': 'Financial Times'
            },
            {
                'title': 'Federal Reserve signals rate cuts',
                'description': 'Central bank suggests potential interest rate reductions',
                'url': 'https://example.com/news2',
                'published_at': '2024-01-01T11:00:00Z',
                'source_name': 'Reuters'
            }
        ]
        
        # Analyze sentiment
        news_sentiment = await analyzer.analyze_news_sentiment(test_news)
        print(f"Analyzed {len(news_sentiment)} news articles")
        
        # Store data
        await analyzer.store_sentiment_data(news_sentiment)
        
        # Get summary
        summary = await analyzer.get_sentiment_summary("1h")
        print(f"Overall sentiment: {summary.overall_sentiment:.2f}")
        print(f"Positive ratio: {summary.positive_ratio:.2%}")
        
        # Generate trading signal
        signal = analyzer.get_sentiment_trading_signal(summary)
        print(f"Trading signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
    
    # Run test
    asyncio.run(test_sentiment_analyzer())
