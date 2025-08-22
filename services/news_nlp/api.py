#!/usr/bin/env python3
"""
SignaMentis - News NLP API Service

This module provides a FastAPI service for news ingestion, sentiment analysis,
and feature extraction. It integrates all news NLP components.

Author: SignaMentis Team
Version: 2.0.0
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
import hashlib
import hmac
import time
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import news NLP components
import sys
sys.path.append('..')
from ingestors.gdelt import GDELTIngestor, GDELTEventType, GDELTEvent
from ingestors.tradingeconomics import TradingEconomicsIngestor, EconomicIndicatorType
from ingestors.newsapi import NewsAPIIngestor, NewsCategory, NewsLanguage
from ingestors.gnews import GNewsIngestor, GNewsCategory, GNewsLanguage
from nlp.finbert import FinBERTSentimentAnalyzer, SentimentLabel
from features.news_features import NewsFeatureExtractor, NewsFeatureType


# Pydantic models for API requests/responses
class NewsIngestionRequest(BaseModel):
    """News ingestion request model."""
    source: str = Field(..., description="News source (gdelt, tradingeconomics, newsapi, gnews)")
    query: Optional[str] = Field(None, description="Search query")
    category: Optional[str] = Field(None, description="News category")
    country: Optional[str] = Field(None, description="Country code")
    hours_back: int = Field(24, ge=1, le=168, description="Hours to look back")
    max_articles: int = Field(50, ge=1, le=1000, description="Maximum articles to retrieve")


class SentimentAnalysisRequest(BaseModel):
    """Sentiment analysis request model."""
    text: str = Field(..., description="Text to analyze")
    batch_mode: bool = Field(False, description="Enable batch processing")


class BatchSentimentRequest(BaseModel):
    """Batch sentiment analysis request model."""
    texts: List[str] = Field(..., description="List of texts to analyze")
    max_batch_size: int = Field(100, ge=1, le=1000, description="Maximum batch size")


class FeatureExtractionRequest(BaseModel):
    """Feature extraction request model."""
    news_data: List[Dict[str, Any]] = Field(..., description="News data for feature extraction")
    target_timestamp: Optional[str] = Field(None, description="Target timestamp (ISO format)")


class NewsSourceStatus(BaseModel):
    """News source status model."""
    source: str
    status: str
    last_update: str
    total_articles: int
    rate_limit_remaining: int
    error_count: int


class SystemHealth(BaseModel):
    """System health model."""
    status: str
    timestamp: str
    uptime_seconds: float
    total_requests: int
    active_connections: int
    component_status: Dict[str, str]


# API service class
class NewsNLPService:
    """
    News NLP API service.
    
    Features:
    - News ingestion from multiple sources
    - Sentiment analysis with FinBERT
    - Feature extraction and processing
    - Health monitoring and status
    - Rate limiting and authentication
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize News NLP service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Service configuration
        self.api_key = self.config.get('api_key', '')
        self.rate_limit = self.config.get('rate_limit', 1000)
        self.enable_auth = self.config.get('enable_auth', True)
        
        # Initialize components
        self.gdelt_ingestor = None
        self.tradingeconomics_ingestor = None
        self.newsapi_ingestor = None
        self.gnews_ingestor = None
        self.sentiment_analyzer = None
        self.feature_extractor = None
        
        # Service state
        self.start_time = datetime.now()
        self.total_requests = 0
        self.active_connections = 0
        self.error_count = 0
        
        # Performance tracking
        self.request_times = []
        self.source_status = {}
        
        logger.info("News NLP Service initialized")
    
    async def initialize(self):
        """Initialize all service components."""
        try:
            # Initialize ingestors
            self.gdelt_ingestor = GDELTIngestor(self.config.get('gdelt', {}))
            self.tradingeconomics_ingestor = TradingEconomicsIngestor(self.config.get('tradingeconomics', {}))
            self.newsapi_ingestor = NewsAPIIngestor(self.config.get('newsapi', {}))
            self.gnews_ingestor = GNewsIngestor(self.config.get('gnews', {}))
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = FinBERTSentimentAnalyzer(self.config.get('finbert', {}))
            await self.sentiment_analyzer.initialize()
            
            # Initialize feature extractor
            self.feature_extractor = NewsFeatureExtractor(self.config.get('features', {}))
            
            logger.info("All News NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize News NLP service: {e}")
            raise
    
    async def get_news(self, request: NewsIngestionRequest) -> Dict[str, Any]:
        """
        Get news from specified source.
        
        Args:
            request: News ingestion request
            
        Returns:
            News data response
        """
        try:
            self.total_requests += 1
            start_time = time.time()
            
            # Route to appropriate ingestor
            if request.source.lower() == 'gdelt':
                news_data = await self._get_gdelt_news(request)
            elif request.source.lower() == 'tradingeconomics':
                news_data = await self._get_tradingeconomics_news(request)
            elif request.source.lower() == 'newsapi':
                news_data = await self._get_newsapi_news(request)
            elif request.source.lower() == 'gnews':
                news_data = await self._get_gnews_news(request)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported news source: {request.source}")
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.request_times.append(processing_time)
            
            # Update source status
            self._update_source_status(request.source, len(news_data), processing_time)
            
            return {
                'source': request.source,
                'total_articles': len(news_data),
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'data': news_data
            }
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to get news from {request.source}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve news: {str(e)}")
    
    async def analyze_sentiment(self, request: SentimentAnalysisRequest) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            request: Sentiment analysis request
            
        Returns:
            Sentiment analysis result
        """
        try:
            self.total_requests += 1
            start_time = time.time()
            
            if request.batch_mode:
                # Batch processing
                result = await self.sentiment_analyzer.analyze_sentiment_batch([request.text])
                sentiment_data = result.results[0] if result.results else None
            else:
                # Single text processing
                sentiment_data = await self.sentiment_analyzer.analyze_sentiment(request.text)
            
            if not sentiment_data:
                raise HTTPException(status_code=500, detail="Sentiment analysis failed")
            
            processing_time = time.time() - start_time
            
            return {
                'text': request.text,
                'sentiment': sentiment_data.sentiment_label.value,
                'confidence': sentiment_data.confidence_score,
                'scores': {
                    'positive': sentiment_data.positive_score,
                    'negative': sentiment_data.negative_score,
                    'neutral': sentiment_data.neutral_score
                },
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to analyze sentiment: {e}")
            raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")
    
    async def analyze_sentiment_batch(self, request: BatchSentimentRequest) -> Dict[str, Any]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            request: Batch sentiment request
            
        Returns:
            Batch sentiment analysis result
        """
        try:
            self.total_requests += 1
            start_time = time.time()
            
            # Process in batches
            all_results = []
            for i in range(0, len(request.texts), request.max_batch_size):
                batch_texts = request.texts[i:i + request.max_batch_size]
                batch_result = await self.sentiment_analyzer.analyze_sentiment_batch(batch_texts)
                all_results.extend(batch_result.results)
            
            processing_time = time.time() - start_time
            
            # Format results
            formatted_results = []
            for result in all_results:
                formatted_results.append({
                    'text': result.text,
                    'sentiment': result.sentiment_label.value,
                    'confidence': result.confidence_score,
                    'scores': {
                        'positive': result.positive_score,
                        'negative': result.negative_score,
                        'neutral': result.neutral_score
                    }
                })
            
            return {
                'total_texts': len(request.texts),
                'processed_texts': len(formatted_results),
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'results': formatted_results
            }
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to analyze sentiment batch: {e}")
            raise HTTPException(status_code=500, detail=f"Batch sentiment analysis failed: {str(e)}")
    
    async def extract_features(self, request: FeatureExtractionRequest) -> Dict[str, Any]:
        """
        Extract news features.
        
        Args:
            request: Feature extraction request
            
        Returns:
            Extracted features
        """
        try:
            self.total_requests += 1
            start_time = time.time()
            
            # Add news data to extractor
            self.feature_extractor.add_news_data(request.news_data)
            
            # Extract features
            target_timestamp = None
            if request.target_timestamp:
                target_timestamp = datetime.fromisoformat(request.target_timestamp.replace('Z', '+00:00'))
            
            features = self.feature_extractor.extract_features(target_timestamp)
            
            processing_time = time.time() - start_time
            
            return {
                'timestamp': features.timestamp.isoformat(),
                'sentiment_score': features.sentiment_score,
                'sentiment_volume': features.sentiment_volume,
                'sentiment_momentum': features.sentiment_momentum,
                'news_volume': features.news_volume,
                'news_urgency': features.news_urgency,
                'source_diversity': features.source_diversity,
                'composite_score': features.composite_score,
                'total_features': len(features.features),
                'processing_time': processing_time,
                'metadata': features.metadata
            }
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to extract features: {e}")
            raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
    
    async def get_system_health(self) -> SystemHealth:
        """Get system health status."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Check component status
            component_status = {}
            
            if self.gdelt_ingestor:
                component_status['gdelt'] = 'active'
            else:
                component_status['gdelt'] = 'inactive'
            
            if self.tradingeconomics_ingestor:
                component_status['tradingeconomics'] = 'active'
            else:
                component_status['tradingeconomics'] = 'inactive'
            
            if self.newsapi_ingestor:
                component_status['newsapi'] = 'active'
            else:
                component_status['newsapi'] = 'inactive'
            
            if self.gnews_ingestor:
                component_status['gnews'] = 'active'
            else:
                component_status['gnews'] = 'inactive'
            
            if self.sentiment_analyzer:
                component_status['sentiment_analyzer'] = 'active'
            else:
                component_status['sentiment_analyzer'] = 'inactive'
            
            if self.feature_extractor:
                component_status['feature_extractor'] = 'active'
            else:
                component_status['feature_extractor'] = 'inactive'
            
            # Determine overall status
            overall_status = 'healthy' if all(status == 'active' for status in component_status.values()) else 'degraded'
            
            return SystemHealth(
                status=overall_status,
                timestamp=datetime.now().isoformat(),
                uptime_seconds=uptime,
                total_requests=self.total_requests,
                active_connections=self.active_connections,
                component_status=component_status
            )
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return SystemHealth(
                status='error',
                timestamp=datetime.now().isoformat(),
                uptime_seconds=0.0,
                total_requests=0,
                active_connections=0,
                component_status={}
            )
    
    async def get_source_status(self) -> List[NewsSourceStatus]:
        """Get status of all news sources."""
        try:
            status_list = []
            
            # GDELT status
            if self.gdelt_ingestor:
                gdelt_stats = self.gdelt_ingestor.get_performance_stats()
                status_list.append(NewsSourceStatus(
                    source='gdelt',
                    status='active',
                    last_update=datetime.now().isoformat(),
                    total_articles=len(self.gdelt_ingestor.events_cache),
                    rate_limit_remaining=gdelt_stats.get('rate_limit_remaining', 0),
                    error_count=0
                ))
            
            # TradingEconomics status
            if self.tradingeconomics_ingestor:
                te_stats = self.tradingeconomics_ingestor.get_performance_stats()
                status_list.append(NewsSourceStatus(
                    source='tradingeconomics',
                    status='active',
                    last_update=datetime.now().isoformat(),
                    total_articles=len(self.tradingeconomics_ingestor.indicators_cache),
                    rate_limit_remaining=te_stats.get('rate_limit_remaining', 0),
                    error_count=0
                ))
            
            # NewsAPI status
            if self.newsapi_ingestor:
                newsapi_stats = self.newsapi_ingestor.get_performance_stats()
                status_list.append(NewsSourceStatus(
                    source='newsapi',
                    status='active',
                    last_update=datetime.now().isoformat(),
                    total_articles=len(self.newsapi_ingestor.articles_cache),
                    rate_limit_remaining=newsapi_stats.get('rate_limit_remaining', 0),
                    error_count=0
                ))
            
            # GNews status
            if self.gnews_ingestor:
                gnews_stats = self.gnews_ingestor.get_performance_stats()
                status_list.append(NewsSourceStatus(
                    source='gnews',
                    status='active',
                    last_update=datetime.now().isoformat(),
                    total_articles=len(self.gnews_ingestor.articles_cache),
                    rate_limit_remaining=gnews_stats.get('rate_limit_remaining', 0),
                    error_count=0
                ))
            
            return status_list
            
        except Exception as e:
            logger.error(f"Failed to get source status: {e}")
            return []
    
    async def _get_gdelt_news(self, request: NewsIngestionRequest) -> List[Dict[str, Any]]:
        """Get news from GDELT."""
        try:
            # Determine event types
            event_types = None
            if request.category:
                # Map category to GDELT event types
                category_mapping = {
                    'economic': [GDELTEventType.ECONOMIC],
                    'financial': [GDELTEventType.FINANCIAL],
                    'political': [GDELTEventType.POLITICAL],
                    'trade': [GDELTEventType.TRADE],
                    'monetary': [GDELTEventType.MONETARY]
                }
                event_types = category_mapping.get(request.category.lower(), None)
            
            # Get events
            if request.query:
                # Search by query
                events = await self.gdelt_ingestor.get_recent_events(
                    event_types=event_types,
                    limit=request.max_articles,
                    time_window=request.hours_back
                )
            else:
                # Get recent events
                events = await self.gdelt_ingestor.get_recent_events(
                    event_types=event_types,
                    limit=request.max_articles,
                    time_window=request.hours_back
                )
            
            # Convert to dictionary format
            news_data = []
            for event in events:
                news_data.append({
                    'id': event.event_id,
                    'title': f"{event.event_type.value} Event",
                    'description': f"Event involving {event.actor1_name} and {event.actor2_name}",
                    'timestamp': event.event_time.isoformat(),
                    'source': 'GDELT',
                    'sentiment_score': event.avg_tone / 10.0,  # Normalize to [-1, 1]
                    'sentiment_label': 'positive' if event.avg_tone > 0 else 'negative' if event.avg_tone < 0 else 'neutral',
                    'relevance_score': 0.8,
                    'volume': 1.0,
                    'raw_data': event.raw_data
                })
            
            return news_data
            
        except Exception as e:
            logger.error(f"Failed to get GDELT news: {e}")
            return []
    
    async def _get_tradingeconomics_news(self, request: NewsIngestionRequest) -> List[Dict[str, Any]]:
        """Get news from TradingEconomics."""
        try:
            # Get economic indicators
            indicators = await self.tradingeconomics_ingestor.get_economic_indicators(
                country=request.country
            )
            
            # Convert to news format
            news_data = []
            for indicator in indicators:
                # Calculate sentiment based on change
                if indicator.forecast_value is not None and indicator.last_value is not None:
                    change = indicator.last_value - indicator.forecast_value
                    sentiment_score = np.tanh(change / max(abs(indicator.last_value), 1))  # Normalize change
                else:
                    sentiment_score = 0.0
                
                news_data.append({
                    'id': indicator.indicator_id,
                    'title': f"{indicator.indicator} - {indicator.country}",
                    'description': f"Current: {indicator.last_value} {indicator.unit}, Previous: {indicator.previous_value}",
                    'timestamp': indicator.last_update.isoformat(),
                    'source': indicator.source,
                    'sentiment_score': sentiment_score,
                    'sentiment_label': 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral',
                    'relevance_score': 0.9,
                    'volume': 1.0,
                    'raw_data': indicator.raw_data
                })
            
            return news_data
            
        except Exception as e:
            logger.error(f"Failed to get TradingEconomics news: {e}")
            return []
    
    async def _get_newsapi_news(self, request: NewsIngestionRequest) -> List[Dict[str, Any]]:
        """Get news from NewsAPI."""
        try:
            # Determine category
            category = None
            if request.category:
                category_mapping = {
                    'business': NewsCategory.BUSINESS,
                    'technology': NewsCategory.TECHNOLOGY,
                    'general': NewsCategory.GENERAL
                }
                category = category_mapping.get(request.category.lower(), NewsCategory.BUSINESS)
            
            # Get articles
            if request.query:
                articles = await self.newsapi_ingestor.get_everything(
                    q=request.query,
                    language=NewsLanguage.ENGLISH,
                    sort_by='publishedAt',
                    page_size=min(request.max_articles, 100)
                )
            else:
                articles = await self.newsapi_ingestor.get_top_headlines(
                    country=request.country,
                    category=category,
                    page_size=min(request.max_articles, 100)
                )
            
            # Convert to news format
            news_data = []
            for article in articles:
                news_data.append({
                    'id': article.article_id,
                    'title': article.title,
                    'description': article.description,
                    'timestamp': article.published_at.isoformat(),
                    'source': article.source_name,
                    'sentiment_score': article.sentiment_score or 0.0,
                    'sentiment_label': article.sentiment_label.value if article.sentiment_label else 'neutral',
                    'relevance_score': article.relevance_score or 0.5,
                    'volume': 1.0,
                    'raw_data': article.raw_data
                })
            
            return news_data
            
        except Exception as e:
            logger.error(f"Failed to get NewsAPI news: {e}")
            return []
    
    async def _get_gnews_news(self, request: NewsIngestionRequest) -> List[Dict[str, Any]]:
        """Get news from GNews."""
        try:
            # Determine category
            category = None
            if request.category:
                category_mapping = {
                    'business': GNewsCategory.BUSINESS,
                    'technology': GNewsCategory.TECHNOLOGY,
                    'general': GNewsCategory.GENERAL
                }
                category = category_mapping.get(request.category.lower(), GNewsCategory.BUSINESS)
            
            # Get articles
            if request.query:
                articles = await self.gnews_ingestor.search_articles(
                    query=request.query,
                    country=request.country,
                    language=GNewsLanguage.ENGLISH,
                    max_articles=min(request.max_articles, 100)
                )
            else:
                articles = await self.gnews_ingestor.get_top_headlines(
                    country=request.country,
                    category=category,
                    max_articles=min(request.max_articles, 100)
                )
            
            # Convert to news format
            news_data = []
            for article in articles:
                news_data.append({
                    'id': article.article_id,
                    'title': article.title,
                    'description': article.description,
                    'timestamp': article.published_at.isoformat(),
                    'source': article.source_name,
                    'sentiment_score': article.sentiment_score or 0.0,
                    'sentiment_label': article.sentiment_label.value if article.sentiment_label else 'neutral',
                    'relevance_score': article.relevance_score or 0.5,
                    'volume': 1.0,
                    'raw_data': article.raw_data
                })
            
            return news_data
            
        except Exception as e:
            logger.error(f"Failed to get GNews news: {e}")
            return []
    
    def _update_source_status(self, source: str, article_count: int, processing_time: float):
        """Update source status tracking."""
        try:
            if source not in self.source_status:
                self.source_status[source] = {
                    'total_articles': 0,
                    'total_processing_time': 0.0,
                    'request_count': 0,
                    'last_update': datetime.now()
                }
            
            self.source_status[source]['total_articles'] += article_count
            self.source_status[source]['total_processing_time'] += processing_time
            self.source_status[source]['request_count'] += 1
            self.source_status[source]['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to update source status: {e}")
    
    async def close(self):
        """Close the service and cleanup resources."""
        try:
            if self.gdelt_ingestor:
                await self.gdelt_ingestor.close()
            if self.tradingeconomics_ingestor:
                await self.tradingeconomics_ingestor.close()
            if self.newsapi_ingestor:
                await self.newsapi_ingestor.close()
            if self.gnews_ingestor:
                await self.gnews_ingestor.close()
            if self.sentiment_analyzer:
                await self.sentiment_analyzer.close()
            
            logger.info("News NLP Service closed")
            
        except Exception as e:
            logger.error(f"Error closing service: {e}")


# Create FastAPI app
app = FastAPI(
    title="SignaMentis News NLP API",
    description="News ingestion, sentiment analysis, and feature extraction service",
    version="2.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Security
security = HTTPBearer()

# Global service instance
news_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    global news_service
    
    try:
        # Load configuration
        config = {
            'api_key': os.getenv('NEWS_NLP_API_KEY', ''),
            'rate_limit': int(os.getenv('NEWS_NLP_RATE_LIMIT', '1000')),
            'enable_auth': os.getenv('NEWS_NLP_ENABLE_AUTH', 'true').lower() == 'true',
            'gdelt': {
                'api_key': os.getenv('GDELT_API_KEY', ''),
                'rate_limit': int(os.getenv('GDELT_RATE_LIMIT', '1000'))
            },
            'tradingeconomics': {
                'api_key': os.getenv('TRADINGECONOMICS_API_KEY', ''),
                'rate_limit': int(os.getenv('TRADINGECONOMICS_RATE_LIMIT', '1000'))
            },
            'newsapi': {
                'api_key': os.getenv('NEWSAPI_API_KEY', ''),
                'rate_limit': int(os.getenv('NEWSAPI_RATE_LIMIT', '100'))
            },
            'gnews': {
                'api_key': os.getenv('GNEWS_API_KEY', ''),
                'rate_limit': int(os.getenv('GNEWS_RATE_LIMIT', '100'))
            },
            'finbert': {
                'model_name': 'ProsusAI/finbert',
                'max_length': 512,
                'batch_size': 8
            },
            'features': {
                'sentiment_window': 24,
                'volume_window': 6,
                'momentum_window': 2
            }
        }
        
        # Initialize service
        news_service = NewsNLPService(config)
        await news_service.initialize()
        
        logger.info("News NLP API service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start News NLP API service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global news_service
    
    if news_service:
        await news_service.close()
        logger.info("News NLP API service shutdown")


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "SignaMentis News NLP API", "version": "2.0.0"}


@app.get("/health", response_model=SystemHealth)
async def health_check():
    """Health check endpoint."""
    if not news_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await news_service.get_system_health()


@app.get("/status", response_model=List[NewsSourceStatus])
async def source_status():
    """Get status of all news sources."""
    if not news_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await news_service.get_source_status()


@app.post("/news/ingest", response_model=Dict[str, Any])
async def ingest_news(request: NewsIngestionRequest):
    """Ingest news from specified source."""
    if not news_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await news_service.get_news(request)


@app.post("/sentiment/analyze", response_model=Dict[str, Any])
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analyze sentiment of text."""
    if not news_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await news_service.analyze_sentiment(request)


@app.post("/sentiment/batch", response_model=Dict[str, Any])
async def analyze_sentiment_batch(request: BatchSentimentRequest):
    """Analyze sentiment of multiple texts."""
    if not news_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await news_service.analyze_sentiment_batch(request)


@app.post("/features/extract", response_model=Dict[str, Any])
async def extract_features(request: FeatureExtractionRequest):
    """Extract news features."""
    if not news_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await news_service.extract_features(request)


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    if not news_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Get performance stats from components
    stats = {
        'service': {
            'total_requests': news_service.total_requests,
            'error_count': news_service.error_count,
            'uptime_seconds': (datetime.now() - news_service.start_time).total_seconds()
        },
        'sentiment_analyzer': news_service.sentiment_analyzer.get_performance_stats() if news_service.sentiment_analyzer else {},
        'feature_extractor': news_service.feature_extractor.get_performance_stats() if news_service.feature_extractor else {}
    }
    
    return stats


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
