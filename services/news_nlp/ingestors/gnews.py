#!/usr/bin/env python3
"""
SignaMentis - GNews Ingestor Service

This module handles ingestion of financial news from GNews (Google News).
It provides access to real-time news articles with advanced filtering and search.

Author: SignaMentis Team
Version: 2.0.0
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging
import time
import json
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNewsCategory(Enum):
    """GNews categories enumeration."""
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    HEALTH = "health"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    GENERAL = "general"
    WORLD = "world"
    NATION = "nation"
    TOP = "top"


class GNewsLanguage(Enum):
    """GNews languages enumeration."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"


@dataclass
class GNewsArticle:
    """GNews article data container."""
    article_id: str
    title: str
    description: str
    content: str
    url: str
    image_url: Optional[str]
    published_at: datetime
    source_name: str
    source_url: str
    author: Optional[str]
    category: GNewsCategory
    language: GNewsLanguage
    sentiment_score: Optional[float]
    relevance_score: Optional[float]
    keywords: List[str]
    country: Optional[str]
    raw_data: Dict[str, Any]


class GNewsIngestor:
    """
    GNews ingestor service.
    
    Features:
    - Real-time news articles
    - Category-based filtering
    - Language support
    - Country-specific news
    - Advanced search capabilities
    - Rate limiting and error handling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize GNews ingestor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # API configuration
        self.base_url = self.config.get('base_url', 'https://gnews.io/api/v4')
        self.api_key = self.config.get('api_key', '')
        self.rate_limit = self.config.get('rate_limit', 100)  # requests per day (free tier)
        self.timeout = self.config.get('timeout', 30)
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = datetime.now()
        self.rate_limit_reset = datetime.now() + timedelta(days=1)
        
        # Data storage
        self.articles_cache = {}
        self.last_update = datetime.now()
        
        # Session management
        self.session = None
        
        logger.info("GNews Ingestor initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def _check_rate_limit(self) -> bool:
        """Check if rate limit allows new request."""
        now = datetime.now()
        
        # Reset counter if day has passed
        if now >= self.rate_limit_reset:
            self.request_count = 0
            self.rate_limit_reset = now + timedelta(days=1)
        
        # Check if limit exceeded
        if self.request_count >= self.rate_limit:
            logger.warning("GNews rate limit exceeded, waiting for reset")
            return False
        
        return True
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make API request with rate limiting and error handling.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response data or None if failed
        """
        # Check rate limit
        if not await self._check_rate_limit():
            return None
        
        try:
            session = await self._get_session()
            
            # Add API key
            params['token'] = self.api_key
            
            # Build URL
            url = f"{self.base_url}/{endpoint}"
            query_string = urlencode(params)
            full_url = f"{url}?{query_string}"
            
            # Make request
            async with session.get(full_url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.request_count += 1
                    return data
                elif response.status == 429:  # Rate limited
                    logger.warning("GNews rate limited")
                    self.rate_limit_reset = datetime.now() + timedelta(hours=1)
                    return None
                elif response.status == 401:  # Unauthorized
                    logger.error("GNews API key invalid")
                    return None
                elif response.status == 400:  # Bad request
                    logger.error("GNews bad request")
                    return None
                else:
                    logger.error(f"GNews error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"GNews request failed: {e}")
            return None
    
    async def get_top_headlines(self,
                               country: Optional[str] = None,
                               category: Optional[GNewsCategory] = None,
                               language: Optional[GNewsLanguage] = None,
                               max_articles: int = 20) -> List[GNewsArticle]:
        """
        Get top headlines from GNews.
        
        Args:
            country: Country code (e.g., 'us', 'gb', 'de')
            category: News category
            language: Language filter
            max_articles: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        try:
            # Build query parameters
            params = {
                'max': min(max_articles, 100)
            }
            
            if country:
                params['country'] = country
            if category:
                params['topic'] = category.value
            if language:
                params['lang'] = language.value
            
            # Make request
            data = await self._make_request('top-headlines', params)
            if not data:
                return []
            
            # Parse articles
            articles = []
            for article_data in data.get('articles', []):
                article = self._parse_article(article_data, category)
                if article:
                    articles.append(article)
            
            logger.info(f"Retrieved {len(articles)} top headlines")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to get top headlines: {e}")
            return []
    
    async def search_articles(self,
                             query: str,
                             country: Optional[str] = None,
                             language: Optional[GNewsLanguage] = None,
                             from_date: Optional[datetime] = None,
                             to_date: Optional[datetime] = None,
                             sort_by: str = 'publishedAt',
                             max_articles: int = 20) -> List[GNewsArticle]:
        """
        Search for articles in GNews.
        
        Args:
            query: Search query string
            country: Country filter
            language: Language filter
            from_date: Start date for search
            to_date: End date for search
            sort_by: Sort by publishedAt or relevance
            max_articles: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        try:
            # Build query parameters
            params = {
                'q': query,
                'sortby': sort_by,
                'max': min(max_articles, 100)
            }
            
            if country:
                params['country'] = country
            if language:
                params['lang'] = language.value
            if from_date:
                params['from'] = from_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            if to_date:
                params['to'] = to_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Make request
            data = await self._make_request('search', params)
            if not data:
                return []
            
            # Parse articles
            articles = []
            for article_data in data.get('articles', []):
                article = self._parse_article(article_data)
                if article:
                    articles.append(article)
            
            logger.info(f"Retrieved {len(articles)} articles for query: {query}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to search articles: {e}")
            return []
    
    async def get_financial_news(self,
                                 keywords: Optional[List[str]] = None,
                                 hours_back: int = 24,
                                 max_articles: int = 50,
                                 country: Optional[str] = None) -> List[GNewsArticle]:
        """
        Get financial news articles.
        
        Args:
            keywords: List of financial keywords
            hours_back: Hours to look back
            max_articles: Maximum number of articles to return
            country: Country filter
            
        Returns:
            List of financial news articles
        """
        try:
            # Default financial keywords
            if not keywords:
                keywords = [
                    'gold', 'XAUUSD', 'forex', 'currency', 'dollar', 'euro',
                    'inflation', 'interest rates', 'Federal Reserve', 'ECB',
                    'trading', 'markets', 'stocks', 'bonds', 'commodities',
                    'precious metals', 'bullion', 'safe haven'
                ]
            
            # Build query string
            query = ' OR '.join([f'"{keyword}"' for keyword in keywords])
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(hours=hours_back)
            
            # Get articles
            articles = await self.search_articles(
                query=query,
                country=country,
                from_date=from_date,
                to_date=to_date,
                sort_by='publishedAt',
                max_articles=min(max_articles, 100)
            )
            
            # Filter and rank by relevance
            financial_articles = []
            for article in articles:
                relevance_score = self._calculate_relevance_score(article, keywords)
                if relevance_score > 0.3:  # Minimum relevance threshold
                    article.relevance_score = relevance_score
                    financial_articles.append(article)
            
            # Sort by relevance
            financial_articles.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            
            # Limit results
            financial_articles = financial_articles[:max_articles]
            
            logger.info(f"Retrieved {len(financial_articles)} financial news articles")
            return financial_articles
            
        except Exception as e:
            logger.error(f"Failed to get financial news: {e}")
            return []
    
    async def get_market_sentiment_news(self,
                                       market: str = 'gold',
                                       hours_back: int = 48,
                                       max_articles: int = 30) -> List[GNewsArticle]:
        """
        Get market-specific sentiment news.
        
        Args:
            market: Market to focus on (gold, forex, stocks, etc.)
            hours_back: Hours to look back
            max_articles: Maximum number of articles to return
            
        Returns:
            List of market sentiment news articles
        """
        try:
            # Market-specific keywords
            market_keywords = {
                'gold': ['gold', 'XAUUSD', 'precious metals', 'bullion', 'safe haven'],
                'forex': ['forex', 'currency', 'dollar', 'euro', 'pound', 'yen'],
                'stocks': ['stocks', 'equity', 'market', 'trading', 'investment'],
                'bonds': ['bonds', 'treasury', 'yield', 'interest rates'],
                'commodities': ['commodities', 'oil', 'gas', 'copper', 'silver']
            }
            
            keywords = market_keywords.get(market.lower(), ['market', 'trading'])
            
            # Get articles
            articles = await self.get_financial_news(
                keywords=keywords,
                hours_back=hours_back,
                max_articles=max_articles
            )
            
            # Filter for sentiment-heavy articles
            sentiment_articles = []
            for article in articles:
                # Check for sentiment indicators in title/description
                text = f"{article.title} {article.description}".lower()
                sentiment_words = [
                    'bullish', 'bearish', 'positive', 'negative', 'optimistic',
                    'pessimistic', 'rally', 'crash', 'surge', 'plunge', 'gain',
                    'loss', 'rise', 'fall', 'strong', 'weak', 'recovery'
                ]
                
                sentiment_count = sum(1 for word in sentiment_words if word in text)
                if sentiment_count > 0:
                    article.sentiment_score = sentiment_count / len(sentiment_words)
                    sentiment_articles.append(article)
            
            # Sort by sentiment score
            sentiment_articles.sort(key=lambda x: x.sentiment_score or 0, reverse=True)
            
            logger.info(f"Retrieved {len(sentiment_articles)} sentiment news articles for {market}")
            return sentiment_articles
            
        except Exception as e:
            logger.error(f"Failed to get market sentiment news: {e}")
            return []
    
    async def get_breaking_news(self,
                                category: Optional[GNewsCategory] = None,
                                country: Optional[str] = None,
                                max_articles: int = 20) -> List[GNewsArticle]:
        """
        Get breaking news articles.
        
        Args:
            category: News category filter
            country: Country filter
            max_articles: Maximum number of articles to return
            
        Returns:
            List of breaking news articles
        """
        try:
            # Get recent articles
            articles = await self.get_top_headlines(
                country=country,
                category=category,
                max_articles=max_articles
            )
            
            # Filter for recent articles (last 2 hours)
            now = datetime.now()
            breaking_articles = []
            
            for article in articles:
                time_diff = now - article.published_at
                if time_diff.total_seconds() <= 7200:  # 2 hours
                    breaking_articles.append(article)
            
            logger.info(f"Retrieved {len(breaking_articles)} breaking news articles")
            return breaking_articles
            
        except Exception as e:
            logger.error(f"Failed to get breaking news: {e}")
            return []
    
    def _parse_article(self, article_data: Dict[str, Any], category: Optional[GNewsCategory] = None) -> Optional[GNewsArticle]:
        """
        Parse article data into GNewsArticle object.
        
        Args:
            article_data: Raw article data
            category: Article category if known
            
        Returns:
            Parsed GNewsArticle or None if parsing failed
        """
        try:
            # Extract basic information
            article_id = article_data.get('url', '')
            title = article_data.get('title', '')
            description = article_data.get('description', '')
            content = article_data.get('content', '')
            url = article_data.get('url', '')
            image_url = article_data.get('image')
            
            # Parse published date
            published_str = article_data.get('publishedAt', '')
            if published_str:
                try:
                    published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                except ValueError:
                    # Try alternative format
                    published_at = datetime.strptime(published_str, '%Y-%m-%dT%H:%M:%S')
            else:
                published_at = datetime.now()
            
            # Extract source information
            source_name = article_data.get('source', {}).get('name', '')
            source_url = article_data.get('source', {}).get('url', '')
            
            # Extract author
            author = article_data.get('author')
            
            # Determine category and language
            if category:
                article_category = category
            else:
                article_category = self._classify_article_category(title, description)
            
            language = self._classify_article_language(title, description)
            
            # Extract country
            country = article_data.get('country')
            
            # Extract keywords
            keywords = self._extract_keywords(title, description, content)
            
            # Create article object
            article = GNewsArticle(
                article_id=article_id,
                title=title,
                description=description,
                content=content,
                url=url,
                image_url=image_url,
                published_at=published_at,
                source_name=source_name,
                source_url=source_url,
                author=author,
                category=article_category,
                language=language,
                sentiment_score=None,  # Will be calculated later
                relevance_score=None,  # Will be calculated later
                keywords=keywords,
                country=country,
                raw_data=article_data
            )
            
            return article
            
        except Exception as e:
            logger.error(f"Failed to parse article: {e}")
            return None
    
    def _classify_article_category(self, title: str, description: str) -> GNewsCategory:
        """
        Classify article category based on content.
        
        Args:
            title: Article title
            description: Article description
            
        Returns:
            Classified category
        """
        text = f"{title} {description}".lower()
        
        # Business/Financial keywords
        business_keywords = [
            'business', 'finance', 'economy', 'market', 'trading', 'investment',
            'stock', 'bond', 'currency', 'forex', 'commodity', 'gold', 'silver',
            'oil', 'gas', 'bank', 'banking', 'loan', 'credit', 'debt',
            'earnings', 'revenue', 'profit', 'loss', 'quarterly', 'annual'
        ]
        
        # Technology keywords
        tech_keywords = [
            'technology', 'tech', 'software', 'hardware', 'ai', 'artificial intelligence',
            'machine learning', 'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum',
            'startup', 'innovation', 'digital', 'cyber', 'internet', 'app', 'platform'
        ]
        
        # World/International keywords
        world_keywords = [
            'world', 'international', 'global', 'foreign', 'diplomacy', 'trade war',
            'sanctions', 'treaty', 'alliance', 'summit', 'meeting', 'negotiation'
        ]
        
        # Check for business keywords
        if any(keyword in text for keyword in business_keywords):
            return GNewsCategory.BUSINESS
        
        # Check for technology keywords
        if any(keyword in text for keyword in tech_keywords):
            return GNewsCategory.TECHNOLOGY
        
        # Check for world keywords
        if any(keyword in text for keyword in world_keywords):
            return GNewsCategory.WORLD
        
        # Default to business for financial news
        return GNewsCategory.BUSINESS
    
    def _classify_article_language(self, title: str, description: str) -> GNewsLanguage:
        """
        Classify article language based on content.
        
        Args:
            title: Article title
            description: Article description
            
        Returns:
            Classified language
        """
        # Simple language detection based on common words
        text = f"{title} {description}".lower()
        
        # Check for non-English characters
        if any(ord(char) > 127 for char in text):
            # This is a simplified approach - in production, use a proper language detection library
            return GNewsLanguage.ENGLISH
        
        return GNewsLanguage.ENGLISH
    
    def _extract_keywords(self, title: str, description: str, content: str) -> List[str]:
        """
        Extract keywords from article content.
        
        Args:
            title: Article title
            description: Article description
            content: Article content
            
        Returns:
            List of extracted keywords
        """
        # Combine all text
        text = f"{title} {description} {content}".lower()
        
        # Common financial keywords
        financial_keywords = [
            'gold', 'silver', 'platinum', 'palladium', 'xauusd', 'xagusd',
            'forex', 'currency', 'dollar', 'euro', 'pound', 'yen', 'franc',
            'inflation', 'deflation', 'interest rate', 'federal reserve', 'ecb',
            'trading', 'market', 'stock', 'bond', 'commodity', 'oil', 'gas',
            'recession', 'growth', 'gdp', 'employment', 'unemployment',
            'bullish', 'bearish', 'rally', 'crash', 'surge', 'plunge'
        ]
        
        # Extract keywords that appear in the text
        keywords = []
        for keyword in financial_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        return keywords
    
    def _calculate_relevance_score(self, article: GNewsArticle, keywords: List[str]) -> float:
        """
        Calculate relevance score for an article.
        
        Args:
            article: News article
            keywords: List of keywords to match against
            
        Returns:
            Relevance score between 0 and 1
        """
        text = f"{article.title} {article.description}".lower()
        
        # Count keyword matches
        matches = 0
        for keyword in keywords:
            if keyword.lower() in text:
                matches += 1
        
        # Calculate score based on matches and text length
        if matches == 0:
            return 0.0
        
        # Normalize by number of keywords and text length
        score = matches / len(keywords)
        
        # Boost score for shorter texts (more focused)
        text_length = len(text.split())
        if text_length < 100:
            score *= 1.2
        elif text_length > 500:
            score *= 0.8
        
        return min(score, 1.0)
    
    async def close(self):
        """Close the ingestor and cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info("GNews Ingestor closed")


# Factory function
def create_gnews_ingestor(config: Optional[Dict] = None) -> GNewsIngestor:
    """
    Create GNews ingestor instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        GNews ingestor instance
    """
    return GNewsIngestor(config)


# Example usage
async def main():
    """Example usage of GNews ingestor."""
    config = {
        'base_url': 'https://gnews.io/api/v4',
        'api_key': 'YOUR_API_KEY_HERE',  # Add your API key here
        'rate_limit': 100,
        'timeout': 30
    }
    
    ingestor = create_gnews_ingestor(config)
    
    try:
        # Get top business headlines
        headlines = await ingestor.get_top_headlines(
            country='us',
            category=GNewsCategory.BUSINESS,
            max_articles=10
        )
        print(f"Retrieved {len(headlines)} business headlines")
        
        # Search for financial news
        financial_news = await ingestor.get_financial_news(
            keywords=['gold', 'forex', 'trading'],
            hours_back=48,
            max_articles=20
        )
        print(f"Retrieved {len(financial_news)} financial news articles")
        
        # Get market sentiment news
        sentiment_news = await ingestor.get_market_sentiment_news(
            market='gold',
            hours_back=24,
            max_articles=15
        )
        print(f"Retrieved {len(sentiment_news)} sentiment news articles")
        
        # Get breaking news
        breaking_news = await ingestor.get_breaking_news(
            category=GNewsCategory.BUSINESS,
            max_articles=10
        )
        print(f"Retrieved {len(breaking_news)} breaking news articles")
        
    finally:
        await ingestor.close()


if __name__ == "__main__":
    asyncio.run(main())
