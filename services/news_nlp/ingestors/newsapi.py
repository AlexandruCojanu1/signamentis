#!/usr/bin/env python3
"""
SignaMentis - NewsAPI Ingestor Service

This module handles ingestion of financial news from NewsAPI.
It provides access to real-time news articles with sentiment analysis.

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


class NewsCategory(Enum):
    """News categories enumeration."""
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    HEALTH = "health"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    GENERAL = "general"


class NewsLanguage(Enum):
    """News languages enumeration."""
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
class NewsArticle:
    """News article data container."""
    article_id: str
    title: str
    description: str
    content: str
    url: str
    image_url: Optional[str]
    published_at: datetime
    source_name: str
    source_id: str
    author: Optional[str]
    category: NewsCategory
    language: NewsLanguage
    sentiment_score: Optional[float]
    relevance_score: Optional[float]
    keywords: List[str]
    raw_data: Dict[str, Any]


class NewsAPIIngestor:
    """
    NewsAPI ingestor service.
    
    Features:
    - Real-time news articles
    - Category-based filtering
    - Language support
    - Sentiment analysis
    - Keyword extraction
    - Rate limiting and error handling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize NewsAPI ingestor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # API configuration
        self.base_url = self.config.get('base_url', 'https://newsapi.org/v2')
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
        
        logger.info("NewsAPI Ingestor initialized")
    
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
            logger.warning("NewsAPI rate limit exceeded, waiting for reset")
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
            params['apiKey'] = self.api_key
            
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
                    logger.warning("NewsAPI rate limited")
                    self.rate_limit_reset = datetime.now() + timedelta(hours=1)
                    return None
                elif response.status == 401:  # Unauthorized
                    logger.error("NewsAPI API key invalid")
                    return None
                elif response.status == 400:  # Bad request
                    logger.error("NewsAPI bad request")
                    return None
                else:
                    logger.error(f"NewsAPI error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"NewsAPI request failed: {e}")
            return None
    
    async def get_top_headlines(self,
                               country: Optional[str] = None,
                               category: Optional[NewsCategory] = None,
                               sources: Optional[List[str]] = None,
                               q: Optional[str] = None,
                               page_size: int = 20,
                               page: int = 1) -> List[NewsArticle]:
        """
        Get top headlines from NewsAPI.
        
        Args:
            country: Country code (e.g., 'us', 'gb', 'de')
            category: News category
            sources: List of source IDs
            q: Query string for search
            page_size: Number of articles per page (max 100)
            page: Page number
            
        Returns:
            List of news articles
        """
        try:
            # Build query parameters
            params = {
                'pageSize': min(page_size, 100),
                'page': page
            }
            
            if country:
                params['country'] = country
            if category:
                params['category'] = category.value
            if sources:
                params['sources'] = ','.join(sources)
            if q:
                params['q'] = q
            
            # Make request
            data = await self._make_request('top-headlines', params)
            if not data or data.get('status') != 'ok':
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
    
    async def get_everything(self,
                            q: str,
                            search_in: Optional[str] = None,
                            sources: Optional[List[str]] = None,
                            domains: Optional[List[str]] = None,
                            exclude_domains: Optional[List[str]] = None,
                            from_date: Optional[datetime] = None,
                            to_date: Optional[datetime] = None,
                            language: Optional[NewsLanguage] = None,
                            sort_by: str = 'relevancy',
                            page_size: int = 20,
                            page: int = 1) -> List[NewsArticle]:
        """
        Search for articles across all sources.
        
        Args:
            q: Query string
            search_in: Search in title, description, or content
            sources: List of source IDs
            domains: List of domains to search
            exclude_domains: List of domains to exclude
            from_date: Start date for search
            to_date: End date for search
            language: Language filter
            sort_by: Sort by relevancy, popularity, or publishedAt
            page_size: Number of articles per page (max 100)
            page: Page number
            
        Returns:
            List of news articles
        """
        try:
            # Build query parameters
            params = {
                'q': q,
                'sortBy': sort_by,
                'pageSize': min(page_size, 100),
                'page': page
            }
            
            if search_in:
                params['searchIn'] = search_in
            if sources:
                params['sources'] = ','.join(sources)
            if domains:
                params['domains'] = ','.join(domains)
            if exclude_domains:
                params['excludeDomains'] = ','.join(exclude_domains)
            if from_date:
                params['from'] = from_date.strftime('%Y-%m-%d')
            if to_date:
                params['to'] = to_date.strftime('%Y-%m-%d')
            if language:
                params['language'] = language.value
            
            # Make request
            data = await self._make_request('everything', params)
            if not data or data.get('status') != 'ok':
                return []
            
            # Parse articles
            articles = []
            for article_data in data.get('articles', []):
                article = self._parse_article(article_data)
                if article:
                    articles.append(article)
            
            logger.info(f"Retrieved {len(articles)} articles for query: {q}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to search articles: {e}")
            return []
    
    async def get_sources(self,
                         category: Optional[NewsCategory] = None,
                         language: Optional[NewsLanguage] = None,
                         country: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available news sources.
        
        Args:
            category: Filter by category
            language: Filter by language
            country: Filter by country
            
        Returns:
            List of news sources
        """
        try:
            # Build query parameters
            params = {}
            if category:
                params['category'] = category.value
            if language:
                params['language'] = language.value
            if country:
                params['country'] = country
            
            # Make request
            data = await self._make_request('sources', params)
            if not data or data.get('status') != 'ok':
                return []
            
            sources = data.get('sources', [])
            logger.info(f"Retrieved {len(sources)} news sources")
            return sources
            
        except Exception as e:
            logger.error(f"Failed to get sources: {e}")
            return []
    
    async def get_financial_news(self,
                                 keywords: Optional[List[str]] = None,
                                 hours_back: int = 24,
                                 max_articles: int = 50) -> List[NewsArticle]:
        """
        Get financial news articles.
        
        Args:
            keywords: List of financial keywords
            hours_back: Hours to look back
            max_articles: Maximum number of articles to return
            
        Returns:
            List of financial news articles
        """
        try:
            # Default financial keywords
            if not keywords:
                keywords = [
                    'gold', 'XAUUSD', 'forex', 'currency', 'dollar', 'euro',
                    'inflation', 'interest rates', 'Federal Reserve', 'ECB',
                    'trading', 'markets', 'stocks', 'bonds', 'commodities'
                ]
            
            # Build query string
            query = ' OR '.join([f'"{keyword}"' for keyword in keywords])
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(hours=hours_back)
            
            # Get articles
            articles = await self.get_everything(
                q=query,
                from_date=from_date,
                to_date=to_date,
                language=NewsLanguage.ENGLISH,
                sort_by='publishedAt',
                page_size=min(max_articles, 100)
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
    
    def _parse_article(self, article_data: Dict[str, Any], category: Optional[NewsCategory] = None) -> Optional[NewsArticle]:
        """
        Parse article data into NewsArticle object.
        
        Args:
            article_data: Raw article data
            category: Article category if known
            
        Returns:
            Parsed NewsArticle or None if parsing failed
        """
        try:
            # Extract basic information
            article_id = article_data.get('url', '')
            title = article_data.get('title', '')
            description = article_data.get('description', '')
            content = article_data.get('content', '')
            url = article_data.get('url', '')
            image_url = article_data.get('urlToImage')
            
            # Parse published date
            published_str = article_data.get('publishedAt', '')
            if published_str:
                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            else:
                published_at = datetime.now()
            
            # Extract source information
            source_data = article_data.get('source', {})
            source_name = source_data.get('name', '')
            source_id = source_data.get('id', '')
            
            # Extract author
            author = article_data.get('author')
            
            # Determine category and language
            if category:
                article_category = category
            else:
                article_category = self._classify_article_category(title, description)
            
            language = self._classify_article_language(title, description)
            
            # Extract keywords
            keywords = self._extract_keywords(title, description, content)
            
            # Create article object
            article = NewsArticle(
                article_id=article_id,
                title=title,
                description=description,
                content=content,
                url=url,
                image_url=image_url,
                published_at=published_at,
                source_name=source_name,
                source_id=source_id,
                author=author,
                category=article_category,
                language=language,
                sentiment_score=None,  # Will be calculated later
                relevance_score=None,  # Will be calculated later
                keywords=keywords,
                raw_data=article_data
            )
            
            return article
            
        except Exception as e:
            logger.error(f"Failed to parse article: {e}")
            return None
    
    def _classify_article_category(self, title: str, description: str) -> NewsCategory:
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
            'oil', 'gas', 'bank', 'banking', 'loan', 'credit', 'debt'
        ]
        
        # Technology keywords
        tech_keywords = [
            'technology', 'tech', 'software', 'hardware', 'ai', 'artificial intelligence',
            'machine learning', 'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum',
            'startup', 'innovation', 'digital', 'cyber', 'internet'
        ]
        
        # Check for business keywords
        if any(keyword in text for keyword in business_keywords):
            return NewsCategory.BUSINESS
        
        # Check for technology keywords
        if any(keyword in text for keyword in tech_keywords):
            return NewsCategory.TECHNOLOGY
        
        # Default to business for financial news
        return NewsCategory.BUSINESS
    
    def _classify_article_language(self, title: str, description: str) -> NewsLanguage:
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
            return NewsLanguage.ENGLISH
        
        return NewsLanguage.ENGLISH
    
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
            'recession', 'growth', 'gdp', 'employment', 'unemployment'
        ]
        
        # Extract keywords that appear in the text
        keywords = []
        for keyword in financial_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        return keywords
    
    def _calculate_relevance_score(self, article: NewsArticle, keywords: List[str]) -> float:
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
        logger.info("NewsAPI Ingestor closed")


# Factory function
def create_newsapi_ingestor(config: Optional[Dict] = None) -> NewsAPIIngestor:
    """
    Create NewsAPI ingestor instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        NewsAPI ingestor instance
    """
    return NewsAPIIngestor(config)


# Example usage
async def main():
    """Example usage of NewsAPI ingestor."""
    config = {
        'base_url': 'https://newsapi.org/v2',
        'api_key': 'YOUR_API_KEY_HERE',  # Add your API key here
        'rate_limit': 100,
        'timeout': 30
    }
    
    ingestor = create_newsapi_ingestor(config)
    
    try:
        # Get top business headlines
        headlines = await ingestor.get_top_headlines(
            country='us',
            category=NewsCategory.BUSINESS,
            page_size=10
        )
        print(f"Retrieved {len(headlines)} business headlines")
        
        # Search for financial news
        financial_news = await ingestor.get_financial_news(
            keywords=['gold', 'forex', 'trading'],
            hours_back=48,
            max_articles=20
        )
        print(f"Retrieved {len(financial_news)} financial news articles")
        
        # Get available sources
        sources = await ingestor.get_sources(
            category=NewsCategory.BUSINESS,
            language=NewsLanguage.ENGLISH
        )
        print(f"Retrieved {len(sources)} news sources")
        
    finally:
        await ingestor.close()


if __name__ == "__main__":
    asyncio.run(main())
