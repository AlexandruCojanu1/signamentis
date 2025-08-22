#!/usr/bin/env python3
"""
SignaMentis - GDELT News Ingestor Service

This module handles ingestion of financial news data from GDELT (Global Database of Events, Language, and Tone).
It provides real-time and historical news data with sentiment analysis.

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


class GDELTEventType(Enum):
    """GDELT event types enumeration."""
    ECONOMIC = "Economic"
    POLITICAL = "Political"
    FINANCIAL = "Financial"
    TRADE = "Trade"
    MONETARY = "Monetary"
    FISCAL = "Fiscal"
    REGULATORY = "Regulatory"


@dataclass
class GDELTEvent:
    """GDELT event data container."""
    event_id: str
    event_time: datetime
    event_type: GDELTEventType
    actor1_name: str
    actor2_name: str
    event_code: str
    goldstein_scale: float
    avg_tone: float
    source_url: str
    mentions: List[str]
    locations: List[str]
    themes: List[str]
    raw_data: Dict[str, Any]


class GDELTIngestor:
    """
    GDELT news ingestor service.
    
    Features:
    - Real-time news ingestion
    - Historical data retrieval
    - Event filtering and categorization
    - Rate limiting and error handling
    - Async data processing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize GDELT ingestor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # API configuration
        self.base_url = self.config.get('base_url', 'https://api.gdeltproject.org/api/v2')
        self.api_key = self.config.get('api_key', '')
        self.rate_limit = self.config.get('rate_limit', 1000)  # requests per hour
        self.timeout = self.config.get('timeout', 30)
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = datetime.now()
        self.rate_limit_reset = datetime.now() + timedelta(hours=1)
        
        # Data storage
        self.events_cache = {}
        self.last_update = datetime.now()
        
        # Session management
        self.session = None
        
        logger.info("GDELT Ingestor initialized")
    
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
        
        # Reset counter if hour has passed
        if now >= self.rate_limit_reset:
            self.request_count = 0
            self.rate_limit_reset = now + timedelta(hours=1)
        
        # Check if limit exceeded
        if self.request_count >= self.rate_limit:
            logger.warning("GDELT rate limit exceeded, waiting for reset")
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
            
            # Add API key if available
            if self.api_key:
                params['apikey'] = self.api_key
            
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
                    logger.warning("GDELT API rate limited")
                    self.rate_limit_reset = datetime.now() + timedelta(minutes=15)
                    return None
                else:
                    logger.error(f"GDELT API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"GDELT request failed: {e}")
            return None
    
    async def get_recent_events(self, 
                               event_types: Optional[List[GDELTEventType]] = None,
                               limit: int = 100,
                               time_window: int = 24) -> List[GDELTEvent]:
        """
        Get recent events from GDELT.
        
        Args:
            event_types: Filter by event types
            limit: Maximum number of events to return
            time_window: Time window in hours
            
        Returns:
            List of GDELT events
        """
        try:
            # Build query parameters
            params = {
                'query': self._build_query_string(event_types),
                'mode': 'artlist',
                'maxrecords': limit,
                'startdatetime': (datetime.now() - timedelta(hours=time_window)).strftime('%Y%m%d%H%M%S'),
                'enddatetime': datetime.now().strftime('%Y%m%d%H%M%S'),
                'sort': 'hybridrel'
            }
            
            # Make request
            data = await self._make_request('doc', params)
            if not data:
                return []
            
            # Parse events
            events = []
            for article in data.get('articles', []):
                event = self._parse_article(article)
                if event:
                    events.append(event)
            
            logger.info(f"Retrieved {len(events)} recent GDELT events")
            return events
            
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []
    
    async def get_historical_events(self,
                                   start_date: datetime,
                                   end_date: datetime,
                                   event_types: Optional[List[GDELTEventType]] = None,
                                   batch_size: int = 1000) -> List[GDELTEvent]:
        """
        Get historical events from GDELT.
        
        Args:
            start_date: Start date for query
            end_date: End date for query
            event_types: Filter by event types
            batch_size: Number of events per batch
            
        Returns:
            List of GDELT events
        """
        try:
            all_events = []
            current_date = start_date
            
            while current_date < end_date:
                # Calculate batch end date
                batch_end = min(current_date + timedelta(days=1), end_date)
                
                # Build query parameters
                params = {
                    'query': self._build_query_string(event_types),
                    'mode': 'artlist',
                    'maxrecords': batch_size,
                    'startdatetime': current_date.strftime('%Y%m%d%H%M%S'),
                    'enddatetime': batch_end.strftime('%Y%m%d%H%M%S'),
                    'sort': 'hybridrel'
                }
                
                # Make request
                data = await self._make_request('doc', params)
                if data:
                    # Parse events
                    for article in data.get('articles', []):
                        event = self._parse_article(article)
                        if event:
                            all_events.append(event)
                
                # Move to next batch
                current_date = batch_end
                
                # Rate limiting delay
                await asyncio.sleep(1)
            
            logger.info(f"Retrieved {len(all_events)} historical GDELT events")
            return all_events
            
        except Exception as e:
            logger.error(f"Failed to get historical events: {e}")
            return []
    
    async def get_event_summary(self,
                               event_types: Optional[List[GDELTEventType]] = None,
                               time_window: int = 24) -> Dict[str, Any]:
        """
        Get event summary statistics.
        
        Args:
            event_types: Filter by event types
            time_window: Time window in hours
            
        Returns:
            Event summary statistics
        """
        try:
            # Build query parameters
            params = {
                'query': self._build_query_string(event_types),
                'mode': 'timelinevolraw',
                'startdatetime': (datetime.now() - timedelta(hours=time_window)).strftime('%Y%m%d%H%M%S'),
                'enddatetime': datetime.now().strftime('%Y%m%d%H%M%S'),
                'timelinesmooth': 0
            }
            
            # Make request
            data = await self._make_request('doc', params)
            if not data:
                return {}
            
            # Parse summary
            summary = {
                'total_events': len(data.get('timeline', [])),
                'event_types': {},
                'tone_distribution': {},
                'source_distribution': {},
                'geographic_distribution': {}
            }
            
            # Process timeline data
            for entry in data.get('timeline', []):
                # Count event types
                event_type = entry.get('event_type', 'Unknown')
                summary['event_types'][event_type] = summary['event_types'].get(event_type, 0) + 1
                
                # Process tone
                tone = entry.get('avg_tone', 0)
                tone_bucket = self._categorize_tone(tone)
                summary['tone_distribution'][tone_bucket] = summary['tone_distribution'].get(tone_bucket, 0) + 1
            
            logger.info("Retrieved GDELT event summary")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get event summary: {e}")
            return {}
    
    def _build_query_string(self, event_types: Optional[List[GDELTEventType]] = None) -> str:
        """
        Build GDELT query string.
        
        Args:
            event_types: Event types to filter by
            
        Returns:
            Query string
        """
        base_query = "domain:finance OR domain:economics OR domain:trading"
        
        if event_types:
            type_filters = []
            for event_type in event_types:
                if event_type == GDELTEventType.ECONOMIC:
                    type_filters.append("(theme:ECONOMIC OR theme:GDP OR theme:INFLATION)")
                elif event_type == GDELTEventType.FINANCIAL:
                    type_filters.append("(theme:FINANCIAL OR theme:STOCK OR theme:BOND)")
                elif event_type == GDELTEventType.MONETARY:
                    type_filters.append("(theme:MONETARY OR theme:INTEREST_RATE OR theme:CENTRAL_BANK)")
                elif event_type == GDELTEventType.TRADE:
                    type_filters.append("(theme:TRADE OR theme:EXPORT OR theme:IMPORT)")
                elif event_type == GDELTEventType.REGULATORY:
                    type_filters.append("(theme:REGULATION OR theme:COMPLIANCE)")
            
            if type_filters:
                base_query += f" AND ({' OR '.join(type_filters)})"
        
        return base_query
    
    def _parse_article(self, article: Dict[str, Any]) -> Optional[GDELTEvent]:
        """
        Parse GDELT article data into event object.
        
        Args:
            article: Raw article data
            
        Returns:
            Parsed GDELT event or None if parsing failed
        """
        try:
            # Extract basic information
            event_id = article.get('url', '')
            event_time = datetime.fromisoformat(article.get('seendate', '').replace('Z', '+00:00'))
            
            # Determine event type
            event_type = self._classify_event_type(article)
            
            # Extract actors
            actor1_name = article.get('actor1name', '')
            actor2_name = article.get('actor2name', '')
            
            # Extract event details
            event_code = article.get('eventcode', '')
            goldstein_scale = float(article.get('goldsteinscale', 0))
            avg_tone = float(article.get('avgtone', 0))
            
            # Extract source and content
            source_url = article.get('url', '')
            mentions = article.get('mentions', [])
            locations = article.get('locations', [])
            themes = article.get('themes', [])
            
            # Create event object
            event = GDELTEvent(
                event_id=event_id,
                event_time=event_time,
                event_type=event_type,
                actor1_name=actor1_name,
                actor2_name=actor2_name,
                event_code=event_code,
                goldstein_scale=goldstein_scale,
                avg_tone=avg_tone,
                source_url=source_url,
                mentions=mentions,
                locations=locations,
                themes=themes,
                raw_data=article
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to parse article: {e}")
            return None
    
    def _classify_event_type(self, article: Dict[str, Any]) -> GDELTEventType:
        """
        Classify event type based on article content.
        
        Args:
            article: Article data
            
        Returns:
            Classified event type
        """
        themes = article.get('themes', [])
        mentions = article.get('mentions', [])
        
        # Check for specific themes
        if any(theme in themes for theme in ['ECONOMIC', 'GDP', 'INFLATION']):
            return GDELTEventType.ECONOMIC
        elif any(theme in themes for theme in ['FINANCIAL', 'STOCK', 'BOND']):
            return GDELTEventType.FINANCIAL
        elif any(theme in themes for theme in ['MONETARY', 'INTEREST_RATE', 'CENTRAL_BANK']):
            return GDELTEventType.MONETARY
        elif any(theme in themes for theme in ['TRADE', 'EXPORT', 'IMPORT']):
            return GDELTEventType.TRADE
        elif any(theme in themes for theme in ['REGULATION', 'COMPLIANCE']):
            return GDELTEventType.REGULATORY
        elif any(theme in themes for theme in ['POLITICAL', 'ELECTION', 'GOVERNMENT']):
            return GDELTEventType.POLITICAL
        
        # Default to economic if no specific classification
        return GDELTEventType.ECONOMIC
    
    def _categorize_tone(self, tone: float) -> str:
        """
        Categorize tone value into buckets.
        
        Args:
            tone: Tone value from GDELT
            
        Returns:
            Tone category
        """
        if tone <= -5:
            return "Very Negative"
        elif tone <= -2:
            return "Negative"
        elif tone <= 2:
            return "Neutral"
        elif tone <= 5:
            return "Positive"
        else:
            return "Very Positive"
    
    async def close(self):
        """Close the ingestor and cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info("GDELT Ingestor closed")


# Factory function
def create_gdelt_ingestor(config: Optional[Dict] = None) -> GDELTIngestor:
    """
    Create GDELT ingestor instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        GDELT ingestor instance
    """
    return GDELTIngestor(config)


# Example usage
async def main():
    """Example usage of GDELT ingestor."""
    config = {
        'base_url': 'https://api.gdeltproject.org/api/v2',
        'api_key': '',  # Add your API key here
        'rate_limit': 1000,
        'timeout': 30
    }
    
    ingestor = create_gdelt_ingestor(config)
    
    try:
        # Get recent events
        events = await ingestor.get_recent_events(
            event_types=[GDELTEventType.ECONOMIC, GDELTEventType.FINANCIAL],
            limit=50,
            time_window=24
        )
        
        print(f"Retrieved {len(events)} events")
        
        # Get event summary
        summary = await ingestor.get_event_summary(
            event_types=[GDELTEventType.ECONOMIC],
            time_window=24
        )
        
        print(f"Event summary: {summary}")
        
    finally:
        await ingestor.close()


if __name__ == "__main__":
    asyncio.run(main())
