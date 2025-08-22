#!/usr/bin/env python3
"""
SignaMentis - TradingEconomics News Ingestor Service

This module handles ingestion of economic data and news from TradingEconomics.
It provides access to economic indicators, calendar events, and financial news.

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


class EconomicIndicatorType(Enum):
    """Economic indicator types enumeration."""
    GDP = "GDP"
    INFLATION = "Inflation"
    INTEREST_RATE = "Interest Rate"
    UNEMPLOYMENT = "Unemployment"
    TRADE_BALANCE = "Trade Balance"
    CONSUMER_SENTIMENT = "Consumer Sentiment"
    MANUFACTURING_PMI = "Manufacturing PMI"
    SERVICES_PMI = "Services PMI"
    RETAIL_SALES = "Retail Sales"
    HOUSING = "Housing"


@dataclass
class EconomicIndicator:
    """Economic indicator data container."""
    indicator_id: str
    country: str
    category: str
    indicator: str
    last_value: float
    previous_value: float
    forecast_value: Optional[float]
    unit: str
    frequency: str
    last_update: datetime
    next_release: Optional[datetime]
    importance: str
    impact: str
    source: str
    raw_data: Dict[str, Any]


@dataclass
class EconomicCalendar:
    """Economic calendar event container."""
    event_id: str
    country: str
    category: str
    event: str
    date: datetime
    time: str
    importance: str
    forecast: Optional[float]
    previous: Optional[float]
    actual: Optional[float]
    unit: str
    source: str
    raw_data: Dict[str, Any]


class TradingEconomicsIngestor:
    """
    TradingEconomics news and data ingestor service.
    
    Features:
    - Economic indicators data
    - Economic calendar events
    - Financial news articles
    - Country-specific data
    - Historical data retrieval
    - Rate limiting and error handling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TradingEconomics ingestor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # API configuration
        self.base_url = self.config.get('base_url', 'https://api.tradingeconomics.com')
        self.api_key = self.config.get('api_key', '')
        self.rate_limit = self.config.get('rate_limit', 1000)  # requests per hour
        self.timeout = self.config.get('timeout', 30)
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = datetime.now()
        self.rate_limit_reset = datetime.now() + timedelta(hours=1)
        
        # Data storage
        self.indicators_cache = {}
        self.calendar_cache = {}
        self.last_update = datetime.now()
        
        # Session management
        self.session = None
        
        logger.info("TradingEconomics Ingestor initialized")
    
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
            logger.warning("TradingEconomics rate limit exceeded, waiting for reset")
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
            params['c'] = self.api_key
            
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
                    logger.warning("TradingEconomics API rate limited")
                    self.rate_limit_reset = datetime.now() + timedelta(minutes=15)
                    return None
                elif response.status == 401:  # Unauthorized
                    logger.error("TradingEconomics API key invalid")
                    return None
                else:
                    logger.error(f"TradingEconomics API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"TradingEconomics request failed: {e}")
            return None
    
    async def get_economic_indicators(self,
                                    country: Optional[str] = None,
                                    indicator: Optional[str] = None,
                                    category: Optional[str] = None) -> List[EconomicIndicator]:
        """
        Get economic indicators data.
        
        Args:
            country: Filter by country code
            indicator: Filter by indicator name
            category: Filter by category
            
        Returns:
            List of economic indicators
        """
        try:
            # Build query parameters
            params = {}
            if country:
                params['country'] = country
            if indicator:
                params['indicator'] = indicator
            if category:
                params['category'] = category
            
            # Make request
            data = await self._make_request('indicators', params)
            if not data:
                return []
            
            # Parse indicators
            indicators = []
            for item in data:
                indicator_obj = self._parse_indicator(item)
                if indicator_obj:
                    indicators.append(indicator_obj)
            
            logger.info(f"Retrieved {len(indicators)} economic indicators")
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to get economic indicators: {e}")
            return []
    
    async def get_economic_calendar(self,
                                   country: Optional[str] = None,
                                   importance: Optional[str] = None,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> List[EconomicCalendar]:
        """
        Get economic calendar events.
        
        Args:
            country: Filter by country code
            importance: Filter by importance (High, Medium, Low)
            start_date: Start date for events
            end_date: End date for events
            
        Returns:
            List of economic calendar events
        """
        try:
            # Build query parameters
            params = {}
            if country:
                params['country'] = country
            if importance:
                params['importance'] = importance
            if start_date:
                params['d1'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['d2'] = end_date.strftime('%Y-%m-%d')
            
            # Make request
            data = await self._make_request('calendar', params)
            if not data:
                return []
            
            # Parse calendar events
            events = []
            for item in data:
                event = self._parse_calendar_event(item)
                if event:
                    events.append(event)
            
            logger.info(f"Retrieved {len(events)} economic calendar events")
            return events
            
        except Exception as e:
            logger.error(f"Failed to get economic calendar: {e}")
            return []
    
    async def get_country_indicators(self, country: str) -> List[EconomicIndicator]:
        """
        Get all indicators for a specific country.
        
        Args:
            country: Country code (e.g., 'US', 'EU', 'GB')
            
        Returns:
            List of economic indicators for the country
        """
        try:
            # Get country indicators
            indicators = await self.get_economic_indicators(country=country)
            
            # Get additional country-specific data
            params = {'country': country}
            
            # Get GDP data
            gdp_data = await self._make_request('country', params)
            if gdp_data:
                # Process GDP data
                pass
            
            # Get inflation data
            inflation_data = await self._make_request('country', params)
            if inflation_data:
                # Process inflation data
                pass
            
            logger.info(f"Retrieved {len(indicators)} indicators for {country}")
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to get country indicators for {country}: {e}")
            return []
    
    async def get_historical_data(self,
                                 country: str,
                                 indicator: str,
                                 start_date: datetime,
                                 end_date: datetime) -> pd.DataFrame:
        """
        Get historical data for a specific indicator.
        
        Args:
            country: Country code
            indicator: Indicator name
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Build query parameters
            params = {
                'country': country,
                'indicator': indicator,
                'd1': start_date.strftime('%Y-%m-%d'),
                'd2': end_date.strftime('%Y-%m-%d')
            }
            
            # Make request
            data = await self._make_request('historical', params)
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Process dates
            if 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df.set_index('DateTime', inplace=True)
            
            # Convert values to numeric
            if 'Value' in df.columns:
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            
            logger.info(f"Retrieved historical data for {indicator} in {country}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    async def get_market_impact_analysis(self,
                                       country: str,
                                       indicator: str,
                                       days_before: int = 5,
                                       days_after: int = 5) -> Dict[str, Any]:
        """
        Analyze market impact of economic releases.
        
        Args:
            country: Country code
            indicator: Indicator name
            days_before: Days before release to analyze
            days_after: Days after release to analyze
            
        Returns:
            Market impact analysis
        """
        try:
            # Get calendar events for the indicator
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_before + days_after)
            
            events = await self.get_economic_calendar(
                country=country,
                start_date=start_date,
                end_date=end_date
            )
            
            # Filter events for the specific indicator
            indicator_events = [e for e in events if indicator.lower() in e.event.lower()]
            
            if not indicator_events:
                return {}
            
            # Analyze market impact
            analysis = {
                'indicator': indicator,
                'country': country,
                'events_analyzed': len(indicator_events),
                'surprise_analysis': [],
                'market_reaction': [],
                'volatility_impact': []
            }
            
            for event in indicator_events:
                # Calculate surprise
                if event.forecast is not None and event.actual is not None:
                    surprise = event.actual - event.forecast
                    surprise_pct = (surprise / event.forecast) * 100 if event.forecast != 0 else 0
                    
                    analysis['surprise_analysis'].append({
                        'date': event.date,
                        'forecast': event.forecast,
                        'actual': event.actual,
                        'surprise': surprise,
                        'surprise_pct': surprise_pct
                    })
            
            logger.info(f"Completed market impact analysis for {indicator} in {country}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze market impact: {e}")
            return {}
    
    def _parse_indicator(self, data: Dict[str, Any]) -> Optional[EconomicIndicator]:
        """
        Parse indicator data into EconomicIndicator object.
        
        Args:
            data: Raw indicator data
            
        Returns:
            Parsed EconomicIndicator or None if parsing failed
        """
        try:
            # Extract basic information
            indicator_id = data.get('ID', '')
            country = data.get('Country', '')
            category = data.get('Category', '')
            indicator = data.get('Indicator', '')
            
            # Extract values
            last_value = float(data.get('LatestValue', 0))
            previous_value = float(data.get('PreviousValue', 0))
            forecast_value = float(data.get('Forecast', 0)) if data.get('Forecast') else None
            
            # Extract metadata
            unit = data.get('Unit', '')
            frequency = data.get('Frequency', '')
            importance = data.get('Importance', '')
            impact = data.get('Impact', '')
            source = data.get('Source', '')
            
            # Parse dates
            last_update = datetime.fromisoformat(data.get('LastUpdate', '').replace('Z', '+00:00'))
            next_release = None
            if data.get('NextRelease'):
                next_release = datetime.fromisoformat(data.get('NextRelease', '').replace('Z', '+00:00'))
            
            # Create indicator object
            indicator_obj = EconomicIndicator(
                indicator_id=indicator_id,
                country=country,
                category=category,
                indicator=indicator,
                last_value=last_value,
                previous_value=previous_value,
                forecast_value=forecast_value,
                unit=unit,
                frequency=frequency,
                last_update=last_update,
                next_release=next_release,
                importance=importance,
                impact=impact,
                source=source,
                raw_data=data
            )
            
            return indicator_obj
            
        except Exception as e:
            logger.error(f"Failed to parse indicator: {e}")
            return None
    
    def _parse_calendar_event(self, data: Dict[str, Any]) -> Optional[EconomicCalendar]:
        """
        Parse calendar event data into EconomicCalendar object.
        
        Args:
            data: Raw calendar event data
            
        Returns:
            Parsed EconomicCalendar or None if parsing failed
        """
        try:
            # Extract basic information
            event_id = data.get('ID', '')
            country = data.get('Country', '')
            category = data.get('Category', '')
            event = data.get('Event', '')
            
            # Parse date and time
            date_str = data.get('Date', '')
            time_str = data.get('Time', '')
            
            # Combine date and time
            if date_str and time_str:
                datetime_str = f"{date_str} {time_str}"
                event_date = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            else:
                event_date = datetime.now()
            
            # Extract values
            forecast = float(data.get('Forecast', 0)) if data.get('Forecast') else None
            previous = float(data.get('Previous', 0)) if data.get('Previous') else None
            actual = float(data.get('Actual', 0)) if data.get('Actual') else None
            
            # Extract metadata
            unit = data.get('Unit', '')
            importance = data.get('Importance', '')
            source = data.get('Source', '')
            
            # Create calendar event object
            calendar_event = EconomicCalendar(
                event_id=event_id,
                country=country,
                category=category,
                event=event,
                date=event_date,
                time=time_str,
                importance=importance,
                forecast=forecast,
                previous=previous,
                actual=actual,
                unit=unit,
                source=source,
                raw_data=data
            )
            
            return calendar_event
            
        except Exception as e:
            logger.error(f"Failed to parse calendar event: {e}")
            return None
    
    async def close(self):
        """Close the ingestor and cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info("TradingEconomics Ingestor closed")


# Factory function
def create_tradingeconomics_ingestor(config: Optional[Dict] = None) -> TradingEconomicsIngestor:
    """
    Create TradingEconomics ingestor instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TradingEconomics ingestor instance
    """
    return TradingEconomicsIngestor(config)


# Example usage
async def main():
    """Example usage of TradingEconomics ingestor."""
    config = {
        'base_url': 'https://api.tradingeconomics.com',
        'api_key': 'YOUR_API_KEY_HERE',  # Add your API key here
        'rate_limit': 1000,
        'timeout': 30
    }
    
    ingestor = create_tradingeconomics_ingestor(config)
    
    try:
        # Get US economic indicators
        indicators = await ingestor.get_economic_indicators(country='US')
        print(f"Retrieved {len(indicators)} US indicators")
        
        # Get economic calendar
        calendar = await ingestor.get_economic_calendar(
            country='US',
            importance='High'
        )
        print(f"Retrieved {len(calendar)} calendar events")
        
        # Get historical GDP data
        gdp_data = await ingestor.get_historical_data(
            country='US',
            indicator='GDP',
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now()
        )
        print(f"Retrieved {len(gdp_data)} GDP data points")
        
    finally:
        await ingestor.close()


if __name__ == "__main__":
    asyncio.run(main())
