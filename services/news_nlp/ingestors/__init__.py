#!/usr/bin/env python3
"""
SignaMentis - News Ingestors Package

This package provides news ingestion from various sources including:
- GDELT (Global Database of Events, Language, and Tone)
- TradingEconomics (Economic indicators and calendar)
- NewsAPI (Financial news articles)
- GNews (Google News integration)

Author: SignaMentis Team
Version: 2.0.0
"""

# Import all ingestors
from .gdelt import GDELTIngestor, GDELTEventType, GDELTEvent
from .tradingeconomics import TradingEconomicsIngestor, EconomicIndicatorType
from .newsapi import NewsAPIIngestor, NewsCategory, NewsLanguage
from .gnews import GNewsIngestor, GNewsCategory, GNewsLanguage

# Package exports
__all__ = [
    'GDELTIngestor',
    'GDELTEventType',
    'GDELTEvent',
    'TradingEconomicsIngestor',
    'EconomicIndicatorType',
    'NewsAPIIngestor',
    'NewsCategory',
    'NewsLanguage',
    'GNewsIngestor',
    'GNewsCategory',
    'GNewsLanguage'
]
