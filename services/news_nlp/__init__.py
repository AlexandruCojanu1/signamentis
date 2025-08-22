#!/usr/bin/env python3
"""
SignaMentis - News NLP Service Package

This package provides news ingestion, sentiment analysis, and feature extraction
for the SignaMentis trading system.

Author: SignaMentis Team
Version: 2.0.0
"""

# Import main components
from .ingestors.gdelt import GDELTIngestor, GDELTEventType, GDELTEvent
from .ingestors.tradingeconomics import TradingEconomicsIngestor, EconomicIndicatorType
from .ingestors.newsapi import NewsAPIIngestor, NewsCategory, NewsLanguage
from .ingestors.gnews import GNewsIngestor, GNewsCategory, GNewsLanguage
from .nlp.finbert import FinBERTSentimentAnalyzer, SentimentLabel
from .features.news_features import NewsFeatureExtractor, NewsFeatureType

# Version information
__version__ = "2.0.0"
__author__ = "SignaMentis Team"

# Package exports
__all__ = [
    # Ingestors
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
    'GNewsLanguage',
    
    # NLP
    'FinBERTSentimentAnalyzer',
    'SentimentLabel',
    
    # Features
    'NewsFeatureExtractor',
    'NewsFeatureType'
]
