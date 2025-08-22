#!/usr/bin/env python3
"""
SignaMentis - News Features Package

This package provides news feature extraction and processing including:
- Sentiment-based features
- Volume and timing features
- Source diversity and reliability features
- Composite scoring and feature aggregation

Author: SignaMentis Team
Version: 2.0.0
"""

# Import feature components
from .news_features import NewsFeatureExtractor, NewsFeatureType

# Package exports
__all__ = [
    'NewsFeatureExtractor',
    'NewsFeatureType'
]
