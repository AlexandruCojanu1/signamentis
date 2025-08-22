#!/usr/bin/env python3
"""
SignaMentis - NLP Package

This package provides natural language processing capabilities including:
- FinBERT sentiment analysis for financial text
- Text preprocessing and normalization
- Batch processing and performance optimization

Author: SignaMentis Team
Version: 2.0.0
"""

# Import NLP components
from .finbert import FinBERTSentimentAnalyzer, SentimentLabel

# Package exports
__all__ = [
    'FinBERTSentimentAnalyzer',
    'SentimentLabel'
]
