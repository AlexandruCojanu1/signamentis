#!/usr/bin/env python3
"""
SignaMentis - FinBERT Sentiment Analysis Service

This module provides financial sentiment analysis using the FinBERT model.
It analyzes news articles and provides sentiment scores and classifications.

Author: SignaMentis Team
Version: 2.0.0
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import time
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Sentiment labels enumeration."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Sentiment analysis result container."""
    text: str
    sentiment_label: SentimentLabel
    confidence_score: float
    positive_score: float
    negative_score: float
    neutral_score: float
    processing_time: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchSentimentResult:
    """Batch sentiment analysis result container."""
    results: List[SentimentResult]
    total_processing_time: float
    average_confidence: float
    sentiment_distribution: Dict[SentimentLabel, int]
    timestamp: datetime


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
    
    Features:
    - Financial domain-specific sentiment analysis
    - Batch processing capabilities
    - Confidence scoring
    - Sentiment classification
    - Performance optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Model configuration
        self.model_name = self.config.get('model_name', 'ProsusAI/finbert')
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 8)
        self.device = self.config.get('device', 'auto')
        
        # Performance settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_size = self.config.get('cache_size', 1000)
        self.enable_batch_processing = self.config.get('enable_batch_processing', True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.executor = None
        self.cache = {}
        
        # Performance tracking
        self.total_requests = 0
        self.average_processing_time = 0.0
        self.start_time = datetime.now()
        
        logger.info("FinBERT Sentiment Analyzer initialized")
    
    async def initialize(self):
        """Initialize the model and tokenizer asynchronously."""
        try:
            # Run initialization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._initialize_model)
            logger.info("FinBERT model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FinBERT model: {e}")
            raise
    
    def _initialize_model(self):
        """Initialize the FinBERT model and tokenizer."""
        try:
            # Determine device
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize thread pool executor
            self.executor = ThreadPoolExecutor(max_workers=4)
            
            logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing FinBERT model: {e}")
            raise
    
    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis result
        """
        try:
            start_time = time.time()
            
            # Check cache first
            if self.enable_caching:
                cache_key = self._generate_cache_key(text)
                if cache_key in self.cache:
                    cached_result = self.cache[cache_key]
                    cached_result.processing_time = 0.001  # Cache hit time
                    return cached_result
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Run analysis in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._analyze_sentiment_sync,
                processed_text
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result object
            sentiment_result = SentimentResult(
                text=text,
                sentiment_label=result['label'],
                confidence_score=result['confidence'],
                positive_score=result['scores']['positive'],
                negative_score=result['scores']['negative'],
                neutral_score=result['scores']['neutral'],
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={'model': self.model_name, 'device': self.device}
            )
            
            # Cache result
            if self.enable_caching:
                self._cache_result(text, sentiment_result)
            
            # Update performance metrics
            self._update_performance_metrics(processing_time)
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            # Return neutral sentiment on error
            return SentimentResult(
                text=text,
                sentiment_label=SentimentLabel.NEUTRAL,
                confidence_score=0.0,
                positive_score=0.33,
                negative_score=0.33,
                neutral_score=0.34,
                processing_time=0.0,
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )
    
    async def analyze_sentiment_batch(self, texts: List[str]) -> BatchSentimentResult:
        """
        Analyze sentiment of multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Batch sentiment analysis result
        """
        try:
            start_time = time.time()
            
            if not self.enable_batch_processing:
                # Process individually
                results = []
                for text in texts:
                    result = await self.analyze_sentiment(text)
                    results.append(result)
            else:
                # Process in batches
                results = await self._process_batch(texts)
            
            # Calculate batch metrics
            total_time = time.time() - start_time
            confidence_scores = [r.confidence_score for r in results]
            average_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # Count sentiment distribution
            sentiment_distribution = {label: 0 for label in SentimentLabel}
            for result in results:
                sentiment_distribution[result.sentiment_label] += 1
            
            # Create batch result
            batch_result = BatchSentimentResult(
                results=results,
                total_processing_time=total_time,
                average_confidence=average_confidence,
                sentiment_distribution=sentiment_distribution,
                timestamp=datetime.now()
            )
            
            logger.info(f"Processed batch of {len(texts)} texts in {total_time:.2f}s")
            return batch_result
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment batch: {e}")
            # Return empty batch result on error
            return BatchSentimentResult(
                results=[],
                total_processing_time=0.0,
                average_confidence=0.0,
                sentiment_distribution={label: 0 for label in SentimentLabel},
                timestamp=datetime.now()
            )
    
    async def _process_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Process texts in batches for efficiency."""
        try:
            results = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Preprocess batch
                processed_batch = [self._preprocess_text(text) for text in batch_texts]
                
                # Run batch analysis
                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(
                    self.executor,
                    self._analyze_sentiment_batch_sync,
                    processed_batch
                )
                
                # Convert to SentimentResult objects
                for j, (text, result) in enumerate(zip(batch_texts, batch_results)):
                    sentiment_result = SentimentResult(
                        text=text,
                        sentiment_label=result['label'],
                        confidence_score=result['confidence'],
                        positive_score=result['scores']['positive'],
                        negative_score=result['scores']['negative'],
                        neutral_score=result['scores']['neutral'],
                        processing_time=0.0,  # Will be calculated in batch
                        timestamp=datetime.now(),
                        metadata={'model': self.model_name, 'device': self.device}
                    )
                    results.append(sentiment_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            return []
    
    def _analyze_sentiment_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous sentiment analysis."""
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Get scores
            scores = {
                'positive': float(probabilities[0][0]),
                'negative': float(probabilities[0][1]),
                'neutral': float(probabilities[0][2])
            }
            
            # Determine label and confidence
            label_idx = torch.argmax(probabilities[0]).item()
            label_map = {0: SentimentLabel.POSITIVE, 1: SentimentLabel.NEGATIVE, 2: SentimentLabel.NEUTRAL}
            label = label_map[label_idx]
            confidence = float(probabilities[0][label_idx])
            
            return {
                'label': label,
                'confidence': confidence,
                'scores': scores
            }
            
        except Exception as e:
            logger.error(f"Error in synchronous sentiment analysis: {e}")
            # Return neutral sentiment on error
            return {
                'label': SentimentLabel.NEUTRAL,
                'confidence': 0.0,
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            }
    
    def _analyze_sentiment_batch_sync(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Synchronous batch sentiment analysis."""
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Process results
            results = []
            for i in range(len(texts)):
                scores = {
                    'positive': float(probabilities[i][0]),
                    'negative': float(probabilities[i][1]),
                    'neutral': float(probabilities[i][2])
                }
                
                label_idx = torch.argmax(probabilities[i]).item()
                label_map = {0: SentimentLabel.POSITIVE, 1: SentimentLabel.NEGATIVE, 2: SentimentLabel.NEUTRAL}
                label = label_map[label_idx]
                confidence = float(probabilities[i][label_idx])
                
                results.append({
                    'label': label,
                    'confidence': confidence,
                    'scores': scores
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in synchronous batch sentiment analysis: {e}")
            # Return neutral sentiment for all texts on error
            return [{
                'label': SentimentLabel.NEUTRAL,
                'confidence': 0.0,
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            } for _ in texts]
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove special characters but keep financial symbols
            text = re.sub(r'[^\w\s\.\,\!\?\-\$\%\@\#\&\*\(\)\[\]\{\}\+\=\|\/\\\:\;\<\>\"\']', '', text)
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            # Truncate if too long
            if len(text) > self.max_length * 4:  # Rough character estimate
                text = text[:self.max_length * 4] + '...'
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Simple hash-based key
        return str(hash(text) % (2**32))
    
    def _cache_result(self, text: str, result: SentimentResult):
        """Cache sentiment analysis result."""
        try:
            cache_key = self._generate_cache_key(text)
            
            # Implement LRU cache
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance tracking metrics."""
        try:
            self.total_requests += 1
            
            # Update average processing time
            if self.total_requests == 1:
                self.average_processing_time = processing_time
            else:
                self.average_processing_time = (
                    (self.average_processing_time * (self.total_requests - 1) + processing_time) 
                    / self.total_requests
                )
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return {
                'total_requests': self.total_requests,
                'average_processing_time': self.average_processing_time,
                'uptime_seconds': uptime,
                'requests_per_second': self.total_requests / uptime if uptime > 0 else 0,
                'cache_size': len(self.cache),
                'cache_hit_rate': 0.0,  # Would need to track cache hits
                'device': self.device,
                'model_name': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the sentiment analysis cache."""
        try:
            self.cache.clear()
            logger.info("Sentiment analysis cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def close(self):
        """Close the analyzer and cleanup resources."""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            
            # Clear cache
            self.clear_cache()
            
            logger.info("FinBERT Sentiment Analyzer closed")
            
        except Exception as e:
            logger.error(f"Error closing analyzer: {e}")


# Factory function
def create_finbert_analyzer(config: Optional[Dict] = None) -> FinBERTSentimentAnalyzer:
    """
    Create FinBERT sentiment analyzer instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FinBERT sentiment analyzer instance
    """
    return FinBERTSentimentAnalyzer(config)


# Example usage
async def main():
    """Example usage of FinBERT sentiment analyzer."""
    config = {
        'model_name': 'ProsusAI/finbert',
        'max_length': 512,
        'batch_size': 8,
        'device': 'auto',
        'enable_caching': True,
        'cache_size': 1000,
        'enable_batch_processing': True
    }
    
    analyzer = create_finbert_analyzer(config)
    
    try:
        # Initialize
        await analyzer.initialize()
        
        # Sample financial texts
        texts = [
            "Gold prices surged to new highs as investors seek safe haven assets.",
            "Stock market crashes amid economic uncertainty and inflation fears.",
            "Federal Reserve announces interest rate hike to combat inflation.",
            "Company reports strong quarterly earnings, beating expectations.",
            "Trade tensions escalate between major economies."
        ]
        
        # Analyze individual text
        result = await analyzer.analyze_sentiment(texts[0])
        print(f"Sentiment: {result.sentiment_label.value}, Confidence: {result.confidence_score:.3f}")
        
        # Analyze batch
        batch_result = await analyzer.analyze_sentiment_batch(texts)
        print(f"Processed {len(batch_result.results)} texts in batch")
        
        # Get performance stats
        stats = analyzer.get_performance_stats()
        print(f"Performance: {stats}")
        
    finally:
        await analyzer.close()


if __name__ == "__main__":
    asyncio.run(main())
