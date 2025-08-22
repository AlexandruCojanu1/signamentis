# SignaMentis News NLP Service

## Overview

The News NLP Service provides comprehensive news ingestion, sentiment analysis, and feature extraction capabilities for the SignaMentis trading system. It integrates multiple news sources and uses advanced NLP techniques to create actionable trading signals.

## Features

### 🗞️ News Ingestion
- **GDELT**: Global Database of Events, Language, and Tone
- **TradingEconomics**: Economic indicators and calendar events
- **NewsAPI**: Real-time financial news articles
- **GNews**: Google News financial content

### 🧠 Sentiment Analysis
- **FinBERT**: Financial domain-specific sentiment analysis
- **Batch Processing**: Efficient processing of multiple texts
- **Confidence Scoring**: Reliability metrics for sentiment predictions
- **Caching**: Performance optimization with result caching

### 📊 Feature Extraction
- **Sentiment Features**: Sentiment scores, volume, and momentum
- **Volume Features**: News volume and intensity metrics
- **Timing Features**: Urgency and breaking news indicators
- **Source Features**: Diversity and reliability scoring
- **Composite Scoring**: Weighted combination of all features

## Architecture

```
services/news_nlp/
├── ingestors/           # News source integrations
│   ├── gdelt.py        # GDELT API client
│   ├── tradingeconomics.py  # TradingEconomics API client
│   ├── newsapi.py      # NewsAPI client
│   └── gnews.py        # GNews client
├── nlp/                 # Natural language processing
│   └── finbert.py      # FinBERT sentiment analyzer
├── features/            # Feature extraction
│   └── news_features.py # News feature processor
├── api.py              # FastAPI service
└── __init__.py         # Package initialization
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GDELT_API_KEY="your_gdelt_key"
export TRADINGECONOMICS_API_KEY="your_te_key"
export NEWSAPI_API_KEY="your_newsapi_key"
export GNEWS_API_KEY="your_gnews_key"
```

### 2. Basic Usage

```python
from services.news_nlp import NewsFeatureExtractor, FinBERTSentimentAnalyzer

# Initialize components
sentiment_analyzer = FinBERTSentimentAnalyzer()
await sentiment_analyzer.initialize()

feature_extractor = NewsFeatureExtractor()

# Analyze sentiment
result = await sentiment_analyzer.analyze_sentiment(
    "Gold prices surge on safe haven demand"
)
print(f"Sentiment: {result.sentiment_label.value}")

# Extract features
features = feature_extractor.extract_features()
print(f"Composite Score: {features.composite_score:.3f}")
```

### 3. API Service

```bash
# Start the API service
cd services/news_nlp
python api.py

# The service will be available at http://localhost:8001
```

## API Endpoints

### News Ingestion
- `POST /news/ingest` - Ingest news from specified source
- `GET /status` - Get status of all news sources

### Sentiment Analysis
- `POST /sentiment/analyze` - Analyze sentiment of single text
- `POST /sentiment/batch` - Analyze sentiment of multiple texts

### Feature Extraction
- `POST /features/extract` - Extract news features

### System Health
- `GET /health` - System health check
- `GET /metrics` - Performance metrics

## Configuration

### Environment Variables

```bash
# API Configuration
NEWS_NLP_API_KEY=your_api_key
NEWS_NLP_RATE_LIMIT=1000
NEWS_NLP_ENABLE_AUTH=true

# GDELT Configuration
GDELT_API_KEY=your_gdelt_key
GDELT_RATE_LIMIT=1000

# TradingEconomics Configuration
TRADINGECONOMICS_API_KEY=your_te_key
TRADINGECONOMICS_RATE_LIMIT=1000

# NewsAPI Configuration
NEWSAPI_API_KEY=your_newsapi_key
NEWSAPI_RATE_LIMIT=100

# GNews Configuration
GNEWS_API_KEY=your_gnews_key
GNEWS_RATE_LIMIT=100
```

### Configuration File

```yaml
# config/news_nlp.yaml
gdelt:
  api_key: "your_key"
  rate_limit: 1000
  timeout: 30

tradingeconomics:
  api_key: "your_key"
  rate_limit: 1000
  timeout: 30

newsapi:
  api_key: "your_key"
  rate_limit: 100
  timeout: 30

gnews:
  api_key: "your_key"
  rate_limit: 100
  timeout: 30

finbert:
  model_name: "ProsusAI/finbert"
  max_length: 512
  batch_size: 8
  device: "auto"

features:
  sentiment_window: 24
  volume_window: 6
  momentum_window: 2
  urgency_threshold: 0.7
```

## Testing

### Run All Tests

```bash
# Run complete test suite
make test

# Run only News NLP tests
make test-news-nlp

# Run specific test file
python test_news_nlp.py
```

### Test Coverage

```bash
# Run tests with coverage
make test-coverage
```

## Performance

### Optimization Features

- **Async Processing**: Non-blocking API calls and processing
- **Batch Processing**: Efficient handling of multiple requests
- **Caching**: LRU cache for sentiment analysis results
- **Rate Limiting**: Intelligent API rate limit management
- **Connection Pooling**: Reusable HTTP connections

### Monitoring

- **Performance Metrics**: Request times, throughput, error rates
- **Health Checks**: Component status and availability
- **Resource Usage**: Memory, CPU, and cache statistics

## Integration

### With Trading System

```python
from services.news_nlp import NewsFeatureExtractor
from scripts.risk_manager import RiskManager

# Initialize components
news_features = NewsFeatureExtractor()
risk_manager = RiskManager()

# Get news features for risk adjustment
features = news_features.extract_features()
risk_multiplier = calculate_news_risk_multiplier(features.composite_score)

# Apply to risk management
risk_manager.adjust_risk_for_news(features)
```

### With Risk Management

```python
# News-driven risk adjustment
if features.news_urgency > 0.8:
    # High urgency news - reduce position sizes
    risk_manager.set_news_risk_multiplier(0.5)
elif features.sentiment_score < -0.5:
    # Negative sentiment - increase stop loss
    risk_manager.adjust_stop_loss_multiplier(1.2)
```

## Development

### Adding New News Sources

1. Create new ingestor in `ingestors/` directory
2. Implement required methods (get_news, get_status, etc.)
3. Add to `NewsNLPService` initialization
4. Update API endpoints and documentation

### Customizing Feature Extraction

1. Modify `NewsFeatureExtractor` class
2. Add new feature types to `NewsFeatureType` enum
3. Implement feature calculation methods
4. Update composite scoring algorithm

### Extending Sentiment Analysis

1. Add new models to `nlp/` directory
2. Implement sentiment analysis interface
3. Update `FinBERTSentimentAnalyzer` or create new analyzer
4. Add configuration options

## Troubleshooting

### Common Issues

1. **API Key Errors**: Verify environment variables are set correctly
2. **Rate Limiting**: Check API rate limits and adjust configuration
3. **Model Loading**: Ensure sufficient memory for FinBERT model
4. **Network Issues**: Verify internet connectivity and firewall settings

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python api.py --log-level debug
```

## Contributing

1. Follow PEP 8 coding standards
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update documentation and README
5. Test with multiple news sources

## License

This project is part of the SignaMentis trading system and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation
3. Check system logs
4. Contact the development team

---

**Version**: 2.0.0  
**Last Updated**: December 2024  
**Author**: SignaMentis Team
