#!/usr/bin/env python3
"""
SignaMentis - Feast Feature Definitions

This module defines all features for the SignaMentis trading system
using Feast feature store framework.

Author: SignaMentis Team
Version: 2.0.0
"""

from datetime import timedelta
from feast import (
    Entity, Feature, FeatureView, FeatureService, Field,
    FileSource, RedisSource, ValueType
)
from feast.types import Float32, Float64, Int64, String
import pandas as pd

# Define entities
symbol_entity = Entity(
    name="symbol",
    value_type=ValueType.STRING,
    description="Trading symbol (e.g., XAUUSD)",
    join_keys=["symbol"]
)

timestamp_entity = Entity(
    name="timestamp",
    value_type=ValueType.UNIX_TIMESTAMP,
    description="Timestamp for the feature",
    join_keys=["timestamp"]
)

# Define data sources
market_data_source = FileSource(
    path="data/processed/market_data.parquet",
    timestamp_field="timestamp",
    created_timestamp_column="created_at"
)

news_data_source = FileSource(
    path="data/processed/news_features.parquet",
    timestamp_field="timestamp",
    created_timestamp_column="created_at"
)

technical_indicators_source = FileSource(
    path="data/processed/technical_indicators.parquet",
    timestamp_field="timestamp",
    created_timestamp_column="created_at"
)

# Market data features
market_data_features = FeatureView(
    name="market_data_features",
    entities=[symbol_entity, timestamp_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="open", dtype=Float64),
        Field(name="high", dtype=Float64),
        Field(name="low", dtype=Float64),
        Field(name="close", dtype=Float64),
        Field(name="volume", dtype=Float64),
        Field(name="spread", dtype=Float64),
        Field(name="bid", dtype=Float64),
        Field(name="ask", dtype=Float64)
    ],
    source=market_data_source,
    online=True
)

# Technical indicator features
technical_indicator_features = FeatureView(
    name="technical_indicator_features",
    entities=[symbol_entity, timestamp_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="sma_20", dtype=Float64),
        Field(name="sma_50", dtype=Float64),
        Field(name="sma_200", dtype=Float64),
        Field(name="ema_12", dtype=Float64),
        Field(name="ema_26", dtype=Float64),
        Field(name="rsi", dtype=Float64),
        Field(name="macd", dtype=Float64),
        Field(name="macd_signal", dtype=Float64),
        Field(name="macd_histogram", dtype=Float64),
        Field(name="bb_upper", dtype=Float64),
        Field(name="bb_middle", dtype=Float64),
        Field(name="bb_lower", dtype=Float64),
        Field(name="bb_width", dtype=Float64),
        Field(name="bb_position", dtype=Float64),
        Field(name="atr", dtype=Float64),
        Field(name="supertrend", dtype=Float64),
        Field(name="stochastic_k", dtype=Float64),
        Field(name="stochastic_d", dtype=Float64),
        Field(name="williams_r", dtype=Float64),
        Field(name="cci", dtype=Float64),
        Field(name="adx", dtype=Float64),
        Field(name="parabolic_sar", dtype=Float64),
        Field(name="ichimoku_a", dtype=Float64),
        Field(name="ichimoku_b", dtype=Float64),
        Field(name="ichimoku_c", dtype=Float64),
        Field(name="ichimoku_d", dtype=Float64)
    ],
    source=technical_indicators_source,
    online=True
)

# News sentiment features
news_sentiment_features = FeatureView(
    name="news_sentiment_features",
    entities=[symbol_entity, timestamp_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="sentiment_score", dtype=Float64),
        Field(name="sentiment_label", dtype=String),
        Field(name="news_volume", dtype=Float64),
        Field(name="source_diversity", dtype=Float64),
        Field(name="relevance_score", dtype=Float64),
        Field(name="momentum_score", dtype=Float64),
        Field(name="composite_score", dtype=Float64),
        Field(name="gdelt_sentiment", dtype=Float64),
        Field(name="newsapi_sentiment", dtype=Float64),
        Field(name="gnews_sentiment", dtype=Float64),
        Field(name="tradingeconomics_sentiment", dtype=Float64)
    ],
    source=news_data_source,
    online=True
)

# Volatility features
volatility_features = FeatureView(
    name="volatility_features",
    entities=[symbol_entity, timestamp_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="realized_volatility", dtype=Float64),
        Field(name="implied_volatility", dtype=Float64),
        Field(name="volatility_ratio", dtype=Float64),
        Field(name="volatility_regime", dtype=String),
        Field(name="volatility_forecast", dtype=Float64),
        Field(name="volatility_skew", dtype=Float64),
        Field(name="volatility_kurtosis", dtype=Float64)
    ],
    source=technical_indicators_source,
    online=True
)

# Market structure features
market_structure_features = FeatureView(
    name="market_structure_features",
    entities=[symbol_entity, timestamp_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="support_level", dtype=Float64),
        Field(name="resistance_level", dtype=Float64),
        Field(name="pivot_point", dtype=Float64),
        Field(name="fibonacci_23", dtype=Float64),
        Field(name="fibonacci_38", dtype=Float64),
        Field(name="fibonacci_50", dtype=Float64),
        Field(name="fibonacci_61", dtype=Float64),
        Field(name="fibonacci_78", dtype=Float64),
        Field(name="order_block_high", dtype=Float64),
        Field(name="order_block_low", dtype=Float64),
        Field(name="fair_value_gap_high", dtype=Float64),
        Field(name="fair_value_gap_low", dtype=Float64),
        Field(name="liquidity_zone_high", dtype=Float64),
        Field(name="liquidity_zone_low", dtype=Float64)
    ],
    source=technical_indicators_source,
    online=True
)

# Time-based features
time_features = FeatureView(
    name="time_features",
    entities=[symbol_entity, timestamp_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="day_of_month", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="quarter", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="is_holiday", dtype=Int64),
        Field(name="session_asia", dtype=Int64),
        Field(name="session_london", dtype=Int64),
        Field(name="session_newyork", dtype=Int64),
        Field(name="session_overlap", dtype=Int64),
        Field(name="time_to_session_open", dtype=Float64),
        Field(name="time_to_session_close", dtype=Float64)
    ],
    source=market_data_source,
    online=True
)

# Correlation features
correlation_features = FeatureView(
    name="correlation_features",
    entities=[symbol_entity, timestamp_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="correlation_usd", dtype=Float64),
        Field(name="correlation_eur", dtype=Float64),
        Field(name="correlation_gbp", dtype=Float64),
        Field(name="correlation_jpy", dtype=Float64),
        Field(name="correlation_aud", dtype=Float64),
        Field(name="correlation_cad", dtype=Float64),
        Field(name="correlation_chf", dtype=Float64),
        Field(name="correlation_nzd", dtype=Float64),
        Field(name="correlation_sp500", dtype=Float64),
        Field(name="correlation_nasdaq", dtype=Float64),
        Field(name="correlation_dow", dtype=Float64),
        Field(name="correlation_vix", dtype=Float64),
        Field(name="correlation_oil", dtype=Float64),
        Field(name="correlation_bonds", dtype=Float64)
    ],
    source=technical_indicators_source,
    online=True
)

# Economic calendar features
economic_calendar_features = FeatureView(
    name="economic_calendar_features",
    entities=[symbol_entity, timestamp_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="next_event_hours", dtype=Float64),
        Field(name="event_importance", dtype=String),
        Field(name="event_currency", dtype=String),
        Field(name="event_impact", dtype=Float64),
        Field(name="consensus_forecast", dtype=Float64),
        Field(name="previous_value", dtype=Float64),
        Field(name="actual_value", dtype=Float64),
        Field(name="surprise", dtype=Float64),
        Field(name="surprise_ratio", dtype=Float64)
    ],
    source=news_data_source,
    online=True
)

# Risk management features
risk_management_features = FeatureView(
    name="risk_management_features",
    entities=[symbol_entity, timestamp_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="current_drawdown", dtype=Float64),
        Field(name="max_drawdown", dtype=Float64),
        Field(name="var_95", dtype=Float64),
        Field(name="var_99", dtype=Float64),
        Field(name="expected_shortfall", dtype=Float64),
        Field(name="kelly_criterion", dtype=Float64),
        Field(name="position_size", dtype=Float64),
        Field(name="risk_per_trade", dtype=Float64),
        Field(name="daily_pnl", dtype=Float64),
        Field(name="monthly_pnl", dtype=Float64),
        Field(name="sharpe_ratio", dtype=Float64),
        Field(name="sortino_ratio", dtype=Float64),
        Field(name="calmar_ratio", dtype=Float64),
        Field(name="profit_factor", dtype=Float64),
        Field(name="win_rate", dtype=Float64),
        Field(name="avg_win", dtype=Float64),
        Field(name="avg_loss", dtype=Float64),
        Field(name="max_consecutive_losses", dtype=Int64)
    ],
    source=technical_indicators_source,
    online=True
)

# Feature services for different use cases
trading_features = FeatureService(
    name="trading_features",
    features=[
        market_data_features,
        technical_indicator_features,
        news_sentiment_features,
        volatility_features,
        market_structure_features,
        time_features
    ]
)

risk_features = FeatureService(
    name="risk_features",
    features=[
        risk_management_features,
        volatility_features,
        correlation_features,
        economic_calendar_features
    ]
)

news_features = FeatureService(
    name="news_features",
    features=[
        news_sentiment_features,
        economic_calendar_features
    ]
)

# Complete feature service
complete_features = FeatureService(
    name="complete_features",
    features=[
        market_data_features,
        technical_indicator_features,
        news_sentiment_features,
        volatility_features,
        market_structure_features,
        time_features,
        correlation_features,
        economic_calendar_features,
        risk_management_features
    ]
)
