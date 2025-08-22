// MongoDB initialization script for SignaMentis
// This script runs when the MongoDB container starts for the first time

// Create the main database
db = db.getSiblingDB('signa_mentis');

// Create collections with proper indexes
db.createCollection('market_data');
db.createCollection('trading_signals');
db.createCollection('portfolio_positions');
db.createCollection('news_articles');
db.createCollection('sentiment_scores');
db.createCollection('model_predictions');
db.createCollection('backtest_results');
db.createCollection('risk_metrics');
db.createCollection('user_preferences');
db.createCollection('system_logs');

// Create indexes for better performance
db.market_data.createIndex({ "symbol": 1, "timestamp": -1 });
db.market_data.createIndex({ "timestamp": -1 });
db.market_data.createIndex({ "symbol": 1, "date": 1 });

db.trading_signals.createIndex({ "symbol": 1, "timestamp": -1 });
db.trading_signals.createIndex({ "timestamp": -1 });
db.trading_signals.createIndex({ "signal_type": 1, "confidence": -1 });

db.portfolio_positions.createIndex({ "symbol": 1, "timestamp": -1 });
db.portfolio_positions.createIndex({ "timestamp": -1 });

db.news_articles.createIndex({ "timestamp": -1 });
db.news_articles.createIndex({ "symbols": 1 });
db.news_articles.createIndex({ "sentiment": 1 });

db.sentiment_scores.createIndex({ "symbol": 1, "timestamp": -1 });
db.sentiment_scores.createIndex({ "timestamp": -1 });

db.model_predictions.createIndex({ "symbol": 1, "timestamp": -1 });
db.model_predictions.createIndex({ "model_name": 1, "timestamp": -1 });

db.backtest_results.createIndex({ "strategy_name": 1, "start_date": -1 });
db.backtest_results.createIndex({ "timestamp": -1 });

db.risk_metrics.createIndex({ "portfolio_id": 1, "timestamp": -1 });
db.risk_metrics.createIndex({ "timestamp": -1 });

db.system_logs.createIndex({ "timestamp": -1 });
db.system_logs.createIndex({ "level": 1, "timestamp": -1 });

// Create a user for the application
db.createUser({
  user: 'signa_mentis_app',
  pwd: 'signa_mentis_app_2024',
  roles: [
    { role: 'readWrite', db: 'signa_mentis' },
    { role: 'dbAdmin', db: 'signa_mentis' }
  ]
});

// Insert initial configuration data
db.system_config.insertOne({
  _id: "system_config",
  version: "1.0.0",
  created_at: new Date(),
  updated_at: new Date(),
  features: {
    real_time_trading: true,
    news_sentiment_analysis: true,
    ml_model_training: true,
    risk_management: true,
    backtesting: true
  },
  settings: {
    max_position_size: 100000,
    risk_per_trade: 0.02,
    max_daily_loss: 0.05,
    trading_hours: {
      start: "09:30",
      end: "16:00"
    }
  }
});

print("MongoDB initialization completed successfully!");
print("Database: signa_mentis");
print("User: signa_mentis_app");
print("Collections created with proper indexes");
