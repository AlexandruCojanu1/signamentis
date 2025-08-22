# SignaMentis - MLflow Setup Guide

## Overview

This document describes the MLflow setup for the SignaMentis trading system, enabling experiment tracking, model versioning, and model deployment.

## Features

- **Experiment Tracking**: Track all AI model training experiments
- **Model Versioning**: Version control for trained models
- **Model Registry**: Centralized model storage and management
- **Artifact Storage**: Store model artifacts, metrics, and visualizations
- **Model Deployment**: Deploy models to production environments

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Training      │    │   MLflow        │    │   Production    │
│   Scripts       │───▶│   Tracking      │───▶│   Environment   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Model         │
                       │   Registry      │
                       └─────────────────┘
```

## Installation

### 1. Install MLflow

```bash
pip install mlflow
```

### 2. Install Additional Dependencies

```bash
pip install boto3  # For S3 backend
pip install azure-storage-blob  # For Azure backend
pip install google-cloud-storage  # For GCS backend
```

### 3. Install MLflow Extras

```bash
pip install mlflow[extras]  # Includes all MLflow features
```

## Configuration

### 1. Environment Variables

```bash
# MLflow Configuration
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export MLFLOW_S3_BUCKET=mlflow-artifacts

# Optional: Azure Storage
export AZURE_STORAGE_CONNECTION_STRING=your_connection_string
export MLFLOW_AZURE_STORAGE_CONTAINER=mlflow-artifacts

# Optional: Google Cloud Storage
export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
export MLFLOW_GCS_BUCKET=mlflow-artifacts
```

### 2. MLflow Configuration File

Create `mlflow.conf`:

```ini
[default]
artifact_root = s3://mlflow-artifacts
default_artifact_root = s3://mlflow-artifacts

[tracking]
tracking_uri = http://localhost:5000

[registry]
registry_uri = sqlite:///mlflow.db

[model_registry]
registry_uri = sqlite:///mlflow.db
```

## Usage

### 1. Start MLflow Server

```bash
# Local server
mlflow server --host 0.0.0.0 --port 5000

# With SQLite backend
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# With S3 backend
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://mlflow-artifacts
```

### 2. Python API Usage

```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Start experiment
mlflow.set_experiment("signa_mentis_trading")

# Log parameters
mlflow.log_param("learning_rate", 0.001)
mlflow.log_param("batch_size", 32)
mlflow.log_param("epochs", 100)

# Log metrics
mlflow.log_metric("train_loss", 0.123)
mlflow.log_metric("val_loss", 0.145)
mlflow.log_metric("accuracy", 0.89)

# Log model
mlflow.pytorch.log_model(model, "model")

# Log artifacts
mlflow.log_artifact("training_plot.png")
mlflow.log_artifact("model_config.yaml")

# End run
mlflow.end_run()
```

### 3. Model Registry

```python
# Register model
model_name = "signa_mentis_bilstm"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/model",
    name=model_name
)

# Load model from registry
loaded_model = mlflow.pytorch.load_model(
    f"models:/{model_name}/Production"
)

# Transition model stage
client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)
```

## Experiment Structure

### 1. Experiment Organization

```
Experiments/
├── signa_mentis_trading/
│   ├── BiLSTM_Model/
│   ├── GRU_Model/
│   ├── Transformer_Model/
│   ├── LNN_Model/
│   └── LTN_Model/
├── signa_mentis_ensemble/
│   ├── Model_Combination/
│   └── Weight_Optimization/
└── signa_mentis_backtesting/
    ├── Strategy_Validation/
    └── Risk_Management/
```

### 2. Run Tags

```python
# Standard tags for all runs
mlflow.set_tag("project", "signa_mentis")
mlflow.set_tag("version", "2.0.0")
mlflow.set_tag("author", "signa_mentis_team")
mlflow.set_tag("model_type", "bilstm")
mlflow.set_tag("symbol", "XAUUSD")
mlflow.set_tag("timeframe", "5m")
mlflow.set_tag("strategy", "supertrend")
```

### 3. Metrics to Track

```python
# Training metrics
mlflow.log_metric("train_loss", train_loss)
mlflow.log_metric("val_loss", val_loss)
mlflow.log_metric("train_accuracy", train_acc)
mlflow.log_metric("val_accuracy", val_acc)

# Trading metrics
mlflow.log_metric("sharpe_ratio", sharpe)
mlflow.log_metric("max_drawdown", max_dd)
mlflow.log_metric("win_rate", win_rate)
mlflow.log_metric("profit_factor", profit_factor)
mlflow.log_metric("total_return", total_return)

# Performance metrics
mlflow.log_metric("training_time", training_time)
mlflow.log_metric("inference_time", inference_time)
mlflow.log_metric("memory_usage", memory_usage)
```

## Model Deployment

### 1. Local Deployment

```python
# Save model locally
mlflow.pytorch.save_model(model, "local_model")

# Load model locally
loaded_model = mlflow.pytorch.load_model("local_model")
```

### 2. Production Deployment

```python
# Deploy to production
mlflow.deploy(
    model_uri=f"models:/{model_name}/Production",
    target="local",
    port=5001
)
```

### 3. Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install MLflow
RUN pip install mlflow[extras]

# Copy model
COPY model/ /app/model/

# Expose port
EXPOSE 5001

# Start MLflow serving
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-p", "5001"]
```

## Monitoring and Logging

### 1. MLflow UI

Access the MLflow UI at `http://localhost:5000` to:
- View experiments and runs
- Compare model performance
- Download models and artifacts
- Manage model registry

### 2. Logging Best Practices

```python
# Log everything important
mlflow.log_param("random_seed", 42)
mlflow.log_param("data_version", "1.0.0")
mlflow.log_param("feature_engineering_version", "2.1.0")

# Log data statistics
mlflow.log_metric("data_size", len(dataset))
mlflow.log_metric("feature_count", X.shape[1])
mlflow.log_metric("class_balance", class_balance)

# Log model architecture
mlflow.log_param("model_architecture", "BiLSTM")
mlflow.log_param("hidden_size", 128)
mlflow.log_param("num_layers", 2)
mlflow.log_param("dropout", 0.2)

# Log training configuration
mlflow.log_param("optimizer", "Adam")
mlflow.log_param("scheduler", "ReduceLROnPlateau")
mlflow.log_param("early_stopping", True)
mlflow.log_param("patience", 10)
```

## Integration with SignaMentis

### 1. Training Scripts

```python
# In training scripts
from mlflow import log_experiment_metrics

def train_model():
    # ... training code ...
    
    # Log to MLflow
    log_experiment_metrics(
        experiment_name="signa_mentis_trading",
        run_name="bilstm_xauusd_5m",
        metrics={
            "train_loss": train_loss,
            "val_loss": val_loss,
            "sharpe_ratio": sharpe_ratio
        },
        params={
            "model_type": "bilstm",
            "symbol": "XAUUSD",
            "timeframe": "5m"
        }
    )
```

### 2. Model Loading

```python
# In production code
def load_trading_model():
    model = mlflow.pytorch.load_model(
        f"models:/signa_mentis_bilstm/Production"
    )
    return model
```

### 3. Experiment Comparison

```python
# Compare experiments
def compare_models():
    client = MlflowClient()
    
    # Get all runs for a specific experiment
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="metrics.sharpe_ratio > 1.5"
    )
    
    # Compare performance
    for run in runs:
        print(f"Run {run.info.run_id}: Sharpe = {run.data.metrics['sharpe_ratio']}")
```

## Troubleshooting

### 1. Common Issues

- **Connection refused**: Check if MLflow server is running
- **Permission denied**: Check file permissions for artifact storage
- **Model not found**: Verify model URI and version

### 2. Debug Commands

```bash
# Check MLflow status
mlflow server --help

# List experiments
mlflow experiments list

# List runs
mlflow runs list --experiment-id 1

# Download model
mlflow models download -m models:/model_name/version
```

## Next Steps

1. Set up MLflow server with persistent storage
2. Integrate MLflow logging into training scripts
3. Set up model registry for production deployment
4. Configure artifact storage (S3, Azure, GCS)
5. Set up monitoring and alerting
6. Implement CI/CD pipeline integration

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Deployment](https://mlflow.org/docs/latest/models.html#deployment)
