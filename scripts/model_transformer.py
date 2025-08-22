"""
SignaMentis Transformer Model Module

This module implements a Transformer neural network for XAU/USD price prediction.
The model uses self-attention mechanisms to capture complex temporal relationships.

Author: SignaMentis Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
import math
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Adds position information to input embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """
    Transformer model for XAU/USD price prediction.
    
    Architecture:
    - Input embedding layer
    - Positional encoding
    - Multi-head self-attention layers
    - Feed-forward networks
    - Output layers for direction, price, and confidence
    """
    
    def __init__(self, 
                 input_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_length: int = 500,
                 num_classes: int = 2):
        """
        Initialize the Transformer model.
        
        Args:
            input_size: Number of input features
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            num_classes: Number of output classes (2 for binary)
        """
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        
        # Input projection layer
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU for better performance
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layers
        # Direction prediction
        self.direction_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Price target prediction
        self.price_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Confidence prediction
        self.confidence_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Transformer model initialized with {self.count_parameters()} parameters")
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor, 
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            Model outputs or tuple with attention weights
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Create attention mask for padding (if needed)
        # For now, assuming no padding is needed
        src_key_padding_mask = None
        
        # Transformer encoder
        transformer_out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Layer normalization
        transformer_out = self.layer_norm(transformer_out)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(transformer_out, dim=1)
        
        # Direction prediction (logits)
        direction_logits = self.direction_fc(pooled)
        
        # Price target prediction
        price_target = self.price_fc(pooled)
        
        # Confidence prediction (0-1)
        confidence = self.confidence_fc(pooled)
        
        if return_attention:
            # For attention visualization, we can return the last layer's attention
            # This is a simplified approach - in practice, you might want to hook into specific layers
            return direction_logits, price_target, confidence, None
        else:
            return direction_logits, price_target, confidence
    
    def predict_direction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trading direction and confidence.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (direction_probabilities, confidence)
        """
        self.eval()
        with torch.no_grad():
            direction_logits, _, confidence = self.forward(x)
            direction_probs = F.softmax(direction_logits, dim=1)
            return direction_probs, confidence
    
    def predict_price(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict price target.
        
        Args:
            x: Input tensor
            
        Returns:
            Price target predictions
        """
        self.eval()
        with torch.no_grad():
            _, price_target, _ = self.forward(x)
            return price_target


class TransformerTrainer:
    """
    Trainer class for the Transformer model.
    
    Handles data preparation, training, validation, and model saving.
    """
    
    def __init__(self, 
                 model: TransformerModel,
                 config: Optional[Dict] = None):
        """
        Initialize the trainer.
        
        Args:
            model: Transformer model instance
            config: Training configuration
        """
        self.model = model
        self.config = config or {}
        
        # Training parameters
        self.learning_rate = self.config.get('learning_rate', 0.0001)
        self.batch_size = self.config.get('batch_size', 64)
        self.epochs = self.config.get('epochs', 200)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 20)
        self.weight_decay = self.config.get('weight_decay', 0.0001)
        self.warmup_steps = self.config.get('warmup_steps', 4000)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and loss functions
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Loss functions
        self.direction_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.price_criterion = nn.HuberLoss(delta=1.0)
        self.confidence_criterion = nn.BCELoss()
        
        # Loss weights
        self.direction_weight = self.config.get('direction_weight', 0.4)
        self.price_weight = self.config.get('price_weight', 0.4)
        self.confidence_weight = self.config.get('confidence_weight', 0.2)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        logger.info(f"Transformer trainer initialized on device: {self.device}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (step - self.warmup_steps) / max(1, self.epochs * 1000 - self.warmup_steps))))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def prepare_data(self, 
                    df: pd.DataFrame,
                    sequence_length: int = 100,
                    target_column: str = 'close',
                    feature_columns: Optional[List[str]] = None,
                    test_size: float = 0.2,
                    val_size: float = 0.2) -> Tuple[torch.Tensor, ...]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            sequence_length: Length of input sequences
            target_column: Name of target column
            feature_columns: List of feature column names
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            
        Returns:
            Tuple of data tensors
        """
        logger.info("Preparing data for Transformer training...")
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Ensure target column is included
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Prepare features and target
        X = df[feature_columns].values
        y_price = df[target_column].values
        
        # Create direction labels (1 for up, 0 for down)
        y_direction = (df[target_column].diff() > 0).astype(int)
        y_direction = y_direction.fillna(0)
        
        # Create confidence labels (based on price movement magnitude)
        price_changes = df[target_column].pct_change().abs()
        y_confidence = np.clip(price_changes * 100, 0, 1)  # Scale to 0-1
        y_confidence = y_confidence.fillna(0)
        
        # Remove rows with NaN values
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y_direction) | np.isnan(y_confidence))
        X = X[valid_indices]
        y_direction = y_direction[valid_indices]
        y_price = y_price[valid_indices]
        y_confidence = y_confidence[valid_indices]
        
        # Create sequences
        X_sequences, y_direction_sequences, y_price_sequences, y_confidence_sequences = [], [], [], []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_direction_sequences.append(y_direction[i])
            y_price_sequences.append(y_price[i])
            y_confidence_sequences.append(y_confidence[i])
        
        X_sequences = np.array(X_sequences)
        y_direction_sequences = np.array(y_direction_sequences)
        y_price_sequences = np.array(y_price_sequences)
        y_confidence_sequences = np.array(y_confidence_sequences)
        
        # Split into train/val/test
        X_temp, X_test, y_direction_temp, y_direction_test, y_price_temp, y_price_test, y_confidence_temp, y_confidence_test = train_test_split(
            X_sequences, y_direction_sequences, y_price_sequences, y_confidence_sequences,
            test_size=test_size, random_state=42, stratify=y_direction_sequences
        )
        
        X_train, X_val, y_direction_train, y_direction_val, y_price_train, y_price_val, y_confidence_train, y_confidence_val = train_test_split(
            X_temp, y_direction_temp, y_price_temp, y_confidence_temp,
            test_size=val_size, random_state=42, stratify=y_direction_temp
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)
        
        # Reshape back to sequences
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Scale price targets
        price_scaler = StandardScaler()
        y_price_train_scaled = price_scaler.fit_transform(y_price_train.reshape(-1, 1)).flatten()
        y_price_val_scaled = price_scaler.transform(y_price_val.reshape(-1, 1)).flatten()
        y_price_test_scaled = price_scaler.transform(y_price_test.reshape(-1, 1)).flatten()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        y_direction_train_tensor = torch.LongTensor(y_direction_train)
        y_direction_val_tensor = torch.LongTensor(y_direction_val)
        y_direction_test_tensor = torch.LongTensor(y_direction_test)
        
        y_price_train_tensor = torch.FloatTensor(y_price_train_scaled)
        y_price_val_tensor = torch.FloatTensor(y_price_val_scaled)
        y_price_test_tensor = torch.FloatTensor(y_price_test_scaled)
        
        y_confidence_train_tensor = torch.FloatTensor(y_confidence_train)
        y_confidence_val_tensor = torch.FloatTensor(y_confidence_val)
        y_confidence_test_tensor = torch.FloatTensor(y_confidence_test)
        
        # Save scalers for later use
        self.feature_scaler = scaler
        self.price_scaler = price_scaler
        
        logger.info(f"Data prepared: {len(X_train_tensor)} train, {len(X_val_tensor)} val, {len(X_test_tensor)} test samples")
        
        return (X_train_tensor, X_val_tensor, X_test_tensor,
                y_direction_train_tensor, y_direction_val_tensor, y_direction_test_tensor,
                y_price_train_tensor, y_price_val_tensor, y_price_test_tensor,
                y_confidence_train_tensor, y_confidence_val_tensor, y_confidence_test_tensor)
    
    def train_epoch(self, 
                   X_train: torch.Tensor,
                   y_direction_train: torch.Tensor,
                   y_price_train: torch.Tensor,
                   y_confidence_train: torch.Tensor) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            Training data tensors
            
        Returns:
            Tuple of (total_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Create batches
        num_batches = len(X_train) // self.batch_size
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            X_batch = X_train[batch_indices].to(self.device)
            y_direction_batch = y_direction_train[batch_indices].to(self.device)
            y_price_batch = y_price_train[batch_indices].to(self.device)
            y_confidence_batch = y_confidence_train[batch_indices].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            direction_logits, price_pred, confidence_pred = self.model(X_batch)
            
            # Calculate losses
            direction_loss = self.direction_criterion(direction_logits, y_direction_batch)
            price_loss = self.price_criterion(price_pred.squeeze(), y_price_batch)
            confidence_loss = self.confidence_criterion(confidence_pred.squeeze(), y_confidence_batch)
            
            # Combined loss
            total_batch_loss = (self.direction_weight * direction_loss + 
                              self.price_weight * price_loss + 
                              self.confidence_weight * confidence_loss)
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(direction_logits, 1)
            correct_predictions += (predicted == y_direction_batch).sum().item()
            total_predictions += y_direction_batch.size(0)
        
        avg_loss = total_loss / num_batches
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, 
                      X_val: torch.Tensor,
                      y_direction_val: torch.Tensor,
                      y_price_val: torch.Tensor,
                      y_confidence_val: torch.Tensor) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            Validation data tensors
            
        Returns:
            Tuple of (total_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            # Create batches
            num_batches = len(X_val) // self.batch_size
            
            for i in range(0, len(X_val), self.batch_size):
                X_batch = X_val[i:i+self.batch_size].to(self.device)
                y_direction_batch = y_direction_val[i:i+self.batch_size].to(self.device)
                y_price_batch = y_price_val[i:i+self.batch_size].to(self.device)
                y_confidence_batch = y_confidence_val[i:i+self.batch_size].to(self.device)
                
                # Forward pass
                direction_logits, price_pred, confidence_pred = self.model(X_batch)
                
                # Calculate losses
                direction_loss = self.direction_criterion(direction_logits, y_direction_batch)
                price_loss = self.price_criterion(price_pred.squeeze(), y_price_batch)
                confidence_loss = self.confidence_criterion(confidence_pred.squeeze(), y_confidence_batch)
                
                # Combined loss
                total_batch_loss = (self.direction_weight * direction_loss + 
                                  self.price_weight * price_loss + 
                                  self.confidence_weight * confidence_loss)
                
                total_loss += total_batch_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(direction_logits, 1)
                correct_predictions += (predicted == y_direction_batch).sum().item()
                total_predictions += y_direction_batch.size(0)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, 
              X_train: torch.Tensor,
              y_direction_train: torch.Tensor,
              y_price_train: torch.Tensor,
              y_confidence_train: torch.Tensor,
              X_val: torch.Tensor,
              y_direction_val: torch.Tensor,
              y_price_val: torch.Tensor,
              y_confidence_val: torch.Tensor) -> Dict:
        """
        Train the model.
        
        Args:
            Training and validation data tensors
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting Transformer training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0
        
        for epoch in range(self.epochs):
            # Training
            train_loss, train_acc = self.train_epoch(
                X_train, y_direction_train, y_price_train, y_confidence_train
            )
            
            # Validation
            val_loss, val_acc = self.validate_epoch(
                X_val, y_direction_val, y_price_val, y_confidence_val
            )
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                          f"LR: {current_lr:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model('best_transformer_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            global_step += 1
        
        logger.info("Transformer training completed!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }
    
    def save_model(self, filepath: str):
        """Save the model and scalers."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'd_model': self.model.d_model,
                'nhead': self.model.nhead,
                'num_layers': self.model.num_layers,
                'dim_feedforward': self.model.dim_feedforward,
                'dropout': self.model.dropout,
                'max_seq_length': self.model.max_seq_length,
                'num_classes': self.model.num_classes
            },
            'feature_scaler': self.feature_scaler,
            'price_scaler': self.price_scaler,
            'training_config': self.config
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Transformer model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model and scalers."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load scalers
        self.feature_scaler = checkpoint['feature_scaler']
        self.price_scaler = checkpoint['price_scaler']
        
        logger.info(f"Transformer model loaded from {filepath}")
    
    def predict(self, X: torch.Tensor) -> Dict:
        """
        Make predictions on new data.
        
        Args:
            X: Input features tensor
            
        Returns:
            Dictionary with predictions
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            direction_logits, price_pred, confidence = self.model(X)
            
            # Get direction probabilities
            direction_probs = F.softmax(direction_logits, dim=1)
            predicted_direction = torch.argmax(direction_probs, dim=1)
            
            # Inverse transform price predictions
            price_pred_original = self.price_scaler.inverse_transform(price_pred.cpu().numpy())
            
            return {
                'direction_probabilities': direction_probs.cpu().numpy(),
                'predicted_direction': predicted_direction.cpu().numpy(),
                'price_targets': price_pred_original.flatten(),
                'confidence': confidence.cpu().numpy().flatten()
            }


# Convenience function for quick model creation
def create_transformer_model(input_size: int, config: Optional[Dict] = None) -> TransformerModel:
    """
    Create a Transformer model with default or custom configuration.
    
    Args:
        input_size: Number of input features
        config: Model configuration dictionary
        
    Returns:
        TransformerModel instance
    """
    default_config = {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_seq_length': 500,
        'num_classes': 2
    }
    
    if config:
        # Filter only model-specific parameters
        model_params = ['d_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout', 'max_seq_length', 'num_classes']
        filtered_config = {k: v for k, v in config.items() if k in model_params}
        default_config.update(filtered_config)
    
    return TransformerModel(input_size=input_size, **default_config)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
    
    sample_data = pd.DataFrame({
        'open': 2000 + np.random.randn(1000).cumsum(),
        'high': 2000 + np.random.randn(1000).cumsum() + 5,
        'low': 2000 + np.random.randn(1000).cumsum() - 5,
        'close': 2000 + np.random.randn(1000).cumsum(),
        'volume': np.random.randint(100, 1000, 1000),
        'rsi': np.random.uniform(0, 100, 1000),
        'macd': np.random.randn(1000),
        'atr': np.random.uniform(1, 10, 1000)
    }, index=dates)
    
    # Create model
    input_size = 4  # rsi, macd, atr, volume
    model = create_transformer_model(input_size)
    
    # Create trainer
    trainer = TransformerTrainer(model)
    
    # Prepare data
    data_tensors = trainer.prepare_data(
        sample_data,
        sequence_length=50,
        feature_columns=['rsi', 'macd', 'atr', 'volume']
    )
    
    # Train model
    history = trainer.train(*data_tensors[:8])  # First 8 tensors for training
    
    # Make predictions
    X_test = data_tensors[2]  # Test features
    predictions = trainer.predict(X_test[:10])  # First 10 test samples
    
    print(f"Transformer Model trained successfully!")
    print(f"Predictions shape: {predictions['direction_probabilities'].shape}")
    print(f"Sample predictions: {predictions['predicted_direction'][:5]}")
    print(f"Sample confidence: {predictions['confidence'][:5]}")
