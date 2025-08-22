"""
SignaMentis LNN (Liquid Neural Network) Model Module

This module implements a Liquid Neural Network for XAU/USD price prediction.
LNN uses continuous-time dynamics and liquid time-constant networks for better temporal modeling.

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


class LiquidTimeConstant(nn.Module):
    """
    Liquid Time-Constant layer for LNN.
    
    Implements continuous-time dynamics with learnable time constants.
    """
    
    def __init__(self, input_size: int, hidden_size: int, alpha: float = 0.1):
        """
        Initialize LTC layer.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            alpha: Leak rate for the liquid state
        """
        super(LiquidTimeConstant, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Hidden state projection
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        
        # Time constant network
        self.time_constant_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # Initialize hidden state
        self.register_buffer('hidden_state', torch.zeros(1, hidden_size))
        
        logger.info(f"LTC layer initialized: {input_size} -> {hidden_size}")
    
    def forward(self, x: torch.Tensor, reset_hidden: bool = False) -> torch.Tensor:
        """
        Forward pass through LTC layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            reset_hidden: Whether to reset hidden state
            
        Returns:
            Output tensor
        """
        batch_size = x.size(0)
        
        # Reset hidden state if requested
        if reset_hidden or self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Input processing
        input_proj = self.input_projection(x)
        
        # Hidden state processing
        hidden_proj = self.hidden_projection(self.hidden_state)
        
        # Combined input
        combined = input_proj + hidden_proj
        
        # Time constant calculation
        time_constants = self.time_constant_net(combined)
        
        # Liquid state update (continuous-time dynamics)
        # dh/dt = alpha * (combined - h)
        # Using Euler discretization: h_{t+1} = h_t + alpha * (combined - h_t)
        new_hidden = self.hidden_state + self.alpha * (combined - self.hidden_state)
        
        # Update hidden state
        self.hidden_state = new_hidden.detach()
        
        # Output
        output = self.output_projection(new_hidden)
        
        return output


class LNNModel(nn.Module):
    """
    Liquid Neural Network model for XAU/USD price prediction.
    
    Architecture:
    - Multiple LTC layers for temporal modeling
    - Attention mechanism for sequence modeling
    - Output layers for direction, price, and confidence
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 dropout: float = 0.2,
                 alpha: float = 0.1,
                 num_classes: int = 2):
        """
        Initialize the LNN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of LTC layers
            dropout: Dropout rate
            alpha: Leak rate for LTC layers
            num_classes: Number of output classes (2 for binary)
        """
        super(LNNModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.alpha = alpha
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # LTC layers
        self.ltc_layers = nn.ModuleList([
            LiquidTimeConstant(hidden_size, hidden_size, alpha)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layers
        # Direction prediction
        self.direction_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # Price target prediction
        self.price_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Confidence prediction
        self.confidence_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"LNN model initialized with {self.count_parameters()} parameters")
    
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
    
    def reset_hidden_states(self):
        """Reset hidden states in all LTC layers."""
        for ltc_layer in self.ltc_layers:
            ltc_layer.hidden_state.zero_()
    
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
        
        # Reset hidden states at the beginning of each sequence
        self.reset_hidden_states()
        
        # Process sequence through LTC layers
        ltc_outputs = []
        
        for t in range(seq_len):
            # Current timestep input
            x_t = x[:, t, :]
            
            # Input projection
            h = self.input_projection(x_t)
            
            # Process through LTC layers
            for i, (ltc_layer, layer_norm) in enumerate(zip(self.ltc_layers, self.layer_norms)):
                h = ltc_layer(h)
                h = layer_norm(h)
                h = self.dropout_layer(h)
            
            ltc_outputs.append(h)
        
        # Stack LTC outputs
        ltc_outputs = torch.stack(ltc_outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        
        # Apply attention mechanism
        attn_out, attn_weights = self.attention(
            ltc_outputs, ltc_outputs, ltc_outputs
        )
        
        # Residual connection
        ltc_outputs = ltc_outputs + attn_out
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(ltc_outputs, dim=1)
        
        # Direction prediction (logits)
        direction_logits = self.direction_fc(pooled)
        
        # Price target prediction
        price_target = self.price_fc(pooled)
        
        # Confidence prediction (0-1)
        confidence = self.confidence_fc(pooled)
        
        if return_attention:
            return direction_logits, price_target, confidence, attn_weights
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


class LNNTrainer:
    """
    Trainer class for the LNN model.
    
    Handles data preparation, training, validation, and model saving.
    """
    
    def __init__(self, 
                 model: LNNModel,
                 config: Optional[Dict] = None):
        """
        Initialize the trainer.
        
        Args:
            model: LNN model instance
            config: Training configuration
        """
        self.model = model
        self.config = config or {}
        
        # Training parameters
        self.learning_rate = self.config.get('learning_rate', 0.0003)
        self.batch_size = self.config.get('batch_size', 64)
        self.epochs = self.config.get('epochs', 200)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 25)
        self.weight_decay = self.config.get('weight_decay', 0.0001)
        
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
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
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
        
        logger.info(f"LNN trainer initialized on device: {self.device}")
    
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
        logger.info("Preparing data for LNN training...")
        
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
        logger.info("Starting LNN training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
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
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
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
                self.save_model('best_lnn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        logger.info("LNN training completed!")
        
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
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'alpha': self.model.alpha,
                'num_classes': self.model.num_classes
            },
            'feature_scaler': self.feature_scaler,
            'price_scaler': self.price_scaler,
            'training_config': self.config
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"LNN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model and scalers."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load scalers
        self.feature_scaler = checkpoint['feature_scaler']
        self.price_scaler = checkpoint['price_scaler']
        
        logger.info(f"LNN model loaded from {filepath}")
    
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
def create_lnn_model(input_size: int, config: Optional[Dict] = None) -> LNNModel:
    """
    Create an LNN model with default or custom configuration.
    
    Args:
        input_size: Number of input features
        config: Model configuration dictionary
        
    Returns:
        LNNModel instance
    """
    default_config = {
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.2,
        'alpha': 0.1,
        'num_classes': 2
    }
    
    if config:
        default_config.update(config)
    
    return LNNModel(input_size=input_size, **default_config)


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
    model = create_lnn_model(input_size)
    
    # Create trainer
    trainer = LNNTrainer(model)
    
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
    
    print(f"LNN Model trained successfully!")
    print(f"Predictions shape: {predictions['direction_probabilities'].shape}")
    print(f"Sample predictions: {predictions['predicted_direction'][:5]}")
    print(f"Sample confidence: {predictions['confidence'][:5]}")
