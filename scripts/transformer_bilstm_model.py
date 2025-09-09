import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class TransformerBiLSTMClassifier(nn.Module):
    def __init__(self, n_features: int, config: Dict):
        super().__init__()
        
        self.d_model = config.get('d_model', 128)
        self.num_layers = config.get('num_layers', 2)
        self.nhead = config.get('nhead', 4)
        self.lstm_hidden = config.get('lstm_hidden', 128)
        self.dropout = config.get('dropout', 0.1)
        self.n_classes = config.get('n_classes', 3)
        
        # Input projection
        self.input_proj = nn.Linear(n_features, self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            dropout=0.0,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification head
        lstm_output_size = self.lstm_hidden * 2  # Bidirectional
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_output_size),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_size, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.n_classes)
        )
        
    def forward(self, x):
        # x: [batch, seq_len, n_features]
        batch_size, seq_len, n_features = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # Transformer encoder
        x = self.transformer(x)  # [batch, seq_len, d_model]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, lstm_hidden*2]
        
        # Take last timestep
        last_output = lstm_out[:, -1, :]  # [batch, lstm_hidden*2]
        
        # Classification head
        logits = self.head(last_output)  # [batch, n_classes]
        
        return logits

class TemperatureScaling(nn.Module):
    """Temperature scaling for probability calibration."""
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, logits):
        return logits / self.temperature
        
    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """Fit temperature using LBFGS."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        return self.temperature.item()
