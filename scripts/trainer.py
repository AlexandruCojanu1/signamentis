import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import os

from .transformer_bilstm_model import TransformerBiLSTMClassifier, TemperatureScaling

logger = logging.getLogger(__name__)

class DirectionTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.temperature_scaler = None
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Setup mixed precision
        self.use_amp = self._setup_mixed_precision()
        
    def _get_device(self):
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS device")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
        return device
        
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        use_amp = self.config.get('mixed_precision', 'auto')
        
        if use_amp == 'auto':
            # Enable AMP only on CUDA
            use_amp = self.device.type == 'cuda'
        elif use_amp == 'on':
            use_amp = True
        else:
            use_amp = False
            
        logger.info(f"Mixed precision: {use_amp}")
        return use_amp
        
    def prepare_model(self, n_features: int):
        """Initialize model."""
        self.model = TransformerBiLSTMClassifier(n_features, self.config)
        self.model.to(self.device)
        
        # Set deterministic behavior
        torch.manual_seed(self.config.get('seed', 42))
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.config.get('seed', 42))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, save_dir: str) -> Dict:
        """Train the model."""
        
        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        batch_size = self.config.get('batch_size', 256)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.get('label_smoothing', 0.05))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('lr', 2e-4),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training loop
        epochs = self.config.get('epochs', 50)
        patience = self.config.get('patience', 8)
        patience_counter = 0
        
        metrics_history = []
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer, scaler)
            
            # Validation phase
            val_loss, val_acc, val_preds, val_true = self._validate_epoch(val_loader, criterion)
            
            epoch_time = time.time() - start_time
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                       f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                       f"time={epoch_time:.1f}s")
            
            # Save metrics
            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'time': epoch_time
            })
            
            # Early stopping and checkpointing
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'config': self.config
                }
                torch.save(checkpoint, os.path.join(save_dir, 'model.pt'))
                logger.info(f"New best validation accuracy: {val_acc:.4f}")
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save metrics history
        metrics_df = pd.DataFrame(metrics_history)
        metrics_df.to_csv(os.path.join(save_dir, 'training_metrics.csv'), index=False)
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'final_epoch': epoch + 1
        }
        
    def _train_epoch(self, train_loader, criterion, optimizer, scaler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
        return total_loss / len(train_loader), correct / total
        
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = criterion(output, target)
                else:
                    output = self.model(data)
                    loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        return total_loss / len(val_loader), accuracy, all_preds, all_targets
        
    def calibrate_temperature(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Calibrate probabilities using temperature scaling."""
        # Load best model
        self.model.eval()
        
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # Get logits
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                logits = self.model(data)
                all_logits.append(logits.cpu())
                all_targets.append(target)
                
        logits_tensor = torch.cat(all_logits, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        
        # Fit temperature
        self.temperature_scaler = TemperatureScaling().to(self.device)
        temperature = self.temperature_scaler.fit(logits_tensor.to(self.device), targets_tensor.to(self.device))
        
        logger.info(f"Temperature scaling: T = {temperature:.3f}")
        return temperature
        
    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray, label_map: Dict) -> Dict:
        """Evaluate model and return metrics."""
        self.model.eval()
        
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                logits = self.model(data)
                
                # Apply temperature scaling if available
                if self.temperature_scaler is not None:
                    logits = self.temperature_scaler(logits)
                
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(target.numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        
        # Classification report
        target_names = [label_map[i] for i in sorted(label_map.keys())]
        class_report = classification_report(all_targets, all_preds, target_names=target_names)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        logger.info(f"Validation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {f1_macro:.4f}")
        logger.info(f"\nClassification Report:\n{class_report}")
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'probabilities': all_probs
        }
