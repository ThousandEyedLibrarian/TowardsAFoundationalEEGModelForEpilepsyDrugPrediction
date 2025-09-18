"""
Enhanced EEG Challenge 2025 Task 1 - S4 Model with PyTorch Lightning
Refactored implementation using Lightning for cleaner training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List, Any
import math
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import datetime
import logging
import sys
import tempfile
import shutil
import argparse
import os
try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# EEG data handling imports
try:
    from eegdash.dataset import EEGChallengeDataset
    EEGDASH_AVAILABLE = True
except ImportError:
    EEGDASH_AVAILABLE = False
    print("Warning: EEGDash not available, will use local data")

from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from braindecode.datasets import BaseConcatDataset
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import mne
try:
    from mne_bids import get_bids_path_from_fname
except ImportError:
    print("Warning: mne_bids not available")
    get_bids_path_from_fname = None

warnings.filterwarnings('ignore')

# ================================
# S4 Components (unchanged from original)
# ================================

class LinearActivation(nn.Module):
    """Linear layer with activation function"""
    def __init__(self, d_input, d_output, bias=True, activation=None, activate=True):
        super().__init__()
        self.linear = nn.Linear(d_input, d_output, bias=bias)
        self.activation = nn.Identity() if activation is None or not activate else self._get_activation(activation)
        
    def _get_activation(self, activation):
        if activation == 'gelu':
            return nn.GELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'silu':
            return nn.SiLU()
        elif activation == 'glu':
            return nn.GLU(dim=-1)
        else:
            return nn.Identity()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class S4Kernel1D(nn.Module):
    """Enhanced 1D S4 kernel for EEG sequences with improved numerical stability"""
    
    def __init__(self, d_model, l_max=None, d_state=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.d_state = d_state
        
        # Improved HiPPO matrix initialization for stability
        A = self._init_hippo_matrix(d_state)
        self.register_buffer("A", A)
        
        # Better initialization for learnable parameters
        self.B = nn.Parameter(torch.randn(d_state, d_model) * (2.0 / (d_state + d_model)) ** 0.5)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * (2.0 / (d_model + d_state)) ** 0.5) 
        
        # Learnable timescale with better initialization
        log_dt = torch.linspace(np.log(dt_min), np.log(dt_max), d_model)
        self.log_dt = nn.Parameter(log_dt)
        
        if lr is not None:
            # Different learning rates for different parameters
            for param in [self.B, self.C]:
                param.lr = lr
    
    def _init_hippo_matrix(self, N):
        """Initialize HiPPO-LegS matrix for better stability"""
        P = torch.sqrt(1 + 2 * torch.arange(N, dtype=torch.float32))
        A = P[:, None] * P[None, :]
        
        # Create HiPPO-LegS structure
        for i in range(N):
            for j in range(N):
                if i > j:
                    A[i, j] = -A[i, j]
                elif i == j:
                    A[i, j] = -(i + 0.5)
        
        return A
                
    def forward(self, L, rate=1.0, state=None):
        """Generate convolution kernel with improved numerical stability"""
        # Clamp dt for stability
        dt = torch.exp(self.log_dt).clamp(1e-4, 0.1)  # (d_model,)
        
        # Improved discretization using bilinear method for stability
        dt_A = dt[0] * self.A / 2.0
        A_discrete = torch.linalg.solve(
            torch.eye(self.d_state, device=self.A.device) - dt_A,
            torch.eye(self.d_state, device=self.A.device) + dt_A
        )
        
        # Compute kernel with numerical stability checks
        k = []
        x = self.C  # (d_model, d_state)
        
        # Add small epsilon for numerical stability
        eps = 1e-8
        
        for i in range(L):
            # k_i = C @ A^i @ B with stability
            kernel_step = torch.sum(x * self.B.T, dim=1)  # (d_model,)
            
            # Clamp to prevent exploding values
            kernel_step = torch.clamp(kernel_step, -10.0, 10.0)
            k.append(kernel_step)
            
            # Update for next step with stability
            x = x @ A_discrete.T
            
            # Prevent gradient explosion
            x = torch.clamp(x, -100.0, 100.0)
            
        k = torch.stack(k, dim=-1)  # (d_model, L)
        
        # Apply exponential decay for stability
        decay = torch.exp(-0.01 * torch.arange(L, device=k.device))
        k = k * decay.unsqueeze(0)
        
        k = k.unsqueeze(0)  # (1, d_model, L)
        
        return k, None  # No state for simplified version

class EnhancedS4Layer(nn.Module):
    """Enhanced S4 layer with proper convolution and activations"""

    def __init__(self, d_model, d_state=64, l_max=None, dropout=0.0, activation='gelu'):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        
        # S4 kernel
        self.kernel = S4Kernel1D(d_model, l_max, d_state)
        
        # Skip connection parameter
        self.D = nn.Parameter(torch.randn(d_model) * 0.1)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == 'gelu' else nn.Identity()

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Add input normalization for stability
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)

        # Get S4 kernel
        k, _ = self.kernel(L=seq_len)  # (1, d_model, seq_len)

        # Transpose for convolution: (batch, d_model, seq_len)
        x_conv = x_norm.transpose(1, 2)

        # FFT-based convolution with stability checks
        # Pad for convolution
        k_f = torch.fft.rfft(k, n=seq_len*2, dim=-1)  # (1, d_model, freq)
        x_f = torch.fft.rfft(x_conv, n=seq_len*2, dim=-1)  # (batch, d_model, freq)

        # Convolution in frequency domain with magnitude clamping
        y_f = x_f * k_f  # Broadcasting: (batch, d_model, freq)
        
        # Prevent numerical issues in IFFT
        y_f_mag = torch.abs(y_f)
        y_f = torch.where(y_f_mag > 1e10, y_f / (y_f_mag + 1e-8) * 1e10, y_f)
        
        y = torch.fft.irfft(y_f, n=seq_len*2, dim=-1)[..., :seq_len]  # (batch, d_model, seq_len)

        # Add skip connection with gating for stability
        skip_gate = torch.sigmoid(self.D).unsqueeze(0).unsqueeze(-1)
        y = y + x_conv * skip_gate

        # Transpose back and apply activation/dropout
        y = y.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Residual connection with original input for stability
        y = self.activation(y) + x * 0.1
        y = self.dropout(y)

        return y

# ========================================
# Domain Adaptation Components
# ========================================

class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for domain adaptation"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DomainAdaptationLayer(nn.Module):
    """Domain adaptation layer with gradient reversal"""
    
    def __init__(self, d_model, n_domains=2, dropout=0.1):
        super().__init__()
        self.gradient_reversal = GradientReversalLayer(alpha=1.0)
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_domains)
        )
        
    def forward(self, x):
        """
        x: (batch, d_model) - pooled features
        Returns: domain predictions (batch, n_domains)
        """
        x = self.gradient_reversal(x)
        domain_pred = self.domain_classifier(x)
        return domain_pred

# ========================================
# Competition Submission Model
# ========================================

class EEGChallengeSubmissionModel(nn.Module):
    """Submission-ready model wrapper for EEG Challenge 2025"""

    def __init__(self, lightning_model: 'LightningEEGS4Model'):
        super().__init__()

        # Extract core model components from Lightning wrapper
        self.n_chans = lightning_model.n_chans
        self.n_times = lightning_model.n_times
        self.d_model = lightning_model.d_model

        # Copy all necessary components
        self.input_projection = lightning_model.input_projection
        self.input_norm = lightning_model.input_norm
        self.pos_encoding = lightning_model.pos_encoding
        self.s4_layers = lightning_model.s4_layers
        self.layer_norms = lightning_model.layer_norms
        self.attention_pool = lightning_model.attention_pool
        self.cls_token = lightning_model.cls_token
        self.main_head = lightning_model.main_head

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using S4 backbone"""
        # Transpose to (batch_size, n_times, n_chans) for temporal modeling
        x = x.transpose(1, 2)  # (batch, n_times, n_chans)

        # Project to d_model dimensions
        x = self.input_projection(x)  # (batch, n_times, d_model)
        x = self.input_norm(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Apply S4 layers with residual connections
        for i, (s4_layer, norm) in enumerate(zip(self.s4_layers, self.layer_norms)):
            residual = x
            x = s4_layer(x)
            x = norm(x + residual)  # Residual connection

        return x  # (batch, n_times, d_model)

    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """Advanced pooling with attention mechanism"""
        batch_size = x.size(0)

        # Add CLS token for attention-based pooling
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x_with_cls = torch.cat([cls_tokens, x], dim=1)  # (batch, n_times+1, d_model)

        # Multi-head attention pooling
        attended, _ = self.attention_pool(cls_tokens, x_with_cls, x_with_cls)  # (batch, 1, d_model)
        pooled = attended.squeeze(1)  # (batch, d_model)

        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for submission
        Args:
            x: EEG tensor of shape (batch_size, n_chans=129, n_times=200)
        Returns:
            Response time predictions (batch_size,)
        """
        # Extract features using S4 backbone
        features = self.extract_features(x)  # (batch, n_times, d_model)

        # Pool features
        pooled_features = self.pool_features(features)  # (batch, d_model)

        # Main task prediction
        predictions = self.main_head(pooled_features).squeeze(-1)  # (batch,)

        return predictions

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction method for challenge submission"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions

# ===================================================
# Lightning Module for S4 EEG Model
# ===================================================

class LightningEEGS4Model(LightningModule):
    """PyTorch Lightning wrapper for Enhanced S4 EEG Model"""
    
    def __init__(
        self,
        # Model architecture parameters
        n_chans: int = 129,
        n_times: int = 200,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        dropout: float = 0.1,
        n_outputs: int = 1,
        # Cross-paradigm parameters
        enable_domain_adaptation: bool = True,
        n_domains: int = 2,
        pretrain_task: str = None,
        # Training parameters
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_t_max: int = 50,
        # Loss parameters
        huber_delta: float = 0.1,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model components
        self.n_chans = n_chans
        self.n_times = n_times
        self.d_model = d_model
        self.enable_domain_adaptation = enable_domain_adaptation
        self.pretrain_task = pretrain_task
        
        # Input projection with better initialization
        self.input_projection = nn.Linear(n_chans, d_model)
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.5)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(n_times, d_model))
        
        # Enhanced S4 backbone
        self.s4_layers = nn.ModuleList([
            EnhancedS4Layer(d_model, d_state, dropout=dropout) for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # Multi-head attention pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Task-specific heads
        if pretrain_task == 'sus':
            self.sus_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 2)
            )

        # Main task head
        self.main_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs)
        )

        # Domain adaptation
        if enable_domain_adaptation:
            self.domain_adapter = DomainAdaptationLayer(d_model, n_domains, dropout)
        
        # Loss functions
        self.huber_loss = nn.HuberLoss(delta=huber_delta)
        self.task_weights = task_weights or {'main': 1.0, 'domain': 0.1, 'sus': 0.5}
        
        # Metrics storage
        self.validation_outputs = []
        self.test_outputs = []
        
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using enhanced S4 backbone"""
        # Transpose to (batch_size, n_times, n_chans) for temporal modelling
        x = x.transpose(1, 2)  # (batch, n_times, n_chans)
        
        # Project to d_model dimensions
        x = self.input_projection(x)  # (batch, n_times, d_model)
        x = self.input_norm(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Apply enhanced S4 layers with residual connections
        for i, (s4_layer, norm) in enumerate(zip(self.s4_layers, self.layer_norms)):
            residual = x
            x = s4_layer(x)
            x = norm(x + residual)  # Residual connection
        
        return x  # (batch, n_times, d_model)
    
    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """Advanced pooling with attention mechanism"""
        batch_size = x.size(0)
        
        # Add CLS token for attention-based pooling
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x_with_cls = torch.cat([cls_tokens, x], dim=1)  # (batch, n_times+1, d_model)
        
        # Multi-head attention pooling
        attended, _ = self.attention_pool(cls_tokens, x_with_cls, x_with_cls)  # (batch, 1, d_model)
        pooled = attended.squeeze(1)  # (batch, d_model)
        
        return pooled
    
    def forward(self, x: torch.Tensor, domain_labels: Optional[torch.Tensor] = None, 
                task: str = 'main') -> Dict[str, torch.Tensor]:
        """Forward pass with multi-task capability"""
        # Extract features using enhanced S4 backbone
        features = self.extract_features(x)  # (batch, n_times, d_model)
        
        # Pool features
        pooled_features = self.pool_features(features)  # (batch, d_model)
        
        outputs = {}
        
        # Main task prediction
        if task in ['main', 'both']:
            main_pred = self.main_head(pooled_features).squeeze(-1)  # (batch,)
            outputs['main_pred'] = main_pred
        
        # SuS task prediction
        if task in ['sus', 'both'] and self.pretrain_task == 'sus' and hasattr(self, 'sus_head'):
            sus_pred = self.sus_head(pooled_features)  # (batch, 2)
            outputs['sus_pred'] = sus_pred
        
        # Domain adaptation
        if self.enable_domain_adaptation and domain_labels is not None and hasattr(self, 'domain_adapter'):
            domain_pred = self.domain_adapter(pooled_features)  # (batch, n_domains)
            outputs['domain_pred'] = domain_pred
            
            # Domain adaptation loss
            domain_loss = F.cross_entropy(domain_pred, domain_labels)
            outputs['domain_loss'] = domain_loss
        
        return outputs
    
    def compute_loss(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        total_loss = 0
        losses = {}
        
        # Main task loss using Huber loss
        if 'main_pred' in predictions and 'main_target' in targets:
            huber_loss = self.huber_loss(predictions['main_pred'], targets['main_target'])
            losses['huber_loss'] = huber_loss
            losses['main_loss'] = huber_loss
            
            # Calculate metrics for logging
            with torch.no_grad():
                mse_loss = F.mse_loss(predictions['main_pred'], targets['main_target'])
                rmse_loss = torch.sqrt(mse_loss)
                mae_loss = F.l1_loss(predictions['main_pred'], targets['main_target'])
                losses['mse_loss'] = mse_loss
                losses['rmse_loss'] = rmse_loss
                losses['mae_loss'] = mae_loss
            
            total_loss += self.task_weights['main'] * huber_loss
        
        # Domain adaptation loss
        if 'domain_loss' in predictions:
            losses['domain_loss'] = predictions['domain_loss']
            total_loss += self.task_weights['domain'] * predictions['domain_loss']
        
        # SuS pretraining loss
        if 'sus_pred' in predictions and 'sus_target' in targets:
            sus_loss = F.cross_entropy(predictions['sus_pred'], targets['sus_target'])
            losses['sus_loss'] = sus_loss
            total_loss += self.task_weights['sus'] * sus_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def training_step(self, batch, batch_idx):
        """Training step for Lightning"""
        X, y = batch[0], batch[1]
        
        # Create domain labels (placeholder for now)
        domain_labels = torch.zeros(X.size(0), dtype=torch.long, device=self.device)
        
        # Forward pass
        outputs = self(X, domain_labels=domain_labels, task='main')
        
        # Prepare targets
        targets = {
            'main_target': y.squeeze(),
            'domain_target': domain_labels
        }
        
        # Calculate losses
        losses = self.compute_loss(outputs, targets)
        
        # Log metrics
        self.log('train/loss', losses['total_loss'], prog_bar=True)
        self.log('train/huber_loss', losses['huber_loss'])
        self.log('train/rmse', losses['rmse_loss'])
        self.log('train/mae', losses['mae_loss'])
        if 'domain_loss' in losses:
            self.log('train/domain_loss', losses['domain_loss'])
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning"""
        X, y = batch[0], batch[1]
        
        # Create domain labels
        domain_labels = torch.zeros(X.size(0), dtype=torch.long, device=self.device)
        
        # Forward pass
        outputs = self(X, domain_labels=domain_labels, task='main')
        predictions = outputs['main_pred']
        
        # Prepare targets
        targets = {
            'main_target': y.squeeze(),
            'domain_target': domain_labels
        }
        
        # Calculate losses
        losses = self.compute_loss(outputs, targets)
        
        # Log metrics
        self.log('val/loss', losses['total_loss'], prog_bar=True)
        self.log('val/huber_loss', losses['huber_loss'])
        self.log('val/rmse', losses['rmse_loss'], prog_bar=True)
        self.log('val/mae', losses['mae_loss'])
        if 'domain_loss' in losses:
            self.log('val/domain_loss', losses['domain_loss'])
        
        # Store outputs for epoch end
        self.validation_outputs.append({
            'predictions': predictions.detach().cpu(),
            'targets': y.squeeze().detach().cpu(),
            'loss': losses['total_loss'].detach().cpu()
        })
        
        return losses['total_loss']
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        if not self.validation_outputs:
            return
        
        # Aggregate predictions and targets
        all_preds = torch.cat([x['predictions'] for x in self.validation_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_outputs])
        
        # Calculate R2
        if len(all_preds) > 1 and torch.std(all_targets) > 0:
            correlation = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1]
            r2 = correlation ** 2 if not torch.isnan(correlation) else 0.0
        else:
            r2 = 0.0
        
        self.log('val/r2', r2)
        
        # Clear outputs
        self.validation_outputs = []
    
    def test_step(self, batch, batch_idx):
        """Test step for Lightning"""
        X, y = batch[0], batch[1]
        
        # Create domain labels
        domain_labels = torch.zeros(X.size(0), dtype=torch.long, device=self.device)
        
        # Forward pass
        outputs = self(X, domain_labels=domain_labels, task='main')
        predictions = outputs['main_pred']
        
        # Prepare targets
        targets = {
            'main_target': y.squeeze(),
            'domain_target': domain_labels
        }
        
        # Calculate losses
        losses = self.compute_loss(outputs, targets)
        
        # Log metrics
        self.log('test/loss', losses['total_loss'])
        self.log('test/huber_loss', losses['huber_loss'])
        self.log('test/rmse', losses['rmse_loss'])
        self.log('test/mae', losses['mae_loss'])
        if 'domain_loss' in losses:
            self.log('test/domain_loss', losses['domain_loss'])
        
        # Store outputs
        self.test_outputs.append({
            'predictions': predictions.detach().cpu(),
            'targets': y.squeeze().detach().cpu()
        })
        
        return losses['total_loss']
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch"""
        if not self.test_outputs:
            return
        
        # Aggregate predictions and targets
        all_preds = torch.cat([x['predictions'] for x in self.test_outputs])
        all_targets = torch.cat([x['targets'] for x in self.test_outputs])
        
        # Calculate final metrics
        mae = torch.mean(torch.abs(all_preds - all_targets))
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))
        
        if len(all_preds) > 1 and torch.std(all_targets) > 0:
            correlation = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1]
            r2 = correlation ** 2 if not torch.isnan(correlation) else 0.0
        else:
            r2 = 0.0
        
        # Log final test metrics
        self.log('test/final_mae', mae)
        self.log('test/final_rmse', rmse)
        self.log('test/final_r2', r2)
        
        print(f"\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}")
        print(f"MAE: {mae:.6f} seconds")
        print(f"RMSE: {rmse:.6f} seconds")
        print(f"R²: {r2:.6f}")
        print(f"Number of test samples: {len(all_preds)}")
        print(f"{'='*60}\n")
        
        # Clear outputs
        self.test_outputs = []
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.scheduler_t_max
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for Lightning"""
        X = batch[0] if isinstance(batch, tuple) else batch
        
        # Forward pass
        outputs = self(X, task='main')
        predictions = outputs['main_pred']
        
        return predictions

# ========================================
# Data Processing Functions (from enhanced S4)
# ========================================

def build_trial_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """Extract trial information with stimulus/response timing"""
    events_df = events_df.copy()
    events_df["onset"] = pd.to_numeric(events_df["onset"], errors="raise")
    events_df = events_df.sort_values("onset", kind="mergesort").reset_index(drop=True)

    trials = events_df[events_df["value"].eq("contrastTrial_start")].copy()
    stimuli = events_df[events_df["value"].isin(["left_target", "right_target"])].copy()
    responses = events_df[events_df["value"].isin(["left_buttonPress", "right_buttonPress"])].copy()

    trials = trials.reset_index(drop=True)
    trials["next_onset"] = trials["onset"].shift(-1)
    trials = trials.dropna(subset=["next_onset"]).reset_index(drop=True)

    rows = []
    for _, tr in trials.iterrows():
        start = float(tr["onset"])
        end   = float(tr["next_onset"])

        stim_block = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
        stim_onset = np.nan if stim_block.empty else float(stim_block.iloc[0]["onset"])

        if not np.isnan(stim_onset):
            resp_block = responses[(responses["onset"] >= stim_onset) & (responses["onset"] < end)]
        else:
            resp_block = responses[(responses["onset"] >= start) & (responses["onset"] < end)]

        if resp_block.empty:
            resp_onset = np.nan
            resp_type  = None
            feedback   = None
        else:
            resp_onset = float(resp_block.iloc[0]["onset"])
            resp_type  = resp_block.iloc[0]["value"]
            feedback   = resp_block.iloc[0]["feedback"]

        rt_from_stim  = (resp_onset - stim_onset) if (not np.isnan(stim_onset) and not np.isnan(resp_onset)) else np.nan
        rt_from_trial = (resp_onset - start)       if not np.isnan(resp_onset) else np.nan

        correct = None
        if isinstance(feedback, str):
            if feedback == "smiley_face": correct = True
            elif feedback == "sad_face":  correct = False

        rows.append({
            "trial_start_onset": start,
            "trial_stop_onset": end,
            "stimulus_onset": stim_onset,
            "response_onset": resp_onset,
            "rt_from_stimulus": rt_from_stim,
            "rt_from_trialstart": rt_from_trial,
            "response_type": resp_type,
            "correct": correct,
        })

    return pd.DataFrame(rows)

def _to_float_or_none(x):
    return None if pd.isna(x) else float(x)

def _to_int_or_none(x):
    if pd.isna(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        return int(x)
    except Exception:
        return None

def _to_str_or_none(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)

def annotate_trials_with_target(raw, target_field="rt_from_stimulus", epoch_length=2.0,
                                require_stimulus=True, require_response=True):
    """Create trial annotations with response time targets"""
    fnames = raw.filenames
    assert len(fnames) == 1, "Expected a single filename"
    if get_bids_path_from_fname:
        bids_path = get_bids_path_from_fname(fnames[0])
        events_file = bids_path.update(suffix="events", extension=".tsv").fpath
    else:
        # Fallback: look for events file in same directory
        base_path = Path(fnames[0])
        events_file = base_path.with_suffix('.tsv')
        if not events_file.exists():
            events_file = base_path.parent / (base_path.stem.replace('_eeg', '_events') + '.tsv')

    events_df = (pd.read_csv(events_file, sep="\t")
                   .assign(onset=lambda d: pd.to_numeric(d["onset"], errors="raise"))
                   .sort_values("onset", kind="mergesort").reset_index(drop=True))

    trials = build_trial_table(events_df)

    if require_stimulus:
        trials = trials[trials["stimulus_onset"].notna()].copy()
    if require_response:
        trials = trials[trials["response_onset"].notna()].copy()

    if target_field not in trials.columns:
        raise KeyError(f"{target_field} not in computed trial table.")
    targets = trials[target_field].astype(float)

    onsets     = trials["trial_start_onset"].to_numpy(float)
    durations  = np.full(len(trials), float(epoch_length), dtype=float)
    descs      = ["contrast_trial_start"] * len(trials)

    extras = []
    for i, v in enumerate(targets):
        row = trials.iloc[i]

        extras.append({
            "target": _to_float_or_none(v),
            "rt_from_stimulus": _to_float_or_none(row["rt_from_stimulus"]),
            "rt_from_trialstart": _to_float_or_none(row["rt_from_trialstart"]),
            "stimulus_onset": _to_float_or_none(row["stimulus_onset"]),
            "response_onset": _to_float_or_none(row["response_onset"]),
            "correct": _to_int_or_none(row["correct"]),
            "response_type": _to_str_or_none(row["response_type"]),
        })

    new_ann = mne.Annotations(onset=onsets, duration=durations, description=descs,
                              orig_time=raw.info["meas_date"], extras=extras)
    raw.set_annotations(new_ann, verbose=False)
    return raw

def add_aux_anchors(raw, stim_desc="stimulus_anchor", resp_desc="response_anchor"):
    """Add stimulus and response anchor events"""
    ann = raw.annotations
    mask = (ann.description == "contrast_trial_start")
    if not np.any(mask):
        return raw

    stim_onsets, resp_onsets = [], []
    stim_extras, resp_extras = [], []

    for idx in np.where(mask)[0]:
        ex = ann.extras[idx] if ann.extras is not None else {}
        t0 = float(ann.onset[idx])

        stim_t = ex.get("stimulus_onset")
        resp_t = ex.get("response_onset")

        if stim_t is None or (isinstance(stim_t, float) and np.isnan(stim_t)):
            rtt = ex.get("rt_from_trialstart")
            rts = ex.get("rt_from_stimulus")
            if rtt is not None and rts is not None:
                stim_t = t0 + float(rtt) - float(rts)

        if resp_t is None or (isinstance(resp_t, float) and np.isnan(resp_t)):
            rtt = ex.get("rt_from_trialstart")
            if rtt is not None:
                resp_t = t0 + float(rtt)

        if (stim_t is not None) and not (isinstance(stim_t, float) and np.isnan(stim_t)):
            stim_onsets.append(float(stim_t))
            stim_extras.append(dict(ex, anchor="stimulus"))
        if (resp_t is not None) and not (isinstance(resp_t, float) and np.isnan(resp_t)):
            resp_onsets.append(float(resp_t))
            resp_extras.append(dict(ex, anchor="response"))

    new_onsets = np.array(stim_onsets + resp_onsets, dtype=float)
    if len(new_onsets):
        aux = mne.Annotations(
            onset=new_onsets,
            duration=np.zeros_like(new_onsets, dtype=float),
            description=[stim_desc]*len(stim_onsets) + [resp_desc]*len(resp_onsets),
            orig_time=raw.info["meas_date"],
            extras=stim_extras + resp_extras,
        )
        raw.set_annotations(ann + aux, verbose=False)
    return raw

def keep_only_recordings_with(desc, concat_ds):
    """Keep only recordings that contain a specific event"""
    kept = []
    for ds in concat_ds.datasets:
        if np.any(ds.raw.annotations.description == desc):
            kept.append(ds)
        else:
            print(f"[warn] Recording {ds.raw.filenames[0]} does not contain event '{desc}'")
    return BaseConcatDataset(kept)

def add_extras_columns(
    windows_concat_ds,
    original_concat_ds,
    desc="contrast_trial_start",
    keys=("target","rt_from_stimulus","rt_from_trialstart","stimulus_onset","response_onset","correct","response_type"),
):
    float_cols = {"target","rt_from_stimulus","rt_from_trialstart","stimulus_onset","response_onset"}

    for win_ds, base_ds in zip(windows_concat_ds.datasets, original_concat_ds.datasets):
        ann = base_ds.raw.annotations
        idx = np.where(ann.description == desc)[0]
        if idx.size == 0:
            continue

        per_trial = [
            {k: (ann.extras[i][k] if ann.extras is not None and k in ann.extras[i] else None) for k in keys}
            for i in idx
        ]

        md = win_ds.metadata.copy()
        first = (md["i_window_in_trial"].to_numpy() == 0)
        trial_ids = first.cumsum() - 1
        n_trials = trial_ids.max() + 1 if len(trial_ids) else 0
        assert n_trials == len(per_trial), f"Trial mismatch: {n_trials} vs {len(per_trial)}"

        for k in keys:
            vals = [per_trial[t][k] if t < len(per_trial) else None for t in trial_ids]
            if k == "correct":
                ser = pd.Series([None if v is None else int(bool(v)) for v in vals],
                                index=md.index, dtype="Int64")
            elif k in float_cols:
                ser = pd.Series([np.nan if v is None else float(v) for v in vals],
                                index=md.index, dtype="Float64")
            else:  # response_type
                ser = pd.Series(vals, index=md.index, dtype="string")

            md[k] = ser

        win_ds.metadata = md.reset_index(drop=True)
        if hasattr(win_ds, "y"):
            y_np = win_ds.metadata["target"].astype(float).to_numpy()
            win_ds.y = y_np[:, None]  # (N, 1)

    return windows_concat_ds

# ========================================
# Lightning DataModule
# ========================================

class LocalEEGDataset(Dataset):
    """Dataset for loading local EEG .set files from BIDS structure"""
    
    def __init__(self, data_dir, task=None, window_size=2.0, sfreq=100):
        self.data_dir = Path(data_dir)
        self.task = task
        self.window_size = window_size
        self.sfreq = sfreq
        self.files = []
        self.windows = []
        self.targets = []
        self.file_info = []  # Store metadata about each window
        
        # Find all .set files (optionally filtered by task)
        self._find_files()
        
        # Load and window the data
        if self.files:
            self._load_and_window_data()
    
    def _find_files(self):
        """Find all .set files in BIDS structure (ds*/sub*/eeg/*.set)"""
        
        # Check for BIDS structure (ds* folders)
        ds_folders = list(self.data_dir.glob("ds*"))
        
        if ds_folders:
            print(f"Found {len(ds_folders)} dataset folders in BIDS structure")
            
            # Navigate BIDS structure: data/ds*/sub*/eeg/*.set
            if self.task:
                # Filter by specific task
                pattern = f"ds*/sub*/eeg/*task-{self.task}*.set"
                self.files = list(self.data_dir.glob(pattern))
                print(f"Found {len(self.files)} files for task '{self.task}'")
            else:
                # Get ALL .set files in BIDS structure
                pattern = "ds*/sub*/eeg/*.set"
                self.files = list(self.data_dir.glob(pattern))
                print(f"Found {len(self.files)} .set files in BIDS structure")
        else:
            # Fallback to simple recursive search if not BIDS structure
            if self.task:
                pattern = f"*task-{self.task}*.set"
                self.files = list(self.data_dir.rglob(pattern))
                print(f"Found {len(self.files)} files for task '{self.task}'")
            else:
                pattern = "*.set"
                self.files = list(self.data_dir.rglob(pattern))
                print(f"Found {len(self.files)} .set files in {self.data_dir}")
        
        # Show the files found with better organization
        if self.files:
            # Group by subject
            subjects = {}
            for f in self.files:
                # Extract subject ID from filename (e.g., sub-NDARDC843HHM)
                parts = f.stem.split('_')
                if parts and parts[0].startswith('sub-'):
                    subj = parts[0]
                    if subj not in subjects:
                        subjects[subj] = []
                    subjects[subj].append(f.name)
            
            print(f"\nFiles organized by subject:")
            for i, (subj, files) in enumerate(subjects.items()):
                if i < 3:  # Show first 3 subjects
                    print(f"  {subj}: {len(files)} runs")
                    for j, fname in enumerate(files[:2]):  # Show first 2 files per subject
                        print(f"    - {fname}")
                    if len(files) > 2:
                        print(f"    ... and {len(files) - 2} more")
                elif i == 3:
                    print(f"  ... and {len(subjects) - 3} more subjects")
    
    def _load_and_window_data(self):
        """Load EEG files and create windows"""
        total_windows = 0
        
        for file_idx, file_path in enumerate(self.files):
            try:
                # Detect file format and load appropriately
                # Check if it's actually a BDF file (common with EEGDash downloads)
                with open(file_path, 'rb') as f:
                    header = f.read(8)
                
                if header.startswith(b'\xff\x42\x49\x4f\x53\x45\x4d\x49') or header[0:1] == b'\xff':
                    # It's a BDF file (Biosemi format), even if it has .set extension
                    # This happens with EEGDash downloads
                    
                    # Create temporary file with .bdf extension
                    with tempfile.NamedTemporaryFile(suffix='.bdf', delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    # Copy the file with proper extension
                    shutil.copy2(file_path, tmp_path)
                    
                    try:
                        raw = mne.io.read_raw_bdf(tmp_path, preload=True, verbose=False)
                    finally:
                        # Clean up temp file
                        Path(tmp_path).unlink(missing_ok=True)
                else:
                    # Try as EEGLAB .set file
                    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
                
                # Get data
                data = raw.get_data()  # (n_channels, n_times)
                n_channels, n_times = data.shape
                
                # Check for corresponding JSON file with metadata
                json_path = file_path.with_suffix('.json')
                has_metadata = json_path.exists()
                
                # Resample if needed
                if raw.info['sfreq'] != self.sfreq:
                    from scipy import signal
                    resample_factor = self.sfreq / raw.info['sfreq']
                    n_times_new = int(n_times * resample_factor)
                    data = signal.resample(data, n_times_new, axis=1)
                
                # Create windows
                window_samples = int(self.window_size * self.sfreq)
                stride_samples = window_samples // 2  # 50% overlap
                
                n_windows_this_file = 0
                for start in range(0, data.shape[1] - window_samples, stride_samples):
                    end = start + window_samples
                    window = data[:, start:end]
                    
                    # Ensure we have 129 channels
                    if window.shape[0] < 129:
                        padding = np.zeros((129 - window.shape[0], window_samples))
                        window = np.vstack([window, padding])
                    elif window.shape[0] > 129:
                        window = window[:129, :]
                    
                    self.windows.append(window)
                    
                    # Create synthetic target (response time between 0.3 and 1.5 seconds)
                    # In real implementation, these would come from trial annotations
                    self.targets.append(np.random.uniform(0.3, 1.5))
                    
                    # Store metadata about this window
                    self.file_info.append({
                        'file': file_path.name,
                        'subject': file_path.stem.split('_')[0],
                        'window_idx': n_windows_this_file,
                        'has_metadata': has_metadata
                    })
                    
                    n_windows_this_file += 1
                    total_windows += 1
                
                if file_idx < 3:  # Show details for first 3 files
                    print(f"  Loaded {file_path.name}: {n_windows_this_file} windows")
                
            except Exception as e:
                print(f"  Error loading {file_path.name}: {str(e)}")
        
        print(f"\nTotal: {total_windows} windows from {len(self.files)} files")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.windows[idx])  # (129, 200)
        y = torch.FloatTensor([self.targets[idx]])  # (1,)
        return X, y

class EEGDataModule(LightningDataModule):
    """Lightning DataModule for EEG data with support for local files and EEGDash"""

    def __init__(
        self,
        data_dir: str = "data",
        task: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 2,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 2025,
        use_eegdash: bool = False,
        eegdash_mini: bool = True,
        download_full: bool = False,
        eegdash_release: str = "R5",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.use_eegdash = use_eegdash and EEGDASH_AVAILABLE
        self.eegdash_mini = eegdash_mini
        self.download_full = download_full
        self.eegdash_release = eegdash_release
        
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training/validation/testing"""
        data_loaded = False

        # Priority 1: Check for local data if it exists and not forced to download
        local_data_path = Path(self.data_dir)
        has_local_data = False

        if local_data_path.exists() and not self.download_full:
            # Check if we have local .set files
            set_files = list(local_data_path.rglob("*.set"))
            bids_structure = len(list(local_data_path.glob("ds*"))) > 0

            if set_files or bids_structure:
                has_local_data = True
                print(f"Found local data in {self.data_dir}")
                if not self.use_eegdash:
                    # Use local data directly
                    data_loaded = False  # Will load local data below

        # Priority 2: Try EEGDash if requested or if downloading full dataset
        if (self.use_eegdash or self.download_full) and EEGDASH_AVAILABLE and not data_loaded:
            try:
                print(f"Loading data from EEGDash (release={self.eegdash_release}, mini={self.eegdash_mini and not self.download_full})...")
                DATA_DIR = Path(self.data_dir)
                DATA_DIR.mkdir(parents=True, exist_ok=True)

                # Determine whether to use mini or full dataset
                use_mini = self.eegdash_mini and not self.download_full

                dataset_ccd = EEGChallengeDataset(
                    task="contrastChangeDetection",
                    release=self.eegdash_release,
                    cache_dir=DATA_DIR,
                    mini=use_mini
                )

                print(f"Dataset loaded: {len(dataset_ccd.datasets)} recordings")

                # Optional: Download all raw files in parallel if full download requested
                if self.download_full:
                    print("Downloading full dataset in parallel...")
                    from joblib import Parallel, delayed

                    def load_raw(d):
                        try:
                            return d.raw
                        except Exception as e:
                            print(f"Error loading raw data: {e}")
                            return None

                    raws = Parallel(n_jobs=-1, verbose=10)(
                        delayed(load_raw)(d) for d in dataset_ccd.datasets
                    )
                    print(f"Downloaded {sum(1 for r in raws if r is not None)} recordings successfully")
                
                # Process EEGDash data
                EPOCH_LEN_S = 2.0
                SFREQ = 100
                
                # Apply preprocessing
                transformation_offline = [
                    Preprocessor(
                        annotate_trials_with_target,
                        target_field="rt_from_stimulus",
                        epoch_length=EPOCH_LEN_S,
                        require_stimulus=True,
                        require_response=True,
                        apply_on_array=False,
                    ),
                    Preprocessor(add_aux_anchors, apply_on_array=False),
                ]
                
                preprocess(dataset_ccd, transformation_offline, n_jobs=1)
                
                # Use stimulus anchor for epoching
                ANCHOR = "stimulus_anchor"
                SHIFT_AFTER_STIM = 0.5
                WINDOW_LEN = 2.0
                
                # Keep only recordings with stimulus anchors
                dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)
                
                # Create stimulus-locked windows
                single_windows = create_windows_from_events(
                    dataset,
                    mapping={ANCHOR: 0},
                    trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
                    trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
                    window_size_samples=int(EPOCH_LEN_S * SFREQ),
                    window_stride_samples=SFREQ,
                    preload=True,
                )
                
                # Add target metadata
                single_windows = add_extras_columns(
                    single_windows,
                    dataset,
                    desc=ANCHOR,
                    keys=("target", "rt_from_stimulus", "rt_from_trialstart",
                          "stimulus_onset", "response_onset", "correct", "response_type")
                )
                
                # Get metadata and create subject-level splits
                meta_information = single_windows.get_metadata()
                
                print(f"Total windows: {len(meta_information)}")
                print(f"Response time range: {meta_information['target'].min():.3f} - {meta_information['target'].max():.3f} seconds")
                
                # Subject-level splits
                subjects = meta_information["subject"].unique()
                print(f"Total subjects: {len(subjects)}")
                
                train_subj, valid_test_subject = train_test_split(
                    subjects, test_size=(self.val_split + self.test_split), 
                    random_state=check_random_state(self.seed), shuffle=True
                )
                
                valid_subj, test_subj = train_test_split(
                    valid_test_subject, test_size=self.test_split/(self.val_split + self.test_split),
                    random_state=check_random_state(self.seed + 1), shuffle=True
                )
                
                # Create splits
                subject_split = single_windows.split("subject")
                train_set = []
                valid_set = []
                test_set = []
                
                for s in subject_split:
                    if s in train_subj:
                        train_set.append(subject_split[s])
                    elif s in valid_subj:
                        valid_set.append(subject_split[s])
                    elif s in test_subj:
                        test_set.append(subject_split[s])
                
                self.train_dataset = BaseConcatDataset(train_set)
                self.val_dataset = BaseConcatDataset(valid_set)
                self.test_dataset = BaseConcatDataset(test_set)
                
                data_loaded = True
                print("Successfully loaded data from EEGDash")
                
            except Exception as e:
                print(f"Could not load from EEGDash: {e}")
                data_loaded = False
        
        # Fall back to local data if EEGDash not available or failed
        if not data_loaded:
            print(f"Loading local data from {self.data_dir}...")
            dataset = LocalEEGDataset(
                data_dir=self.data_dir,
                task=self.task,
                window_size=2.0,
                sfreq=100
            )
            
            if len(dataset) == 0:
                raise ValueError(f"No data loaded from {self.data_dir}. Please provide valid EEG files.")
            
            # Create splits
            n_samples = len(dataset)
            indices = np.arange(n_samples)
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            
            train_size = int((1 - self.val_split - self.test_split) * n_samples)
            val_size = int(self.val_split * n_samples)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Create subset datasets
            from torch.utils.data import Subset
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
            self.test_dataset = Subset(dataset, test_indices)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# ========================================
# Main Training Script
# ========================================

def create_parser():
    """Create command-line argument parser with standard ML training options"""
    parser = argparse.ArgumentParser(
        description='EEG Challenge 2025 S4 Model Training with PyTorch Lightning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data-dir', type=str, default='data',
                          help='Directory for storing/loading data')
    data_group.add_argument('--task', type=str, default=None,
                          help='Specific task to load (e.g., contrastChangeDetection)')
    data_group.add_argument('--download-full', action='store_true',
                          help='Download full dataset from EEGDash (requires eegdash package)')
    data_group.add_argument('--use-mini', action='store_true', default=True,
                          help='Use mini dataset for faster testing')
    data_group.add_argument('--eegdash-release', type=str, default='R5',
                          help='EEGDash release version to use')
    data_group.add_argument('--force-eegdash', action='store_true',
                          help='Force use of EEGDash even if local data exists')

    # Model arguments
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--n-chans', type=int, default=129,
                           help='Number of EEG channels')
    model_group.add_argument('--n-times', type=int, default=200,
                           help='Number of time points (2s @ 100Hz)')
    model_group.add_argument('--d-model', type=int, default=128,
                           help='Hidden dimension size')
    model_group.add_argument('--n-layers', type=int, default=4,
                           help='Number of S4 layers')
    model_group.add_argument('--d-state', type=int, default=64,
                           help='State dimension for S4')
    model_group.add_argument('--dropout', type=float, default=0.1,
                           help='Dropout rate')
    model_group.add_argument('--enable-domain-adaptation', action='store_true', default=True,
                           help='Enable domain adaptation layers')
    model_group.add_argument('--n-domains', type=int, default=2,
                           help='Number of domains for adaptation')

    # Training arguments
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--batch-size', type=int, default=32,
                           help='Batch size for training')
    train_group.add_argument('--lr', '--learning-rate', type=float, default=1e-3,
                           help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=1e-4,
                           help='Weight decay for AdamW optimizer')
    train_group.add_argument('--epochs', type=int, default=50,
                           help='Number of training epochs')
    train_group.add_argument('--num-workers', type=int, default=2,
                           help='Number of data loading workers')
    train_group.add_argument('--val-split', type=float, default=0.15,
                           help='Validation split ratio')
    train_group.add_argument('--test-split', type=float, default=0.15,
                           help='Test split ratio')
    train_group.add_argument('--seed', type=int, default=2025,
                           help='Random seed for reproducibility')
    train_group.add_argument('--gradient-clip', type=float, default=1.0,
                           help='Gradient clipping value')
    train_group.add_argument('--huber-delta', type=float, default=0.1,
                           help='Delta parameter for Huber loss')

    # Callbacks and logging
    callback_group = parser.add_argument_group('Callbacks and Logging')
    callback_group.add_argument('--patience', type=int, default=10,
                              help='Early stopping patience')
    callback_group.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                              help='Directory for saving model checkpoints')
    callback_group.add_argument('--log-dir', type=str, default='lightning_logs',
                              help='Directory for tensorboard logs')
    callback_group.add_argument('--save-top-k', type=int, default=3,
                              help='Number of best models to keep')
    callback_group.add_argument('--log-every-n-steps', type=int, default=10,
                              help='Logging frequency')

    # Hardware arguments
    hw_group = parser.add_argument_group('Hardware Configuration')
    hw_group.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs to use (None for auto)')
    hw_group.add_argument('--accelerator', type=str, default='auto',
                        choices=['auto', 'gpu', 'cpu', 'tpu'],
                        help='Accelerator to use')
    hw_group.add_argument('--precision', type=str, default='32',
                        choices=['16', '32', 'bf16', '16-mixed', 'bf16-mixed'],
                        help='Training precision')
    hw_group.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic algorithms')

    # Experiment management
    exp_group = parser.add_argument_group('Experiment Management')
    exp_group.add_argument('--experiment-name', type=str, default='eeg_s4',
                         help='Name for this experiment run')
    exp_group.add_argument('--resume-from', type=str, default=None,
                         help='Path to checkpoint to resume training from')
    exp_group.add_argument('--test-only', action='store_true',
                         help='Only run testing (requires --resume-from)')
    exp_group.add_argument('--profile', action='store_true',
                         help='Enable profiling for performance analysis')
    exp_group.add_argument('--save-submission', action='store_true',
                         help='Save model in competition submission format after training')
    exp_group.add_argument('--submission-path', type=str, default=None,
                         help='Path for submission model (default: eeg_challenge_submission_<experiment>.pth)')

    # Advanced options
    adv_group = parser.add_argument_group('Advanced Options')
    adv_group.add_argument('--accumulate-grad-batches', type=int, default=1,
                         help='Number of batches to accumulate gradients')
    adv_group.add_argument('--mixed-precision', action='store_true',
                         help='Use automatic mixed precision training')
    adv_group.add_argument('--find-lr', action='store_true',
                         help='Run learning rate finder before training')
    adv_group.add_argument('--auto-batch-size', action='store_true',
                         help='Automatically find optimal batch size')
    adv_group.add_argument('--debug', action='store_true',
                         help='Enable debug mode with faster training')

    # Hyperparameter optimization
    hpo_group = parser.add_argument_group('Hyperparameter Optimization')
    hpo_group.add_argument('--optuna-trials', type=int, default=None,
                         help='Number of Optuna trials for hyperparameter search')
    hpo_group.add_argument('--optuna-study-name', type=str, default='eeg_s4_hpo',
                         help='Name for Optuna study')
    hpo_group.add_argument('--optuna-storage', type=str, default=None,
                         help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    hpo_group.add_argument('--optuna-pruning', action='store_true',
                         help='Enable Optuna pruning for early stopping of bad trials')
    hpo_group.add_argument('--hpo-metric', type=str, default='val/rmse',
                         help='Metric to optimize during hyperparameter search')
    hpo_group.add_argument('--hpo-direction', type=str, default='minimize',
                         choices=['minimize', 'maximize'],
                         help='Direction for metric optimization')
    hpo_group.add_argument('--hpo-epochs', type=int, default=20,
                         help='Number of epochs per trial during HPO')

    return parser

def create_optuna_objective(args, data_module):
    """Create Optuna objective function for hyperparameter optimization"""

    def objective(trial):
        """Optuna objective function for model training"""

        # Suggest hyperparameters
        suggested_lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        suggested_batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        suggested_d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
        suggested_n_layers = trial.suggest_int('n_layers', 2, 8)
        suggested_d_state = trial.suggest_categorical('d_state', [16, 32, 64, 128])
        suggested_dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        suggested_weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        suggested_huber_delta = trial.suggest_float('huber_delta', 0.05, 0.5)

        # Update data module batch size
        data_module.batch_size = suggested_batch_size

        # Create model with suggested hyperparameters
        model = LightningEEGS4Model(
            n_chans=args.n_chans,
            n_times=args.n_times,
            d_model=suggested_d_model,
            n_layers=suggested_n_layers,
            d_state=suggested_d_state,
            dropout=suggested_dropout,
            enable_domain_adaptation=args.enable_domain_adaptation,
            n_domains=args.n_domains,
            learning_rate=suggested_lr,
            weight_decay=suggested_weight_decay,
            scheduler_t_max=args.hpo_epochs,
            huber_delta=suggested_huber_delta,
        )

        # Set up callbacks for this trial
        callbacks = []

        # Add pruning callback if enabled
        if args.optuna_pruning and OPTUNA_AVAILABLE:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor=args.hpo_metric))

        # Simple checkpoint for best model in trial
        checkpoint_callback = ModelCheckpoint(
            monitor=args.hpo_metric,
            mode='min' if args.hpo_direction == 'minimize' else 'max',
            save_top_k=1,
            filename=f'trial_{trial.number}_{{epoch:02d}}_{{val_rmse:.4f}}',
            dirpath=f'optuna_checkpoints/{args.optuna_study_name}',
            verbose=False
        )
        callbacks.append(checkpoint_callback)

        # Early stopping for individual trials
        early_stopping = EarlyStopping(
            monitor=args.hpo_metric,
            patience=5,
            mode='min' if args.hpo_direction == 'minimize' else 'max',
            verbose=False
        )
        callbacks.append(early_stopping)

        # Configure trainer for HPO
        trainer = Trainer(
            max_epochs=args.hpo_epochs,
            accelerator=args.accelerator,
            devices=args.gpus if args.gpus else 'auto',
            callbacks=callbacks,
            enable_progress_bar=False,  # Disable for cleaner output
            enable_model_summary=False,
            logger=False,  # Disable logging during HPO
            deterministic=args.deterministic,
            gradient_clip_val=args.gradient_clip,
        )

        # Train model
        trainer.fit(model, data_module)

        # Return the metric to optimize
        metric_value = trainer.callback_metrics[args.hpo_metric].item()

        # Log hyperparameters to trial
        trial.set_user_attr('best_epoch', trainer.current_epoch)
        trial.set_user_attr('d_model', suggested_d_model)
        trial.set_user_attr('n_layers', suggested_n_layers)
        trial.set_user_attr('lr', suggested_lr)
        trial.set_user_attr('batch_size', suggested_batch_size)

        return metric_value

    return objective

def run_hyperparameter_optimization(args, data_module):
    """Run Optuna hyperparameter optimization"""

    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Install with: pip install optuna")

    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*60)
    print(f"Study name: {args.optuna_study_name}")
    print(f"Number of trials: {args.optuna_trials}")
    print(f"Optimization metric: {args.hpo_metric}")
    print(f"Direction: {args.hpo_direction}")
    print(f"Epochs per trial: {args.hpo_epochs}")
    print("="*60 + "\n")

    # Create or load study
    if args.optuna_storage:
        study = optuna.create_study(
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
            direction=args.hpo_direction,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner() if args.optuna_pruning else optuna.pruners.NopPruner()
        )
    else:
        study = optuna.create_study(
            study_name=args.optuna_study_name,
            direction=args.hpo_direction,
            pruner=optuna.pruners.MedianPruner() if args.optuna_pruning else optuna.pruners.NopPruner()
        )

    # Create objective function
    objective = create_optuna_objective(args, data_module)

    # Run optimization
    study.optimize(
        objective,
        n_trials=args.optuna_trials,
        show_progress_bar=True
    )

    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)

    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Trial number: {trial.number}")
    print(f"  {args.hpo_metric}: {trial.value:.6f}")

    print(f"\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Print top 5 trials
    print(f"\nTop 5 trials:")
    trials_df = study.trials_dataframe().sort_values('value', ascending=(args.hpo_direction == 'minimize'))
    print(trials_df[['number', 'value', 'params_lr', 'params_batch_size', 'params_d_model', 'params_n_layers']].head())

    # Save results
    results_file = Path(args.log_dir) / args.optuna_study_name / 'optimization_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'study_name': args.optuna_study_name,
        'n_trials': len(study.trials),
        'best_trial': {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'user_attrs': trial.user_attrs
        },
        'optimization_history': [
            {'trial': t.number, 'value': t.value, 'params': t.params}
            for t in study.trials
        ]
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Generate final training command with best hyperparameters
    print("\n" + "="*60)
    print("RECOMMENDED TRAINING COMMAND")
    print("="*60)
    print("\nTrain final model with best hyperparameters:")
    print(f"python3 {sys.argv[0]} \\")
    print(f"    --lr {trial.params['lr']:.6f} \\")
    print(f"    --batch-size {trial.params['batch_size']} \\")
    print(f"    --d-model {trial.params['d_model']} \\")
    print(f"    --n-layers {trial.params['n_layers']} \\")
    print(f"    --d-state {trial.params['d_state']} \\")
    print(f"    --dropout {trial.params['dropout']} \\")
    print(f"    --weight-decay {trial.params['weight_decay']:.6f} \\")
    print(f"    --huber-delta {trial.params['huber_delta']:.3f} \\")
    print(f"    --epochs {args.epochs} \\")
    print(f"    --experiment-name best_model \\")
    print(f"    --save-submission")
    print("="*60 + "\n")

    return study

def main():
    """Main training script using PyTorch Lightning with CLI arguments"""

    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Print configuration
    print("\n" + "="*60)
    print("EEG S4 MODEL TRAINING CONFIGURATION")
    print("="*60)
    for arg_group in parser._action_groups:
        if arg_group.title not in ['positional arguments', 'optional arguments']:
            print(f"\n{arg_group.title}:")
            for action in arg_group._group_actions:
                if action.dest != 'help':
                    value = getattr(args, action.dest)
                    print(f"  {action.dest}: {value}")
    print("="*60 + "\n")

    # Determine data source priority
    use_eegdash = args.force_eegdash or args.download_full

    # Set up data module with CLI arguments
    data_module = EEGDataModule(
        data_dir=args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        use_eegdash=use_eegdash,
        eegdash_mini=args.use_mini and not args.download_full,
        download_full=args.download_full,
        eegdash_release=args.eegdash_release,
    )

    # Initialize model with CLI arguments
    model = LightningEEGS4Model(
        n_chans=args.n_chans,
        n_times=args.n_times,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        dropout=args.dropout,
        enable_domain_adaptation=args.enable_domain_adaptation,
        n_domains=args.n_domains,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler_t_max=args.epochs,
        huber_delta=args.huber_delta,
    )
    
    # Load from checkpoint if specified
    if args.resume_from:
        print(f"Loading model from checkpoint: {args.resume_from}")
        model = LightningEEGS4Model.load_from_checkpoint(args.resume_from)

    # Set up callbacks
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/rmse',
        mode='min',
        save_top_k=args.save_top_k,
        filename=f'{args.experiment_name}-{{epoch:02d}}-{{val_rmse:.4f}}',
        dirpath=args.checkpoint_dir,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    if not args.test_only:
        early_stopping = EarlyStopping(
            monitor='val/rmse',
            patience=args.patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Set up loggers
    tb_logger = TensorBoardLogger(args.log_dir, name=args.experiment_name)
    csv_logger = CSVLogger(args.log_dir, name=args.experiment_name)
    
    # Configure trainer arguments
    trainer_kwargs = {
        'max_epochs': args.epochs if not args.test_only else 0,
        'accelerator': args.accelerator,
        'devices': args.gpus if args.gpus else 'auto',
        'callbacks': callbacks,
        'logger': [tb_logger, csv_logger],
        'gradient_clip_val': args.gradient_clip,
        'log_every_n_steps': args.log_every_n_steps,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'deterministic': args.deterministic,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'precision': args.precision if not args.mixed_precision else '16-mixed',
        'profiler': 'simple' if args.profile else None,
    }

    # Add debug-specific settings
    if args.debug:
        trainer_kwargs.update({
            'fast_dev_run': 5,  # Run only 5 batches for debugging
            'num_sanity_val_steps': 2,
            'log_every_n_steps': 1,
        })

    # Initialize trainer
    trainer = Trainer(**trainer_kwargs)
    
    # Run learning rate finder if requested
    if args.find_lr and not args.test_only:
        print("\nRunning learning rate finder...")
        lr_finder = trainer.tuner.lr_find(model, data_module)

        # Plot and suggest LR
        fig = lr_finder.plot(suggest=True)
        suggested_lr = lr_finder.suggestion()
        print(f"Suggested learning rate: {suggested_lr}")

        # Update model's learning rate
        model.hparams.learning_rate = suggested_lr

    # Auto-scale batch size if requested
    if args.auto_batch_size and not args.test_only:
        print("\nFinding optimal batch size...")
        trainer.tuner.scale_batch_size(model, data_module, mode='power')

    # Run hyperparameter optimization if requested
    if args.optuna_trials:
        study = run_hyperparameter_optimization(args, data_module)
        return  # Exit after optimization

    # Train or test based on arguments
    if args.test_only:
        if not args.resume_from:
            raise ValueError("--test-only requires --resume-from checkpoint path")
        print("\nRunning test evaluation only...")
        trainer.test(model, data_module)
    else:
        # Train model
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)

        trainer.fit(model, data_module)

        # Test model after training
        print("\nRunning test evaluation...")
        trainer.test(model, data_module, ckpt_path='best')

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
        if hasattr(checkpoint_callback, 'best_model_score') and checkpoint_callback.best_model_score is not None:
            print(f"Best validation RMSE: {checkpoint_callback.best_model_score:.6f}")

        # Save final results summary
        results_file = Path(args.log_dir) / args.experiment_name / 'final_results.json'
        results_file.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'experiment_name': args.experiment_name,
            'best_checkpoint': str(checkpoint_callback.best_model_path),
            'best_val_rmse': float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
            'epochs_trained': trainer.current_epoch,
            'configuration': vars(args)
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        # Save submission model if requested
        if args.save_submission:
            save_submission_model(model, checkpoint_callback, args)

def save_submission_model(model: LightningEEGS4Model, checkpoint_callback, args):
    """Save model in competition submission format"""

    print("\n" + "="*60)
    print("SAVING SUBMISSION MODEL")
    print("="*60)

    # Load best checkpoint if available
    if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path and Path(checkpoint_callback.best_model_path).exists():
        print(f"Loading best checkpoint: {checkpoint_callback.best_model_path}")
        model = LightningEEGS4Model.load_from_checkpoint(checkpoint_callback.best_model_path)
    else:
        print("Using current model state (no best checkpoint found)")

    # Create submission model wrapper
    submission_model = EEGChallengeSubmissionModel(model)
    submission_model.eval()

    # Determine save path
    if args.submission_path:
        submission_path = Path(args.submission_path)
    else:
        submission_path = Path(f"eeg_challenge_submission_{args.experiment_name}.pth")

    # Save the submission model
    torch.save(submission_model.state_dict(), submission_path)
    print(f"Submission model saved to: {submission_path}")

    # Also save a full model with metadata
    full_submission_path = submission_path.with_suffix('.full.pth')
    torch.save({
        'model_state_dict': submission_model.state_dict(),
        'model_config': {
            'n_chans': model.n_chans,
            'n_times': model.n_times,
            'd_model': model.d_model,
            'n_layers': len(model.s4_layers),
            'd_state': model.hparams.d_state,
            'dropout': model.hparams.dropout,
            'n_outputs': 1
        },
        'training_config': {
            'epochs_trained': trainer.current_epoch if 'trainer' in locals() else args.epochs,
            'best_val_rmse': float(checkpoint_callback.best_model_score) if hasattr(checkpoint_callback, 'best_model_score') and checkpoint_callback.best_model_score else None,
            'experiment_name': args.experiment_name,
            'data_config': {
                'used_full_dataset': args.download_full,
                'used_mini': args.use_mini and not args.download_full,
                'eegdash_release': args.eegdash_release if hasattr(args, 'eegdash_release') else 'R5'
            }
        }
    }, full_submission_path)
    print(f"Full submission model with metadata saved to: {full_submission_path}")

    # Test the submission model with dummy input
    print("\nTesting submission model...")
    dummy_input = torch.randn(4, 129, 200)  # Batch of 4 samples
    with torch.no_grad():
        test_output = submission_model(dummy_input)
    print(f"Test input shape: {dummy_input.shape}")
    print(f"Test output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")

    # Create a standalone loading script
    loading_script_path = submission_path.with_suffix('.load.py')
    loading_script = f'''"""Loading script for EEG Challenge 2025 submission model"""

import torch
import torch.nn as nn
from eeg_s4_lightning import EEGChallengeSubmissionModel, LightningEEGS4Model

def load_submission_model(model_path="{submission_path}", device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load the submission model for inference"""

    # Initialize Lightning model with saved config
    lightning_model = LightningEEGS4Model(
        n_chans=129,
        n_times=200,
        d_model={model.d_model},
        n_layers={len(model.s4_layers)},
        d_state={model.hparams.d_state},
        dropout=0.0,  # No dropout for inference
    )

    # Create submission wrapper
    submission_model = EEGChallengeSubmissionModel(lightning_model)

    # Load weights
    submission_model.load_state_dict(torch.load(model_path, map_location=device))
    submission_model.to(device)
    submission_model.eval()

    return submission_model

if __name__ == "__main__":
    # Example usage
    model = load_submission_model()

    # Test with dummy input
    dummy_input = torch.randn(1, 129, 200)
    with torch.no_grad():
        output = model.predict(dummy_input)

    print(f"Model loaded successfully!")
    print(f"Input shape: {{dummy_input.shape}}")
    print(f"Output: {{output}}")
'''

    with open(loading_script_path, 'w') as f:
        f.write(loading_script)
    print(f"\nLoading script saved to: {loading_script_path}")

    print("\n" + "="*60)
    print("SUBMISSION MODEL READY")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  1. Model weights: {submission_path}")
    print(f"  2. Full model with metadata: {full_submission_path}")
    print(f"  3. Loading script: {loading_script_path}")
    print(f"\nTo use in competition:")
    print(f"  - Submit '{submission_path}' as your model file")
    print(f"  - Use the loading script to verify model works correctly")
    print("="*60)

if __name__ == "__main__":
    main()