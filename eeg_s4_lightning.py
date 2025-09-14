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
        
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training/validation/testing"""
        data_loaded = False
        
        # Try EEGDash first if requested
        if self.use_eegdash and EEGDASH_AVAILABLE:
            try:
                print("Loading data from EEGDash...")
                DATA_DIR = Path(self.data_dir)
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                
                dataset_ccd = EEGChallengeDataset(
                    task="contrastChangeDetection",
                    release="R5", 
                    cache_dir=DATA_DIR,
                    mini=self.eegdash_mini
                )
                
                print(f"Dataset loaded: {len(dataset_ccd.datasets)} recordings")
                
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

def main():
    """Main training script using PyTorch Lightning"""
    
    # Set up data module
    data_module = EEGDataModule(
        data_dir="data",
        task=None,  # Use all available files
        batch_size=32,
        num_workers=2
    )
    
    # Initialize model
    model = LightningEEGS4Model(
        n_chans=129,
        n_times=200,
        d_model=128,
        n_layers=4,
        d_state=64,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        scheduler_t_max=50
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/rmse',
        mode='min',
        save_top_k=3,
        filename='eeg-s4-{epoch:02d}-{val_rmse:.4f}',
        dirpath='checkpoints',
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val/rmse',
        patience=10,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Set up loggers
    tb_logger = TensorBoardLogger('lightning_logs', name='eeg_s4')
    csv_logger = CSVLogger('lightning_logs', name='eeg_s4')
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=50,
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=[tb_logger, csv_logger],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True
    )
    
    # Train model
    print("Starting PyTorch Lightning training...")
    trainer.fit(model, data_module)
    
    # Test model
    print("\nRunning test evaluation...")
    trainer.test(model, data_module)
    
    print("\nTraining complete!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation RMSE: {checkpoint_callback.best_model_score:.6f}")

if __name__ == "__main__":
    main()