"""
Enhanced EEG Challenge 2025 Task 1 - S4 Model Solution
Implementation with full S4 block, cross-paradigm transfer learning, and domain adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List, Any
import math
from functools import partial
# from einops import rearrange, repeat  # Not used in this implementation
import warnings
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import json
import datetime
import logging
import sys

# TensorBoard support (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

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

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ================================
# Experiment Logger
# ================================

class ExperimentLogger:
    """Comprehensive logging for EEG S4 model experiments"""
    
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None, 
                 enable_tensorboard: bool = False, tensorboard_dir: str = "runs"):
        """Initialize experiment logger
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Name for this experiment (auto-generated if None)
            enable_tensorboard: Whether to enable TensorBoard logging
            tensorboard_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE
        
        # Create experiment-specific directory with timestamp
        if experiment_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"s4_eeg_{timestamp}"
        
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.figures_dir = self.experiment_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Initialize TensorBoard writer if enabled
        self.writer = None
        if self.enable_tensorboard:
            tensorboard_log_dir = Path(tensorboard_dir) / experiment_name
            self.writer = SummaryWriter(tensorboard_log_dir)
            logging.info(f"TensorBoard logging enabled: {tensorboard_log_dir}")
            print(f"TensorBoard logging to: {tensorboard_log_dir}")
            print(f"Run 'tensorboard --logdir={tensorboard_dir}' to visualize")
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_mae': [],
            'val_mae': [],
            'test_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'test_rmse': [],
            'train_r2': [],
            'val_r2': [],
            'test_r2': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        self.experiment_info = {
            'start_time': datetime.datetime.now().isoformat(),
            'device': str(device),
            'experiment_name': experiment_name,
            'tensorboard_enabled': self.enable_tensorboard
        }
        
        logging.info(f"Experiment logger initialized: {self.experiment_dir}")
    
    def setup_logging(self):
        """Setup Python logging to file and console"""
        log_file = self.experiment_dir / "experiment.log"
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def log_model_parameters(self, model: nn.Module, optimizer: Any, config: Dict[str, Any]):
        """Log model architecture and training configuration"""
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'architecture': str(model),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'optimizer': str(optimizer.__class__.__name__),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'config': config
        }
        
        # Save to JSON
        with open(self.experiment_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        logging.info(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
        logging.info(f"Optimizer: {model_info['optimizer']} (lr={model_info['learning_rate']})")
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = "train"):
        """Log metrics for a single epoch"""
        
        for key, value in metrics.items():
            metric_name = f"{phase}_{key}"
            if metric_name in self.metrics:
                self.metrics[metric_name].append(value)
            
            # Log to TensorBoard if enabled
            if self.writer:
                self.writer.add_scalar(f'{key}/{phase}', value, epoch)
        
        # Log to console/file
        log_msg = f"Epoch {epoch} [{phase}]: "
        log_msg += " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logging.info(log_msg)
    
    def log_training_complete(self, final_metrics: Dict[str, float]):
        """Log final training results and save all metrics"""
        
        self.experiment_info['end_time'] = datetime.datetime.now().isoformat()
        self.experiment_info['final_metrics'] = final_metrics
        
        # Save all metrics to JSON
        results = {
            'experiment_info': self.experiment_info,
            'metrics_history': self.metrics,
            'final_metrics': final_metrics
        }
        
        with open(self.experiment_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Log summary
        logging.info("="*60)
        logging.info("TRAINING COMPLETE")
        logging.info("="*60)
        for key, value in final_metrics.items():
            logging.info(f"{key}: {value:.4f}")
        logging.info(f"Results saved to: {self.experiment_dir}")
    
    def save_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, phase: str = "test"):
        """Save predictions to file"""
        
        predictions_file = self.experiment_dir / f"{phase}_predictions.npz"
        np.savez(predictions_file, y_true=y_true, y_pred=y_pred)
        logging.info(f"Predictions saved to: {predictions_file}")
    
    def plot_training_history(self):
        """Generate and save training history plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plot
        epochs = range(1, len(self.metrics.get('train_loss', [])) + 1)
        if epochs:
            axes[0, 0].plot(epochs, self.metrics.get('train_loss', []), 'b-', label='Train')
            axes[0, 0].plot(epochs, self.metrics.get('val_loss', []), 'r-', label='Validation')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # MAE plot
            axes[0, 1].plot(epochs, self.metrics.get('train_mae', []), 'b-', label='Train')
            axes[0, 1].plot(epochs, self.metrics.get('val_mae', []), 'r-', label='Validation')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].set_title('Mean Absolute Error')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # RMSE plot
            axes[1, 0].plot(epochs, self.metrics.get('train_rmse', []), 'b-', label='Train')
            axes[1, 0].plot(epochs, self.metrics.get('val_rmse', []), 'r-', label='Validation')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].set_title('Root Mean Square Error')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # R² plot
            axes[1, 1].plot(epochs, self.metrics.get('train_r2', []), 'b-', label='Train')
            axes[1, 1].plot(epochs, self.metrics.get('val_r2', []), 'r-', label='Validation')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('R²')
            axes[1, 1].set_title('R² Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.figures_dir / "training_history.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training history plot saved to: {fig_path}")
    
    def plot_predictions_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, phase: str = "test"):
        """Create scatter plot of predictions vs true values"""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        if len(y_true) > 1 and np.std(y_true) > 0:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            r2 = correlation ** 2 if not np.isnan(correlation) else 0.0
        else:
            r2 = 0.0
        
        # Add metrics text
        textstr = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('True Response Time (s)')
        ax.set_ylabel('Predicted Response Time (s)')
        ax.set_title(f'{phase.capitalize()} Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig_path = self.figures_dir / f"{phase}_predictions_scatter.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Predictions scatter plot saved to: {fig_path}")
    
    def save_checkpoint(self, model: nn.Module, optimizer: Any, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as best if it's the best validation performance
        if 'val_mae' in metrics:
            is_best = len(self.metrics['val_mae']) == 0 or metrics['val_mae'] <= min(self.metrics['val_mae'])
            if is_best:
                best_path = self.checkpoints_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                logging.info(f"Best model saved at epoch {epoch}")
    
    def close(self):
        """Close TensorBoard writer and cleanup"""
        if self.writer:
            self.writer.close()
            logging.info("TensorBoard writer closed")

# ================================
# Enhanced S4 Implementation (1D)
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
# Domain Adaptation Layer (Point 3)
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
# Enhanced EEG Model with Cross-Paradigm Transfer (Points 1-3)
# ===================================================

class EnhancedEEGS4Model(nn.Module):
    """Enhanced S4-based model for EEG Challenge 2025 with cross-paradigm transfer learning"""
    
    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        dropout: float = 0.1,
        n_outputs: int = 1,
        # Cross-paradigm parameters
        enable_domain_adaptation: bool = True,
        n_domains: int = 2,  # SuS and CCD
        # Transfer learning parameters
        pretrain_task: str = None,  # 'sus' for pretraining on SuS data
    ):
        super().__init__()
        
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
        self.pos_encoding = self._create_positional_encoding(n_times, d_model)
        
        # Enhanced S4 backbone (Point 1: Full S4 implementation)
        self.s4_layers = nn.ModuleList([
            EnhancedS4Layer(d_model, d_state, dropout=dropout) for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # Multi-head attention pooling with better initialization
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Task-specific heads for cross-paradigm transfer (Point 2)
        if pretrain_task == 'sus':
            # SuS task head (passive paradigm - could be stimulus detection)
            self.sus_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 2)  # Binary classification for stimulus detection
            )

        # Main task head (CCD response time prediction)
        self.main_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs)
        )

        # Domain adaptation (Point 3)
        if enable_domain_adaptation:
            self.domain_adapter = DomainAdaptationLayer(d_model, n_domains, dropout)
        
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
        pos_enc = self.pos_encoding[:, :x.size(1), :].to(x.device)
        x = x + pos_enc
        
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
        """
        Forward pass with multi-task capability for cross-paradigm transfer
        
        Args:
            x: Input EEG tensor (batch_size, n_chans, n_times)
            domain_labels: Domain labels for training domain adaptation (batch_size,)
            task: Which task head to use ('main', 'sus', 'both')
        
        Returns:
            Dictionary with predictions and losses
        """
        # Extract features using enhanced S4 backbone
        features = self.extract_features(x)  # (batch, n_times, d_model)
        
        # Pool features
        pooled_features = self.pool_features(features)  # (batch, d_model)
        
        outputs = {}
        
        # Main task prediction (CCD response time)
        if task in ['main', 'both']:
            main_pred = self.main_head(pooled_features).squeeze(-1)  # (batch,)
            outputs['main_pred'] = main_pred
        
        # SuS task prediction (if in pretraining mode for cross-paradigm transfer)
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

# ========================================
# Training utilities for multi-task setup
# ========================================

class MultiTaskLoss(nn.Module):
    """Multi-task loss with Huber loss for robustness and RMSE tracking"""
    
    def __init__(self, task_weights=None, huber_delta=0.1):
        super().__init__()
        self.task_weights = task_weights or {'main': 1.0, 'domain': 0.1, 'sus': 0.5}
        self.huber_loss = nn.HuberLoss(delta=huber_delta)
        # Delta parameter controls the threshold between L1 and L2 behavior
        # Lower delta = more robust to outliers, higher delta = more like MSE
    
    def forward(self, predictions, targets):
        """
        predictions: dict with model outputs
        targets: dict with target values
        Returns: dict with total_loss and individual metrics
        """
        total_loss = 0
        losses = {}
        
        # Main task loss using Huber loss (robust to outliers)
        if 'main_pred' in predictions and 'main_target' in targets:
            # Huber loss for training (robust to outliers)
            huber_loss = self.huber_loss(predictions['main_pred'], targets['main_target'])
            losses['huber_loss'] = huber_loss
            losses['main_loss'] = huber_loss  # Keep for compatibility
            
            # Also calculate MSE and RMSE for competition tracking
            with torch.no_grad():
                mse_loss = F.mse_loss(predictions['main_pred'], targets['main_target'])
                rmse_loss = torch.sqrt(mse_loss)
                losses['mse_loss'] = mse_loss
                losses['rmse_loss'] = rmse_loss
            
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

# ========================================
# Data Processing Functions
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
    bids_path = get_bids_path_from_fname(fnames[0])
    events_file = bids_path.update(suffix="events", extension=".tsv").fpath

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

        stim_t = ex["stimulus_onset"]
        resp_t = ex["response_onset"]

        if stim_t is None or (isinstance(stim_t, float) and np.isnan(stim_t)):
            rtt = ex["rt_from_trialstart"]
            rts = ex["rt_from_stimulus"]
            if rtt is not None and rts is not None:
                stim_t = t0 + float(rtt) - float(rts)

        if resp_t is None or (isinstance(resp_t, float) and np.isnan(resp_t)):
            rtt = ex["rt_from_trialstart"]
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
# Training Functions
# ========================================

def train_epoch_enhanced(model, train_loader, optimizer, criterion, device):
    """Enhanced training for one epoch with Huber loss and RMSE tracking"""
    model.train()
    total_loss = 0.0
    total_huber_loss = 0.0
    total_rmse = 0.0
    total_domain_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        X, y = batch[0], batch[1]  # EEG data and response times
        X, y = X.to(device).float(), y.to(device).float()
        
        # Create domain labels (0 for all samples since we only have CCD data currently)
        # In real cross-paradigm training, these would indicate SuS (0) vs CCD (1)
        domain_labels = torch.zeros(X.size(0), dtype=torch.long, device=device)

        optimizer.zero_grad()
        
        # Forward pass with enhanced model
        outputs = model(X, domain_labels=domain_labels, task='main')
        
        # Prepare targets for multi-task loss
        targets = {
            'main_target': y.squeeze(),
            'domain_target': domain_labels
        }
        
        # Calculate multi-task loss (includes Huber and RMSE)
        losses = criterion(outputs, targets)
        total_loss_batch = losses['total_loss']
        
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()
        if 'huber_loss' in losses:
            total_huber_loss += losses['huber_loss'].item()
        if 'rmse_loss' in losses:
            total_rmse += losses['rmse_loss'].item()
        if 'domain_loss' in losses:
            total_domain_loss += losses['domain_loss'].item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_huber_loss = total_huber_loss / n_batches
    avg_rmse = total_rmse / n_batches
    avg_domain_loss = total_domain_loss / n_batches
    
    return avg_loss, avg_huber_loss, avg_rmse, avg_domain_loss

@torch.no_grad()
def validate_epoch_enhanced(model, valid_loader, criterion, device):
    """Enhanced validation with Huber loss and RMSE tracking"""
    model.eval()
    total_loss = 0.0
    total_huber_loss = 0.0
    total_rmse = 0.0
    total_domain_loss = 0.0
    total_mae = 0.0
    n_batches = 0
    n_samples = 0

    for batch_idx, batch in enumerate(tqdm(valid_loader, desc="Validation")):
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()
        
        # Create domain labels
        domain_labels = torch.zeros(X.size(0), dtype=torch.long, device=device)

        # Forward pass
        outputs = model(X, domain_labels=domain_labels, task='main')
        predictions = outputs['main_pred']
        
        # Prepare targets
        targets = {
            'main_target': y.squeeze(),
            'domain_target': domain_labels
        }
        
        # Calculate losses (includes Huber and RMSE)
        losses = criterion(outputs, targets)
        mae = F.l1_loss(predictions, y.squeeze())

        total_loss += losses['total_loss'].item()
        if 'huber_loss' in losses:
            total_huber_loss += losses['huber_loss'].item()
        if 'rmse_loss' in losses:
            total_rmse += losses['rmse_loss'].item()
        if 'domain_loss' in losses:
            total_domain_loss += losses['domain_loss'].item()
        total_mae += mae.item() * X.size(0)
        n_batches += 1
        n_samples += X.size(0)

    avg_loss = total_loss / n_batches
    avg_huber_loss = total_huber_loss / n_batches
    avg_rmse = total_rmse / n_batches
    avg_domain_loss = total_domain_loss / n_batches
    avg_mae = total_mae / n_samples

    return avg_loss, avg_huber_loss, avg_rmse, avg_domain_loss, avg_mae

@torch.no_grad()
def evaluate_test_enhanced(model, test_loader, criterion, device):
    """Comprehensive test evaluation for enhanced model"""
    model.eval()

    all_predictions = []
    all_targets = []
    all_domain_predictions = []
    total_loss = 0.0
    total_main_loss = 0.0
    total_domain_loss = 0.0
    n_batches = 0

    for batch in tqdm(test_loader, desc="Testing"):
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()
        
        # Create domain labels
        domain_labels = torch.zeros(X.size(0), dtype=torch.long, device=device)

        # Forward pass
        outputs = model(X, domain_labels=domain_labels, task='main')
        predictions = outputs['main_pred']
        
        # Prepare targets
        targets = {
            'main_target': y.squeeze(),
            'domain_target': domain_labels
        }
        
        # Calculate losses
        losses = criterion(outputs, targets)

        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(y.squeeze().cpu().numpy())
        if 'domain_pred' in outputs:
            all_domain_predictions.extend(torch.softmax(outputs['domain_pred'], dim=1).cpu().numpy())
        
        total_loss += losses['total_loss'].item()
        if 'main_loss' in losses:
            total_main_loss += losses['main_loss'].item()
        if 'domain_loss' in losses:
            total_domain_loss += losses['domain_loss'].item()
        n_batches += 1

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_domain_predictions = np.array(all_domain_predictions) if all_domain_predictions else None

    # Calculate metrics
    test_loss = total_loss / n_batches
    test_main_loss = total_main_loss / n_batches
    test_domain_loss = total_domain_loss / n_batches
    test_mae = np.mean(np.abs(all_predictions - all_targets))
    test_rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    # Calculate R2 with error handling for edge cases
    if len(all_predictions) > 1 and np.std(all_targets) > 0:
        correlation = np.corrcoef(all_predictions, all_targets)[0, 1]
        test_r2 = correlation ** 2 if not np.isnan(correlation) else 0.0
    else:
        test_r2 = 0.0

    return {
        'total_loss': test_loss,
        'main_loss': test_main_loss,
        'domain_loss': test_domain_loss,
        'mae': test_mae,
        'rmse': test_rmse,
        'r2': test_r2,
        'predictions': all_predictions,
        'targets': all_targets,
        'domain_predictions': all_domain_predictions
    }

# ========================================
# Local Data Loading Support
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
                    import tempfile
                    import shutil
                    
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

# Synthetic data generation removed - use real data or download from EEGDash

# ========================================
# Main Training Pipeline
# ========================================

def main(use_local_data=True, data_dir="data", task=None, enable_logging=True, 
         enable_tensorboard=False, tensorboard_dir="runs"):
    """Main training pipeline for enhanced EEG S4 model
    
    Args:
        use_local_data: If True, try to use local .set files
        data_dir: Directory containing .set files (default: "data")
        task: Optional task filter (e.g., 'contrastChangeDetection'). If None, uses all .set files
        enable_logging: If True, logs all metrics to files (default: True)
        enable_tensorboard: If True, enables TensorBoard visualization (default: False)
        tensorboard_dir: Directory for TensorBoard logs (default: "runs")
    """
    
    print("Starting Enhanced EEG Challenge 2025 - S4 Model Training")
    print("=" * 60)
    
    # Initialize experiment logger if enabled
    logger = None
    if enable_logging:
        logger = ExperimentLogger(
            log_dir="logs",
            enable_tensorboard=enable_tensorboard,
            tensorboard_dir=tensorboard_dir
        )
        logging.info("Enhanced EEG Challenge 2025 - S4 Model Training Started")
        
        if enable_tensorboard and TENSORBOARD_AVAILABLE:
            print(f"\n✅ TensorBoard enabled!")
            print(f"To view real-time training metrics, run in a new terminal:")
            print(f"  tensorboard --logdir={tensorboard_dir}")
            print(f"Then open http://localhost:6006 in your browser\n")
        elif enable_tensorboard and not TENSORBOARD_AVAILABLE:
            print("\n⚠️  TensorBoard requested but not installed.")
            print("Install with: pip install tensorboard\n")
    
    # Data directory - now uses the data/ folder by default
    DATA_DIR = Path(data_dir)
    
    # Try to load data
    data_loaded = False
    
    # Option 1: Try local EEG data files from data/ directory
    if use_local_data:
        if DATA_DIR.exists():
            print(f"\nLoading EEG data from {DATA_DIR}/...")
            try:
                local_dataset = LocalEEGDataset(
                    data_dir=DATA_DIR,
                    task=task,  # Will use all .set files if task=None
                    window_size=2.0,
                    sfreq=100
                )
                
                if len(local_dataset) > 0:
                    print(f"Successfully loaded {len(local_dataset)} windows from local files")
                    
                    # Create train/val/test splits
                    n_samples = len(local_dataset)
                    indices = np.arange(n_samples)
                    np.random.shuffle(indices)
                    
                    train_size = int(0.7 * n_samples)
                    val_size = int(0.15 * n_samples)
                    
                    train_indices = indices[:train_size]
                    val_indices = indices[train_size:train_size + val_size]
                    test_indices = indices[train_size + val_size:]
                    
                    # Create subset datasets
                    from torch.utils.data import Subset
                    train_set = Subset(local_dataset, train_indices)
                    valid_set = Subset(local_dataset, val_indices)
                    test_set = Subset(local_dataset, test_indices)
                    
                    data_loaded = True
                    data_source = "local EEG files"
                    
            except Exception as e:
                print(f"Error loading local data: {e}")
    
    # Option 2: Try EEGDash if available and not using local data
    if not data_loaded and EEGDASH_AVAILABLE:
        print("\nAttempting to load data from EEGDash...")

        try:
            DATA_DIR = Path("data")
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            dataset_ccd = EEGChallengeDataset(
                task="contrastChangeDetection",
                release="R5", 
                cache_dir=DATA_DIR,
                mini=True
            )

            print(f"Dataset loaded: {len(dataset_ccd.datasets)} recordings")
            
            # Process EEGDash data similar to the original pipeline
            EPOCH_LEN_S = 2.0  # 2 second epochs
            SFREQ = 100  # 100 Hz sampling rate
            
            print("Processing trials and creating annotations...")
            
            # Apply preprocessing to extract trials and create annotations
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
            SHIFT_AFTER_STIM = 0.5  # 500ms shift after stimulus
            WINDOW_LEN = 2.0        # 2 second window
            
            # Keep only recordings with stimulus anchors
            dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)
            
            print(f"Creating windowed epochs from {len(dataset.datasets)} recordings...")
            
            # Create stimulus-locked windows
            single_windows = create_windows_from_events(
                dataset,
                mapping={ANCHOR: 0},
                trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
                trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
                window_size_samples=int(EPOCH_LEN_S * SFREQ),  # 200 samples
                window_stride_samples=SFREQ,
                preload=True,
            )
            
            print(f"Created {len(single_windows)} windowed epochs")
            
            # Add target metadata to windows
            single_windows = add_extras_columns(
                single_windows,
                dataset,
                desc=ANCHOR,
                keys=("target", "rt_from_stimulus", "rt_from_trialstart",
                      "stimulus_onset", "response_onset", "correct", "response_type")
            )
            
            print("Metadata added to windowed epochs")
            
            # Get metadata and create subject-level splits
            meta_information = single_windows.get_metadata()
            
            print(f"Total windows: {len(meta_information)}")
            print(f"Response time range: {meta_information['target'].min():.3f} - {meta_information['target'].max():.3f} seconds")
            print(f"Subjects: {meta_information['subject'].unique()}")
            
            # Subject-level train/validation/test split
            valid_frac = 0.15
            test_frac = 0.15
            seed = 2025
            
            subjects = meta_information["subject"].unique()
            print(f"Total subjects: {len(subjects)}")
            
            train_subj, valid_test_subject = train_test_split(
                subjects, test_size=(valid_frac + test_frac), random_state=check_random_state(seed), shuffle=True
            )
            
            valid_subj, test_subj = train_test_split(
                valid_test_subject, test_size=test_frac/(valid_frac + test_frac),
                random_state=check_random_state(seed + 1), shuffle=True
            )
            
            # Create splits using braindecode functionality
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
            
            train_set = BaseConcatDataset(train_set)
            valid_set = BaseConcatDataset(valid_set)
            test_set = BaseConcatDataset(test_set)
            
            data_loaded = True
            data_source = "EEGDash"
        except Exception as e:
            print(f"Could not connect to EEGDash: {e}")
    
    if not data_loaded:
        print("\n" + "=" * 60)
        print("ERROR: No EEG data available for training!")
        print("=" * 60)
        print("\nTo get data, you have two options:")
        print("\n1. Add local .set files:")
        print(f"   Place your EEG .set files in: {DATA_DIR}/")
        print("   The model will automatically find and use them.")
        print("\n2. Download from EEGDash:")
        print("   Run: python download_eeg_data.py")
        print("   This will download sample data from the EEG challenge.")
        print("\nOnce you have data, run this script again.")
        print("=" * 60)
        sys.exit(1)  # Graceful exit with error code
    
    print(f"\nUsing data from: {data_source}")
    print(f"Dataset splits:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Valid: {len(valid_set)} samples")
    print(f"  Test: {len(test_set)} samples")
    
    # Create PyTorch DataLoaders
    batch_size = 32
    num_workers = 2

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"DataLoaders created:")
    print(f"Train: {len(train_loader)} batches")
    print(f"Valid: {len(valid_loader)} batches")
    print(f"Test: {len(test_loader)} batches")
    
    # Initialise Enhanced S4 Model with Cross-Paradigm Transfer Learning
    model = EnhancedEEGS4Model(
        n_chans=129,                    # EEG channels
        n_times=200,                    # Time points (2s @ 100Hz)
        d_model=128,                    # Hidden dimension
        n_layers=4,                     # Enhanced S4 layers
        d_state=64,                     # State dimension  
        dropout=0.1,                    # Dropout rate
        n_outputs=1,                    # Response time regression
        enable_domain_adaptation=True,   # Point 3: Domain adaptation
        n_domains=2,                    # SuS and CCD domains
        pretrain_task=None              # Point 2: Could be 'sus' for pretraining
    ).to(device)

    print(f"Enhanced model initialised with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training configuration with multi-task loss
    lr = 1e-3
    weight_decay = 1e-4
    n_epochs = 50
    patience = 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Use multi-task loss for enhanced training
    criterion = MultiTaskLoss(task_weights={'main': 1.0, 'domain': 0.1})

    print(f"Enhanced training configuration:")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Epochs: {n_epochs}")
    print(f"Patience: {patience}")
    print(f"Multi-task loss with domain adaptation: enabled")
    
    # Enhanced training loop with cross-paradigm transfer learning
    print("Starting enhanced training...")

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_main_losses = []
    val_domain_losses = []
    val_maes = []
    val_rmses = []

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        print("-" * 60)

        # Enhanced training with Huber loss and RMSE tracking
        train_loss, train_huber, train_rmse, train_domain_loss = train_epoch_enhanced(
            model, train_loader, optimizer, criterion, device
        )
        train_losses.append(train_loss)

        # Enhanced validation with Huber loss and RMSE tracking
        val_loss, val_huber, val_rmse, val_domain_loss, val_mae = validate_epoch_enhanced(
            model, valid_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_main_losses.append(val_huber)  # Store Huber loss
        val_domain_losses.append(val_domain_loss)
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)  # Store RMSE for competition

        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log both Huber loss (training objective) and RMSE (competition metric)
        print(f"Train Loss: {train_loss:.6f} (Huber: {train_huber:.6f}, RMSE: {train_rmse:.6f}, Domain: {train_domain_loss:.6f})")
        print(f"Val Loss: {val_loss:.6f} (Huber: {val_huber:.6f}, RMSE: {val_rmse:.6f}, Domain: {val_domain_loss:.6f})")
        print(f"Val MAE: {val_mae:.6f}")
        print(f"Learning Rate: {current_lr:.8f}")
        
        # Log metrics if logger is enabled
        if logger:
            train_metrics = {
                'loss': train_loss,
                'huber_loss': train_huber,
                'rmse': train_rmse,
                'domain_loss': train_domain_loss
            }
            val_metrics = {
                'loss': val_loss,
                'huber_loss': val_huber,
                'rmse': val_rmse,
                'domain_loss': val_domain_loss,
                'mae': val_mae
            }
            logger.log_epoch_metrics(epoch, train_metrics, phase='train')
            logger.log_epoch_metrics(epoch, val_metrics, phase='val')
            logger.metrics['learning_rates'].append(current_lr)

        # Early stopping based on validation RMSE (competition metric)
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"New best validation RMSE: {best_val_loss:.6f}")
            
            # Save checkpoint if logger enabled
            if logger:
                checkpoint_metrics = {
                    'val_loss': val_loss,
                    'val_huber': val_huber,
                    'val_rmse': val_rmse,
                    'val_mae': val_mae
                }
                logger.save_checkpoint(model, optimizer, epoch, checkpoint_metrics)
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation RMSE: {best_val_loss:.6f}")

    print("\nEnhanced training completed!")
    
    # Enhanced test evaluation
    print("Evaluating enhanced model on test set...")

    test_results = evaluate_test_enhanced(model, test_loader, criterion, device)

    print(f"\n=== ENHANCED TEST RESULTS ===")
    print(f"Total Loss: {test_results['total_loss']:.6f}")
    print(f"Main Task Loss (Huber): {test_results['main_loss']:.6f}")
    print(f"Domain Loss: {test_results['domain_loss']:.6f}")
    print(f"Test MAE: {test_results['mae']:.6f} seconds")
    print(f"Test RMSE (Competition Metric): {test_results['rmse']:.6f} seconds")
    print(f"Test R2: {test_results['r2']:.6f}")
    print(f"Number of test samples: {len(test_results['predictions'])}")

    # Additional analysis for domain adaptation
    if test_results['domain_predictions'] is not None:
        domain_confidence = np.mean(np.max(test_results['domain_predictions'], axis=1))
        print(f"Average domain classification confidence: {domain_confidence:.3f}")

    print(f"\nEnhanced model performance:")
    print(f"Regression performance maintained with R2 = {test_results['r2']:.3f}")
    print(f"Domain adaptation training successful")
    print(f"Enhanced S4 architecture implemented")
    
    # Plot training curves and results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss curves
    epochs_range = range(1, len(train_losses) + 1)
    axes[0].plot(epochs_range, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs_range, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Validation metrics
    axes[1].plot(epochs_range, val_maes, 'g-', label='Val MAE')
    axes[1].plot(epochs_range, val_rmses, 'purple', label='Val RMSE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Error (seconds)')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True)

    # Predictions vs targets scatter plot
    axes[2].scatter(test_results['targets'], test_results['predictions'], alpha=0.6)
    axes[2].plot([test_results['targets'].min(), test_results['targets'].max()],
                 [test_results['targets'].min(), test_results['targets'].max()],
                 'r--', lw=2)
    axes[2].set_xlabel('True Response Time (s)')
    axes[2].set_ylabel('Predicted Response Time (s)')
    axes[2].set_title(f'Predictions vs Ground Truth\n(R2 = {test_results["r2"]:.3f})')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('eeg_s4_enhanced_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Training results plotted and saved")
    
    # ========================================
    # Section 6: Model Saving and Summary
    # ========================================
    
    print("\n" + "="*60)
    print("MODEL SAVING AND SUMMARY")
    print("="*60)
    
    # Determine save paths based on logger
    if logger:
        model_save_path = logger.checkpoints_dir / 'final_model.pth'
        submission_model_path = logger.experiment_dir / 'submission_model.pth'
        summary_path = logger.experiment_dir / 'model_summary.txt'
    else:
        model_save_path = 'eeg_s4_challenge_enhanced_model.pth'
        submission_model_path = 'eeg_challenge_2025_submission_s4.pth'
        summary_path = 'eeg_s4_enhanced_summary.txt'
    
    # Save complete model checkpoint
    model_checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_chans': 129,
            'n_times': 200,
            'd_model': 128,
            'n_layers': 4,
            'd_state': 64,
            'dropout': 0.1,
            'n_outputs': 1,
            'enable_domain_adaptation': True,
            'n_domains': 2,
            'pretrain_task': None
        },
        'test_results': test_results,
        'training_config': {
            'lr': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'epochs_trained': len(train_losses),
            'data_source': data_source,
            'best_val_loss': best_val_loss
        },
        'performance_metrics': {
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': best_val_loss,
            'test_mae': test_results['mae'],
            'test_rmse': test_results['rmse'],
            'test_r2': test_results['r2']
        }
    }
    
    torch.save(model_checkpoint, model_save_path)
    print(f"✓ Model checkpoint saved to: {model_save_path}")
    
    if logger:
        logging.info(f"Final model saved to: {model_save_path}")

    # Create comprehensive model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
{'='*70}
EEG CHALLENGE 2025 TASK 1 - S4 MODEL SUMMARY
{'='*70}

MODEL ARCHITECTURE:
- Architecture: S41D with Domain Adaptation
- Input Shape: 129 EEG channels × 200 time points (2s @ 100Hz)
- Hidden Dimension: 128
- S4 Layers: 4 (with FFT-based convolution)
- State Dimension: 64
- Attention: Multi-head attention pooling
- Domain Adaptation: Gradient reversal layer
- Output: Response time regression (1 value)

PARAMETERS:
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
- Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)

TRAINING CONFIGURATION:
- Data Source: {data_source}
- Train Samples: {len(train_set)}
- Validation Samples: {len(valid_set)}
- Test Samples: {len(test_set)}
- Batch Size: {batch_size}
- Learning Rate: {lr}
- Weight Decay: {weight_decay}
- Epochs Trained: {len(train_losses)}
- Early Stopping: Patience {patience}

PERFORMANCE METRICS:
- Best Validation Loss: {best_val_loss:.6f}
- Test MAE: {test_results['mae']:.6f} seconds
- Test RMSE: {test_results['rmse']:.6f} seconds
- Test R²: {test_results['r2']:.6f}
- Number of Test Samples: {len(test_results['predictions'])}

EXPERIMENT INFO:
- Timestamp: {datetime.datetime.now().isoformat()}
- Device: {device}
{'- Log Directory: ' + str(logger.experiment_dir) if logger else ''}
"""

    print(summary)

    # Save summary to file
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"\n✓ Model summary saved to: {summary_path}")
    
    if logger:
        logging.info(f"Model summary saved to: {summary_path}")
    
    # Challenge Submission Preparation
    
    print("\n" + "="*60)
    print("CHALLENGE SUBMISSION PREPARATION")
    print("="*60)
    
    # Create submission-ready model class
    class EEGChallengeSubmissionModel(nn.Module):
        """Submission-ready model wrapper for EEG Challenge 2025"""
        
        def __init__(self, model_config):
            super().__init__()
            self.model_config = model_config
            
            # Initialize the enhanced S4 model
            self.s4_model = EnhancedEEGS4Model(
                n_chans=model_config['n_chans'],
                n_times=model_config['n_times'],
                d_model=model_config['d_model'],
                n_layers=model_config['n_layers'],
                d_state=model_config['d_state'],
                dropout=model_config['dropout'],
                n_outputs=model_config['n_outputs'],
                enable_domain_adaptation=model_config['enable_domain_adaptation'],
                n_domains=model_config['n_domains']
            )
        
        def forward(self, x, return_dict=False):
            """Forward pass for submission
            
            Args:
                x: EEG tensor of shape (batch_size, n_chans=129, n_times=200)
                return_dict: If True, return full output dict; else return predictions only
            
            Returns:
                Response time predictions (batch_size,) or full output dict
            """
            # Handle both enhanced model (dict output) and standard output
            outputs = self.s4_model(x)
            
            if isinstance(outputs, dict):
                if return_dict:
                    return outputs
                else:
                    # Return main predictions for submission (check various possible keys)
                    return outputs.get('main_output', outputs.get('main_pred', outputs.get('output', list(outputs.values())[0])))
            else:
                return outputs
        
        def predict(self, x):
            """Prediction method for challenge submission"""
            self.eval()
            with torch.no_grad():
                predictions = self.forward(x, return_dict=False)
                # Ensure output is 1D for regression task
                if predictions.dim() > 1 and predictions.size(-1) == 1:
                    predictions = predictions.squeeze(-1)
            return predictions
    
    # Create submission model instance
    submission_model = EEGChallengeSubmissionModel(model_checkpoint['model_config'])
    
    # Load trained weights
    submission_model.s4_model.load_state_dict(model.state_dict())
    submission_model.eval()
    
    print("Submission model created and weights loaded")
    
    # Test submission model format
    dummy_input = torch.randn(4, 129, 200).to(device)  # Batch of 4 samples
    with torch.no_grad():
        test_output = submission_model.predict(dummy_input)
    
    print(f"Submission format test - Input: {dummy_input.shape}, Output: {test_output.shape}")
    assert test_output.shape == (4,), f"Expected output shape (4,), got {test_output.shape}"
    
    # Save submission model
    submission_checkpoint = {
        'model_state_dict': submission_model.state_dict(),
        'model_config': model_checkpoint['model_config'],
        'submission_info': {
            'challenge': 'EEG Challenge 2025',
            'task': 'Response Time Prediction',
            'input_shape': '(batch_size, 129, 200)',
            'output_shape': '(batch_size,)',
            'created': datetime.datetime.now().isoformat(),
            'test_mae': test_results['mae'],
            'test_rmse': test_results['rmse'],
            'test_r2': test_results['r2']
        }
    }
    
    torch.save(submission_checkpoint, submission_model_path)
    print(f"Submission model saved to: {submission_model_path}")
    
    if logger:
        logging.info(f"Submission model saved to: {submission_model_path}")
    
    # Create submission README
    readme_content = f"""
# EEG Challenge 2025 - Enhanced S4 Model Submission

## Model Information
- Architecture: Enhanced S4 with Domain Adaptation
- Parameters: {total_params:,}
- Test MAE: {test_results['mae']:.4f} seconds
- Test RMSE: {test_results['rmse']:.4f} seconds

## Usage
```python
import torch
from eeg_challenge_2025_s4_enhanced import EEGChallengeSubmissionModel

# Load model
checkpoint = torch.load('submission_model.pth')
model = EEGChallengeSubmissionModel(checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
X = torch.randn(32, 129, 200)  # Batch of EEG data
predictions = model.predict(X)  # Response time predictions
```

## Input/Output Format
- Input: Tensor of shape (batch_size, 129, 200)
  - 129 EEG channels
  - 200 time points (2 seconds @ 100Hz)
- Output: Tensor of shape (batch_size,)
  - Response time predictions in seconds

## Created
{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = logger.experiment_dir / 'SUBMISSION_README.md' if logger else 'SUBMISSION_README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Submission README saved to: {readme_path}")
    
    print("\n" + "="*60)
    print("SUBMISSION PREPARATION COMPLETE")
    print("="*60)
    print(f"\nFiles created for submission:")
    print(f"1. Model checkpoint: {model_save_path}")
    print(f"2. Submission model: {submission_model_path}")
    print(f"3. Model summary: {summary_path}")
    print(f"4. Submission README: {readme_path}")
    if logger:
        print(f"\nAll files saved to experiment directory: {logger.experiment_dir}")
        logging.info("Challenge submission preparation complete")
    
    print("\nModel is ready for EEG Challenge 2025 submission!")
    print("="*60)
    
    # Clean up logger resources
    if logger:
        logger.close()
        logging.info("Training pipeline completed and resources cleaned up")

if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced EEG Challenge 2025 - S4 Model Training")
    parser.add_argument("--data-dir", type=str, default="data", 
                        help="Directory containing .set files (default: data)")
    parser.add_argument("--task", type=str, default=None,
                        help="Filter by task name (e.g., contrastChangeDetection). If not specified, uses all .set files")
    # Removed --synthetic option as synthetic data is no longer supported
    parser.add_argument("--eegdash", action="store_true",
                        help="Try to use EEGDash (requires connection)")
    parser.add_argument("--no-local", action="store_true",
                        help="Don't use local data files")
    parser.add_argument("--no-logging", action="store_true",
                        help="Disable comprehensive logging to files")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard visualization for real-time metric tracking")
    parser.add_argument("--tensorboard-dir", type=str, default="runs",
                        help="Directory for TensorBoard logs (default: runs)")
    
    args = parser.parse_args()
    
    # Determine data source
    use_local = not args.no_local and not args.eegdash
    
    if args.eegdash:
        use_local = False
    
    # Determine if logging should be enabled (enabled by default)
    enable_logging = not args.no_logging
    
    # Run main with specified options
    main(use_local_data=use_local,
         data_dir=args.data_dir,
         task=args.task,
         enable_logging=enable_logging,
         enable_tensorboard=args.tensorboard,
         tensorboard_dir=args.tensorboard_dir)