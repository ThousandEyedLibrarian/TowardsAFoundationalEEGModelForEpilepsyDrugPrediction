#!/usr/bin/env python3
"""
End-to-End Optimal Model Training Pipeline for EEG Challenge 2025
=================================================================

This script provides a complete training pipeline based on extensive optimisation
experiments. It trains the optimal S4D model architecture and automatically
creates a competition-ready submission package.

Author: Carter
Date: October 2025
Version: 1.0.0

"""

import os
import sys
import json
import time
import shutil
import zipfile
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')

# Import model architectures
from submission import OptimizedS4DEEG


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """
    Central configuration for the training pipeline.
    """

    # Model Architecture Configuration
    MODEL_CONFIG = {
        'd_model': 96,      # Hidden dimension (was 128)
        'n_layers': 8,      # Number of S4D layers (increased for better capacity)
        'd_state': 384,     # State dimension (4x d_model as recommended)
        'n_heads': 6,       # Attention heads (was 8)
        'bidirectional': True,
        'dropout': 0.20,    # Optimal dropout rate from experiments
        'use_demographics': False,  # Enable when demographic data is available
        'demographic_dim': 5,  # Expected dimension of demographic features

        # Moving Z-Score Normalization
        'use_moving_zscore': True,  # Enable moving z-score normalization
        'zscore_window_size': 50,    # Window size in samples (0.5s at 100Hz)
        'zscore_window_type': 'center',  # 'center', 'causal', or 'backward'
        'zscore_use_median': False   # Use median instead of mean (more robust)
    }

    # Training Parameters (balanced for stability and performance)
    TRAINING_CONFIG = {
        'learning_rate': 4e-4,
        'weight_decay': 0.012,
        'batch_size': 32,
        'max_epochs': 60,  # Increased from 40 for better convergence
        'patience': 20,
        'gradient_clip': 0.5
    }

    # Data Split Ratios
    DATA_CONFIG = {
        'val_size': 0.18,
        'test_size': 0.15,
        'seed': 456  # Best performing seed from experiments
    }

    # Paths
    DATA_PATH = 'r5_full_data.npz'  # Default, can be overridden
    OUTPUT_DIR = Path('optimal_model_output')
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    LOGS_DIR = OUTPUT_DIR / 'logs'
    SUBMISSION_DIR = OUTPUT_DIR / 'submission'


# ==============================================================================
# DATA MODULE
# ==============================================================================

class OptimalEEGDataset(Dataset):
    """
    Dataset implementation with optimal preprocessing and augmentation strategies.

    Rationale: The augmentation probabilities and strengths were calibrated to
    provide sufficient regularisation without destabilising training.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, demographics: Optional[np.ndarray] = None,
                 augment: bool = False,
                 use_moving_zscore: bool = False,
                 zscore_window_size: int = 50,
                 zscore_window_type: str = 'center',
                 zscore_use_median: bool = False):
        """
        Initialise the dataset.

        Args:
            X: EEG data of shape (n_samples, 129, 200)
            y: Response times of shape (n_samples, 1)
            demographics: Optional demographic features of shape (n_samples, demographic_dim)
            augment: Whether to apply data augmentation
            use_moving_zscore: Use moving z-score instead of global normalization
            zscore_window_size: Window size for moving z-score (in samples)
            zscore_window_type: Window type ('center', 'causal', 'backward')
            zscore_use_median: Use median instead of mean in moving z-score
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.demographics = torch.FloatTensor(demographics) if demographics is not None else None
        self.augment = augment

        # Store normalization parameters
        self.use_moving_zscore = use_moving_zscore
        self.zscore_window_size = zscore_window_size
        self.zscore_window_type = zscore_window_type
        self.zscore_use_median = zscore_use_median

        # Only compute global statistics if not using moving z-score
        if not use_moving_zscore:
            # Robust normalisation using median and standard deviation
            # This approach is less sensitive to outliers than mean normalisation
            self.median = torch.median(self.X, dim=0)[0].median(dim=-1)[0]
            self.std = self.X.std(dim=(0, 2)) + 1e-6

    def __len__(self) -> int:
        return len(self.X)

    def _apply_moving_zscore(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply moving z-score normalization to EEG data.

        This method normalizes each timepoint using statistics from a sliding window,
        making it adaptive to non-stationary signals like EEG.

        Args:
            x: Input tensor of shape (n_channels, n_timepoints)

        Returns:
            Normalized tensor of shape (n_channels, n_timepoints)
        """
        n_channels, n_timepoints = x.shape
        window_size = self.zscore_window_size

        # Determine padding based on window type
        # For unfold with step=1: num_windows = padded_length - window_size + 1
        # We want num_windows = n_timepoints, so:
        # padded_length = n_timepoints + window_size - 1
        # pad_before + n_timepoints + pad_after = n_timepoints + window_size - 1
        # pad_before + pad_after = window_size - 1
        if self.zscore_window_type == 'center':
            pad_before = window_size // 2
            pad_after = window_size - 1 - pad_before
        elif self.zscore_window_type == 'causal':
            pad_before = window_size - 1
            pad_after = 0
        elif self.zscore_window_type == 'backward':
            pad_before = window_size - 1
            pad_after = 0
        else:
            raise ValueError(f"Unknown window_type: {self.zscore_window_type}")

        # Pad signal to handle boundaries (reflection padding preserves signal properties)
        x_padded = F.pad(x, (pad_before, pad_after), mode='reflect')

        # Use unfold to create sliding windows: (n_channels, n_timepoints, window_size)
        # This is much faster than looping through each timepoint
        windows = x_padded.unfold(dimension=1, size=window_size, step=1)

        # Compute statistics across window dimension
        if self.zscore_use_median:
            # Use median for center and MAD for scale (more robust to outliers)
            center = windows.median(dim=2)[0]  # (n_channels, n_timepoints)
            # MAD = Median Absolute Deviation
            mad = (windows - center.unsqueeze(2)).abs().median(dim=2)[0]
            scale = mad * 1.4826 + 1e-6  # Convert MAD to std equivalent
        else:
            # Use mean and std (standard z-score)
            center = windows.mean(dim=2)  # (n_channels, n_timepoints)
            scale = windows.std(dim=2) + 1e-6  # Add epsilon for numerical stability

        # Normalize
        x_norm = (x - center) / scale

        return x_norm

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                              Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        x = self.X[idx].clone()
        y = self.y[idx].clone()
        demographics = self.demographics[idx].clone() if self.demographics is not None else None

        # Normalise channels
        if self.use_moving_zscore:
            # Moving z-score normalization (adaptive to temporal drift)
            x = self._apply_moving_zscore(x)
        else:
            # Global z-score normalization (original approach)
            x = (x - self.median.unsqueeze(-1)) / self.std.unsqueeze(-1)

        if self.augment:
            # 1. Gaussian noise (helps generalisation)
            if torch.rand(1) < 0.5:
                noise_level = torch.rand(1) * 0.015  # 0-1.5% noise
                x = x + torch.randn_like(x) * noise_level

            # 2. Channel dropout (simulates bad electrodes)
            if torch.rand(1) < 0.4:
                n_drop = torch.randint(1, 6, (1,)).item()
                drop_idx = torch.randperm(129)[:n_drop]
                x[drop_idx, :] = 0

            # 3. Temporal shift (accounts for timing variations)
            if torch.rand(1) < 0.4:
                shift = torch.randint(-8, 9, (1,)).item()
                if shift > 0:
                    x[:, shift:] = x[:, :-shift]
                    x[:, :shift] = 0
                elif shift < 0:
                    x[:, :shift] = x[:, -shift:]
                    x[:, shift:] = 0

            # 4. Amplitude scaling (accounts for individual differences)
            if torch.rand(1) < 0.4:
                scale = 0.8 + torch.rand(1) * 0.4  # Scale 0.8-1.2
                x = x * scale

            # Temporal masking removed - not necessary with z-score normalization

        # Return demographics if available
        if demographics is not None:
            return (x, demographics), y
        return x, y


class EEGDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for organised data handling.

    Rationale: Using a data module ensures consistent data processing across
    training, validation, and testing phases.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.data_loaded = False

    def prepare_data(self):
        """Check if data exists and validate format."""
        if not Path(self.config.DATA_PATH).exists():
            raise FileNotFoundError(
                f"Data file not found: {self.config.DATA_PATH}\n"
                f"Please ensure the dataset is available.\n"
                f"Options:\n"
                f"  1. Use existing: r5_full_data.npz (1080 samples)\n"
                f"  2. Use processed BDF: r5_l100_full.npz (large dataset)\n"
                f"  3. Process BDF files: python3 process_r5_bdf.py"
            )

    def setup(self, stage: Optional[str] = None):
        """Load and split the data."""
        if self.data_loaded:
            return

        # Load data
        print(f"Loading data from {self.config.DATA_PATH}...")
        data = np.load(self.config.DATA_PATH, allow_pickle=True)

        # Get normalization config
        use_moving_zscore = self.config.MODEL_CONFIG.get('use_moving_zscore', False)
        zscore_window_size = self.config.MODEL_CONFIG.get('zscore_window_size', 50)
        zscore_window_type = self.config.MODEL_CONFIG.get('zscore_window_type', 'center')
        zscore_use_median = self.config.MODEL_CONFIG.get('zscore_use_median', False)

        # Log normalization configuration
        print("\nNormalization Configuration:")
        if use_moving_zscore:
            print(f"  Type: Moving Z-Score (adaptive to non-stationary signals)")
            print(f"  Window size: {zscore_window_size} samples ({zscore_window_size/100:.2f}s at 100Hz)")
            print(f"  Window type: {zscore_window_type}")
            print(f"  Statistic: {'Median + MAD' if zscore_use_median else 'Mean + Std'}")
        else:
            print(f"  Type: Global Z-Score (median + std across dataset)")

        # Check if pre-split data is available
        if 'X_train' in data.keys():
            print("Using pre-split data from file...")
            self.train_dataset = OptimalEEGDataset(
                data['X_train'], data['y_train'],
                augment=True,
                use_moving_zscore=use_moving_zscore,
                zscore_window_size=zscore_window_size,
                zscore_window_type=zscore_window_type,
                zscore_use_median=zscore_use_median
            )
            self.val_dataset = OptimalEEGDataset(
                data['X_val'], data['y_val'],
                augment=False,
                use_moving_zscore=use_moving_zscore,
                zscore_window_size=zscore_window_size,
                zscore_window_type=zscore_window_type,
                zscore_use_median=zscore_use_median
            )
            self.test_dataset = OptimalEEGDataset(
                data['X_test'], data['y_test'],
                augment=False,
                use_moving_zscore=use_moving_zscore,
                zscore_window_size=zscore_window_size,
                zscore_window_type=zscore_window_type,
                zscore_use_median=zscore_use_median
            )

            print(f"\nData splits loaded from file:")
            print(f"  Train: {len(data['X_train']):,} samples")
            print(f"  Val: {len(data['X_val']):,} samples")
            print(f"  Test: {len(data['X_test']):,} samples")
            print(f"Response time range: [{data['y'].min():.3f}, {data['y'].max():.3f}] seconds")
        else:
            # Create splits from full dataset
            X = data['X']
            y = data['y']

            print(f"Data shape: X={X.shape}, y={y.shape}")
            print(f"Response time range: [{y.min():.3f}, {y.max():.3f}] seconds")

            # Create splits
            self._create_splits(X, y)
        self.data_loaded = True

    def _create_splits(self, X: np.ndarray, y: np.ndarray):
        """Create train/val/test splits."""
        np.random.seed(self.config.DATA_CONFIG['seed'])
        n_samples = len(X)
        indices = np.random.permutation(n_samples)

        n_test = int(n_samples * self.config.DATA_CONFIG['test_size'])
        n_val = int(n_samples * self.config.DATA_CONFIG['val_size'])
        n_train = n_samples - n_test - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        # Get normalization config
        use_moving_zscore = self.config.MODEL_CONFIG.get('use_moving_zscore', False)
        zscore_window_size = self.config.MODEL_CONFIG.get('zscore_window_size', 50)
        zscore_window_type = self.config.MODEL_CONFIG.get('zscore_window_type', 'center')
        zscore_use_median = self.config.MODEL_CONFIG.get('zscore_use_median', False)

        # Create datasets
        self.train_dataset = OptimalEEGDataset(
            X[train_idx], y[train_idx],
            augment=True,
            use_moving_zscore=use_moving_zscore,
            zscore_window_size=zscore_window_size,
            zscore_window_type=zscore_window_type,
            zscore_use_median=zscore_use_median
        )
        self.val_dataset = OptimalEEGDataset(
            X[val_idx], y[val_idx],
            augment=False,
            use_moving_zscore=use_moving_zscore,
            zscore_window_size=zscore_window_size,
            zscore_window_type=zscore_window_type,
            zscore_use_median=zscore_use_median
        )
        self.test_dataset = OptimalEEGDataset(
            X[test_idx], y[test_idx],
            augment=False,
            use_moving_zscore=use_moving_zscore,
            zscore_window_size=zscore_window_size,
            zscore_window_type=zscore_window_type,
            zscore_use_median=zscore_use_median
        )

        print(f"\nData splits created:")
        print(f"  Train: {len(train_idx):,} samples")
        print(f"  Val: {len(val_idx):,} samples")
        print(f"  Test: {len(test_idx):,} samples")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.TRAINING_CONFIG['batch_size'],
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.TRAINING_CONFIG['batch_size'],
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.TRAINING_CONFIG['batch_size'],
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )


# ==============================================================================
# MODEL MODULE
# ==============================================================================

class OptimalS4DLightning(pl.LightningModule):
    """
    PyTorch Lightning module implementing the optimal S4D architecture.

    Rationale: This module incorporates all successful training strategies
    identified through experimentation incl. adaptive learning rate scheduling.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Initialise model
        print("\nUsing Optimised S4D Model")

        # Filter out dataset-specific parameters (normalization config)
        model_params = {
            k: v for k, v in config.MODEL_CONFIG.items()
            if k not in ['use_moving_zscore', 'zscore_window_size',
                         'zscore_window_type', 'zscore_use_median']
        }

        self.model = OptimizedS4DEEG(
            n_chans=129,
            n_outputs=1,
            n_times=200,
            **model_params
        )

        # Track performance metrics
        self.val_rmse_history = []
        self.train_rmse_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Handle batches with or without demographics
        if len(batch) == 2 and isinstance(batch[0], tuple):
            # Batch contains demographics: ((x, demographics), y)
            (x, demographics), y = batch
        else:
            # Standard batch: (x, y)
            x, y = batch
            demographics = None

        # Check if model returns entropy loss (for MoE)
        if hasattr(self.model, 'use_demographics') and self.model.use_demographics and demographics is not None:
            result = self.model(x, demographics=demographics, return_entropy_loss=True)
        else:
            result = self.model(x, return_entropy_loss=True) if hasattr(self.model, 'use_moe') else self(x)

        if isinstance(result, tuple):
            pred, entropy_loss = result
            # Use Huber loss for robustness to outliers
            huber_loss = F.huber_loss(pred, y, reduction='mean', delta=1.0)
            # Combine Huber loss with entropy regularization (weighted at 0.2)
            loss = huber_loss + 0.2 * entropy_loss
            self.log('entropy_loss', entropy_loss, prog_bar=False)
        else:
            pred = result
            # Use Huber loss for robustness to outliers
            loss = F.huber_loss(pred, y, reduction='mean', delta=1.0)
            huber_loss = loss

        # Log metrics (still compute RMSE for comparison)
        mse_for_rmse = F.mse_loss(pred, y)
        true_rmse = torch.sqrt(mse_for_rmse)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_rmse', true_rmse, prog_bar=True)
        self.train_rmse_history.append(true_rmse.item())

        return loss

    def validation_step(self, batch, batch_idx):
        # Handle batches with or without demographics
        if len(batch) == 2 and isinstance(batch[0], tuple):
            (x, demographics), y = batch
        else:
            x, y = batch
            demographics = None

        # Make prediction
        if hasattr(self.model, 'use_demographics') and self.model.use_demographics and demographics is not None:
            pred = self.model(x, demographics=demographics)
        else:
            pred = self(x)
        loss = F.mse_loss(pred, y)
        rmse = torch.sqrt(loss)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_rmse', rmse, prog_bar=True)
        self.val_rmse_history.append(rmse.item())

        return loss

    def test_step(self, batch, batch_idx):
        # Handle batches with or without demographics
        if len(batch) == 2 and isinstance(batch[0], tuple):
            (x, demographics), y = batch
        else:
            x, y = batch
            demographics = None

        # Make prediction without any augmentation (test should use clean data)
        if hasattr(self.model, 'use_demographics') and self.model.use_demographics and demographics is not None:
            pred = self.model(x, demographics=demographics)
        else:
            pred = self(x)

        # Calculate loss
        loss = F.mse_loss(pred, y)

        self.log('test_loss', loss)
        self.log('test_rmse', torch.sqrt(loss))

        return loss

    def configure_optimizers(self):
        # AdamW optimiser with proper weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.TRAINING_CONFIG['learning_rate'],
            weight_decay=self.config.TRAINING_CONFIG['weight_decay']
        )

        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,     # First restart after 10 epochs
            T_mult=1,   # Don't increase period
            eta_min=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def on_validation_epoch_end(self):
        """Monitor training stability."""
        if len(self.val_rmse_history) > 3:
            recent_val = self.val_rmse_history[-3:]
            recent_train = self.train_rmse_history[-3:]

            val_trend = recent_val[-1] - recent_val[0]
            train_trend = recent_train[-1] - recent_train[0]

            if val_trend > 0.02 and train_trend < 0:
                print(f"\n[WARNING] Validation diverging "
                      f"(val: +{val_trend:.4f}, train: {train_trend:.4f})")


# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================

class TrainingPipeline:
    """
    Complete training pipeline with logging and submission creation.

    Rationale: This class encapsulates the entire training process, making it
    easy to run end-to-end experiments with consistent configuration.
    """

    def __init__(self, config: Config):
        self.config = config
        self._setup_directories()
        self._setup_reproducibility()

    def _setup_directories(self):
        """Create necessary directories."""
        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
        self.config.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.config.LOGS_DIR.mkdir(exist_ok=True)
        self.config.SUBMISSION_DIR.mkdir(exist_ok=True)

    def _setup_reproducibility(self):
        """Set seeds for reproducible results."""
        import random
        seed = self.config.DATA_CONFIG['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def train(self) -> Dict[str, float]:
        """
        Execute the training pipeline.

        Returns:
            Dictionary containing performance metrics
        """
        print("\n" + "="*70)
        print("OPTIMAL MODEL TRAINING PIPELINE")
        print("="*70)

        # Initialise data module
        data_module = EEGDataModule(self.config)
        data_module.setup()

        # Initialise model
        model = OptimalS4DLightning(self.config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel Parameters: {total_params:,}")
        print(f"Expected Performance: 0.87-0.89 Val RMSE")

        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.CHECKPOINT_DIR,
            filename='optimal_{epoch:02d}_{val_rmse:.3f}',
            monitor='val_rmse',
            mode='min',
            save_top_k=2,
            verbose=True
        )

        early_stop = EarlyStopping(
            monitor='val_rmse',
            patience=self.config.TRAINING_CONFIG['patience'],
            mode='min',
            verbose=True,
            min_delta=0.001
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # Setup logger
        logger = TensorBoardLogger(
            save_dir=self.config.LOGS_DIR,
            name='optimal_training',
            version=datetime.now().strftime('%Y%m%d_%H%M%S')
        )

        # Initialise trainer
        trainer = Trainer(
            max_epochs=self.config.TRAINING_CONFIG['max_epochs'],
            accelerator='auto',
            devices=1,
            callbacks=[checkpoint_callback, early_stop, lr_monitor],
            logger=logger,
            gradient_clip_val=self.config.TRAINING_CONFIG['gradient_clip'],
            precision='16-mixed' if torch.cuda.is_available() else 32,
            enable_progress_bar=True,
            deterministic=False
        )

        # Train
        print("\n" + "-"*70)
        print("Starting Training...")
        print(f"TensorBoard: tensorboard --logdir={logger.log_dir}")
        print("-"*70)

        trainer.fit(model, data_module)

        # Test
        print("\n" + "-"*70)
        print("Evaluating on Test Set...")
        print("-"*70)

        test_results = trainer.test(
            model,
            data_module,
            ckpt_path=checkpoint_callback.best_model_path
        )[0]

        # Extract and save best weights
        best_weights = self._extract_best_weights(checkpoint_callback.best_model_path)

        # Print results
        results = {
            'best_val_rmse': checkpoint_callback.best_model_score,
            'test_rmse': test_results['test_rmse'],
            'epochs_trained': trainer.current_epoch,
            'best_checkpoint': checkpoint_callback.best_model_path
        }

        self._print_results(results)
        self._save_results(results)

        return results, best_weights

    def _extract_best_weights(self, checkpoint_path: str) -> Dict:
        """Extract model weights from checkpoint."""
        # PyTorch 2.8 compatibility: set weights_only=False for complex checkpoints
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']

        # Remove 'model.' prefix
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_state_dict[k[6:]] = v
            else:
                clean_state_dict[k] = v

        return clean_state_dict

    def _print_results(self, results: Dict):
        """Print training results."""
        print("\n" + "="*70)
        print("TRAINING RESULTS")
        print("="*70)
        print(f"Best Validation RMSE: {results['best_val_rmse']:.5f}")
        print(f"Test RMSE: {results['test_rmse']:.5f}")
        print(f"Epochs Trained: {results['epochs_trained']}")
        print("="*70)

    def _save_results(self, results: Dict):
        """Save results to JSON."""
        results_path = self.config.OUTPUT_DIR / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")

    def create_submission(self, model_weights: Dict):
        """
        Create competition submission package.

        Args:
            model_weights: Trained model weights
        """
        print("\n" + "="*70)
        print("CREATING SUBMISSION PACKAGE")
        print("="*70)

        # Save weights for both challenges
        weights_path_1 = self.config.SUBMISSION_DIR / 'weights_challenge_1.pt'
        weights_path_2 = self.config.SUBMISSION_DIR / 'weights_challenge_2.pt'

        torch.save(model_weights, weights_path_1)
        torch.save(model_weights, weights_path_2)

        # Copy submission.py
        submission_src = Path('submission.py')
        if submission_src.exists():
            shutil.copy(submission_src, self.config.SUBMISSION_DIR / 'submission.py')

        # Create requirements.txt
        requirements = [
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "pytorch-lightning>=2.0.0"
        ]

        req_path = self.config.SUBMISSION_DIR / 'requirements.txt'
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))

        # Create submission zip
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_path = self.config.OUTPUT_DIR / f'submission_{timestamp}.zip'

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in self.config.SUBMISSION_DIR.iterdir():
                zipf.write(file, file.name)

        # Calculate file sizes
        zip_size = zip_path.stat().st_size / (1024 * 1024)

        print(f"\nSubmission package created: {zip_path}")
        print(f"   Size: {zip_size:.2f} MB")
        print(f"   Contents:")
        print(f"     - submission.py")
        print(f"     - weights_challenge_1.pt")
        print(f"     - weights_challenge_2.pt")
        print(f"     - requirements.txt")

        return zip_path


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="End-to-End Optimal Model Training for EEG Challenge 2025"
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='r5_full_data.npz',
        help='Path to the R5 dataset'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=60,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--no-submission',
        action='store_true',
        help='Skip submission package creation'
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=4,
        choices=[3, 4, 6, 8],
        help='Number of S4D layers to use'
    )
    parser.add_argument(
        '--no-moe',
        action='store_true',
        help='Disable Mixture of Experts module'
    )
    parser.add_argument(
        '--use-demographics',
        action='store_true',
        help='Enable demographic MLP branch (requires demographic data)'
    )
    parser.add_argument(
        '--demographic-dim',
        type=int,
        default=5,
        help='Dimension of demographic features (default: 5)'
    )
    parser.add_argument(
        '--use-moving-zscore',
        action='store_true',
        help='Use moving z-score normalization instead of global (better for non-stationary signals)'
    )
    parser.add_argument(
        '--zscore-window-size',
        type=int,
        default=50,
        help='Window size for moving z-score in samples (default: 50 = 0.5s at 100Hz)'
    )
    parser.add_argument(
        '--zscore-window-type',
        type=str,
        default='center',
        choices=['center', 'causal', 'backward'],
        help='Window type for moving z-score: center (symmetric), causal (past+present), backward (past only)'
    )
    parser.add_argument(
        '--zscore-use-median',
        action='store_true',
        help='Use median instead of mean in moving z-score (more robust to outliers)'
    )
    parser.add_argument(
        '--run-local-scoring',
        action='store_true',
        help='Run local scoring evaluation after training using local_scoring.py'
    )
    parser.add_argument(
        '--scoring-data-dir',
        type=str,
        default=None,
        help='Path to data directory for local scoring (default: same as data-path parent dir)'
    )
    parser.add_argument(
        '--fast-scoring',
        action='store_true',
        help='Run scoring on single subject only (fast validation)'
    )

    args = parser.parse_args()

    # Update configuration with command line arguments
    config = Config()
    config.DATA_PATH = args.data_path
    config.TRAINING_CONFIG['max_epochs'] = args.epochs
    config.TRAINING_CONFIG['batch_size'] = args.batch_size
    config.MODEL_CONFIG['n_layers'] = args.n_layers
    config.MODEL_CONFIG['use_moe'] = not args.no_moe
    config.MODEL_CONFIG['use_demographics'] = args.use_demographics
    config.MODEL_CONFIG['demographic_dim'] = args.demographic_dim

    # Moving z-score normalization configuration
    config.MODEL_CONFIG['use_moving_zscore'] = args.use_moving_zscore
    config.MODEL_CONFIG['zscore_window_size'] = args.zscore_window_size
    config.MODEL_CONFIG['zscore_window_type'] = args.zscore_window_type
    config.MODEL_CONFIG['zscore_use_median'] = args.zscore_use_median

    print("\n" + "="*70)
    print("SELECTED MODEL: Optimised S4D")
    print(f"  n_layers: {args.n_layers}")
    print(f"  d_state: {config.MODEL_CONFIG['d_state']}")
    print(f"  d_model: {config.MODEL_CONFIG['d_model']}")
    print(f"  MoE: {'Enabled' if not args.no_moe else 'Disabled'}")
    print(f"  Demographics: {'Enabled' if args.use_demographics else 'Disabled'}")
    if args.use_demographics:
        print(f"  Demographic dim: {args.demographic_dim}")
    print(f"  Normalization: {'Moving Z-Score' if args.use_moving_zscore else 'Global Z-Score'}")
    if args.use_moving_zscore:
        print(f"    Window: {args.zscore_window_size} samples ({args.zscore_window_size/100:.2f}s)")
    print("="*70)

    # Initialise pipeline
    pipeline = TrainingPipeline(config)

    try:
        # Train model
        results, model_weights = pipeline.train()

        # Create submission if requested
        submission_path = None
        if not args.no_submission:
            submission_path = pipeline.create_submission(model_weights)
            print(f"\nTraining complete! Submission ready at: {submission_path}")
        else:
            print("\nTraining complete!")

        # Run local scoring if requested
        scoring_results = None
        if args.run_local_scoring:
            if not submission_path and not args.no_submission:
                print("\n[WARNING] --run-local-scoring requires a submission package")
                print("          Run without --no-submission to create the package first")
            elif submission_path:
                # Determine data directory for scoring
                if args.scoring_data_dir:
                    scoring_data_dir = args.scoring_data_dir
                else:
                    # Try to infer from data path
                    data_path = Path(args.data_path)
                    if data_path.parent.name == 'data' or 'data' in str(data_path.parent):
                        scoring_data_dir = str(data_path.parent)
                    else:
                        scoring_data_dir = str(data_path.parent / 'data')
                    print(f"\nInferred scoring data directory: {scoring_data_dir}")

                # Run scoring
                scoring_results = run_local_scoring(
                    submission_path,
                    scoring_data_dir,
                    fast_dev_run=args.fast_scoring
                )

        # Final summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Final Test RMSE: {results['test_rmse']:.5f}")
        print(f"Output Directory: {config.OUTPUT_DIR}")
        print(f"TensorBoard Logs: {config.LOGS_DIR}")
        if submission_path:
            print(f"Submission Package: {submission_path}")
        if scoring_results:
            print(f"\nLocal Scoring Results:")
            print(f"  Challenge 1 NRMSE: {scoring_results.get('challenge1', 'N/A')}")
            print(f"  Challenge 2 NRMSE: {scoring_results.get('challenge2', 'N/A')}")
            print(f"  Overall Score (30%/70%): {scoring_results.get('overall', 'N/A')}")
        print("="*70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_local_scoring(submission_zip: Path, data_dir: str, fast_dev_run: bool = False):
    """
    Run local scoring evaluation on the submission package.

    Args:
        submission_zip: Path to submission ZIP file
        data_dir: Path to data directory containing R5 dataset
        fast_dev_run: If True, evaluate on single subject only

    Returns:
        Dict with scores for both challenges
    """
    import subprocess
    import json

    print("\n" + "="*70)
    print("RUNNING LOCAL SCORING EVALUATION")
    print("="*70)

    # Build command
    cmd = [
        'python3', 'local_scoring.py',
        '--submission-zip', str(submission_zip),
        '--data-dir', data_dir,
        '--output-dir', str(submission_zip.parent / 'scoring_output')
    ]

    if fast_dev_run:
        cmd.append('--fast-dev-run')
        print("Note: Running fast validation on single subject only")

    print(f"\nRunning: {' '.join(cmd)}")
    print()

    try:
        # Run local_scoring.py
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Display output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            print(f"\n[ERROR] Local scoring exited with code {result.returncode}")
            return None

        # Try to parse scores from output
        scores = {}
        for line in result.stdout.split('\n'):
            if 'NRMSE:' in line and 'Challenge 1' in result.stdout[:result.stdout.find(line)+100]:
                try:
                    scores['challenge1'] = float(line.split('NRMSE:')[1].split()[0])
                except:
                    pass
            elif 'NRMSE:' in line and 'Challenge 2' in result.stdout[:result.stdout.find(line)+100]:
                try:
                    scores['challenge2'] = float(line.split('NRMSE:')[1].split()[0])
                except:
                    pass
            elif 'NRMSE challenge 1' in line and 'NRMSE challenge 2' in line:
                try:
                    scores['overall'] = float(line.split(':')[1].strip())
                except:
                    pass

        print("\n" + "="*70)
        print("LOCAL SCORING RESULTS")
        print("="*70)
        if scores:
            print(f"Challenge 1 NRMSE: {scores.get('challenge1', 'N/A')}")
            print(f"Challenge 2 NRMSE: {scores.get('challenge2', 'N/A')}")
            print(f"Overall Score: {scores.get('overall', 'N/A')}")
        else:
            print("Scores could not be parsed from output")
        print("="*70)

        return scores

    except subprocess.TimeoutExpired:
        print("\n[ERROR] Local scoring timed out after 10 minutes")
        return None
    except FileNotFoundError:
        print("\n[ERROR] local_scoring.py not found in current directory")
        print("        Please ensure local_scoring.py is in the same directory")
        return None
    except Exception as e:
        print(f"\n[ERROR] Error running local scoring: {e}")
        return None


if __name__ == "__main__":
    main()