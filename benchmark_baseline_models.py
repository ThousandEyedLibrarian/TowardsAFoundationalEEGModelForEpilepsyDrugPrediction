#!/usr/bin/env python3
"""
Baseline Model Benchmarking for EEG Challenge 2025
==================================================

This script benchmarks pre-made models from the braindecode library against
the custom OptimizedS4DEEG model on both Challenge 1 (response time prediction)
and Challenge 2 (externalizing score prediction).

Models tested:
- EEGNet (Lawhern et al., 2018)
- ShallowFBCSPNet (Schirrmeister et al., 2017)
- Deep4Net (Schirrmeister et al., 2017)
- EEGConformer (Song et al., 2022)
- TIDNet (Thinker Invariance DenseNet)
- Custom OptimizedS4DEEG (baseline comparison)

Author: Carter
Date: October 2025
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Import braindecode models
try:
    from braindecode.models import (
        EEGNetv4,
        ShallowFBCSPNet,
        Deep4Net,
        EEGConformer,
        TIDNet,
        ATCNet,
        TCN,
        EEGITNet,
        TSception,
        EEGNeX,
        FBCNet,
    )
    BRAINDECODE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Could not import braindecode models: {e}")
    print("          Install with: pip install braindecode")
    BRAINDECODE_AVAILABLE = False

# Import custom model
try:
    from submission import OptimizedS4DEEG
    CUSTOM_MODEL_AVAILABLE = True
except ImportError:
    print("[WARNING] Could not import OptimizedS4DEEG from submission.py")
    CUSTOM_MODEL_AVAILABLE = False


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class BenchmarkConfig:
    """Configuration for benchmarking experiments"""

    # Data
    DATA_PATH = 'r5_full_data.npz'
    VAL_SIZE = 0.18
    TEST_SIZE = 0.15
    SEED = 456

    # Training
    BATCH_SIZE = 32
    MAX_EPOCHS = 60
    PATIENCE = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP = 0.5

    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # Output
    OUTPUT_DIR = Path('benchmark_results')
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'

    # EEG specs
    N_CHANS = 129
    N_TIMES = 200
    SFREQ = 100
    N_OUTPUTS = 1  # Regression task


# ==============================================================================
# DATASET
# ==============================================================================

class BenchmarkEEGDataset(Dataset):
    """Simple dataset for benchmarking - no augmentation"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: EEG data of shape (n_samples, 129, 200)
            y: Target values of shape (n_samples,) or (n_samples, 1)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

        # Global z-score normalization
        self.mean = self.X.mean(dim=(0, 2), keepdim=True)
        self.std = self.X.std(dim=(0, 2), keepdim=True) + 1e-6

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # self.mean and self.std have shape (1, 129, 1)
        # self.X[idx] has shape (129, 200)
        # We need to broadcast mean/std to (129, 1) for proper channel-wise normalization
        mean = self.mean.squeeze(0)  # (129, 1)
        std = self.std.squeeze(0)    # (129, 1)
        X = (self.X[idx] - mean) / std
        y = self.y[idx]
        return X, y


# ==============================================================================
# MODEL WRAPPERS
# ==============================================================================

class BraindecodeRegressionWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for braindecode models doing regression"""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = nn.HuberLoss(delta=1.0)

        # Metrics storage
        self.train_losses = []
        self.val_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        X, y = batch
        y_pred = self.forward(X)
        loss = self.loss_fn(y_pred, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        X, y = batch
        y_pred = self.forward(X)
        loss = self.loss_fn(y_pred, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_losses.append(loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


class CustomModelWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for OptimizedS4DEEG"""

    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 4e-4,
        weight_decay: float = 0.012
    ):
        super().__init__()
        self.save_hyperparameters()

        # Filter out dataset-specific parameters
        model_params = {
            k: v for k, v in model_config.items()
            if k not in ['use_moving_zscore', 'zscore_window_size',
                         'zscore_window_type', 'zscore_use_median']
        }

        self.model = OptimizedS4DEEG(
            n_chans=BenchmarkConfig.N_CHANS,
            n_outputs=BenchmarkConfig.N_OUTPUTS,
            n_times=BenchmarkConfig.N_TIMES,
            **model_params
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = nn.HuberLoss(delta=1.0)

        self.train_losses = []
        self.val_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        X, y = batch
        y_pred = self.forward(X)
        loss = self.loss_fn(y_pred, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        X, y = batch
        y_pred = self.forward(X)
        loss = self.loss_fn(y_pred, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_losses.append(loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


# ==============================================================================
# MODEL FACTORY
# ==============================================================================

def create_model(model_name: str, config: BenchmarkConfig) -> Optional[nn.Module]:
    """
    Create a model instance by name.

    Args:
        model_name: Name of the model to create
        config: Configuration object

    Returns:
        Model instance or None if model not available
    """
    if not BRAINDECODE_AVAILABLE and model_name != 'OptimizedS4DEEG':
        print(f"[ERROR] Braindecode not available, cannot create {model_name}")
        return None

    try:
        if model_name == 'EEGNetv4':
            model = EEGNetv4(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS,
                n_times=config.N_TIMES,
                final_conv_length='auto',
                drop_prob=0.25
            )

        elif model_name == 'ShallowFBCSPNet':
            model = ShallowFBCSPNet(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS,
                n_times=config.N_TIMES,
                final_conv_length='auto',
                drop_prob=0.25
            )

        elif model_name == 'Deep4Net':
            model = Deep4Net(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS,
                n_times=config.N_TIMES,
                final_conv_length='auto',
                drop_prob=0.25
            )

        elif model_name == 'EEGConformer':
            model = EEGConformer(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS,
                n_times=config.N_TIMES,
                drop_prob=0.25
            )

        elif model_name == 'TIDNet':
            model = TIDNet(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS,
                n_times=config.N_TIMES,
                drop_prob=0.25
            )

        elif model_name == 'ATCNet':
            model = ATCNet(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS,
                n_times=config.N_TIMES
            )

        elif model_name == 'TCN':
            model = TCN(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS
            )

        elif model_name == 'EEGITNet':
            model = EEGITNet(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS,
                n_times=config.N_TIMES,
                drop_prob=0.25
            )

        elif model_name == 'TSception':
            model = TSception(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS,
                n_times=config.N_TIMES,
                sfreq=config.SFREQ,
                drop_prob=0.25
            )

        elif model_name == 'EEGNeX':
            model = EEGNeX(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS,
                n_times=config.N_TIMES,
                drop_prob=0.25
            )

        elif model_name == 'FBCNet':
            model = FBCNet(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS,
                n_times=config.N_TIMES,
                sfreq=config.SFREQ
            )

        elif model_name == 'OptimizedS4DEEG':
            if not CUSTOM_MODEL_AVAILABLE:
                print("[ERROR] OptimizedS4DEEG not available")
                return None

            # Use optimal config from train_optimal_model.py
            model_config = {
                'd_model': 96,
                'n_layers': 8,
                'd_state': 384,
                'n_heads': 6,
                'bidirectional': True,
                'dropout': 0.20,
                'use_demographics': False,
                'demographic_dim': 5,
            }

            model = OptimizedS4DEEG(
                n_chans=config.N_CHANS,
                n_outputs=config.N_OUTPUTS,
                n_times=config.N_TIMES,
                **model_config
            )

        else:
            print(f"[ERROR] Unknown model: {model_name}")
            return None

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Created {model_name} with {n_params:,} parameters")

        return model

    except Exception as e:
        print(f"[ERROR] Failed to create {model_name}: {e}")
        return None


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_model(
    model: pl.LightningModule,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Args:
        model: Trained PyTorch Lightning model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dictionary of metrics
    """
    model.eval()
    model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0).flatten()
    y_pred = np.concatenate(y_pred, axis=0).flatten()

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / y_true.std()
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))

    return {
        'rmse': float(rmse),
        'nrmse': float(nrmse),
        'r2': float(r2),
        'mae': float(mae),
        'n_samples': len(y_true)
    }


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_split_data(
    data_path: str,
    config: BenchmarkConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load data and create train/val/test splits.

    Args:
        data_path: Path to .npz data file
        config: Configuration object

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print(f"Loading data from {data_path}")
    data = np.load(data_path)

    X = data['X']  # (n_samples, 129, 200)
    y = data['y']  # (n_samples,)

    print(f"  Data shape: X={X.shape}, y={y.shape}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(config.VAL_SIZE + config.TEST_SIZE),
        random_state=config.SEED
    )

    val_ratio = config.VAL_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=config.SEED
    )

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Create datasets
    train_dataset = BenchmarkEEGDataset(X_train, y_train)
    val_dataset = BenchmarkEEGDataset(X_val, y_val)
    test_dataset = BenchmarkEEGDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ==============================================================================
# TRAINING
# ==============================================================================

def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: BenchmarkConfig
) -> Optional[pl.LightningModule]:
    """
    Train a single model.

    Args:
        model_name: Name of the model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object

    Returns:
        Trained model or None if training failed
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")

    # Create model
    base_model = create_model(model_name, config)
    if base_model is None:
        return None

    # Wrap model
    if model_name == 'OptimizedS4DEEG':
        model_config = {
            'd_model': 96,
            'n_layers': 8,
            'd_state': 384,
            'n_heads': 6,
            'bidirectional': True,
            'dropout': 0.20,
            'use_demographics': False,
            'demographic_dim': 5,
        }
        wrapped_model = CustomModelWrapper(
            model_config=model_config,
            learning_rate=4e-4,
            weight_decay=0.012
        )
    else:
        wrapped_model = BraindecodeRegressionWrapper(
            model=base_model,
            model_name=model_name,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR / model_name,
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=False
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        mode='min',
        verbose=False
    )

    # Create trainer
    trainer = Trainer(
        max_epochs=config.MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=config.GRADIENT_CLIP,
        deterministic=True,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False
    )

    # Train
    try:
        trainer.fit(wrapped_model, train_loader, val_loader)

        # Model checkpoint callback saves best model
        # but we'll use the trained model directly from memory for evaluation
        # (Loading from checkpoint can have issues with model initialization args)
        return wrapped_model

    except Exception as e:
        print(f"[ERROR] Training failed for {model_name}: {e}")
        return None


# ==============================================================================
# BENCHMARKING
# ==============================================================================

def run_benchmark(
    data_path: str,
    models_to_test: Optional[List[str]] = None,
    config: Optional[BenchmarkConfig] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run full benchmark comparison.

    Args:
        data_path: Path to data file
        models_to_test: List of model names to test (None = all)
        config: Configuration object (None = default)

    Returns:
        Dictionary mapping model names to their metrics
    """
    if config is None:
        config = BenchmarkConfig()

    # Default models to test
    if models_to_test is None:
        models_to_test = [
            'EEGNetv4',
            'ShallowFBCSPNet',
            'Deep4Net',
            'EEGConformer',
            'TIDNet',
            'ATCNet',
            'TCN',
            'EEGITNet',
            'TSception',
            'EEGNeX',
            'FBCNet',
            'OptimizedS4DEEG'
        ]

    print(f"\n{'='*70}")
    print("BENCHMARK: Baseline Model Comparison")
    print(f"{'='*70}")
    print(f"Data: {data_path}")
    print(f"Models to test: {', '.join(models_to_test)}")
    print(f"Device: {config.DEVICE}")
    print(f"{'='*70}\n")

    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = load_and_split_data(data_path, config)

    # Train and evaluate each model
    results = {}

    for model_name in models_to_test:
        # Train model
        trained_model = train_model(model_name, train_loader, val_loader, config)

        if trained_model is None:
            print(f"[WARNING] Skipping {model_name} (training failed)")
            continue

        # Evaluate on test set
        print(f"\nEvaluating {model_name} on test set...")
        metrics = evaluate_model(trained_model, test_loader, config.DEVICE)

        print(f"  RMSE:  {metrics['rmse']:.4f}")
        print(f"  NRMSE: {metrics['nrmse']:.4f}")
        print(f"  R²:    {metrics['r2']:.4f}")
        print(f"  MAE:   {metrics['mae']:.4f}")

        results[model_name] = metrics

    # Save results
    results_file = config.OUTPUT_DIR / f'benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Results saved to {results_file}")

    # Print summary table
    print_results_table(results)

    return results


def print_results_table(results: Dict[str, Dict[str, float]]):
    """Print formatted results table"""

    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}")

    # Check if we have any results
    if not results:
        print("[WARNING] No models completed training successfully")
        print(f"{'='*70}\n")
        return

    print(f"{'Model':<20} {'RMSE':<10} {'NRMSE':<10} {'R²':<10} {'MAE':<10}")
    print(f"{'-'*70}")

    # Sort by NRMSE (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['nrmse'])

    for model_name, metrics in sorted_results:
        print(f"{model_name:<20} {metrics['rmse']:<10.4f} {metrics['nrmse']:<10.4f} "
              f"{metrics['r2']:<10.4f} {metrics['mae']:<10.4f}")

    print(f"{'='*70}\n")

    # Highlight best model
    best_model = sorted_results[0][0]
    best_nrmse = sorted_results[0][1]['nrmse']
    print(f"[SUCCESS] Best model: {best_model} (NRMSE: {best_nrmse:.4f})")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Benchmark baseline models for EEG Challenge 2025',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='r5_full_data.npz',
        help='Path to preprocessed data file'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Models to benchmark (default: all). Options: EEGNetv4, ShallowFBCSPNet, '
             'Deep4Net, EEGConformer, TIDNet, ATCNet, TCN, EEGITNet, TSception, '
             'EEGNeX, FBCNet, OptimizedS4DEEG'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=60,
        help='Maximum number of training epochs (default: 60)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='Output directory for results (default: benchmark_results)'
    )

    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig()
    config.DATA_PATH = args.data_path
    config.MAX_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.OUTPUT_DIR = Path(args.output_dir)
    config.CHECKPOINT_DIR = config.OUTPUT_DIR / 'checkpoints'

    # Check dependencies
    if not BRAINDECODE_AVAILABLE:
        print("[ERROR] Braindecode not available. Install with:")
        print("        pip install braindecode")
        return

    if not CUSTOM_MODEL_AVAILABLE:
        print("[WARNING] OptimizedS4DEEG not available (submission.py not found)")
        print("          Will only benchmark braindecode models")

    # Run benchmark
    results = run_benchmark(
        data_path=args.data_path,
        models_to_test=args.models,
        config=config
    )

    print("\n[SUCCESS] Benchmarking complete!")


if __name__ == '__main__':
    main()
