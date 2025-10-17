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

Australian English Documentation
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
from typing import Tuple, Optional, Dict, Any

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
from spatial_s4d_improved import SpatialS4DEEG


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """
    Central configuration for the training pipeline.

    Rationale: These hyperparameters were determined through extensive
    experimentation and represent the optimal balance between model capacity,
    regularisation, and training stability.
    """

    # Model Architectures
    # Original S4D Model Configuration (35% smaller than original to prevent overfitting)
    ORIGINAL_MODEL_CONFIG = {
        'd_model': 96,      # Hidden dimension (was 128)
        'n_layers': 3,      # Number of S4D layers (was 4) #FIXME: Go bigger 6, 8, 16, test each
        'd_state': 48,      # State dimension (was 64) #FIXME: Set to 4x d_model
        'n_heads': 6,       # Attention heads (was 8)
        'bidirectional': True,
        'dropout': 0.22     # Optimal dropout rate from experiments
    }

    # Spatial S4D Model Configuration (EEG-specific architecture)
    SPATIAL_MODEL_CONFIG = {
        'spatial_filters': 36,  # Spatial feature dimension
        'd_state': 32,          # State dimension for S4D blocks #FIXME: Set to 4x d_model
        'n_layers': 2,          # Fewer layers due to better features
    }

    # Default to original model
    MODEL_CONFIG = ORIGINAL_MODEL_CONFIG
    MODEL_TYPE = 'original'  # Will be updated based on command line

    # Training Parameters (balanced for stability and performance)
    TRAINING_CONFIG = {
        'learning_rate': 4e-4,
        'weight_decay': 0.012,
        'batch_size': 32, #
        'max_epochs': 40, #FIXME: Maybe too low
        'patience': 6,
        'gradient_clip': 0.5
    }
    
    #FIXME: eeg_pretraiing pretrain 2d branch base block.py for s4d base block, normalization should be at very end, 
    # i think i'm doing dropout before any layer normalization when i should be other way round (lines 150 and 149?)
    # FIXME: set bidirectional to true
    # FIXME: change num of predicted classes to 1 from 64 at submission.py
    #FIXME: Re-add huber loss after model fixes and try it again
    #FIXME: Add the fuseMOE stuff duong sent, set num experts to 16, k to 4, remove num modalities, add to line 174 of
    # submission.py after pooling and then pass to main_head after
    # outs, ent_loss = moe(self.pooling)
    #loss = mse ...
    #loss = loss + 0.2 * ent_loss
    #look at mse for val and entropy of training only
    

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

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        """
        Initialise the dataset.

        Args:
            X: EEG data of shape (n_samples, 129, 200)
            y: Response times of shape (n_samples, 1)
            augment: Whether to apply data augmentation
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment

        # Robust normalisation using median and standard deviation
        # This approach is less sensitive to outliers than mean normalisation
        self.median = torch.median(self.X, dim=0)[0].median(dim=-1)[0]
        self.std = self.X.std(dim=(0, 2)) + 1e-6

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        y = self.y[idx].clone()

        # Normalise channels
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

            # 5. Temporal masking (forces model to use partial information)
            # FIXME: Remove, good for autoencoders but no necessary for z-score normalization
            if torch.rand(1) < 0.3:
                mask_len = torch.randint(10, 25, (1,)).item()
                mask_start = torch.randint(0, 200 - mask_len, (1,)).item()
                x[:, mask_start:mask_start + mask_len] *= 0.1

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

        # Check if pre-split data is available
        if 'X_train' in data.keys():
            print("Using pre-split data from file...")
            self.train_dataset = OptimalEEGDataset(
                data['X_train'], data['y_train'], augment=True
            )
            self.val_dataset = OptimalEEGDataset(
                data['X_val'], data['y_val'], augment=False
            )
            self.test_dataset = OptimalEEGDataset(
                data['X_test'], data['y_test'], augment=False
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

        # Create datasets
        self.train_dataset = OptimalEEGDataset(X[train_idx], y[train_idx], augment=True)
        self.val_dataset = OptimalEEGDataset(X[val_idx], y[val_idx], augment=False)
        self.test_dataset = OptimalEEGDataset(X[test_idx], y[test_idx], augment=False)

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

        # Initialise model based on type
        if config.MODEL_TYPE == 'spatial':
            print("\nUsing Spatial S4D Model (EEG-specific architecture)")
            self.model = SpatialS4DEEG(
                n_chans=129,
                n_outputs=1,
                n_times=200,
                **config.MODEL_CONFIG
            )
        else:  # original
            print("\nUsing Original Optimised S4D Model")
            self.model = OptimizedS4DEEG(
                n_chans=129,
                n_outputs=1,
                n_times=200,
                **config.MODEL_CONFIG
            )

        # Track performance metrics
        self.val_rmse_history = []
        self.train_rmse_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss = F.mse_loss(pred, y)

        # L2 regularisation
        l2_reg = sum(p.pow(2).sum() for p in self.model.parameters())
        weight_decay = self.config.TRAINING_CONFIG['weight_decay'] #FIXME: Move to lower section if not required here re handling

        # Log metrics
        true_rmse = torch.sqrt(F.mse_loss(pred, y))
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_rmse', true_rmse, prog_bar=True)
        self.train_rmse_history.append(true_rmse.item())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        rmse = torch.sqrt(loss)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_rmse', rmse, prog_bar=True)
        self.val_rmse_history.append(rmse.item())

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Test-Time Augmentation (TTA)
        preds = []

        # Original prediction
        preds.append(self(x))

        # Light noise augmentations
        for noise_level in [0.003, 0.005]:
            x_aug = x + torch.randn_like(x) * noise_level
            preds.append(self(x_aug))

        # Light amplitude scaling
        for scale in [0.97, 1.03]:
            x_aug = x * scale
            preds.append(self(x_aug))

        # Average predictions
        pred = torch.stack(preds).mean(dim=0)
        loss = F.mse_loss(pred, y)

        self.log('test_loss', loss)
        self.log('test_rmse', torch.sqrt(loss))

        return loss

    def configure_optimizers(self):
        # AdamW optimiser
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.TRAINING_CONFIG['learning_rate'],
            weight_decay=0  # We handle weight decay manually FIXME: Double check handling elsewhere or needed here
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
        default=40,
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
        '--model',
        type=str,
        choices=['original', 'spatial'],
        default='original',
        help='Model architecture to use: original (Optimised S4D) or spatial (Spatial S4D)'
    )

    args = parser.parse_args()

    # Update configuration with command line arguments
    config = Config()
    config.DATA_PATH = args.data_path
    config.TRAINING_CONFIG['max_epochs'] = args.epochs
    config.TRAINING_CONFIG['batch_size'] = args.batch_size

    # Set model type and configuration
    config.MODEL_TYPE = args.model
    if args.model == 'spatial':
        config.MODEL_CONFIG = config.SPATIAL_MODEL_CONFIG
        print("\n" + "="*70)
        print("SELECTED MODEL: Spatial S4D (EEG-specific architecture)")
        print("="*70)
    else:
        config.MODEL_CONFIG = config.ORIGINAL_MODEL_CONFIG
        print("\n" + "="*70)
        print("SELECTED MODEL: Original Optimised S4D")
        print("="*70)

    # Initialise pipeline
    pipeline = TrainingPipeline(config)

    try:
        # Train model
        results, model_weights = pipeline.train()

        # Create submission if requested
        if not args.no_submission:
            submission_path = pipeline.create_submission(model_weights)
            print(f"\nTraining complete! Submission ready at: {submission_path}")
        else:
            print("\nTraining complete!")

        # Final summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Final Test RMSE: {results['test_rmse']:.5f}")
        print(f"Output Directory: {config.OUTPUT_DIR}")
        print(f"TensorBoard Logs: {config.LOGS_DIR}")
        print("="*70)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()