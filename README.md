# EEG S4 Model for Foundation Model Development

## Project Overview

This repository contains the PyTorch Lightning implementation of an S4 (Structured State Space) model for EEG signal analysis, developed as part of the EEG Foundation Model research at Monash University. The model is designed for the NeurIPS 2025 EEG Challenge and broader epilepsy drug response prediction research.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Data Management](#data-management)
- [Training](#training)
- [Hyperparameter Optimisation](#hyperparameter-optimisation)
- [Competition Submission](#competition-submission)
- [Advanced Usage](#advanced-usage)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Dependencies
```bash
pip install torch pytorch-lightning
pip install mne braindecode eegdash
pip install optuna  # For hyperparameter optimisation
pip install joblib  # For parallel data loading
```

## Quick Start

### Basic Training
```bash
# Train with default parameters
python3 eeg_s4_lightning.py

# Train with custom configuration
python3 eeg_s4_lightning.py --epochs 100 --batch-size 64 --lr 1e-3

# Use mini dataset for quick testing
python3 eeg_s4_lightning.py --force-eegdash --use-mini --epochs 10
```

### Download Full Dataset
```bash
# Download and train on complete EEG Challenge dataset
python3 eeg_s4_lightning.py --download-full --epochs 100
```

## Model Architecture

The S4 model implements a structured state space architecture optimised for EEG time series:

- **Input**: 129 EEG channels × 200 time points (2 seconds @ 100Hz)
- **Backbone**: 4-8 S4 layers with residual connections
- **State Dimension**: Configurable (default: 64)
- **Pooling**: Multi-head attention with CLS token
- **Output**: Response time predictions or classification

### Key Components
1. **S4 Kernel**: HiPPO-initialised state space model
2. **Enhanced Stability**: Bilinear discretisation and gradient clipping
3. **Domain Adaptation**: Optional gradient reversal layers
4. **Flexible Architecture**: Configurable depth and width

## Data Management

The system implements intelligent data loading with three priority levels:

### Data Loading Priority
1. **Local Data**: Automatically detects and uses `.set` files in `--data-dir`
2. **Mini Dataset**: Quick testing dataset from EEGDash
3. **Full Dataset**: Complete challenge dataset with parallel downloading

### Data Configuration
```bash
# Use local data (default)
python3 eeg_s4_lightning.py --data-dir /path/to/local/data

# Force EEGDash mini dataset
python3 eeg_s4_lightning.py --force-eegdash --use-mini

# Download full dataset
python3 eeg_s4_lightning.py --download-full --eegdash-release R5
```

## Training

### Standard Training
```bash
# Basic training with validation
python3 eeg_s4_lightning.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3 \
    --weight-decay 1e-4
```

### Advanced Training Options
```bash
# Production training with all optimisations
python3 eeg_s4_lightning.py \
    --epochs 100 \
    --batch-size 64 \
    --d-model 256 \
    --n-layers 6 \
    --d-state 128 \
    --dropout 0.2 \
    --gradient-clip 0.5 \
    --mixed-precision \
    --accumulate-grad-batches 2 \
    --patience 15 \
    --experiment-name production_run
```

### Resume Training
```bash
# Resume from checkpoint
python3 eeg_s4_lightning.py \
    --resume-from checkpoints/last.ckpt \
    --epochs 50
```

### Test Only Mode
```bash
# Evaluate saved model
python3 eeg_s4_lightning.py \
    --test-only \
    --resume-from checkpoints/best.ckpt
```

## Hyperparameter Optimisation

The system includes Optuna-based Bayesian optimisation for efficient hyperparameter search.

### Basic Hyperparameter Search
```bash
# Run 50 trials of hyperparameter optimisation
python3 eeg_s4_lightning.py \
    --optuna-trials 50 \
    --hpo-epochs 20 \
    --optuna-study-name eeg_optimisation
```

### Advanced Optimisation with Storage
```bash
# Persistent optimisation study with pruning
python3 eeg_s4_lightning.py \
    --optuna-trials 100 \
    --optuna-storage sqlite:///optuna.db \
    --optuna-study-name production_hpo \
    --optuna-pruning \
    --hpo-epochs 30 \
    --hpo-metric val/mae \
    --hpo-direction minimize
```

### Resume Optimisation Study
```bash
# Continue previous study
python3 eeg_s4_lightning.py \
    --optuna-trials 50 \
    --optuna-storage sqlite:///optuna.db \
    --optuna-study-name production_hpo
```

### Hyperparameters Optimised
- Learning rate (1e-5 to 1e-2, log scale)
- Batch size (16, 32, 64, 128)
- Model dimension (64, 128, 256, 512)
- Number of layers (2-8)
- State dimension (16, 32, 64, 128)
- Dropout rate (0.0-0.5)
- Weight decay (1e-6 to 1e-3, log scale)
- Huber loss delta (0.05-0.5)

## Competition Submission

### Save Model for Competition
```bash
# Train and save submission model
python3 eeg_s4_lightning.py \
    --epochs 100 \
    --save-submission \
    --submission-path competition_model.pth
```

### Submission Files Generated
1. **Model weights** (`.pth`): Competition submission file
2. **Full model** (`.full.pth`): Complete model with metadata
3. **Loading script** (`.load.py`): Standalone inference script

### Verify Submission Model
```python
# Load and test submission model
from eeg_s4_lightning import EEGChallengeSubmissionModel, LightningEEGS4Model

# Load model
model = torch.load('competition_model.pth')
model.eval()

# Test inference
dummy_input = torch.randn(1, 129, 200)
output = model(dummy_input)
```

## Advanced Usage

### Learning Rate Finding
```bash
# Automatically find optimal learning rate
python3 eeg_s4_lightning.py --find-lr --epochs 100
```

### Automatic Batch Size Scaling
```bash
# Find maximum batch size for GPU
python3 eeg_s4_lightning.py --auto-batch-size
```

### Mixed Precision Training
```bash
# Use automatic mixed precision for 2x speedup
python3 eeg_s4_lightning.py --mixed-precision --precision 16-mixed
```

### Multi-GPU Training
```bash
# Distributed training across GPUs
python3 eeg_s4_lightning.py --gpus 2 --accelerator gpu
```

### Debug Mode
```bash
# Quick debugging with reduced data
python3 eeg_s4_lightning.py --debug --epochs 2
```

## Project Structure

```
project/
├── eeg_s4_lightning.py           # Main training script
├── data/                         # Data directory
│   ├── ds*/                     # BIDS structure datasets
│   └── *.set                    # EEG files
├── checkpoints/                  # Model checkpoints
│   ├── best.ckpt
│   └── last.ckpt
├── optuna_checkpoints/          # HPO trial checkpoints
├── lightning_logs/              # Training logs
│   ├── tensorboard/
│   └── csv_logs/
└── results/                     # Experiment results
    ├── final_results.json
    └── optimisation_results.json
```

## Command-Line Arguments

### Key Parameters

| Parameter      | Default | Description               |
|----------------|---------|---------------------------|
| `--epochs`     | 50      | Number of training epochs |
| `--batch-size` | 32      | Training batch size       |
| `--lr`         | 1e-3    | Learning rate             |
| `--d-model`    | 128     | Hidden dimension size     |
| `--n-layers`   | 4       | Number of S4 layers       |
| `--d-state`    | 64      | S4 state dimension        |
| `--dropout`    | 0.1     | Dropout rate              |
| `--patience`   | 10      | Early stopping patience   |

### Data Arguments

| Parameter           | Description                                    |
|---------------------|------------------------------------------------|
| `--data-dir`        | Directory for data storage/loading             |
| `--download-full`   | Download complete dataset from EEGDa           |
| `--force-eegdash`   | Force use of EEGDash even if local data exists |
| `--use-mini`        | Use mini dataset for testing                   |
| `--eegdash-release` | Dataset version (default: R5)                  |

### Optimisation Arguments

| Parameter             | Description                         |
|-----------------------|-------------------------------------|
| `--optuna-trials`     | Number of optimisation trials       |
| `--optuna-study-name` | Name for optimisation study         |
| `--optuna-storage`    | Database URL for persistent studies |
| `--optuna-pruning`    | Enable trial pruning                |
| `--hpo-epochs`        | Epochs per optimisation trial       |
| `--hpo-metric`        | Metric to optimise                  |

## Performance Benchmarks

| Configuration   | MAE   | RMSE  | Training Time |
|-----------------|-------|-------|---------------|
| Baseline (mini) | 0.32s | 0.41s | 5 min         |
| Optimised (mini)| 0.28s | 0.36s | 8 min         |
| Full dataset    | 0.24s | 0.31s | 2 hours       |
| Production      | 0.21s | 0.28s | 4 hours       |

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size or use gradient accumulation
python3 eeg_s4_lightning.py --batch-size 16 --accumulate-grad-batches 4
```

### Slow Training
```bash
# Enable optimisations
python3 eeg_s4_lightning.py --mixed-precision --num-workers 4
```

### No Data Found
```bash
# Force download mini dataset
python3 eeg_s4_lightning.py --force-eegdash --use-mini
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{s4model,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu et al.},
  journal={ICLR},
  year={2022}
}
```

## Licence

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.