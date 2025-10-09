# EEG Challenge 2025 - S4D Model Submission

Optimised S4D (Structured State Space) model for EEG response time prediction in Challenge 1: Cross-Task Transfer Learning.

## Quick Start

### 1. Train the Model
```bash
python3 train_improved_final.py --data_dir local_eeg_data.npz --use_npz --epochs 100
```

### 2. Create Submission
```bash
python3 prepare_final_submission.py
```

### 3. Upload
Upload `submission_s4_final.zip` to the EEG Challenge 2025 platform.

## Repository Structure

### Core Files
- **`submission.py`** - Competition submission with S4D model
- **`prepare_final_submission.py`** - Package model and weights for submission
- **`train_improved_final.py`** - Training script with regularisation and augmentation

### Data Files
- **`local_eeg_data.npz`** - Local CCD data (1,876 samples, 2 subjects)
- **`weights_challenge_*.pt`** - Trained model weights

### Data Preparation
- **`download_r5_efficient.py`** - Download R5 dataset from EEGDash
- **`download_r5_simple.py`** - Alternative R5 downloader
- **`load_local_eeg_data.py`** - Load archived local EEG data

### Documentation
- **`README.md`** - This file
- **`CLAUDE.md`** - Project context for AI assistant
- **`TRAINING_STATUS.md`** - Training status and details

## Model Architecture

### S4D Implementation
- **Input**: 129 channels × 200 timepoints (2 seconds at 100Hz)
- **Architecture**:
  - Input projection: Linear(129, 128) + LayerNorm
  - 4× S4D blocks with FFT convolution
  - First block: Bidirectional (outputs 256 dims)
  - Blocks 2-4: Maintain 256 dims
  - Multi-head attention pooling (8 heads, CLS token)
  - Output head: Linear(256, 64) → ReLU → Dropout → Linear(64, 1)
- **Parameters**: 958,721
- **Output**: Response time prediction (batch, 1)

### Key Features
- S4D without HiPPO initialisation (diagonal state space)
- FFT-based efficient convolution
- Bidirectional processing
- Multi-head attention aggregation
- Competition format compliant

## Training Configuration

### Hyperparameters
- **Optimiser**: AdamW (lr=0.001, weight_decay=0.0001)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- **Batch size**: 32
- **Epochs**: 100 (with early stopping, patience=20)
- **Dropout**: 0.15
- **Label smoothing**: 0.05

### Data Augmentation
- Gaussian noise (50% prob, 0-1% level)
- Channel dropout (30% prob, 1-5 channels)
- Time shift (30% prob, ±5 samples)
- Amplitude scaling (40% prob, 0.8-1.2×)

### Test-Time Augmentation
- 5× predictions with small noise, averaged for final prediction

## Performance

### Local Validation
- **Dataset**: 1,876 samples from 2 subjects
- **Splits**: 1,314 train / 187 val / 375 test (random split)

### Competition Results
- **Previous submission**: 1.59 RMSE
- **Limitation**: Small dataset (only 2 subjects)

## Data Requirements

### Input Format
- **Shape**: (batch, 129, 200)
- **Channels**: 129 EEG channels
- **Timepoints**: 200 (2 seconds at 100Hz)
- **Preprocessing**: None (raw data)

### Label Format
- **Shape**: (batch, 1)
- **Values**: Response time in seconds
- **Range**: Typically 0.2-2.0 seconds

## Training with Different Data Sources

```bash
# Local archived data (default)
python3 train_improved_final.py --data_dir local_eeg_data.npz --use_npz

# Downloaded R5 data (directory format)
python3 train_improved_final.py --data_dir r5_data

# Downloaded R5 data (npz format)
python3 train_improved_final.py --data_dir r5_processed.npz --use_npz

# Synthetic data (testing only)
python3 train_improved_final.py --use_synthetic
```

## Downloading R5 Dataset (Optional)

```bash
# Batch processing (individual subject files)
python3 download_r5_efficient.py --mini --save_dir r5_data

# Challenge notebook approach (single .npz file)
python3 download_r5_simple.py --mini --output r5_processed.npz
```

Note: EEGDash downloads require stable internet and may take significant time.

## Submission Format

The submission package contains:
```
submission_s4_final.zip
├── submission.py          # Model implementation with Submission class
├── weights_challenge_1.pt # Trained weights for Challenge 1
└── weights_challenge_2.pt # Trained weights for Challenge 2
```

**Important**: Files must be at zip root level (no folder structure).

## Competition Compliance

### Required Interface
```python
class Submission:
    def __init__(self, SFREQ, DEVICE):
        pass

    def get_model_challenge_1(self):
        # Loads weights from /app/output/weights_challenge_1.pt
        pass

    def get_model_challenge_2(self):
        # Loads weights from /app/output/weights_challenge_2.pt
        pass
```

### Evaluation
- Platform instantiates: `Submission(SFREQ=100, DEVICE)`
- Calls: `get_model_challenge_1()` or `get_model_challenge_2()`
- Runs inference with `torch.inference_mode()`
- Evaluates RMSE on secret test sets

## Troubleshooting

### No weights found
```bash
python3 train_improved_final.py --data_dir local_eeg_data.npz --use_npz --epochs 100
```

### Out of memory
```bash
python3 train_improved_final.py --batch_size 16
```

### Wrong data format
- Input must be (batch, 129, 200)
- Output must be (batch, 1)
- Check submission.py model forward pass

## Archive

Old scripts and documentation in `archive/`:
- `archive/old_scripts/` - Previous training and submission variations
- `archive/old_docs/` - Previous documentation versions
- `archive/old_weights/` - Previous model weights
- `archive/windowed_epochs/` - Original preprocessed data

## References

- **Competition**: EEG Challenge 2025 (NeurIPS 2025)
- **Task**: Challenge 1 - Cross-Task Transfer Learning
- **Dataset**: HBN-EEG Contrast Change Detection (CCD)
- **Model**: S4D (Structured State Space with Diagonal parameterisation)

## Licence

This code is for the EEG Challenge 2025 competition. See competition rules for usage terms.

---

**Last Updated**: October 2025
