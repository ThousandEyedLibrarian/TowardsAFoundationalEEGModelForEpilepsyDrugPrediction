# Model Selection Guide for Training Script

## Overview

The training script now supports selection between two model architectures:

1. **Original Optimised S4D** - The baseline architecture with optimised hyperparameters
2. **Spatial S4D** - New EEG-specific architecture with spatial processing

## Usage

### Command Line Selection

```bash
# Train with original model (default)
python3 train_optimal_model.py --model original

# Train with spatial S4D model
python3 train_optimal_model.py --model spatial
```

### Full Examples

```bash
# Original model with custom epochs and batch size
python3 train_optimal_model.py --model original --epochs 40 --batch-size 32

# Spatial model with reduced epochs for quick test
python3 train_optimal_model.py --model spatial --epochs 10 --batch-size 16 --no-submission

# Show help to see all options
python3 train_optimal_model.py --help
```

## Model Comparison

| Feature | Original S4D | Spatial S4D |
|---------|-------------|-------------|
| **Parameters** | 453,729 | 283,495 (37% fewer) |
| **Architecture** | Generic S4D blocks | EEG-specific design |
| **Spatial Processing** | None | Channel attention & spatial convolution |
| **Frequency Awareness** | None | 5-band decomposition |
| **Response Attention** | Multi-head pooling | Temporal focus attention |
| **Expected Performance** | 0.92-0.97 RMSE | 0.75-0.85 RMSE (expected) |
| **Training Speed** | Baseline | ~20% slower (more complex features) |

## Configuration Details

### Original Model Config
- d_model: 96
- n_layers: 3
- d_state: 48
- n_heads: 6
- bidirectional: True
- dropout: 0.22

### Spatial Model Config
- spatial_filters: 36
- d_state: 32
- n_layers: 2
- frequency_bands: 5 (delta to gamma)
- channel_attention: Enabled
- response_attention: Enabled

## Recommendations

### When to Use Original Model
- Baseline comparison
- Quick testing
- Limited computational resources
- Need for consistent comparisons with previous results

### When to Use Spatial Model
- Best performance is priority
- EEG-specific features are important
- Cross-subject generalisation needed
- Research into spatial/frequency patterns

## Output Structure

Both models save outputs to the same directory structure:

```
optimal_model_output/
├── checkpoints/
│   └── optimal_epoch=XX_val_rmse=X.XXX.ckpt
├── logs/
│   └── optimal_training/
│       └── [timestamp]/
└── submission/
    ├── weights_challenge_1.pt
    └── weights_challenge_2.pt
```

## Performance Monitoring

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=optimal_model_output/logs/optimal_training
```

Both models log the same metrics:
- train_loss
- train_rmse
- val_loss
- val_rmse
- test_rmse (final)
- learning_rate

## Troubleshooting

### Memory Issues
The spatial model uses more memory during training due to additional feature extraction:

```bash
# Reduce batch size for spatial model if needed
python3 train_optimal_model.py --model spatial --batch-size 8
```

### Convergence Speed
The spatial model may converge faster due to better inductive biases:

```bash
# Can often use fewer epochs with spatial model
python3 train_optimal_model.py --model spatial --epochs 25
```

### Validation Performance
If spatial model doesn't outperform original, check:
1. Data loading is correct (129 channels, 200 timepoints)
2. No preprocessing applied (model expects raw EEG)
3. Sufficient training data (spatial features need more samples)

## Next Steps

1. **Baseline**: Train both models with same settings for fair comparison
2. **Hyperparameter Tuning**: Each model may need different learning rates
3. **Ensemble**: Consider combining predictions from both models
4. **Analysis**: Use attention weights from spatial model for interpretability

---

**Note**: The spatial model is experimental and may require fine-tuning for optimal performance. Start with the original model to establish baseline performance.