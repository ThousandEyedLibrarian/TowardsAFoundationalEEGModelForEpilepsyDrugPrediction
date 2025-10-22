# Benchmark Usage Examples

This file contains practical examples for using `benchmark_baseline_models.py`.

## Quick Start

### 1. Quick Test (Fastest - ~10 min)

Test with just two models and minimal epochs:

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 OptimizedS4DEEG \
    --epochs 10 \
    --output-dir quick_test
```

Or use the provided script:
```bash
./quick_benchmark_test.sh
```

### 2. Compare Against Best Baseline (~30 min)

Compare your model against the best-performing baseline (EEGConformer):

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGConformer OptimizedS4DEEG \
    --epochs 40 \
    --output-dir s4d_vs_conformer
```

### 3. Full Benchmark (~2-3 hours)

Run all models with full training:

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --epochs 60
```

## Model Comparisons

### Lightweight Models Only

Compare computational-efficient models:

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 ShallowFBCSPNet \
    --epochs 60 \
    --output-dir lightweight_comparison
```

**Parameter Counts:**
- EEGNetv4: ~3K params
- ShallowFBCSPNet: ~208K params

### CNN-Based Models

Compare convolutional architectures:

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 ShallowFBCSPNet Deep4Net \
    --epochs 60 \
    --output-dir cnn_comparison
```

### Modern Architectures

Compare state-of-the-art models:

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGConformer TIDNet OptimizedS4DEEG \
    --epochs 60 \
    --output-dir modern_comparison
```

## Training Variations

### Fast Training (Development)

Quick iteration for debugging:

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models OptimizedS4DEEG \
    --epochs 5 \
    --batch-size 64 \
    --output-dir dev_test
```

### Extended Training

Longer training for better convergence:

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models OptimizedS4DEEG EEGConformer \
    --epochs 100 \
    --output-dir extended_training
```

### Large Batch Training

Faster training with larger batches (requires more GPU memory):

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 ShallowFBCSPNet \
    --batch-size 128 \
    --epochs 60 \
    --output-dir large_batch
```

### Small Batch Training

Better generalization with smaller batches:

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models OptimizedS4DEEG \
    --batch-size 16 \
    --epochs 60 \
    --output-dir small_batch
```

## Learning Rate Experiments

### Conservative Learning Rate

More stable training:

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models OptimizedS4DEEG \
    --learning-rate 1e-4 \
    --epochs 60 \
    --output-dir lr_conservative
```

### Aggressive Learning Rate

Faster convergence (may be less stable):

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 \
    --learning-rate 5e-3 \
    --epochs 60 \
    --output-dir lr_aggressive
```

## Data Variations

### Different Datasets

Test on Challenge 1 vs Challenge 2 data:

```bash
# Challenge 1 data
python3 benchmark_baseline_models.py \
    --data-path challenge1_data.npz \
    --models OptimizedS4DEEG EEGConformer \
    --output-dir challenge1_benchmark

# Challenge 2 data
python3 benchmark_baseline_models.py \
    --data-path challenge2_data.npz \
    --models OptimizedS4DEEG EEGConformer \
    --output-dir challenge2_benchmark
```

## Systematic Comparisons

### Ablation Study: Model Size

Compare how model size affects performance:

```bash
# Small model
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 \
    --output-dir size_small

# Medium model
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models ShallowFBCSPNet Deep4Net \
    --output-dir size_medium

# Large model
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models OptimizedS4DEEG EEGConformer \
    --output-dir size_large
```

Then compare results:
```bash
echo "Small Models:"
cat size_small/benchmark_results_*.json

echo "Medium Models:"
cat size_medium/benchmark_results_*.json

echo "Large Models:"
cat size_large/benchmark_results_*.json
```

### Cross-Validation Setup

Run multiple seeds for robust comparison:

```bash
# Modify script to change seed, or run multiple times
for seed in 42 123 456 789 999; do
    # Note: This requires modifying BenchmarkConfig.SEED in the script
    python3 benchmark_baseline_models.py \
        --data-path r5_full_data.npz \
        --models OptimizedS4DEEG EEGConformer \
        --output-dir cv_seed_${seed}
done

# Average results
python3 -c "
import json
import numpy as np
from pathlib import Path

results = {}
for seed in [42, 123, 456, 789, 999]:
    with open(f'cv_seed_{seed}/benchmark_results_*.json') as f:
        data = json.load(f)
        for model, metrics in data.items():
            if model not in results:
                results[model] = []
            results[model].append(metrics['nrmse'])

for model, nrmses in results.items():
    print(f'{model}: {np.mean(nrmses):.4f} ± {np.std(nrmses):.4f}')
"
```

## Batch Processing

### Compare All Models Individually

Run each model separately for isolated results:

```bash
models=("EEGNetv4" "ShallowFBCSPNet" "Deep4Net" "EEGConformer" "TIDNet" "OptimizedS4DEEG")

for model in "${models[@]}"; do
    echo "Benchmarking $model..."
    python3 benchmark_baseline_models.py \
        --data-path r5_full_data.npz \
        --models $model \
        --epochs 60 \
        --output-dir individual_${model}
done

# Aggregate results
echo "Aggregated Results:"
for model in "${models[@]}"; do
    echo "=== $model ==="
    cat individual_${model}/benchmark_results_*.json
    echo ""
done
```

### Parameter Sweep

Test multiple hyperparameter combinations:

```bash
# Learning rate sweep
for lr in 1e-4 5e-4 1e-3 5e-3; do
    python3 benchmark_baseline_models.py \
        --data-path r5_full_data.npz \
        --models OptimizedS4DEEG \
        --learning-rate $lr \
        --epochs 60 \
        --output-dir lr_${lr}
done

# Batch size sweep
for bs in 16 32 64 128; do
    python3 benchmark_baseline_models.py \
        --data-path r5_full_data.npz \
        --models OptimizedS4DEEG \
        --batch-size $bs \
        --epochs 60 \
        --output-dir bs_${bs}
done
```

## Production Runs

### Paper/Thesis Benchmark

Comprehensive benchmark for publication:

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 ShallowFBCSPNet Deep4Net EEGConformer TIDNet OptimizedS4DEEG \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --output-dir paper_results
```

**Recommended**: Run with multiple seeds for error bars.

### Competition Submission Validation

Validate your model before competition submission:

```bash
# First, train your optimal model
python3 train_optimal_model.py \
    --data-path r5_full_data.npz \
    --epochs 60 \
    --create-submission

# Then benchmark against baselines
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGConformer OptimizedS4DEEG \
    --epochs 60 \
    --output-dir competition_validation

# Check if OptimizedS4DEEG outperforms baseline
echo "If NRMSE(OptimizedS4DEEG) < NRMSE(EEGConformer), you're competitive!"
```

## Output Analysis

### View Results

```bash
# Latest results
cat benchmark_results/benchmark_results_*.json | tail -1 | python3 -m json.tool

# Specific model
cat benchmark_results/benchmark_results_*.json | python3 -c "
import json
import sys
data = json.load(sys.stdin)
print(f\"OptimizedS4DEEG NRMSE: {data['OptimizedS4DEEG']['nrmse']:.4f}\")
"

# Compare top 2 models
cat benchmark_results/benchmark_results_*.json | python3 -c "
import json
import sys
data = json.load(sys.stdin)
sorted_models = sorted(data.items(), key=lambda x: x[1]['nrmse'])
print(f\"1st: {sorted_models[0][0]} - {sorted_models[0][1]['nrmse']:.4f}\")
print(f\"2nd: {sorted_models[1][0]} - {sorted_models[1][1]['nrmse']:.4f}\")
"
```

### Load Best Checkpoint

```python
import torch
from benchmark_baseline_models import CustomModelWrapper

# Load best OptimizedS4DEEG checkpoint
checkpoint_path = "benchmark_results/checkpoints/OptimizedS4DEEG/best-epoch=38-val_loss=1.1234.ckpt"
model = CustomModelWrapper.load_from_checkpoint(checkpoint_path)

# Use for inference
model.eval()
with torch.no_grad():
    predictions = model(X_test)
```

## Performance Tips

### GPU Optimization

```bash
# Use all available GPU memory
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --batch-size 64

# Limit to specific GPU
CUDA_VISIBLE_DEVICES=0 \
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz
```

### CPU-Only Mode

```bash
# Force CPU (useful for debugging)
CUDA_VISIBLE_DEVICES="" \
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 \
    --epochs 5
```

### Parallel Runs

Run different models on different GPUs:

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 ShallowFBCSPNet \
    --output-dir gpu0_results &

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGConformer OptimizedS4DEEG \
    --output-dir gpu1_results &

wait
```

## Troubleshooting Examples

### Debug Mode

Test script functionality:

```bash
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 \
    --epochs 1 \
    --batch-size 8 \
    --output-dir debug
```

### Memory-Constrained Environment

```bash
# Reduce batch size and test one model at a time
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 \
    --batch-size 8 \
    --epochs 60 \
    --output-dir low_memory
```

### Verify Data Loading

```bash
python3 -c "
from benchmark_baseline_models import load_and_split_data, BenchmarkConfig

config = BenchmarkConfig()
config.DATA_PATH = 'r5_full_data.npz'

train_loader, val_loader, test_loader = load_and_split_data(
    config.DATA_PATH, config
)

print(f'Train batches: {len(train_loader)}')
print(f'Val batches: {len(val_loader)}')
print(f'Test batches: {len(test_loader)}')

# Test one batch
X, y = next(iter(train_loader))
print(f'Batch X shape: {X.shape}')
print(f'Batch y shape: {y.shape}')
"
```

## Integration Examples

### Use with Main Training Pipeline

```bash
# 1. Train optimal model
python3 train_optimal_model.py \
    --data-path r5_full_data.npz \
    --epochs 60 \
    --create-submission

# 2. Benchmark comparison
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGConformer OptimizedS4DEEG \
    --epochs 60

# 3. Run local scoring
python3 train_optimal_model.py \
    --data-path r5_full_data.npz \
    --epochs 60 \
    --create-submission \
    --run-local-scoring
```

### Generate Comparison Report

```bash
# Run benchmark
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --output-dir report_benchmark

# Generate markdown report
python3 -c "
import json
from pathlib import Path

result_file = list(Path('report_benchmark').glob('benchmark_results_*.json'))[0]
with open(result_file) as f:
    data = json.load(f)

print('# Benchmark Results\\n')
print('| Model | NRMSE | RMSE | R² | MAE |')
print('|-------|-------|------|----|----|')

sorted_data = sorted(data.items(), key=lambda x: x[1]['nrmse'])
for model, metrics in sorted_data:
    print(f\"| {model} | {metrics['nrmse']:.4f} | {metrics['rmse']:.4f} | \"
          f\"{metrics['r2']:.4f} | {metrics['mae']:.4f} |\")
" > BENCHMARK_REPORT.md

cat BENCHMARK_REPORT.md
```

## Recommended Workflow

For thesis/paper preparation:

```bash
# 1. Quick validation (are baselines working?)
./quick_benchmark_test.sh

# 2. Full baseline comparison (which baseline is best?)
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 ShallowFBCSPNet Deep4Net EEGConformer TIDNet \
    --output-dir baseline_comparison

# 3. Custom model vs best baseline (is our model better?)
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGConformer OptimizedS4DEEG \
    --epochs 100 \
    --output-dir custom_vs_best

# 4. Multiple seeds for error bars (is improvement consistent?)
# (Requires modifying script for different seeds)

# 5. Generate final results table
# (Use Python script above or manual analysis)
```

This gives you:
1. Validation that baselines work
2. Identification of best baseline
3. Direct comparison with your model
4. Statistical significance testing
5. Publication-ready results

## Summary

**Quick test**: `./quick_benchmark_test.sh`

**Full benchmark**: `python3 benchmark_baseline_models.py --data-path r5_full_data.npz`

**Custom comparison**: Select specific models with `--models` flag

**Adjust training**: Use `--epochs`, `--batch-size`, `--learning-rate` flags

**All results saved** to JSON files for further analysis!
