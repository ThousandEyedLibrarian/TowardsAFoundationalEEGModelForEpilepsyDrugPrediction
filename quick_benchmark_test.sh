#!/bin/bash
# Quick Benchmark Test Script
# Tests the benchmark_baseline_models.py with minimal configuration

echo "========================================================================"
echo "Quick Benchmark Test"
echo "========================================================================"
echo ""
echo "This script runs a minimal benchmark test with:"
echo "  - 2 models: EEGNetv4 (baseline) and OptimizedS4DEEG (custom)"
echo "  - 10 epochs (fast)"
echo "  - Default data path: r5_full_data.npz"
echo ""
echo "Estimated time: ~10-15 minutes on GPU, ~30-45 minutes on CPU"
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."

# Run benchmark with minimal config
python3 benchmark_baseline_models.py \
    --data-path r5_full_data.npz \
    --models EEGNetv4 OptimizedS4DEEG \
    --epochs 10 \
    --batch-size 32 \
    --output-dir benchmark_test_output

echo ""
echo "========================================================================"
echo "Test Complete!"
echo "========================================================================"
echo ""
echo "Results saved to: benchmark_test_output/"
echo ""
echo "To run full benchmark (all models, 60 epochs):"
echo "  python3 benchmark_baseline_models.py --data-path r5_full_data.npz"
echo ""
