## Quick Start Checklist

### Setup
- [ ] Clone repository
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Data Preparation
- [ ] Download R5 challenge data: `python process_r5_bdf.py --download`
- [ ] Process BDF to NPZ format: `python process_r5_bdf.py --process`
- [ ] Verify data files exist: `r5_l100_full.npz` and `r5_l100_full.pkl`

### Model Training
- [ ] Train optimal model: `python train_optimal_model.py --data-path ./r5_l100_full.npz --epochs 100`
- [ ] Monitor training progress in `optimal_model_output/logs/`
- [ ] Check saved checkpoints in `optimal_model_output/checkpoints/`

### Model Submission
- [ ] Prepare submission: `python train_optimal_model.py --create_submission`
- [ ] Locate submission zip in `optimal_model_output/submission_*.zip`
- [ ] Upload to challenge platform

## File Structure
```
.
├── train_optimal_model.py    # Main training script
├── submission.py             # Model implementation for submission
├── spatial_s4d_improved.py   # Enhanced S4D model variant
├── process_r5_bdf.py        # Data download and processing
├── r5_l100_full.npz         # Processed training data
├── r5_l100_test.npz         # Test data subset
└── optimal_model_output/     # Training outputs and submissions
```

## Quick Commands

Train with default settings:
```bash
python train_optimal_model.py
```

Train with custom epochs:
```bash
python train_optimal_model.py --epochs 50 --batch_size 64
```

Create submission package:
```bash
python train_optimal_model.py --create_submission --checkpoint optimal_model_output/checkpoints/best_model.pt
```

Process new data:
```bash
python process_r5_bdf.py --input_dir /path/to/bdf/files --output_path custom_data.npz
```
