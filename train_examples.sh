#!/bin/bash
# Training examples for EEG S4 Lightning model

echo "=============================================="
echo "EEG S4 Lightning Training Examples"
echo "=============================================="

# 1. Basic training with local data
echo "Example 1: Standard training"
echo "python3 eeg_s4_lightning.py --epochs 50 --batch-size 32"

# 2. Quick test with mini dataset
echo -e "\nExample 2: Quick test with mini dataset"
echo "python3 eeg_s4_lightning.py --force-eegdash --use-mini --epochs 10"

# 3. Download and train on full dataset
echo -e "\nExample 3: Full dataset training"
echo "python3 eeg_s4_lightning.py --download-full --epochs 100 --batch-size 64"

# 4. Hyperparameter optimization with Optuna
echo -e "\nExample 4: Hyperparameter optimization (50 trials)"
cat << 'EOF'
python3 eeg_s4_lightning.py \
    --optuna-trials 50 \
    --hpo-epochs 20 \
    --optuna-study-name eeg_hpo \
    --optuna-pruning
EOF

# 5. Advanced hyperparameter optimization with persistent storage
echo -e "\nExample 5: Advanced HPO with database storage"
cat << 'EOF'
python3 eeg_s4_lightning.py \
    --optuna-trials 100 \
    --optuna-storage sqlite:///optuna_studies.db \
    --optuna-study-name production_hpo \
    --optuna-pruning \
    --hpo-epochs 30 \
    --hpo-metric val/rmse \
    --hpo-direction minimize
EOF

# 6. Resume hyperparameter optimization
echo -e "\nExample 6: Resume previous optimization study"
cat << 'EOF'
python3 eeg_s4_lightning.py \
    --optuna-trials 50 \
    --optuna-storage sqlite:///optuna_studies.db \
    --optuna-study-name production_hpo
EOF

# 7. Production training with optimized hyperparameters
echo -e "\nExample 7: Production training with best hyperparameters"
cat << 'EOF'
python3 eeg_s4_lightning.py \
    --lr 0.000523 \
    --batch-size 64 \
    --d-model 256 \
    --n-layers 6 \
    --d-state 64 \
    --dropout 0.2 \
    --weight-decay 0.000156 \
    --huber-delta 0.125 \
    --epochs 100 \
    --mixed-precision \
    --gradient-clip 0.5 \
    --patience 15 \
    --experiment-name optimized_model \
    --save-submission
EOF

# 8. Multi-GPU training
echo -e "\nExample 8: Distributed training on multiple GPUs"
cat << 'EOF'
python3 eeg_s4_lightning.py \
    --gpus 2 \
    --accelerator gpu \
    --batch-size 128 \
    --epochs 100 \
    --mixed-precision
EOF

# 9. Learning rate finder
echo -e "\nExample 9: Find optimal learning rate"
echo "python3 eeg_s4_lightning.py --find-lr --epochs 50"

# 10. Auto batch size scaling
echo -e "\nExample 10: Find maximum batch size for GPU"
echo "python3 eeg_s4_lightning.py --auto-batch-size --epochs 50"

# 11. Resume from checkpoint
echo -e "\nExample 11: Resume interrupted training"
echo "python3 eeg_s4_lightning.py --resume-from checkpoints/last.ckpt --epochs 50"

# 12. Test only mode
echo -e "\nExample 12: Evaluate trained model"
echo "python3 eeg_s4_lightning.py --test-only --resume-from checkpoints/best.ckpt"

# 13. Debug mode
echo -e "\nExample 13: Quick debugging"
echo "python3 eeg_s4_lightning.py --debug --epochs 2 --batch-size 8"

# 14. Competition submission
echo -e "\nExample 14: Train and save competition model"
cat << 'EOF'
python3 eeg_s4_lightning.py \
    --download-full \
    --epochs 100 \
    --save-submission \
    --submission-path eeg_challenge_final.pth \
    --experiment-name competition_submission
EOF

echo -e "\n=============================================="
echo "Notes:"
echo "1. Install Optuna for HPO: pip install optuna"
echo "2. Data priority: Local -> Mini -> Full dataset"
echo "3. Use --help for all available options"
echo "=============================================="