#!/usr/bin/env python3
"""
Prepare final submission for EEG Challenge 2025
Supports both single model and ensemble submissions
"""

import os
import shutil
import zipfile
from pathlib import Path
import sys


def prepare_ensemble_submission(output_name="submission_ensemble_final.zip"):
    """Prepare ensemble submission package"""
    
    print("\n" + "="*60)
    print("Preparing ENSEMBLE Submission")
    print("="*60 + "\n")
    
    # Check for ensemble weights
    ensemble_dir = Path("ensemble_weights")
    if not ensemble_dir.exists():
        print("[ERROR] No ensemble_weights/ directory found!")
        print("   Run: bash train_ensemble_fixed.sh")
        return False
    
    # Find all ensemble weights
    ch1_weights = list(ensemble_dir.glob("weights_ch1_seed*.pt"))
    ch2_weights = list(ensemble_dir.glob("weights_ch2_seed*.pt"))
    
    if len(ch1_weights) == 0:
        print("[ERROR] No ensemble weights found!")
        print("   Expected files like: ensemble_weights/weights_ch1_seed42.pt")
        return False
    
    print(f"[OK] Found {len(ch1_weights)} ensemble models")
    
    # Check submission file exists
    if not Path("submission_ensemble.py").exists():
        print("[ERROR] submission_ensemble.py not found!")
        return False
    
    print("[OK] submission_ensemble.py found")
    
    # Create output directory
    output_dir = Path("submission_package")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    # Copy submission file (rename to submission.py for competition)
    shutil.copy("submission_ensemble.py", output_dir / "submission.py")
    
    # Copy all ensemble weights
    for weight_file in ch1_weights:
        shutil.copy(weight_file, output_dir / weight_file.name)
    
    for weight_file in ch2_weights:
        shutil.copy(weight_file, output_dir / weight_file.name)
    
    # Also copy the original submission.py (needed for model definition)
    if Path("submission.py").exists():
        shutil.copy("submission.py", output_dir / "submission_base.py")
    
    # Test the submission
    print("\n[OK] Testing submission locally...")
    os.chdir(output_dir)
    test_result = os.system("python3 submission.py > /dev/null 2>&1")
    os.chdir("..")
    
    if test_result != 0:
        print("[WARN] Submission test had errors")
    else:
        print("[OK] Submission test passed")
    
    # Create zip file
    print("\n[OK] Creating submission zip...")
    if Path(output_name).exists():
        os.remove(output_name)
    
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_dir.glob("*"):
            if file.is_file():
                zipf.write(file, file.name)
    
    # Verify zip structure
    print("\n[OK] Verifying zip structure...")
    with zipfile.ZipFile(output_name, 'r') as zipf:
        files = zipf.namelist()
        print("   Files in zip:")
        total_size = 0
        for f in files:
            info = zipf.getinfo(f)
            size_mb = info.file_size / (1024 * 1024)
            total_size += size_mb
            print(f"   - {f} ({size_mb:.2f} MB)")
    
    zip_size = Path(output_name).stat().st_size / (1024 * 1024)
    
    print("\n" + "="*60)
    print("Ensemble Submission Successfully Prepared!")
    print("="*60)
    print(f"\nFile: {output_name}")
    print(f"Size: {zip_size:.2f} MB")
    print(f"Models: {len(ch1_weights)} ensemble members")
    
    # Calculate estimated parameters
    params_per_model = 450_000  # Approximate for smaller model
    total_params = params_per_model * len(ch1_weights)
    
    print(f"\nArchitecture:")
    print(f"  - Model type: Optimised S4D")
    print(f"  - Parameters per model: ~{params_per_model:,}")
    print(f"  - Ensemble size: {len(ch1_weights)} models")
    print(f"  - Total parameters: ~{total_params:,}")
    print(f"  - Prediction: Median of {len(ch1_weights)} outputs")
    
    print(f"\nExpected benefits:")
    print(f"  - Reduced variance through averaging")
    print(f"  - More robust predictions")
    print(f"  - Better generalisation")
    
    return True


def prepare_single_submission(output_name="submission_single_final.zip"):
    """Prepare single model submission (fallback)"""
    
    print("\n" + "="*60)
    print("Preparing SINGLE Model Submission")
    print("="*60 + "\n")
    
    # Find best single model
    best_weights = None
    best_val_rmse = float('inf')
    
    ensemble_dir = Path("ensemble_weights")
    if ensemble_dir.exists():
        # Check ensemble results
        results_file = Path("ensemble_results.txt")
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    if "Seed" in line and "Val RMSE" in line:
                        try:
                            seed = int(line.split("Seed")[1].split(":")[0].strip())
                            rmse = float(line.split("=")[1].strip())
                            if rmse < best_val_rmse:
                                best_val_rmse = rmse
                                best_weights = f"ensemble_weights/weights_ch1_seed{seed}.pt"
                        except:
                            pass
    
    if best_weights and Path(best_weights).exists():
        print(f"[OK] Using best model: {best_weights}")
        print(f"  Validation RMSE: {best_val_rmse:.4f}")
        
        shutil.copy(best_weights, "weights_challenge_1.pt")
        shutil.copy(best_weights.replace("_ch1_", "_ch2_"), "weights_challenge_2.pt")
    elif Path("weights_challenge_1_improved.pt").exists():
        print("[OK] Using improved weights")
        shutil.copy("weights_challenge_1_improved.pt", "weights_challenge_1.pt")
        shutil.copy("weights_challenge_2_improved.pt", "weights_challenge_2.pt")
    else:
        print("[ERROR] No weights found!")
        return False
    
    # Create zip
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write("submission.py", "submission.py")
        zipf.write("weights_challenge_1.pt", "weights_challenge_1.pt")
        zipf.write("weights_challenge_2.pt", "weights_challenge_2.pt")
    
    print(f"\n[OK] Created: {output_name}")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare EEG Challenge submission")
    parser.add_argument('--ensemble', action='store_true',
                        help='Create ensemble submission')
    parser.add_argument('--single', action='store_true',
                        help='Create single model submission (use best from ensemble)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output zip filename')
    
    args = parser.parse_args()
    
    # Default to ensemble if trained
    if not args.ensemble and not args.single:
        if Path("ensemble_weights").exists():
            args.ensemble = True
            print("[INFO] Detected ensemble weights, creating ensemble submission")
        else:
            args.single = True
            print("[INFO] No ensemble found, creating single model submission")
    
    if args.ensemble:
        output = args.output or "submission_ensemble_final.zip"
        success = prepare_ensemble_submission(output)
    else:
        output = args.output or "submission_single_final.zip"
        success = prepare_single_submission(output)
    
    if success:
        print("\n" + "="*60)
        print("[SUCCESS] Ready to upload to competition platform!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("[FAILED] Submission preparation failed")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
