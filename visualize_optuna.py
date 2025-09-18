#!/usr/bin/env python3
"""Visualize Optuna hyperparameter optimization results"""

import argparse
import json
from pathlib import Path

try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
    import plotly.io as pio
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Install with: pip install optuna plotly")

def main():
    parser = argparse.ArgumentParser(description="Visualize Optuna optimization results")
    parser.add_argument("--study-name", type=str, required=True,
                        help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, required=True,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument("--output-dir", type=str, default="optuna_plots",
                        help="Directory to save plots")
    parser.add_argument("--format", type=str, default="html",
                        choices=["html", "png", "pdf"],
                        help="Output format for plots")
    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is required. Install with: pip install optuna plotly")
        return

    # Load study
    print(f"Loading study '{args.study_name}' from {args.storage}")
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print study summary
    print(f"\nStudy Summary:")
    print(f"  Number of trials: {len(study.trials)}")
    print(f"  Best value: {study.best_value:.6f}")
    print(f"  Best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Generate plots
    print(f"\nGenerating plots...")

    # 1. Optimization history
    fig = plot_optimization_history(study)
    fig.update_layout(title="Optimization History", height=500)
    file_path = output_dir / f"optimization_history.{args.format}"
    if args.format == "html":
        fig.write_html(str(file_path))
    else:
        pio.write_image(fig, str(file_path))
    print(f"  Saved: {file_path}")

    # 2. Parameter importances
    try:
        fig = plot_param_importances(study)
        fig.update_layout(title="Hyperparameter Importances", height=500)
        file_path = output_dir / f"param_importances.{args.format}"
        if args.format == "html":
            fig.write_html(str(file_path))
        else:
            pio.write_image(fig, str(file_path))
        print(f"  Saved: {file_path}")
    except Exception as e:
        print(f"  Could not generate parameter importances: {e}")

    # 3. Parallel coordinate plot
    fig = plot_parallel_coordinate(study)
    fig.update_layout(title="Parallel Coordinate Plot", height=600)
    file_path = output_dir / f"parallel_coordinate.{args.format}"
    if args.format == "html":
        fig.write_html(str(file_path))
    else:
        pio.write_image(fig, str(file_path))
    print(f"  Saved: {file_path}")

    # 4. Save best trial details
    best_trial = study.best_trial
    best_trial_info = {
        "trial_number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
        "datetime_start": str(best_trial.datetime_start),
        "datetime_complete": str(best_trial.datetime_complete),
    }

    json_path = output_dir / "best_trial.json"
    with open(json_path, "w") as f:
        json.dump(best_trial_info, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\nAll plots saved to: {output_dir}/")

if __name__ == "__main__":
    main()