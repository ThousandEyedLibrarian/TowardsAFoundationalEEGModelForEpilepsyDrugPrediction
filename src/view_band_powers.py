#!/usr/bin/env python3
"""
Script to properly view and interpret your EEG band powers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def view_band_powers(csv_file='./results/eeg_features.csv'):
    """
    Load and properly display band power results
    """
    print("="*60)
    print("EEG Band Power Analysis Results")
    print("="*60)
    
    # Load the features
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        print("Run the analysis first: python eeg_analysis.py your_file.set")
        return
    
    features = pd.read_csv(csv_file, index_col=0)
    
    # Extract band powers
    band_cols = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_powers = features[band_cols]
    
    print(f"\nData from: {csv_file}")
    print(f"Number of channels: {len(band_powers)}")
    
    # Display overall statistics with proper formatting
    print("\n" + "="*60)
    print("BAND POWER STATISTICS (All Channels)")
    print("="*60)
    
    print("\nAverage Power (V²/Hz):")
    print("-"*40)
    for band in band_cols:
        mean_val = band_powers[band].mean()
        std_val = band_powers[band].std()
        min_val = band_powers[band].min()
        max_val = band_powers[band].max()
        
        print(f"{band.upper():8s}:")
        print(f"  Mean ± SD:  {mean_val:.2e} ± {std_val:.2e}")
        print(f"  Range:      [{min_val:.2e}, {max_val:.2e}]")
    
    # Relative power (percentage)
    print("\n" + "="*60)
    print("RELATIVE BAND POWER (Percentage of Total)")
    print("="*60)
    
    total_power = band_powers.sum(axis=1)
    relative_powers = {}
    
    print("\nAverage Relative Power (%):")
    print("-"*40)
    for band in band_cols:
        relative = (band_powers[band] / total_power * 100)
        relative_powers[band] = relative
        
        mean_rel = relative.mean()
        std_rel = relative.std()
        
        print(f"{band.upper():8s}: {mean_rel:6.2f}% ± {std_rel:5.2f}%")
    
    # Find dominant frequency band for each channel
    print("\n" + "="*60)
    print("DOMINANT FREQUENCY BANDS BY CHANNEL")
    print("="*60)
    
    dominant_bands = band_powers.idxmax(axis=1)
    band_counts = dominant_bands.value_counts()
    
    print("\nNumber of channels dominated by each band:")
    print("-"*40)
    for band in band_cols:
        count = band_counts.get(band, 0)
        percentage = (count / len(dominant_bands)) * 100
        print(f"{band.upper():8s}: {count:3d} channels ({percentage:5.1f}%)")
    
    # Top channels by alpha power (commonly analyzed)
    print("\n" + "="*60)
    print("TOP 10 CHANNELS BY ALPHA POWER")
    print("="*60)
    
    alpha_sorted = band_powers['alpha'].sort_values(ascending=False)
    for i, (channel, power) in enumerate(alpha_sorted.head(10).items(), 1):
        print(f"{i:2d}. {channel:15s}: {power:.2e} V²/Hz")
    
    # Create visualization
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Average band powers (bar plot)
    ax1 = axes[0, 0]
    mean_powers = [band_powers[band].mean() for band in band_cols]
    std_powers = [band_powers[band].std() for band in band_cols]
    colors = ['purple', 'blue', 'green', 'orange', 'red']
    
    bars = ax1.bar(band_cols, mean_powers, yerr=std_powers, 
                    color=colors, alpha=0.7, capsize=5)
    ax1.set_ylabel('Power (V²/Hz)')
    ax1.set_title('Average Band Powers Across All Channels')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, mean_powers):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1e}', ha='center', va='bottom', fontsize=8)
    
    # 2. Relative power pie chart
    ax2 = axes[0, 1]
    mean_relative = [relative_powers[band].mean() for band in band_cols]
    ax2.pie(mean_relative, labels=[b.upper() for b in band_cols], 
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Average Relative Band Power Distribution')
    
    # 3. Channel-wise band power heatmap
    ax3 = axes[1, 0]
    # Normalize for better visualization
    band_powers_norm = np.log10(band_powers + 1e-15)  # Add small value to avoid log(0)
    im = ax3.imshow(band_powers_norm.T, aspect='auto', cmap='hot')
    ax3.set_yticks(range(len(band_cols)))
    ax3.set_yticklabels([b.upper() for b in band_cols])
    ax3.set_xlabel('Channel Index')
    ax3.set_title('Band Powers by Channel (log scale)')
    plt.colorbar(im, ax=ax3, label='log₁₀(Power)')
    
    # 4. Distribution of alpha power
    ax4 = axes[1, 1]
    ax4.hist(band_powers['alpha'], bins=30, color='green', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Alpha Power (V²/Hz)')
    ax4.set_ylabel('Number of Channels')
    ax4.set_title('Distribution of Alpha Power Across Channels')
    ax4.axvline(band_powers['alpha'].mean(), color='red', 
                linestyle='--', label=f'Mean: {band_powers["alpha"].mean():.2e}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('EEG Band Power Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = 'band_power_summary.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved as: {output_file}")
    
    # Clinical interpretation guide
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    
    print("""
Normal EEG Power Distribution (approximate):
- Delta (0.5-4 Hz):  High during deep sleep, low when awake
- Theta (4-8 Hz):    Drowsiness, meditation, memory
- Alpha (8-13 Hz):   Relaxed wakefulness, eyes closed
- Beta (13-30 Hz):   Active thinking, concentration
- Gamma (30-45 Hz):  Cognitive processing, attention

Your Data Summary:""")
    
    # Determine likely state based on power distribution
    alpha_rel = relative_powers['alpha'].mean()
    beta_rel = relative_powers['beta'].mean()
    delta_rel = relative_powers['delta'].mean()
    
    if alpha_rel > 30:
        print("  → High alpha suggests relaxed, eyes-closed state")
    elif beta_rel > 30:
        print("  → High beta suggests active, alert state")
    elif delta_rel > 50:
        print("  → High delta suggests sleep or artifact")
    else:
        print("  → Mixed frequency pattern - typical for active wakefulness")
    
    print("\nNote: These are typical values for clean EEG data.")
    print("Artifacts, filtering, and reference can affect these values.")
    
    return band_powers, relative_powers

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = './results/eeg_features.csv'
    
    view_band_powers(csv_file)
