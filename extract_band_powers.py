#!/usr/bin/env python3
"""
Minimal script to extract just band powers from EEG data
Perfect for when you only need frequency band analysis
Supports multiple formats: .set, .fif, .edf, .bdf, .vhdr, .cnt
"""

import mne
import numpy as np
import pandas as pd
import sys
import os

def extract_band_powers(filename, freq_bands=None):
    """
    Extract band powers from an EEG file
    
    Parameters:
    -----------
    filename : str
        Path to EEG file (.set, .fif, .edf, .bdf, .vhdr, .cnt)
    freq_bands : dict
        Dictionary of frequency bands (name: (low_freq, high_freq))
        Default: standard EEG bands
    
    Returns:
    --------
    pandas.DataFrame with band powers for each channel
    """
    
    # Default frequency bands
    if freq_bands is None:
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    
    print(f"Loading {filename}...")
    
    # Detect file type and load accordingly
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext == '.set':
        raw = mne.io.read_raw_eeglab(filename, preload=True, verbose=False)
    elif file_ext == '.fif':
        raw = mne.io.read_raw_fif(filename, preload=True, verbose=False)
    elif file_ext == '.edf':
        raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
    elif file_ext == '.bdf':
        raw = mne.io.read_raw_bdf(filename, preload=True, verbose=False)
    elif file_ext == '.vhdr':
        raw = mne.io.read_raw_brainvision(filename, preload=True, verbose=False)
    elif file_ext == '.cnt':
        raw = mne.io.read_raw_cnt(filename, preload=True, verbose=False)
    else:
        # Try auto-detection
        raw = mne.io.read_raw(filename, preload=True, verbose=False)
    
    # Basic preprocessing
    print("Preprocessing...")
    raw.filter(0.5, 45, fir_design='firwin', verbose=False)
    raw.notch_filter(50, verbose=False)  # 50 Hz for Europe/Asia, use 60 for US
    
    # Set average reference
    raw.set_eeg_reference('average', projection=True, verbose=False)
    raw.apply_proj()
    
    # Compute PSD
    print("Computing power spectral density...")
    psd = raw.compute_psd(
        method='welch',
        fmin=0.5,
        fmax=45,
        n_fft=2048,
        n_overlap=1024,
        verbose=False
    )
    
    # Get PSD data
    psds, freqs = psd.get_data(return_freqs=True)
    
    # Calculate band powers
    print("Calculating band powers...")
    band_powers = {}
    
    for band_name, (fmin, fmax) in freq_bands.items():
        # Find frequency indices
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        # Calculate mean power in band
        band_powers[band_name] = psds[:, freq_mask].mean(axis=1)
    
    # Create DataFrame
    df = pd.DataFrame(band_powers, index=raw.ch_names)
    
    # Add relative powers
    total_power = psds.mean(axis=1)
    for band_name in freq_bands.keys():
        df[f'{band_name}_relative'] = df[band_name] / total_power
    
    return df, raw.info

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_band_powers.py <eeg_file> [output.csv]")
        print("Supported formats: .set, .fif, .edf, .bdf, .vhdr, .cnt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "band_powers.csv"
    
    # Extract band powers
    band_powers, info = extract_band_powers(input_file)
    
    # Save to CSV
    band_powers.to_csv(output_file)
    print(f"\nBand powers saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("BAND POWER SUMMARY (averaged across channels)")
    print("=" * 50)
    
    # Absolute powers
    print("\nAbsolute Power (V²/Hz):")
    for col in band_powers.columns:
        if not col.endswith('_relative'):
            mean_power = band_powers[col].mean()
            std_power = band_powers[col].std()
            print(f"  {col:8s}: {mean_power:.2e} ± {std_power:.2e}")
    
    # Relative powers
    print("\nRelative Power (%):")
    for col in band_powers.columns:
        if col.endswith('_relative'):
            band = col.replace('_relative', '')
            mean_power = band_powers[col].mean() * 100
            std_power = band_powers[col].std() * 100
            print(f"  {band:8s}: {mean_power:.1f}% ± {std_power:.1f}%")
    
    print("\n" + "=" * 50)
    print(f"Data: {len(info['ch_names'])} channels, {info['sfreq']:.1f} Hz sampling rate")
    
    return band_powers

if __name__ == "__main__":
    main()
