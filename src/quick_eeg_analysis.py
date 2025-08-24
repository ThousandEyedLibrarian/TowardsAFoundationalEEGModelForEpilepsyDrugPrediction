#!/usr/bin/env python3
"""
Quick EEG analysis with automatic handling of low sampling rates
"""

from eeg_analysis import AnalysisConfig, EEGAnalyzer
import sys
import os
import mne

def quick_analysis(filename):
    """
    Perform a quick analysis with automatic adjustment for sampling rate
    """
    
    # Get base name for output directory
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # First, quickly check the sampling rate to adjust parameters
    print(f"Checking data parameters...")
    file_ext = os.path.splitext(filename)[1].lower()
    
    # Load data briefly to check parameters
    if file_ext == '.set':
        raw_check = mne.io.read_raw_eeglab(filename, preload=False, verbose=False)
    elif file_ext == '.fif':
        raw_check = mne.io.read_raw_fif(filename, preload=False, verbose=False)
    elif file_ext == '.edf':
        raw_check = mne.io.read_raw_edf(filename, preload=False, verbose=False)
    else:
        raw_check = mne.io.read_raw(filename, preload=False, verbose=False)
    
    sfreq = raw_check.info['sfreq']
    nyquist = sfreq / 2
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Nyquist frequency: {nyquist} Hz")
    
    # Adjust parameters based on sampling rate
    if sfreq <= 128:
        print(f"  Detected low sampling rate - adjusting parameters...")
        # For low sampling rates, adjust filter settings
        h_freq = min(45.0, nyquist - 2)  # Leave 2 Hz margin
        notch_freq = None if nyquist <= 51 else 50.0  # Skip notch if too close to Nyquist
        resample = False  # Don't resample low rate data
    else:
        h_freq = 45.0
        notch_freq = 50.0  # or 60.0 for US
        resample = False
    
    # Create configuration with adjusted settings
    config = AnalysisConfig(
        input_file=filename,
        output_dir=f"./{base_name}_results",
        
        # Adjusted preprocessing
        apply_filter=True,
        l_freq=0.5,
        h_freq=h_freq,
        notch_freq=notch_freq,
        resample=resample,
        
        # Skip time-consuming steps for quick analysis
        apply_ica=False,
        compute_connectivity=False,
        
        # Simple visualizations
        plot_raw=False,
        plot_psd=True,
        plot_topomap=True,
        plot_connectivity=False,
        
        # Export results
        export_features=True,
        export_format='csv'
    )
    
    # Print adjusted configuration
    print(f"\nAdjusted configuration for {sfreq} Hz data:")
    print(f"  High-pass filter: {config.l_freq} Hz")
    print(f"  Low-pass filter: {config.h_freq} Hz")
    print(f"  Notch filter: {config.notch_freq if config.notch_freq else 'Skipped'}")
    
    # Create analyzer
    analyzer = EEGAnalyzer(config)
    
    # Run analysis steps
    print(f"\nAnalyzing {filename}...")
    print("-" * 40)
    
    try:
        # Load and preprocess
        analyzer.load_data()
        analyzer.preprocess_data()
        
        # Compute features
        print("\nComputing features...")
        analyzer.compute_spectral_features()
        analyzer.compute_temporal_features()
        
        # Create visualizations
        print("\nCreating visualizations...")
        analyzer.visualize_results()
        
        # Export results
        print("\nExporting results...")
        analyzer.export_features()
        
        print("\n" + "=" * 40)
        print("✓ Analysis complete!")
        print(f"Results saved to: {config.output_dir}")
        print("=" * 40)
        
        # Print quick summary
        if 'band_powers' in analyzer.features:
            print("\nQuick Summary - Average Band Powers:")
            band_powers = analyzer.features['band_powers']
            for band in band_powers.columns:
                avg_power = band_powers[band].mean()
                print(f"  {band:8s}: {avg_power:.2e} V²/Hz")
        
        # Print warnings if any
        if sfreq <= 128:
            print("\n⚠ Note: Low sampling rate detected")
            print("  - Some frequency features may be limited")
            print("  - Consider resampling source data to 250+ Hz if possible")
        
        return analyzer
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the file format is supported")
        print("2. Ensure the file is not corrupted")
        print("3. Try running with --no-filter flag")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_analysis.py <eeg_file>")
        print("Supported formats: .set, .fif, .edf, .bdf, .vhdr, .cnt")
        print("\nThis script automatically adjusts for low sampling rates.")
        sys.exit(1)
    
    quick_analysis(sys.argv[1])
