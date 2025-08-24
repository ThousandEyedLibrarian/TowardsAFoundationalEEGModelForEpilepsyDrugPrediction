#!/usr/bin/env python3
"""
Test script to verify EEG analysis installation and create sample data
"""

import numpy as np
import mne
import os
import sys

def test_mne_installation():
    """Test if MNE is properly installed"""
    print("Testing MNE installation...")
    print(f"MNE version: {mne.__version__}")
    
    # Check for mne-connectivity
    try:
        import mne_connectivity
        print(f"mne-connectivity version: {mne_connectivity.__version__}")
    except ImportError:
        print("⚠ mne-connectivity not installed (optional for connectivity analysis)")
        print("  Install with: pip install mne-connectivity")
    
    # Test PSD computation methods
    try:
        # Create sample data
        sfreq = 250  # Hz
        times = np.arange(0, 10, 1/sfreq)  # 10 seconds
        n_channels = 5
        
        # Generate random EEG-like data
        data = np.random.randn(n_channels, len(times)) * 1e-6  # in Volts
        
        # Create channel names and types
        ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels
        
        # Create info structure
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # Create Raw object
        raw = mne.io.RawArray(data, info)
        
        # Test PSD computation
        print("\nTesting PSD computation...")
        psd = raw.compute_psd(method='welch', fmin=0.5, fmax=45, verbose=False)
        psds, freqs = psd.get_data(return_freqs=True)
        print(f"✓ PSD computation successful. Shape: {psds.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def create_sample_eeg_file(filename="sample_eeg.fif"):
    """Create a sample EEG file for testing"""
    print(f"\nCreating sample EEG file: {filename}")
    
    # Parameters
    sfreq = 250  # Sampling frequency
    duration = 60  # seconds
    n_channels = 19  # Standard 10-20 system subset
    
    # Create time vector
    times = np.arange(0, duration, 1/sfreq)
    n_samples = len(times)
    
    # Channel names (10-20 system)
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
                'Fz', 'Cz', 'Pz']
    
    # Generate realistic EEG data with different frequency components
    data = np.zeros((n_channels, n_samples))
    
    for i in range(n_channels):
        # Add different frequency components
        # Delta (0.5-4 Hz)
        data[i] += 30e-6 * np.sin(2 * np.pi * 2 * times + np.random.rand())
        # Theta (4-8 Hz)
        data[i] += 20e-6 * np.sin(2 * np.pi * 6 * times + np.random.rand())
        # Alpha (8-13 Hz)
        data[i] += 50e-6 * np.sin(2 * np.pi * 10 * times + np.random.rand())
        # Beta (13-30 Hz)
        data[i] += 10e-6 * np.sin(2 * np.pi * 20 * times + np.random.rand())
        # Add some noise
        data[i] += 5e-6 * np.random.randn(n_samples)
    
    # Create MNE Info structure
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )
    
    # Set standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    
    # Create Raw object
    raw = mne.io.RawArray(data, info)
    
    # Add some annotations (like eye blinks)
    annotations = mne.Annotations(
        onset=[10, 25, 40, 55],  # in seconds
        duration=[0.5, 0.5, 0.5, 0.5],
        description=['blink', 'blink', 'blink', 'blink']
    )
    raw.set_annotations(annotations)
    
    # Save the file
    raw.save(filename, overwrite=True, verbose=False)
    print(f"✓ Sample EEG file created: {filename}")
    print(f"  - Channels: {n_channels}")
    print(f"  - Duration: {duration} seconds")
    print(f"  - Sampling rate: {sfreq} Hz")
    
    return filename

def test_analysis_pipeline(eeg_file=None):
    """Test the analysis pipeline with sample data"""
    print("\n" + "="*50)
    print("Testing EEG Analysis Pipeline")
    print("="*50)
    
    # Import the analysis module
    try:
        from eeg_analysis import AnalysisConfig, EEGAnalyzer
        print("✓ EEG analysis module imported successfully")
    except ImportError as e:
        print(f"✗ Could not import eeg_analysis module: {e}")
        print("Make sure eeg_analysis.py is in the current directory")
        return False
    
    # Create sample data if no file provided
    if eeg_file is None:
        eeg_file = create_sample_eeg_file()
    
    # Create configuration
    print("\nCreating analysis configuration...")
    config = AnalysisConfig(
        input_file=eeg_file,
        output_dir="./test_output",
        apply_filter=True,
        l_freq=0.5,
        h_freq=45.0,
        notch_freq=50.0,
        apply_ica=False,  # Skip ICA for quick test
        compute_connectivity=False,  # Skip connectivity for quick test
        plot_raw=False,  # Skip plots for quick test
        plot_psd=True,
        plot_topomap=True,
        plot_connectivity=False,
        export_features=True
    )
    
    print("✓ Configuration created")
    
    # Run analysis
    print("\nRunning analysis...")
    try:
        analyzer = EEGAnalyzer(config)
        
        # Test each step
        print("  - Loading data...")
        analyzer.load_data()
        
        print("  - Preprocessing...")
        analyzer.preprocess_data()
        
        print("  - Computing spectral features...")
        analyzer.compute_spectral_features()
        
        print("  - Computing temporal features...")
        analyzer.compute_temporal_features()
        
        print("  - Exporting features...")
        analyzer.export_features()
        
        print("\n✓ Analysis completed successfully!")
        print(f"  Results saved to: {config.output_dir}")
        
        # Print feature summary
        if analyzer.features:
            print("\nComputed features:")
            for feature_type, features in analyzer.features.items():
                if hasattr(features, 'shape'):
                    print(f"  - {feature_type}: shape {features.shape}")
                elif hasattr(features, '__len__'):
                    print(f"  - {feature_type}: {len(features)} features")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("EEG Analysis Pipeline Test Suite")
    print("=" * 50)
    
    # Test MNE installation
    if not test_mne_installation():
        print("\n⚠ MNE installation test failed!")
        print("Please install MNE with: pip install mne")
        return 1
    
    # Test analysis pipeline
    if not test_analysis_pipeline():
        print("\n⚠ Analysis pipeline test failed!")
        return 1
    
    print("\n" + "=" * 50)
    print("✓ All tests passed successfully!")
    print("\nYou can now use the analysis script with your own .set files:")
    print("  python eeg_analysis.py your_file.set")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
