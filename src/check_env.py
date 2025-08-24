#!/usr/bin/env python3
"""
Environment checker for EEG Analysis Pipeline
This script verifies that all dependencies are properly installed
"""

import sys
import importlib
from packaging import version

def check_package(package_name, min_version=None, optional=False):
    """Check if a package is installed and meets version requirements"""
    try:
        module = importlib.import_module(package_name)
        installed_version = getattr(module, '__version__', 'unknown')
        
        status = "✅"
        if min_version and installed_version != 'unknown':
            if version.parse(installed_version) < version.parse(min_version):
                status = "⚠️ "
                print(f"{status} {package_name:20s} {installed_version:10s} (minimum: {min_version})")
            else:
                print(f"{status} {package_name:20s} {installed_version:10s}")
        else:
            print(f"{status} {package_name:20s} {installed_version:10s}")
        return True
    except ImportError:
        if optional:
            print(f"⚠️  {package_name:20s} not installed (optional)")
        else:
            print(f"❌ {package_name:20s} not installed (REQUIRED)")
        return False

def check_mne_api():
    """Check if MNE API is up to date"""
    print("\n" + "="*50)
    print("Checking MNE API compatibility...")
    print("="*50)
    
    try:
        import mne
        
        # Test new PSD API
        print("Testing PSD computation API...")
        import numpy as np
        
        # Create test data
        sfreq = 250
        data = np.random.randn(3, 1000) * 1e-6
        info = mne.create_info(ch_names=['EEG1', 'EEG2', 'EEG3'], 
                               sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        # Try new API
        try:
            psd = raw.compute_psd(method='welch', verbose=False)
            psds, freqs = psd.get_data(return_freqs=True)
            print("✅ New PSD API (compute_psd) working correctly")
        except AttributeError:
            print("❌ Old MNE version detected - please upgrade with: pip install --upgrade mne")
            return False
            
        # Test EEGLAB import
        print("Testing EEGLAB import capability...")
        try:
            from mne.io import read_raw_eeglab
            print("✅ EEGLAB import available")
        except ImportError:
            print("❌ EEGLAB import not available")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing MNE API: {e}")
        return False

def main():
    print("="*50)
    print("EEG Analysis Pipeline - Environment Check")
    print("="*50)
    print(f"Python version: {sys.version}")
    print("\n" + "="*50)
    print("Checking installed packages...")
    print("="*50)
    
    # Required packages
    print("\nREQUIRED packages:")
    required_ok = all([
        check_package('mne', '1.0.0'),
        check_package('numpy', '1.21.0'),
        check_package('scipy', '1.7.0'),
        check_package('pandas', '1.3.0'),
        check_package('matplotlib', '3.4.0'),
    ])
    
    # Optional packages
    print("\nOPTIONAL packages (for additional features):")
    check_package('mne_connectivity', '0.5.0', optional=True)
    check_package('seaborn', '0.11.0', optional=True)
    check_package('sklearn', '1.0.0', optional=True)
    check_package('h5py', '3.0.0', optional=True)
    check_package('pymatreader', optional=True)
    check_package('openpyxl', '3.0.0', optional=True)
    check_package('joblib', '1.0.0', optional=True)
    
    # Check MNE API
    api_ok = check_mne_api()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if required_ok and api_ok:
        print("✅ All required packages are installed and working!")
        print("✅ Your environment is ready for EEG analysis")
        
        # Check for optional features
        try:
            import mne_connectivity
            print("✅ Connectivity analysis available")
        except ImportError:
            print("⚠️  Connectivity analysis not available (install mne-connectivity)")
            
        try:
            import sklearn
            print("✅ ICA artifact removal available")
        except ImportError:
            print("⚠️  ICA artifact removal not available (install scikit-learn)")
            
    else:
        print("❌ Some required packages are missing or outdated")
        print("\nTo fix missing packages, run:")
        print("  pip install -r requirements.txt")
        print("\nOr install individually:")
        print("  pip install mne numpy scipy pandas matplotlib")
        
    print("\n" + "="*50)
    print("Next steps:")
    print("1. Run test script: python test_eeg_analysis.py")
    print("2. Try sample analysis: python quick_analysis.py your_file.set")
    print("="*50)

if __name__ == "__main__":
    main()
