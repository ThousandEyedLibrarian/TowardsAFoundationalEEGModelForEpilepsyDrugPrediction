# EEG Analysis Pipeline for Multiple EEG Formats

A robust, configurable Python script for comprehensive EEG analysis using MNE-Python.

## Supported File Formats

✅ **EEGLAB** (`.set` files with optional `.fdt`)  
✅ **MNE** (`.fif` files)  
✅ **European Data Format** (`.edf`)  
✅ **BioSemi** (`.bdf`)  
✅ **BrainVision** (`.vhdr`)  
✅ **Neuroscan** (`.cnt`)  

The script automatically detects the file format based on the extension.

## Compatibility

✅ **Tested with MNE 1.10.1** (Latest as of 2025)  
✅ **Supports mne-connectivity 0.5+**  
✅ **Python 3.7+**

## Quick Start

### 1. Installation

```bash
# Install required packages (including connectivity analysis)
pip install -r requirements.txt

# Or install core dependencies only
pip install mne numpy scipy pandas matplotlib seaborn

# For connectivity analysis (optional but recommended)
pip install mne-connectivity

# Test your installation
python test_eeg_analysis.py
```

### 2. Basic Usage

```bash
# Analyze a .set file with default settings
python eeg_analysis.py your_file.set

# With custom output directory
python eeg_analysis.py your_file.set --output-dir ./my_results

# With ICA artifact removal
python eeg_analysis.py your_file.set --ica

# Using a configuration file
python eeg_analysis.py your_file.set --config config.json
```

## Compatibility Note

This script has been updated to work with **MNE 1.0+**. The main changes from older versions:
- PSD computation now uses `raw.compute_psd()` instead of deprecated `psd_welch()`
- Better error handling for optional features
- Automatic API compatibility detection

## Features

### Data Processing
- ✅ Automatic loading of EEGLAB .set files (with .fdt support)
- ✅ Configurable preprocessing pipeline
- ✅ Band-pass, high-pass, low-pass filtering
- ✅ Notch filter for line noise removal
- ✅ Optional ICA for artifact removal
- ✅ Automatic epoching for connectivity analysis

### Feature Extraction

#### Spectral Features
- Power Spectral Density (Welch or multitaper methods)
- Band power (delta, theta, alpha, beta, gamma)
- Relative band power
- Spectral entropy

#### Temporal Features
- Statistical measures (mean, std, variance, skewness, kurtosis)
- RMS amplitude
- Peak-to-peak amplitude
- Zero-crossing rate
- Hjorth parameters (mobility and complexity)

#### Connectivity Features
- Channel connectivity matrices
- Connectivity strength metrics

### Visualization
- Raw data plots
- PSD plots and band power heatmaps
- Topographic maps (when channel locations available)
- Connectivity matrices

### Export Options
- CSV or Excel export of all features
- Comprehensive analysis report
- All plots saved as PNG files

## Configuration

### Using JSON Configuration Files

Create a `config.json` file:

```json
{
  "input_file": "path/to/your/file.set",
  "output_dir": "./results",
  "apply_filter": true,
  "l_freq": 0.5,
  "h_freq": 45.0,
  "notch_freq": 50.0,
  "apply_ica": false,
  "freq_bands": {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "alpha": [8, 13],
    "beta": [13, 30],
    "gamma": [30, 45]
  }
}
```

### Python API Usage

```python
from eeg_analysis import AnalysisConfig, EEGAnalyzer

# Configure analysis
config = AnalysisConfig(
    input_file="your_file.set",
    output_dir="./results",
    apply_ica=True,
    compute_connectivity=True
)

# Run analysis
analyzer = EEGAnalyzer(config)
features, results = analyzer.run_full_analysis()

# Access specific features
band_powers = features['band_powers']
temporal_features = features['temporal']
```

## Low Sampling Rate Data

The script automatically handles low sampling rate data (e.g., 100 Hz, 128 Hz):

### Automatic Adjustments:
- **Nyquist frequency checking** - Prevents filter errors
- **Notch filter skipping** - Automatically skipped when too close to Nyquist
- **Filter adjustment** - High-pass filter adjusted to stay below Nyquist
- **FFT size optimization** - Automatically reduced for low sampling rates

### For 100 Hz Data:
```bash
# Use the automatic adjustment script
python quick_eeg_analysis_fixed.py your_100hz_file.set

# Or use the low sampling rate configuration
python eeg_analysis.py your_file.set --config config_low_srate.json
```

### Manual Configuration for Low Sampling Rates:
```python
config = AnalysisConfig(
    input_file="your_100hz_file.set",
    h_freq=40.0,  # Below Nyquist (50 Hz)
    notch_freq=None,  # Skip notch filter
    n_fft=512,  # Smaller FFT for low rates
)
```

### Common Issues

1. **ImportError for psd_welch**
   - This means you have an older MNE version
   - Solution: Update MNE with `pip install --upgrade mne`

2. **No module named 'mne'**
   - Solution: Install MNE with `pip install mne`

3. **Memory issues with large files**
   - Try setting `resample=True` with a lower `resample_freq` (e.g., 250 Hz)
   - Disable connectivity analysis if not needed
   - Process files in smaller segments

4. **Topomap plotting fails**
   - This happens when channel locations are not available
   - The script will skip topomaps and continue with other analyses

### Testing Your Setup

Run the test script to verify everything is working:

```bash
python test_eeg_analysis.py
```

This will:
- Check MNE installation
- Create a sample EEG file
- Run a basic analysis
- Report any issues

## Output Files

After analysis, you'll find in your output directory:

```
output_dir/
├── eeg_features.csv         # All computed features
├── analysis_report.txt      # Summary report
├── raw_data.png             # Raw EEG visualization
├── psd_analysis.png         # PSD plots
├── topomap_bands.png        # Topographic maps
└── connectivity_matrix.png  # Connectivity visualization
```

## Advanced Usage

### Batch Processing

```python
import glob
from eeg_analysis import AnalysisConfig, EEGAnalyzer

# Process multiple files
for file in glob.glob("*.set"):
    config = AnalysisConfig(
        input_file=file,
        output_dir=f"./results_{file.split('.')[0]}"
    )
    analyzer = EEGAnalyzer(config)
    analyzer.run_full_analysis()
```

### Custom Frequency Bands

```python
config = AnalysisConfig(
    input_file="your_file.set",
    freq_bands={
        'low_delta': (0.5, 2),
        'high_delta': (2, 4),
        'low_theta': (4, 6),
        'high_theta': (6, 8),
        'low_alpha': (8, 10),
        'high_alpha': (10, 13),
        'low_beta': (13, 20),
        'high_beta': (20, 30),
        'gamma': (30, 45)
    }
)
```

### Selective Feature Computation

```python
analyzer = EEGAnalyzer(config)
analyzer.load_data()
analyzer.preprocess_data()

# Compute only specific features
analyzer.compute_spectral_features()
# Skip temporal features
# Skip connectivity

analyzer.export_features()
```

## Requirements

- Python 3.7+
- MNE 1.0+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- scikit-learn (for ICA)

## Citation

If you use this script in your research, please cite:
- MNE-Python: [Gramfort et al., 2013](https://doi.org/10.3389/fnins.2013.00267)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Run the test script to diagnose problems
3. Ensure all dependencies are correctly installed
4. Check that your .set file is valid and not corrupted

## License

This script is provided as-is for research and educational purposes.
