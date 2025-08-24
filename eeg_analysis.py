#!/usr/bin/env python3
"""
Robust EEG Feature Analysis Script for Multiple EEG Formats
Author: EEG Analysis Pipeline
Description: Comprehensive analysis of EEG data with configurable parameters
Compatible with MNE 1.0+ and supports multiple file formats (.set, .fif, .edf, .bdf, .vhdr, .cnt)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import logging

# Check MNE version
try:
    import mne
    mne_version = mne.__version__
    print(f"Using MNE version: {mne_version}")
except ImportError:
    raise ImportError("MNE-Python is required. Install it with: pip install mne")

# MNE imports
from mne.preprocessing import ICA
# Try to import connectivity from mne-connectivity (separate package)
try:
    from mne_connectivity import spectral_connectivity_epochs
    CONNECTIVITY_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old location in case of older MNE versions
        from mne.connectivity import spectral_connectivity_epochs
        CONNECTIVITY_AVAILABLE = True
    except ImportError:
        CONNECTIVITY_AVAILABLE = False
        print("Warning: mne-connectivity not installed. Connectivity analysis will be skipped.")
        print("Install with: pip install mne-connectivity")
from scipy import signal, stats
from scipy.stats import kurtosis, skew

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration class for EEG analysis parameters"""
    
    # File paths
    input_file: str = ""
    output_dir: str = "./eeg_analysis_output"
    
    # Preprocessing
    apply_filter: bool = True
    l_freq: float = 0.5  # High-pass filter
    h_freq: float = 45.0  # Low-pass filter
    notch_freq: Optional[float] = 50.0  # Notch filter (50 or 60 Hz)
    
    # Resampling
    resample: bool = False
    resample_freq: float = 250.0
    
    # ICA
    apply_ica: bool = False
    n_ica_components: Optional[int] = None
    ica_method: str = 'fastica'
    
    # Spectral Analysis
    psd_method: str = 'welch'  # 'welch' or 'multitaper'
    psd_fmin: float = 0.5
    psd_fmax: float = 45.0
    n_fft: int = 2048
    
    # Frequency Bands
    freq_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    })
    
    # Connectivity
    compute_connectivity: bool = True
    connectivity_method: str = 'coh'  # coherence
    
    # Epochs
    epoch_duration: float = 2.0  # seconds
    epoch_overlap: float = 0.5  # overlap ratio
    
    # Visualization
    plot_raw: bool = True
    plot_psd: bool = True
    plot_topomap: bool = True
    plot_connectivity: bool = False
    save_plots: bool = True
    
    # Export
    export_features: bool = True
    export_format: str = 'csv'  # 'csv' or 'excel'

class EEGAnalyzer:
    """Main class for EEG feature analysis"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.raw = None
        self.epochs = None
        self.features = {}
        self.results = {}
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> mne.io.Raw:
        """Load EEG data from .set file or other supported formats"""
        try:
            logger.info(f"Loading EEG data from {self.config.input_file}")
            
            # Check if file exists
            if not os.path.exists(self.config.input_file):
                raise FileNotFoundError(f"File not found: {self.config.input_file}")
            
            # Detect file type and load accordingly
            file_ext = os.path.splitext(self.config.input_file)[1].lower()
            
            if file_ext == '.set':
                # Load EEGLAB file
                self.raw = mne.io.read_raw_eeglab(self.config.input_file, preload=True)
            elif file_ext == '.fif':
                # Load MNE FIF file
                self.raw = mne.io.read_raw_fif(self.config.input_file, preload=True)
            elif file_ext == '.edf':
                # Load EDF file
                self.raw = mne.io.read_raw_edf(self.config.input_file, preload=True)
            elif file_ext == '.bdf':
                # Load BDF file
                self.raw = mne.io.read_raw_bdf(self.config.input_file, preload=True)
            elif file_ext == '.vhdr':
                # Load BrainVision file
                self.raw = mne.io.read_raw_brainvision(self.config.input_file, preload=True)
            elif file_ext == '.cnt':
                # Load Neuroscan CNT file
                self.raw = mne.io.read_raw_cnt(self.config.input_file, preload=True)
            else:
                # Try to auto-detect format
                logger.warning(f"Unknown file extension {file_ext}, attempting auto-detection...")
                self.raw = mne.io.read_raw(self.config.input_file, preload=True)
            
            # Print basic info
            logger.info(f"Data loaded successfully:")
            logger.info(f"  - Format: {file_ext if file_ext else 'auto-detected'}")
            logger.info(f"  - Channels: {len(self.raw.ch_names)}")
            logger.info(f"  - Sampling rate: {self.raw.info['sfreq']} Hz")
            logger.info(f"  - Duration: {self.raw.times[-1]:.2f} seconds")
            
            return self.raw
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self):
        """Apply preprocessing steps to raw data"""
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Starting preprocessing...")
        
        # Apply filters
        if self.config.apply_filter:
            # Check if high-pass frequency is appropriate for the sampling rate
            nyquist_freq = self.raw.info['sfreq'] / 2
            h_freq = min(self.config.h_freq, nyquist_freq - 1)
            
            if h_freq != self.config.h_freq:
                logger.warning(f"Adjusting high-pass filter from {self.config.h_freq} Hz to {h_freq} Hz due to sampling rate")
            
            logger.info(f"Applying band-pass filter: {self.config.l_freq}-{h_freq} Hz")
            self.raw.filter(self.config.l_freq, h_freq, fir_design='firwin')
        
        # Apply notch filter
        if self.config.notch_freq:
            # Check if notch frequency is below Nyquist frequency
            nyquist_freq = self.raw.info['sfreq'] / 2
            
            if self.config.notch_freq >= nyquist_freq - 1:
                logger.warning(f"Notch filter at {self.config.notch_freq} Hz too close to Nyquist frequency ({nyquist_freq} Hz)")
                logger.warning("Skipping notch filter for this data.")
            else:
                logger.info(f"Applying notch filter at {self.config.notch_freq} Hz")
                # Only apply fundamental frequency, skip harmonics for low sampling rates
                self.raw.notch_filter(self.config.notch_freq, filter_length='auto', phase='zero')
        
        # Resample if requested
        if self.config.resample:
            logger.info(f"Resampling to {self.config.resample_freq} Hz")
            self.raw.resample(self.config.resample_freq)
        
        # Apply ICA for artifact removal
        if self.config.apply_ica:
            self._apply_ica()
        
        # Set EEG reference (average reference)
        self.raw.set_eeg_reference('average', projection=True)
        self.raw.apply_proj()
        
        logger.info("Preprocessing completed")
    
    def _apply_ica(self):
        """Apply ICA for artifact removal"""
        logger.info("Applying ICA for artifact removal...")
        
        try:
            # Create ICA object
            ica = ICA(
                n_components=self.config.n_ica_components,
                method=self.config.ica_method,
                random_state=42,
                max_iter='auto'
            )
            
            # Fit ICA
            ica.fit(self.raw)
            
            # Automatic component selection (EOG and ECG artifacts)
            # This is a simplified version - you might want to add manual inspection
            eog_indices = []
            ecg_indices = []
            
            # Find EOG components
            eog_channels = [ch for ch in self.raw.ch_names if 'EOG' in ch.upper()]
            if eog_channels:
                try:
                    eog_indices, _ = ica.find_bads_eog(self.raw, ch_name=eog_channels[0])
                except:
                    logger.warning("Could not automatically detect EOG artifacts")
            
            # Find ECG components  
            ecg_channels = [ch for ch in self.raw.ch_names if 'ECG' in ch.upper()]
            if ecg_channels:
                try:
                    ecg_indices, _ = ica.find_bads_ecg(self.raw, ch_name=ecg_channels[0])
                except:
                    logger.warning("Could not automatically detect ECG artifacts")
            
            # Exclude components
            ica.exclude = list(set(eog_indices + ecg_indices))
            logger.info(f"Excluding {len(ica.exclude)} ICA components")
            
            # Apply ICA
            self.raw = ica.apply(self.raw)
            
        except Exception as e:
            logger.error(f"ICA failed: {e}")
            logger.warning("Continuing without ICA artifact removal")
    
    def create_epochs(self):
        """Create epochs from continuous data"""
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Creating epochs...")
        
        # Calculate epoch parameters
        duration = self.config.epoch_duration
        overlap = self.config.epoch_overlap
        step = duration * (1 - overlap)
        
        # Create events at regular intervals
        sfreq = self.raw.info['sfreq']
        n_samples = len(self.raw.times)
        event_samples = np.arange(0, n_samples - duration * sfreq, step * sfreq).astype(int)
        events = np.column_stack([
            event_samples,
            np.zeros(len(event_samples), dtype=int),
            np.ones(len(event_samples), dtype=int)
        ])
        
        # Create epochs
        self.epochs = mne.Epochs(
            self.raw, events, tmin=0, tmax=duration,
            baseline=None, preload=True, reject_by_annotation=True
        )
        
        logger.info(f"Created {len(self.epochs)} epochs of {duration}s duration")
        
        return self.epochs
    
    def compute_spectral_features(self):
        """Compute spectral features (PSD, band power)"""
        logger.info("Computing spectral features...")
        
        # Compute PSD using the new MNE API
        if self.config.psd_method == 'welch':
            psd = self.raw.compute_psd(
                method='welch',
                fmin=self.config.psd_fmin,
                fmax=self.config.psd_fmax,
                n_fft=self.config.n_fft,
                n_overlap=self.config.n_fft // 2,
                verbose=False
            )
        else:  # multitaper
            psd = self.raw.compute_psd(
                method='multitaper',
                fmin=self.config.psd_fmin,
                fmax=self.config.psd_fmax,
                adaptive=True,
                verbose=False
            )
        
        # Get the data from PSD object
        psds, freqs = psd.get_data(return_freqs=True)
        
        # Store PSD results
        self.results['psd'] = {
            'data': psds,
            'freqs': freqs,
            'ch_names': self.raw.ch_names
        }
        
        # Compute band power for each channel
        band_powers = {}
        for band_name, (fmin, fmax) in self.config.freq_bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_powers[band_name] = psds[:, freq_mask].mean(axis=1)
        
        self.features['band_powers'] = pd.DataFrame(
            band_powers,
            index=self.raw.ch_names
        )
        
        # Compute relative band power
        total_power = psds.mean(axis=1)
        relative_band_powers = {}
        for band_name in self.config.freq_bands.keys():
            relative_band_powers[f'{band_name}_relative'] = band_powers[band_name] / total_power
        
        self.features['relative_band_powers'] = pd.DataFrame(
            relative_band_powers,
            index=self.raw.ch_names
        )
        
        # Compute spectral entropy
        spectral_entropy = []
        for ch_psd in psds:
            psd_norm = ch_psd / ch_psd.sum()
            spectral_entropy.append(-np.sum(psd_norm * np.log2(psd_norm + 1e-15)))
        
        self.features['spectral_entropy'] = pd.Series(
            spectral_entropy,
            index=self.raw.ch_names,
            name='spectral_entropy'
        )
        
        logger.info("Spectral features computed")
    
    def compute_temporal_features(self):
        """Compute temporal domain features"""
        logger.info("Computing temporal features...")
        
        # Get data
        data = self.raw.get_data()
        
        temporal_features = {}
        
        # Statistical features for each channel
        temporal_features['mean'] = np.mean(data, axis=1)
        temporal_features['std'] = np.std(data, axis=1)
        temporal_features['variance'] = np.var(data, axis=1)
        temporal_features['skewness'] = skew(data, axis=1)
        temporal_features['kurtosis'] = kurtosis(data, axis=1)
        temporal_features['rms'] = np.sqrt(np.mean(data**2, axis=1))
        
        # Peak-to-peak amplitude
        temporal_features['ptp'] = np.ptp(data, axis=1)
        
        # Zero crossing rate
        zero_crossings = []
        for ch_data in data:
            zero_cross = np.sum(np.diff(np.sign(ch_data)) != 0) / len(ch_data)
            zero_crossings.append(zero_cross)
        temporal_features['zero_crossing_rate'] = np.array(zero_crossings)
        
        # Hjorth parameters
        mobility, complexity = self._compute_hjorth_parameters(data)
        temporal_features['hjorth_mobility'] = mobility
        temporal_features['hjorth_complexity'] = complexity
        
        # Convert to DataFrame
        self.features['temporal'] = pd.DataFrame(
            temporal_features,
            index=self.raw.ch_names
        )
        
        logger.info("Temporal features computed")
    
    def _compute_hjorth_parameters(self, data):
        """Compute Hjorth mobility and complexity"""
        # First derivative
        diff1 = np.diff(data, axis=1)
        # Second derivative
        diff2 = np.diff(diff1, axis=1)
        
        # Variances
        var_data = np.var(data, axis=1)
        var_diff1 = np.var(diff1, axis=1)
        var_diff2 = np.var(diff2, axis=1)
        
        # Mobility
        mobility = np.sqrt(var_diff1 / var_data)
        
        # Complexity
        complexity = np.sqrt(var_diff2 / var_diff1) / mobility
        
        return mobility, complexity
    
    def compute_connectivity(self):
        """Compute connectivity measures between channels"""
        if not self.config.compute_connectivity:
            return
        
        if not CONNECTIVITY_AVAILABLE:
            logger.warning("mne-connectivity package not installed. Skipping connectivity analysis.")
            logger.warning("Install with: pip install mne-connectivity")
            return
        
        logger.info("Computing connectivity measures...")
        
        try:
            # Create epochs if not already created
            if self.epochs is None:
                self.create_epochs()
            
            # Compute connectivity
            con = spectral_connectivity_epochs(
                self.epochs,
                method=self.config.connectivity_method,
                mode='multitaper',
                fmin=self.config.psd_fmin,
                fmax=self.config.psd_fmax,
                faverage=True,
                verbose=False
            )
            
            # Store connectivity matrix
            # Get data with proper indexing for the connectivity result
            conn_data = con.get_data(output='dense')
            if conn_data.ndim == 3:
                conn_matrix = conn_data[:, :, 0]
            else:
                conn_matrix = conn_data
            
            self.results['connectivity'] = {
                'data': conn_matrix,
                'method': self.config.connectivity_method,
                'ch_names': self.raw.ch_names
            }
            
            # Compute connectivity strength for each channel
            conn_strength = np.mean(conn_matrix, axis=0)
            
            self.features['connectivity_strength'] = pd.Series(
                conn_strength,
                index=self.raw.ch_names,
                name='connectivity_strength'
            )
            
            logger.info("Connectivity measures computed")
            
        except Exception as e:
            logger.warning(f"Could not compute connectivity: {e}")
            logger.warning("This might be due to MNE version differences or missing mne-connectivity package.")
    
    def visualize_results(self):
        """Create visualizations of analysis results"""
        if not any([self.config.plot_raw, self.config.plot_psd, 
                   self.config.plot_topomap, self.config.plot_connectivity]):
            return
        
        logger.info("Creating visualizations...")
        
        # Raw data plot
        if self.config.plot_raw:
            fig = self.raw.plot(duration=30, n_channels=30, show=False)
            if self.config.save_plots:
                fig.savefig(os.path.join(self.config.output_dir, 'raw_data.png'), dpi=100)
            plt.close(fig)
        
        # PSD plot
        if self.config.plot_psd and 'psd' in self.results:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot average PSD
            psds = self.results['psd']['data']
            freqs = self.results['psd']['freqs']
            
            # Average PSD across channels
            avg_psd = 10 * np.log10(psds.mean(axis=0))
            axes[0].plot(freqs, avg_psd)
            axes[0].set_xlabel('Frequency (Hz)')
            axes[0].set_ylabel('Power (dB)')
            axes[0].set_title('Average Power Spectral Density')
            axes[0].grid(True, alpha=0.3)
            
            # Band powers heatmap
            if 'band_powers' in self.features:
                band_powers_norm = self.features['band_powers'].T
                band_powers_norm = (band_powers_norm - band_powers_norm.mean()) / band_powers_norm.std()
                
                im = axes[1].imshow(band_powers_norm, aspect='auto', cmap='RdBu_r')
                axes[1].set_yticks(range(len(self.config.freq_bands)))
                axes[1].set_yticklabels(list(self.config.freq_bands.keys()))
                axes[1].set_xticks(range(0, len(self.raw.ch_names), max(1, len(self.raw.ch_names)//10)))
                axes[1].set_xticklabels(self.raw.ch_names[::max(1, len(self.raw.ch_names)//10)], rotation=45)
                axes[1].set_title('Normalized Band Powers by Channel')
                plt.colorbar(im, ax=axes[1], label='Z-score')
            
            plt.tight_layout()
            if self.config.save_plots:
                plt.savefig(os.path.join(self.config.output_dir, 'psd_analysis.png'), dpi=100)
            plt.close()
        
        # Topographic maps
        if self.config.plot_topomap and 'band_powers' in self.features:
            try:
                # Check if we have channel locations
                picks = mne.pick_types(self.raw.info, meg=False, eeg=True)
                if len(picks) == 0:
                    logger.warning("No EEG channels found. Skipping topomap.")
                elif not any(ch['loc'][:3].any() for ch in self.raw.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH):
                    logger.warning("No channel locations found. Skipping topomap.")
                else:
                    n_bands = len(self.config.freq_bands)
                    fig, axes = plt.subplots(1, n_bands, figsize=(15, 3))
                    if n_bands == 1:
                        axes = [axes]  # Make it iterable
                    
                    for idx, (band_name, _) in enumerate(self.config.freq_bands.items()):
                        band_power = self.features['band_powers'][band_name].values
                        
                        # Only plot EEG channels
                        im, _ = mne.viz.plot_topomap(
                            band_power[picks],
                            self.raw.info,
                            axes=axes[idx],
                            show=False,
                            cmap='RdBu_r'
                        )
                        axes[idx].set_title(f'{band_name} band')
                    
                    plt.suptitle('Topographic Distribution of Band Powers')
                    plt.tight_layout()
                    if self.config.save_plots:
                        plt.savefig(os.path.join(self.config.output_dir, 'topomap_bands.png'), dpi=100)
                    plt.close()
            except Exception as e:
                logger.warning(f"Could not create topomap: {e}")
        
        # Connectivity matrix
        if self.config.plot_connectivity and 'connectivity' in self.results:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            conn_matrix = self.results['connectivity']['data']
            im = ax.imshow(conn_matrix, cmap='hot', aspect='auto')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Channel')
            ax.set_title(f'Connectivity Matrix ({self.config.connectivity_method})')
            plt.colorbar(im, ax=ax)
            
            if self.config.save_plots:
                plt.savefig(os.path.join(self.config.output_dir, 'connectivity_matrix.png'), dpi=100)
            plt.close()
        
        logger.info("Visualizations created")
    
    def export_features(self):
        """Export computed features to file"""
        if not self.config.export_features or not self.features:
            return
        
        logger.info("Exporting features...")
        
        # Combine all features into one DataFrame
        all_features = pd.DataFrame(index=self.raw.ch_names)
        
        for feature_type, feature_df in self.features.items():
            if isinstance(feature_df, pd.DataFrame):
                all_features = all_features.join(feature_df, rsuffix=f'_{feature_type}')
            elif isinstance(feature_df, pd.Series):
                all_features[feature_type] = feature_df
        
        # Export based on format
        output_file = os.path.join(self.config.output_dir, f'eeg_features.{self.config.export_format}')
        
        if self.config.export_format == 'csv':
            all_features.to_csv(output_file)
        elif self.config.export_format == 'excel':
            all_features.to_excel(output_file)
        
        logger.info(f"Features exported to {output_file}")
        
        # Also save a summary report
        self._create_summary_report(all_features)
    
    def _create_summary_report(self, features_df):
        """Create a summary report of the analysis"""
        report_file = os.path.join(self.config.output_dir, 'analysis_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("EEG ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Data info
            f.write("DATA INFORMATION:\n")
            f.write(f"  File: {self.config.input_file}\n")
            f.write(f"  Channels: {len(self.raw.ch_names)}\n")
            f.write(f"  Sampling rate: {self.raw.info['sfreq']} Hz\n")
            f.write(f"  Duration: {self.raw.times[-1]:.2f} seconds\n\n")
            
            # Preprocessing
            f.write("PREPROCESSING:\n")
            f.write(f"  Filter applied: {self.config.apply_filter}\n")
            if self.config.apply_filter:
                f.write(f"    Frequency range: {self.config.l_freq}-{self.config.h_freq} Hz\n")
            f.write(f"  Notch filter: {self.config.notch_freq} Hz\n")
            f.write(f"  ICA applied: {self.config.apply_ica}\n")
            f.write(f"  Resampled: {self.config.resample}\n\n")
            
            # Feature summary
            f.write("FEATURES COMPUTED:\n")
            f.write(f"  Total features: {len(features_df.columns)}\n")
            f.write(f"  Feature categories: {list(self.features.keys())}\n\n")
            
            # Statistical summary
            f.write("FEATURE STATISTICS:\n")
            f.write("-" * 40 + "\n")
            
            # Summary statistics for key features
            if 'band_powers' in self.features:
                f.write("\nBand Power Statistics (averaged across channels):\n")
                for band in self.config.freq_bands.keys():
                    values = self.features['band_powers'][band]
                    f.write(f"  {band:8s}: mean={values.mean():.4f}, std={values.std():.4f}\n")
            
            if 'temporal' in self.features:
                f.write("\nTemporal Feature Statistics (averaged across channels):\n")
                for col in ['mean', 'std', 'rms', 'hjorth_mobility']:
                    if col in self.features['temporal'].columns:
                        values = self.features['temporal'][col]
                        f.write(f"  {col:20s}: mean={values.mean():.4f}, std={values.std():.4f}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Analysis completed successfully\n")
        
        logger.info(f"Summary report saved to {report_file}")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            # Load data
            self.load_data()
            
            # Preprocess
            self.preprocess_data()
            
            # Compute features
            self.compute_spectral_features()
            self.compute_temporal_features()
            self.compute_connectivity()
            
            # Visualize
            self.visualize_results()
            
            # Export
            self.export_features()
            
            logger.info("Full analysis completed successfully!")
            
            return self.features, self.results
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise


def main():
    """Main function to run the analysis"""
    
    # Example configuration
    config = AnalysisConfig(
        input_file="path/to/your/eeg_file.set",  # Supports .set, .fif, .edf, .bdf, .vhdr, .cnt
        output_dir="./eeg_analysis_output",
        apply_filter=True,
        l_freq=0.5,
        h_freq=45.0,
        notch_freq=50.0,  # Use 60.0 for US
        apply_ica=False,  # Set to True for artifact removal
        compute_connectivity=True,
        plot_raw=True,
        plot_psd=True,
        plot_topomap=True,
        plot_connectivity=True,
        export_features=True
    )
    
    # Create analyzer and run analysis
    analyzer = EEGAnalyzer(config)
    features, results = analyzer.run_full_analysis()
    
    print("\nAnalysis complete! Check the output directory for results.")
    print(f"Output directory: {config.output_dir}")
    
    return analyzer


def load_config_from_json(json_file: str) -> AnalysisConfig:
    """Load configuration from JSON file"""
    with open(json_file, 'r') as f:
        config_dict = json.load(f)
    
    # Convert frequency bands from dict format if needed
    if 'freq_bands' in config_dict:
        freq_bands = {}
        for band, freqs in config_dict['freq_bands'].items():
            if isinstance(freqs, list):
                freq_bands[band] = tuple(freqs)
            else:
                freq_bands[band] = freqs
        config_dict['freq_bands'] = freq_bands
    
    return AnalysisConfig(**config_dict)


def save_config_to_json(config: AnalysisConfig, json_file: str):
    """Save configuration to JSON file"""
    config_dict = asdict(config)
    
    # Convert tuples to lists for JSON serialization
    if 'freq_bands' in config_dict:
        freq_bands = {}
        for band, freqs in config_dict['freq_bands'].items():
            freq_bands[band] = list(freqs)
        config_dict['freq_bands'] = freq_bands
    
    with open(json_file, 'w') as f:
        json.dump(config_dict, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EEG Feature Analysis Tool - Supports .set, .fif, .edf, .bdf, .vhdr, .cnt formats")
    parser.add_argument("input_file", help="Path to EEG file (.set, .fif, .edf, .bdf, .vhdr, or .cnt)")
    parser.add_argument("--output-dir", default="./eeg_analysis_output", help="Output directory")
    parser.add_argument("--config", help="Path to JSON configuration file")
    parser.add_argument("--save-config", help="Save configuration to JSON file")
    parser.add_argument("--no-filter", action="store_true", help="Skip filtering")
    parser.add_argument("--ica", action="store_true", help="Apply ICA for artifact removal")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config_from_json(args.config)
        config.input_file = args.input_file  # Override with command line argument
    else:
        config = AnalysisConfig(
            input_file=args.input_file,
            output_dir=args.output_dir,
            apply_filter=not args.no_filter,
            apply_ica=args.ica,
            plot_raw=not args.no_plots,
            plot_psd=not args.no_plots,
            plot_topomap=not args.no_plots,
            plot_connectivity=not args.no_plots
        )
    
    # Save configuration if requested
    if args.save_config:
        save_config_to_json(config, args.save_config)
        print(f"Configuration saved to {args.save_config}")
    
    # Run analysis
    analyzer = EEGAnalyzer(config)
    analyzer.run_full_analysis()
