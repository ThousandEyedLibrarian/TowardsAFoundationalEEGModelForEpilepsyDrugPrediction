#!/usr/bin/env python3
"""
Professional EEG Preprocessing Pipeline for R5 BDF Data
=========================================================
Leverages eegdash for dataset management and braindecode for preprocessing.
Processes R5 BDF format data (already at 100 Hz) into challenge-compliant format.

Author: EEG Challenge Team
Date: 2024
"""

import os
import argparse
import logging
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# MNE imports
import mne
from mne.preprocessing import ICA

# Braindecode imports
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
    exponential_moving_standardize
)
from braindecode.datasets import BaseConcatDataset, BaseDataset

# EEGDash imports
try:
    from eegdash.dataset import EEGChallengeDataset
    from eegdash.hbn.windows import (
        annotate_trials_with_target,
        add_aux_anchors,
        add_extras_columns,
        keep_only_recordings_with
    )
    EEGDASH_AVAILABLE = True
except ImportError:
    EEGDASH_AVAILABLE = False
    warnings.warn("eegdash not installed. Using fallback BIDS loader.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    # Data paths
    data_dir: str
    output_dir: str

    # Task configuration
    task: str = 'contrastChangeDetection'
    release: str = 'R5'

    # Window parameters (no resampling - data already at 100 Hz)
    sampling_rate: int = 100  # Fixed at 100 Hz
    window_length: float = 2.0
    window_stride: float = 1.0

    # Filtering parameters
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    notch_freqs: List[float] = None

    # Quality control
    min_trials: int = 10
    max_bad_channels: float = 0.2
    reject_amplitude: float = 200.0  # μV

    # Demographics extraction
    extract_demographics: bool = True

    # Output configuration
    output_format: str = 'npz'
    split_ratio: str = '70:15:15'
    save_metadata: bool = True

    # Performance options
    n_jobs: int = -1
    batch_size: int = 100
    memory_efficient: bool = False

    # Preprocessing pipeline
    preprocessing_pipeline: str = 'standard'

    # Debugging options
    verbose: bool = False
    debug: bool = False
    visualize: bool = False
    dry_run: bool = False

    def __post_init__(self):
        if self.notch_freqs is None:
            self.notch_freqs = [50, 60]


class EEGDashDataLoader:
    """Dataset loader using eegdash"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.dataset = None

    def load_dataset(self) -> Any:
        """Load dataset using eegdash"""
        if not EEGDASH_AVAILABLE:
            logger.warning("eegdash not available, using fallback loader")
            return self._load_fallback()

        logger.info(f"Loading {self.config.task} data from release {self.config.release}")

        try:
            self.dataset = EEGChallengeDataset(
                task=self.config.task,
                release=self.config.release,
                cache_dir=self.config.data_dir,
                mini=self.config.debug  # Use mini dataset for debugging
            )

            logger.info(f"Loaded {len(self.dataset.datasets)} recordings")
            return self.dataset

        except Exception as e:
            logger.error(f"Failed to load with eegdash: {e}")
            return self._load_fallback()

    def _load_fallback(self):
        """Fallback BIDS loader using MNE - parses sub-XXX directories directly"""
        logger.info("Using fallback BIDS loader to parse subdirectories")

        from process_bdf import BIDSDirectoryParser
        parser = BIDSDirectoryParser(self.config)
        return parser.parse_bids_directory()

    def extract_demographics(self, dataset) -> Dict[str, List]:
        """Extract demographic information from dataset"""
        demographics = {
            'subject_id': [],
            'age': [],
            'sex': [],
            'handedness': [],
            'diagnosis': [],
            'medication': [],
            'task': [],
            'session': [],
            'run': [],
            'p_factor': [],
            'adhd_status': [],
            'anxiety_status': [],
            'depression_status': [],
        }

        if not self.config.extract_demographics:
            return demographics

        logger.info("Extracting demographic information")

        for recording in dataset.datasets:
            if hasattr(recording, 'description'):
                info = recording.description
                demographics['subject_id'].append(info.get('subject', 'unknown'))
                demographics['age'].append(info.get('age', np.nan))
                demographics['sex'].append(info.get('sex', 'unknown'))
                demographics['task'].append(info.get('task', self.config.task))

                # Extract additional fields if available
                for field in ['handedness', 'diagnosis', 'medication',
                             'session', 'run', 'p_factor', 'adhd_status',
                             'anxiety_status', 'depression_status']:
                    demographics[field].append(info.get(field, np.nan))

        return demographics


class EEGPreprocessor:
    """Main preprocessing class using braindecode"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.preprocessors = self._get_preprocessors()

    def _get_preprocessors(self) -> List[Preprocessor]:
        """Get preprocessing steps based on pipeline choice"""

        if self.config.preprocessing_pipeline == 'minimal':
            return self._get_minimal_pipeline()
        elif self.config.preprocessing_pipeline == 'standard':
            return self._get_standard_pipeline()
        elif self.config.preprocessing_pipeline == 'advanced':
            return self._get_advanced_pipeline()
        elif self.config.preprocessing_pipeline == 'custom':
            return self._load_custom_pipeline()
        else:
            raise ValueError(f"Unknown pipeline: {self.config.preprocessing_pipeline}")

    def _get_minimal_pipeline(self) -> List[Preprocessor]:
        """Minimal preprocessing pipeline"""
        return [
            # Channel selection
            Preprocessor('pick_types', eeg=True, exclude='bads'),

            # Basic filtering
            Preprocessor('filter',
                        l_freq=self.config.bandpass_low,
                        h_freq=self.config.bandpass_high),

            # Reference to average
            Preprocessor('set_eeg_reference', ref_channels='average'),
        ]

    def _get_standard_pipeline(self) -> List[Preprocessor]:
        """Standard preprocessing pipeline (no resampling - data already at 100 Hz)"""
        return [
            # 1. Channel selection (129 channels)
            Preprocessor('pick_types', eeg=True, exclude='bads'),

            # 2. NO RESAMPLING - Data is already at 100 Hz

            # 3. Bandpass filtering
            Preprocessor('filter',
                        l_freq=self.config.bandpass_low,
                        h_freq=self.config.bandpass_high,
                        method='fir',
                        fir_design='firwin'),

            # 4. Notch filter for powerline noise
            Preprocessor('notch_filter',
                        freqs=self.config.notch_freqs,
                        method='fir'),

            # 5. Re-referencing to common average
            Preprocessor('set_eeg_reference', ref_channels='average'),

            # 6. Bad channel detection
            Preprocessor(self._detect_bad_channels, apply_on_array=False),

            # 7. Artifact rejection
            Preprocessor(self._reject_bad_epochs, apply_on_array=False),

            # 8. Exponential moving standardization
            Preprocessor(exponential_moving_standardize,
                        factor_new=0.001,
                        init_block_size=1000,
                        apply_on_array=True)
        ]

    def _get_advanced_pipeline(self) -> List[Preprocessor]:
        """Advanced pipeline with ICA"""
        pipeline = self._get_standard_pipeline()

        # Add ICA for artifact removal
        pipeline.insert(-1, Preprocessor(self._apply_ica, apply_on_array=False))

        return pipeline

    def _load_custom_pipeline(self) -> List[Preprocessor]:
        """Load custom pipeline from JSON configuration"""
        config_path = Path(self.config.data_dir) / 'preprocessing_config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Custom config not found: {config_path}")

        with open(config_path, 'r') as f:
            custom_config = json.load(f)

        # Parse and create preprocessors from config
        # Implementation would go here
        raise NotImplementedError("Custom pipeline loading not yet implemented")

    def _detect_bad_channels(self, raw):
        """Detect and interpolate bad channels"""
        # Detect flat channels
        flat_chans = mne.preprocessing.find_bad_channels_flat(raw)[0]

        # Detect noisy channels
        noisy_chans = mne.preprocessing.find_bad_channels_maxwell(
            raw,
            return_scores=False,
            limit=5
        )[0] if hasattr(raw.info, 'dev_head_t') else []

        # Mark bad channels
        raw.info['bads'] = list(set(flat_chans + noisy_chans))

        # Check if too many bad channels
        n_bad = len(raw.info['bads'])
        n_total = len(raw.ch_names)

        if n_bad / n_total > self.config.max_bad_channels:
            logger.warning(f"Too many bad channels ({n_bad}/{n_total})")

        # Interpolate bad channels
        if raw.info['bads']:
            raw.interpolate_bads(reset_bads=True)

        return raw

    def _reject_bad_epochs(self, epochs):
        """Reject bad epochs based on amplitude threshold"""
        reject_criteria = dict(
            eeg=self.config.reject_amplitude * 1e-6  # Convert to V
        )

        epochs.drop_bad(reject=reject_criteria)

        # Check if enough epochs remain
        if len(epochs) < self.config.min_trials:
            logger.warning(f"Too few epochs remaining: {len(epochs)}")

        return epochs

    def _apply_ica(self, raw):
        """Apply ICA for artifact removal"""
        ica = ICA(n_components=20, random_state=42, max_iter=800)
        ica.fit(raw)

        # Find EOG/ECG artifacts
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=None)
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name=None)

        # Exclude components
        ica.exclude = list(set(eog_indices + ecg_indices))

        # Apply ICA
        raw_clean = ica.apply(raw.copy())

        return raw_clean

    def process_dataset(self, dataset):
        """Apply preprocessing to dataset"""
        logger.info("Applying preprocessing pipeline")

        # Apply preprocessors
        preprocessed = preprocess(
            dataset,
            preprocessors=self.preprocessors,
            n_jobs=self.config.n_jobs
        )

        return preprocessed


class WindowCreator:
    """Create windows from continuous data"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def create_windows(self, dataset):
        """Create windows from events"""
        logger.info("Creating windows from events")

        window_size_samples = int(self.config.window_length * self.config.sampling_rate)
        window_stride_samples = int(self.config.window_stride * self.config.sampling_rate)

        # Create windows with proper anchoring
        windows_dataset = create_windows_from_events(
            dataset,
            mapping={'stimulus_anchor': 0},  # Map events to labels
            trial_start_offset_samples=int(0.5 * self.config.sampling_rate),
            trial_stop_offset_samples=int(2.5 * self.config.sampling_rate),
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            preload=not self.config.memory_efficient
        )

        return windows_dataset


class QualityController:
    """Quality control and metrics"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.metrics = {}

    def compute_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for processed data"""
        metrics = {}

        # Signal-to-noise ratio
        signal_power = np.mean(np.square(data))
        noise_power = np.mean(np.square(np.diff(data, axis=-1)))
        metrics['snr'] = 10 * np.log10(signal_power / (noise_power + 1e-10))

        # Channel correlations
        if data.ndim == 3:  # (samples, channels, time)
            channel_data = data.reshape(data.shape[0], data.shape[1], -1)
            correlations = []
            for sample in channel_data:
                corr_matrix = np.corrcoef(sample)
                # Get upper triangle (excluding diagonal)
                upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                correlations.append(np.mean(np.abs(upper_tri)))
            metrics['mean_channel_correlation'] = np.mean(correlations)

        # Temporal consistency
        if data.ndim >= 2:
            temporal_diff = np.diff(data, axis=-1)
            metrics['temporal_consistency'] = 1.0 - np.std(temporal_diff)

        # Data range
        metrics['min_value'] = float(np.min(data))
        metrics['max_value'] = float(np.max(data))
        metrics['mean_value'] = float(np.mean(data))
        metrics['std_value'] = float(np.std(data))

        self.metrics = metrics
        return metrics

    def validate_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate processed data"""
        checks = []

        # Check shapes
        checks.append(X.ndim == 3)  # (samples, channels, time)
        checks.append(y.ndim in [1, 2])  # (samples,) or (samples, 1)
        checks.append(X.shape[0] == len(y))  # Same number of samples

        # Check for NaN/Inf
        checks.append(not np.any(np.isnan(X)))
        checks.append(not np.any(np.isinf(X)))
        checks.append(not np.any(np.isnan(y)))

        # Check data range
        checks.append(np.abs(X).max() < 1000)  # Reasonable amplitude range

        # Check minimum samples
        checks.append(X.shape[0] >= self.config.min_trials)

        return all(checks)


class BIDSDirectoryParser:
    """Parse BIDS directory structure and load BDF files"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.processed_subjects = []
        self.failed_subjects = []

    def parse_bids_directory(self) -> Dict[str, Any]:
        """Parse all subject directories and load data"""
        logger.info(f"Scanning BIDS directory: {self.data_dir}")

        # Discover all BIDS datasets
        bids_datasets = self._discover_bids_datasets()

        # Process all datasets
        all_data = {
            'X': [],
            'y': [],
            'subjects': [],
            'demographics': []
        }

        for dataset_dir in bids_datasets:
            logger.info(f"Processing BIDS dataset: {dataset_dir.name}")
            self.data_dir = dataset_dir

            # Discover subjects in this dataset
            subjects = self._discover_subjects_in_dataset()
            logger.info(f"Found {len(subjects)} subjects in {dataset_dir.name}")

            if self.config.debug:
                subjects = subjects[:5]
                logger.info(f"Debug mode: processing only {len(subjects)} subjects from {dataset_dir.name}")

            # Process each subject
            for subject_id in tqdm(subjects, desc=f"Loading BDF files from {dataset_dir.name}"):
                try:
                    subject_data = self._process_subject(subject_id)
                    if subject_data is not None:
                        all_data['X'].append(subject_data['X'])
                        all_data['y'].append(subject_data['y'])
                        # Track dataset in subject ID
                        full_subject_id = f"{dataset_dir.name}_{subject_id}"
                        all_data['subjects'].extend([full_subject_id] * len(subject_data['X']))
                        all_data['demographics'].extend(subject_data.get('demographics', []))
                        self.processed_subjects.append(full_subject_id)
                except Exception as e:
                    logger.warning(f"Failed to process {subject_id}: {e}")
                    self.failed_subjects.append((subject_id, str(e)))

        # Concatenate all data
        if all_data['X']:
            all_data['X'] = np.concatenate(all_data['X'], axis=0)
            all_data['y'] = np.concatenate(all_data['y'], axis=0)
            all_data['subjects'] = np.array(all_data['subjects'])
            logger.info(f"Loaded {len(all_data['X'])} samples from {len(self.processed_subjects)} subjects across {len(bids_datasets)} datasets")
        else:
            raise ValueError("No data was loaded successfully")

        return all_data

    def _discover_bids_datasets(self) -> List[Path]:
        """Discover all BIDS datasets in the data directory"""
        bids_datasets = []

        # First, check if the current directory itself is a BIDS dataset
        dataset_desc = self.data_dir / "dataset_description.json"
        participants_tsv = self.data_dir / "participants.tsv"
        has_subjects = any(p.is_dir() and p.name.startswith("sub-")
                          for p in self.data_dir.iterdir())

        if (dataset_desc.exists() or participants_tsv.exists()) and has_subjects:
            # The data_dir itself is a BIDS dataset
            logger.info(f"Data directory {self.data_dir.name} is a BIDS dataset")
            bids_datasets.append(self.data_dir)
        else:
            # Look for BIDS datasets in subdirectories
            logger.info("Looking for BIDS datasets in subdirectories...")
            for path in sorted(self.data_dir.iterdir()):
                if path.is_dir():
                    # Check for BIDS dataset markers
                    dataset_desc = path / "dataset_description.json"
                    participants_tsv = path / "participants.tsv"
                    # Also check if it has subject directories
                    has_subjects = any(p.is_dir() and p.name.startswith("sub-")
                                      for p in path.iterdir())

                    if (dataset_desc.exists() or participants_tsv.exists()) and has_subjects:
                        bids_datasets.append(path)
                        logger.info(f"Found BIDS dataset: {path.name}")

        if not bids_datasets:
            raise ValueError(
                f"No BIDS datasets found in {self.data_dir}\n"
                f"Expected either:\n"
                f"  1. {self.data_dir} to be a BIDS dataset with sub-XXX directories\n"
                f"  2. {self.data_dir} to contain BIDS dataset subdirectories (e.g., ds005509/)\n\n"
                f"BIDS datasets should have:\n"
                f"  - dataset_description.json or participants.tsv\n"
                f"  - sub-XXX/ directories containing EEG data"
            )

        logger.info(f"Found {len(bids_datasets)} BIDS dataset(s): {[d.name for d in bids_datasets]}")
        return bids_datasets

    def _discover_subjects_in_dataset(self) -> List[str]:
        """Discover all subject directories in the current dataset"""
        subjects = []
        for path in sorted(self.data_dir.iterdir()):
            if path.is_dir() and path.name.startswith("sub-"):
                subjects.append(path.name)

        if not subjects:
            raise ValueError(
                f"No subject directories (sub-XXX) found in {self.data_dir}\n"
                f"Expected BIDS format:\n"
                f"  {self.data_dir}/sub-001/eeg/*.bdf\n"
                f"  {self.data_dir}/sub-002/eeg/*.bdf\n"
                f"  ..."
            )

        return subjects

    def _process_subject(self, subject_id: str) -> Optional[Dict]:
        """Process a single subject's BDF files"""
        subject_dir = self.data_dir / subject_id / "eeg"

        if not subject_dir.exists():
            logger.warning(f"EEG directory not found for {subject_id}")
            return None

        # Find BDF files for the task
        task_pattern = f"*task-{self.config.task}*.bdf"
        bdf_files = sorted(subject_dir.glob(task_pattern))

        if not bdf_files:
            logger.warning(f"No BDF files found for {subject_id} task {self.config.task}")
            return None

        all_X = []
        all_y = []
        all_metadata = []

        for bdf_file in bdf_files:
            # Load BDF file
            logger.debug(f"Loading {bdf_file.name}")
            raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose=False)

            # Ensure correct sampling rate (should already be 100 Hz)
            if raw.info['sfreq'] != self.config.sampling_rate:
                logger.warning(f"Unexpected sampling rate {raw.info['sfreq']} Hz, expected {self.config.sampling_rate} Hz")
                # Note: per plan, we don't resample

            # Find corresponding events file
            events_file = self._find_events_file(bdf_file)
            if events_file is None or not events_file.exists():
                logger.warning(f"Events file not found for {bdf_file.name}")
                continue

            # Load events and create epochs
            try:
                X_run, y_run, metadata_run = self._extract_epochs(raw, events_file)
                if X_run is not None and len(X_run) > 0:
                    all_X.append(X_run)
                    all_y.append(y_run)
                    all_metadata.extend(metadata_run)
            except Exception as e:
                logger.warning(f"Failed to extract epochs from {bdf_file.name}: {e}")

        if all_X:
            X = np.concatenate(all_X, axis=0)
            y = np.concatenate(all_y, axis=0)

            # Ensure correct number of channels (129 for challenge)
            if X.shape[1] != 129:
                logger.warning(f"{subject_id}: Expected 129 channels, got {X.shape[1]}")
                if X.shape[1] > 129:
                    X = X[:, :129, :]
                else:
                    # Pad with zeros
                    pad_width = ((0, 0), (0, 129 - X.shape[1]), (0, 0))
                    X = np.pad(X, pad_width, mode='constant')

            # Load demographics if available
            demographics = self._load_demographics(subject_id)

            return {
                'X': X,
                'y': y,
                'metadata': all_metadata,
                'demographics': [demographics] * len(X) if demographics else []
            }

        return None

    def _find_events_file(self, bdf_file: Path) -> Optional[Path]:
        """Find corresponding events.tsv file for a BDF file"""
        # BDF file: sub-XXX_task-YYY_eeg.bdf
        # Events file: sub-XXX_task-YYY_events.tsv
        events_file = bdf_file.parent / bdf_file.name.replace('_eeg.bdf', '_events.tsv')
        return events_file if events_file.exists() else None

    def _extract_epochs(self, raw: mne.io.Raw, events_file: Path) -> Tuple[np.ndarray, np.ndarray, List]:
        """Extract epochs from raw data using events file"""
        # Load events
        events_df = pd.read_csv(events_file, sep='\t')

        # Build trial table (following challenge notebook approach)
        trials = self._build_trial_table(events_df)

        # Filter valid trials
        trials = trials[
            (trials['stimulus_onset'].notna()) &
            (trials['response_onset'].notna()) &
            (trials['rt_from_stimulus'] >= 0.1) &
            (trials['rt_from_stimulus'] <= 3.0)
        ]

        if trials.empty:
            return None, None, []

        # Extract epochs
        X = []
        y = []
        metadata = []

        shift_after_stim = 0.5  # 0.5s shift after stimulus
        window_len = self.config.window_length

        for _, trial in trials.iterrows():
            # Calculate epoch timing
            epoch_start = trial['stimulus_onset'] + shift_after_stim
            epoch_end = epoch_start + window_len

            try:
                start_sample = int(epoch_start * self.config.sampling_rate)
                end_sample = int(epoch_end * self.config.sampling_rate)

                # Extract data segment
                data, times = raw[:, start_sample:end_sample]

                # Ensure correct length (2s at 100 Hz = 200 samples)
                expected_samples = int(window_len * self.config.sampling_rate)
                if data.shape[1] == expected_samples:
                    X.append(data)
                    y.append(trial['rt_from_stimulus'])
                    metadata.append({
                        'correct': trial.get('correct', None),
                        'response_type': trial.get('response_type', None),
                        'trial_start': trial['trial_start_onset'],
                        'stimulus_onset': trial['stimulus_onset'],
                        'response_onset': trial['response_onset']
                    })
            except Exception:
                continue

        if X:
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32).reshape(-1, 1)
            return X, y, metadata

        return None, None, []

    def _build_trial_table(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Build trial table from events (from challenge notebook)"""
        events_df = events_df.copy()
        events_df["onset"] = pd.to_numeric(events_df["onset"], errors="coerce")
        events_df = events_df.sort_values("onset").reset_index(drop=True)

        # Extract trial starts
        trials = events_df[events_df["value"].eq("contrastTrial_start")].copy()
        if trials.empty:
            trials = events_df[events_df["trial_type"].eq("contrastTrial_start")].copy()
        if trials.empty:
            return pd.DataFrame()

        # Extract stimuli and responses
        stimuli = events_df[events_df["value"].isin(["left_target", "right_target"])].copy()
        responses = events_df[events_df["value"].isin(["left_buttonPress", "right_buttonPress"])].copy()

        # Alternative column names
        if stimuli.empty:
            stimuli = events_df[events_df["trial_type"].isin(["left_target", "right_target"])].copy()
        if responses.empty:
            responses = events_df[events_df["trial_type"].isin(["left_buttonPress", "right_buttonPress"])].copy()

        trials = trials.reset_index(drop=True)
        trials["next_onset"] = trials["onset"].shift(-1)
        trials = trials.dropna(subset=["next_onset"]).reset_index(drop=True)

        rows = []
        for _, tr in trials.iterrows():
            start = float(tr["onset"])
            end = float(tr["next_onset"])

            # Find stimulus and response in this trial
            stim_block = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
            stim_onset = np.nan if stim_block.empty else float(stim_block.iloc[0]["onset"])

            if not np.isnan(stim_onset):
                resp_block = responses[(responses["onset"] >= stim_onset) & (responses["onset"] < end)]
            else:
                resp_block = responses[(responses["onset"] >= start) & (responses["onset"] < end)]

            if resp_block.empty:
                resp_onset = np.nan
                resp_type = None
                correct = None
            else:
                resp_onset = float(resp_block.iloc[0]["onset"])
                resp_type = resp_block.iloc[0].get("value", resp_block.iloc[0].get("trial_type", None))

                # Determine correctness if available
                feedback = resp_block.iloc[0].get("feedback", None)
                correct = None
                if feedback == "smiley_face":
                    correct = True
                elif feedback == "sad_face":
                    correct = False

            rt_from_stim = (resp_onset - stim_onset) if (not np.isnan(stim_onset) and not np.isnan(resp_onset)) else np.nan
            rt_from_trial = (resp_onset - start) if not np.isnan(resp_onset) else np.nan

            rows.append({
                "trial_start_onset": start,
                "trial_stop_onset": end,
                "stimulus_onset": stim_onset,
                "response_onset": resp_onset,
                "rt_from_stimulus": rt_from_stim,
                "rt_from_trialstart": rt_from_trial,
                "response_type": resp_type,
                "correct": correct,
            })

        return pd.DataFrame(rows)

    def _load_demographics(self, subject_id: str) -> Optional[Dict]:
        """Load demographic data for a subject from participants.tsv"""
        participants_file = self.data_dir / "participants.tsv"

        if not participants_file.exists():
            return None

        try:
            df = pd.read_csv(participants_file, sep='\t')
            subject_row = df[df['participant_id'] == subject_id]

            if subject_row.empty:
                return None

            demographics = {
                'age': subject_row.get('age', np.nan).values[0] if 'age' in subject_row.columns else np.nan,
                'sex': subject_row.get('sex', 'unknown').values[0] if 'sex' in subject_row.columns else 'unknown',
                'p_factor': subject_row.get('p_factor', np.nan).values[0] if 'p_factor' in subject_row.columns else np.nan,
                # FIX Bug #4: Use correct column names from actual dataset
                'attention': subject_row.get('attention', np.nan).values[0] if 'attention' in subject_row.columns else np.nan,
                'internalizing': subject_row.get('internalizing', np.nan).values[0] if 'internalizing' in subject_row.columns else np.nan,
                'externalizing': subject_row.get('externalizing', np.nan).values[0] if 'externalizing' in subject_row.columns else np.nan,
            }

            return demographics
        except Exception as e:
            logger.warning(f"Failed to load demographics for {subject_id}: {e}")
            return None


class DataSplitter:
    """Split data into train/val/test sets"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.split_ratios = self._parse_split_ratio()

    def _parse_split_ratio(self) -> Tuple[float, float, float]:
        """Parse split ratio string"""
        parts = self.config.split_ratio.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid split ratio: {self.config.split_ratio}")

        ratios = [float(p) for p in parts]
        total = sum(ratios)
        return tuple(r / total for r in ratios)

    def split_data(self, X: np.ndarray, y: np.ndarray,
                   demographics: Optional[np.ndarray] = None,
                   subjects: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Split data into train/val/test sets"""
        n_samples = len(X)
        train_ratio, val_ratio, test_ratio = self.split_ratios

        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        # Create indices
        indices = np.arange(n_samples)

        # If we have subjects, do subject-wise splitting
        if subjects is not None:
            unique_subjects = np.unique(subjects)
            np.random.shuffle(unique_subjects)

            n_subjects = len(unique_subjects)
            train_subjects_end = int(n_subjects * train_ratio)
            val_subjects_end = train_subjects_end + int(n_subjects * val_ratio)

            train_subjects = unique_subjects[:train_subjects_end]
            val_subjects = unique_subjects[train_subjects_end:val_subjects_end]
            test_subjects = unique_subjects[val_subjects_end:]

            train_mask = np.isin(subjects, train_subjects)
            val_mask = np.isin(subjects, val_subjects)
            test_mask = np.isin(subjects, test_subjects)

            train_indices = indices[train_mask]
            val_indices = indices[val_mask]
            test_indices = indices[test_mask]
        else:
            # Random splitting
            np.random.shuffle(indices)
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]

        # Split data
        split_data = {
            'X_train': X[train_indices],
            'y_train': y[train_indices],
            'X_val': X[val_indices],
            'y_val': y[val_indices],
            'X_test': X[test_indices],
            'y_test': y[test_indices],
        }

        # Split demographics if available
        if demographics is not None:
            split_data['demographics_train'] = demographics[train_indices]
            split_data['demographics_val'] = demographics[val_indices]
            split_data['demographics_test'] = demographics[test_indices]

        # Add subject IDs if available
        if subjects is not None:
            split_data['subjects_train'] = subjects[train_indices]
            split_data['subjects_val'] = subjects[val_indices]
            split_data['subjects_test'] = subjects[test_indices]

        return split_data


class R5BDFProcessor:
    """Main processor coordinating all components"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.loader = EEGDashDataLoader(config)
        self.preprocessor = EEGPreprocessor(config)
        self.window_creator = WindowCreator(config)
        self.quality_controller = QualityController(config)
        self.splitter = DataSplitter(config)

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def process(self):
        """Main processing pipeline"""
        logger.info("Starting EEG preprocessing pipeline")
        start_time = datetime.now()

        try:
            # 1. Load dataset
            logger.info("Stage 1: Loading dataset")
            dataset = self.loader.load_dataset()

            # Check if we got a dictionary (from BIDS parser) or dataset object (from eegdash)
            if isinstance(dataset, dict):
                # Data already parsed from BIDS directory
                logger.info("Using pre-parsed BIDS data")
                X = dataset['X']
                y = dataset['y']
                subjects = dataset['subjects']
                demographics = None  # Will be extracted from demographics field if available

                # Extract demographics if present
                if 'demographics' in dataset and dataset['demographics']:
                    demographics = self._process_bids_demographics(dataset['demographics'], subjects)

            else:
                # Use eegdash/braindecode pipeline
                logger.info("Using eegdash/braindecode pipeline")

                # 2. Extract demographics
                logger.info("Stage 2: Extracting demographics")
                demographics_dict = self.loader.extract_demographics(dataset)

                # 3. Apply preprocessing
                logger.info("Stage 3: Applying preprocessing")
                preprocessed = self.preprocessor.process_dataset(dataset)

                # 4. Create windows
                logger.info("Stage 4: Creating windows")
                windows = self.window_creator.create_windows(preprocessed)

                # 5. Extract data arrays
                logger.info("Stage 5: Extracting data arrays")
                X, y, subjects = self._extract_arrays(windows)

                # 6. Process demographics
                demographics = self._process_demographics(demographics_dict, subjects)

            # 7. Quality control
            logger.info("Stage 6: Quality control")
            if not self.quality_controller.validate_data(X, y):
                raise ValueError("Data validation failed")

            metrics = self.quality_controller.compute_metrics(X)
            logger.info(f"Quality metrics: {metrics}")

            # 8. Split data
            logger.info("Stage 7: Splitting data")
            split_data = self.splitter.split_data(X, y, demographics, subjects)

            # 9. Save data
            if not self.config.dry_run:
                logger.info("Stage 8: Saving processed data")
                self._save_data(X, y, subjects, demographics, split_data, metrics)

            # Report completion
            elapsed_time = datetime.now() - start_time
            logger.info(f"Processing completed in {elapsed_time}")
            logger.info(f"Processed {len(X)} samples from {len(np.unique(subjects))} subjects")

            return True

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def _extract_arrays(self, windows) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract numpy arrays from windows dataset"""
        X_list = []
        y_list = []
        subjects_list = []

        for i in range(len(windows)):
            x, y = windows[i]
            X_list.append(x)
            y_list.append(y)

            # Extract subject ID from metadata
            if hasattr(windows.datasets[i], 'description'):
                subject = windows.datasets[i].description.get('subject', i)
            else:
                subject = i
            subjects_list.append(subject)

        X = np.stack(X_list)
        y = np.array(y_list)
        subjects = np.array(subjects_list)

        return X, y, subjects

    def _process_demographics(self, demographics_dict: Dict[str, List],
                            subjects: np.ndarray) -> Optional[np.ndarray]:
        """Process demographics into array format"""
        if not self.config.extract_demographics:
            return None

        # Convert to DataFrame for easier processing
        demo_df = pd.DataFrame(demographics_dict)

        # Encode categorical variables
        categorical_cols = ['sex', 'handedness', 'diagnosis', 'medication']
        for col in categorical_cols:
            if col in demo_df.columns:
                demo_df[col] = pd.Categorical(demo_df[col]).codes

        # Select numeric columns
        numeric_cols = ['age', 'p_factor', 'adhd_status',
                       'anxiety_status', 'depression_status']

        # Create demographic matrix
        demo_matrix = []
        for col in numeric_cols:
            if col in demo_df.columns:
                values = demo_df[col].fillna(0).values
                demo_matrix.append(values)

        if demo_matrix:
            demographics = np.stack(demo_matrix, axis=1)

            # Align with subjects
            aligned_demographics = np.zeros((len(subjects), demographics.shape[1]))
            for i, subject in enumerate(subjects):
                if subject < len(demographics):
                    aligned_demographics[i] = demographics[subject]

            return aligned_demographics

        return None

    def _process_bids_demographics(self, demographics_list: List[Dict],
                                  subjects: np.ndarray) -> Optional[np.ndarray]:
        """Process demographics from BIDS parser into array format"""
        if not self.config.extract_demographics or not demographics_list:
            return None

        # Extract numeric values from demographic dictionaries
        demo_arrays = []
        for demo in demographics_list:
            if demo is None:
                demo_arrays.append([0, 0, 0, 0, 0])  # Default values
            else:
                # FIX Bug #4: Use correct column names
                values = [
                    float(demo.get('age', 0)) if not pd.isna(demo.get('age')) else 0,
                    float(demo.get('p_factor', 0)) if not pd.isna(demo.get('p_factor')) else 0,
                    float(demo.get('attention', 0)) if not pd.isna(demo.get('attention')) else 0,
                    float(demo.get('internalizing', 0)) if not pd.isna(demo.get('internalizing')) else 0,
                    float(demo.get('externalizing', 0)) if not pd.isna(demo.get('externalizing')) else 0,
                ]
                demo_arrays.append(values)

        if demo_arrays:
            return np.array(demo_arrays, dtype=np.float32)

        return None

    def _save_data(self, X: np.ndarray, y: np.ndarray,
                  subjects: np.ndarray, demographics: Optional[np.ndarray],
                  split_data: Dict[str, np.ndarray],
                  metrics: Dict[str, float]):
        """Save processed data"""
        output_path = Path(self.config.output_dir) / f"r5_processed.{self.config.output_format}"

        # Prepare save dict
        save_dict = {
            # Full data
            'X': X,
            'y': y,
            'subjects': subjects,

            # Split data
            **split_data,

            # Metadata
            'sampling_rate': self.config.sampling_rate,
            'window_length': self.config.window_length,
            'window_stride': self.config.window_stride,
            'preprocessing_info': {
                'pipeline': self.config.preprocessing_pipeline,
                'bandpass': [self.config.bandpass_low, self.config.bandpass_high],
                'notch_freqs': self.config.notch_freqs,
                'reference': 'average',
            },
            'quality_metrics': metrics,
            'channel_names': ['CH' + str(i) for i in range(X.shape[1])],  # Placeholder
        }

        # Add demographics if available
        if demographics is not None:
            save_dict['demographics'] = demographics
            # FIX Bug #4: Use correct demographic names
            save_dict['demographic_names'] = ['age', 'p_factor', 'attention',
                                             'internalizing', 'externalizing']

        # Save based on format
        if self.config.output_format == 'npz':
            np.savez_compressed(output_path, **save_dict)
        elif self.config.output_format == 'hdf5':
            import h5py
            with h5py.File(output_path, 'w') as f:
                for key, value in save_dict.items():
                    if isinstance(value, dict):
                        grp = f.create_group(key)
                        for k, v in value.items():
                            grp.create_dataset(k, data=v)
                    else:
                        f.create_dataset(key, data=value)
        else:
            raise ValueError(f"Unknown output format: {self.config.output_format}")

        logger.info(f"Data saved to {output_path}")

        # Save metadata if requested
        if self.config.save_metadata:
            metadata_path = Path(self.config.output_dir) / "preprocessing_metadata.json"
            metadata = {
                'config': self.config.__dict__,
                'metrics': metrics,
                'data_shape': {
                    'X': X.shape,
                    'y': y.shape,
                    'n_subjects': len(np.unique(subjects)),
                },
                'timestamp': datetime.now().isoformat(),
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Metadata saved to {metadata_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Professional EEG Preprocessing Pipeline for R5 BDF Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to BIDS format data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for processed data')

    # Task configuration
    parser.add_argument('--task', type=str, default='contrastChangeDetection',
                       choices=['contrastChangeDetection', 'RestingState',
                               'surroundSupp', 'symbolSearch', 'all'],
                       help='Task to process')
    parser.add_argument('--release', type=str, default='R5',
                       choices=['R1', 'R2', 'R3', 'R4', 'R5'],
                       help='Data release version')

    # Window parameters (no sampling rate - fixed at 100 Hz)
    parser.add_argument('--window-length', type=float, default=2.0,
                       help='Window length in seconds')
    parser.add_argument('--window-stride', type=float, default=1.0,
                       help='Window stride in seconds')

    # Filtering parameters
    parser.add_argument('--bandpass-low', type=float, default=0.5,
                       help='Lower frequency for bandpass filter')
    parser.add_argument('--bandpass-high', type=float, default=45,
                       help='Upper frequency for bandpass filter')

    # Demographics
    parser.add_argument('--extract-demographics', action='store_true', default=True,
                       help='Extract demographic information')

    # Preprocessing pipeline
    parser.add_argument('--preprocessing-pipeline', type=str, default='standard',
                       choices=['minimal', 'standard', 'advanced', 'custom'],
                       help='Preprocessing pipeline to use')

    # Quality control
    parser.add_argument('--min-trials', type=int, default=10,
                       help='Minimum trials per subject')
    parser.add_argument('--max-bad-channels', type=float, default=0.2,
                       help='Maximum proportion of bad channels')
    parser.add_argument('--reject-amplitude', type=float, default=200,
                       help='Amplitude threshold for rejection (μV)')

    # Performance options
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing')
    parser.add_argument('--memory-efficient', action='store_true',
                       help='Use memory-efficient processing')

    # Output options
    parser.add_argument('--output-format', type=str, default='npz',
                       choices=['npz', 'hdf5', 'zarr'],
                       help='Output file format')
    parser.add_argument('--split-ratio', type=str, default='70:15:15',
                       help='Train:Val:Test split ratio')
    # FIX Bug #5: Save metadata by default
    parser.add_argument('--save-metadata', action='store_true', default=True,
                       help='Save preprocessing metadata (default: True)')
    parser.add_argument('--no-save-metadata', dest='save_metadata', action='store_false',
                       help='Do not save preprocessing metadata')

    # Debugging options
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (process only 5 subjects)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform dry run without saving')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create config
    config = PreprocessingConfig(**vars(args))

    # Run processing
    processor = R5BDFProcessor(config)
    processor.process()


if __name__ == '__main__':
    main()