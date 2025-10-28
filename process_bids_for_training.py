#!/usr/bin/env python3
"""
Hybrid BIDS EEG Preprocessing for Training
==========================================
Combines directory parsing from preprocess_multichannel_npz.py with
time-domain processing from process_bdf.py to create training data
in the format expected by train_optimal_model.py

Output format: (n_samples, 129, 200) with response time labels

Author: Carter
Date: October 2025
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import mne

warnings.filterwarnings('ignore')


class BIDSDataProcessor:
    """Process BIDS-formatted EEG data for training"""

    def __init__(self,
                 data_folder: str,
                 output_file: str,
                 task: str = 'contrastChangeDetection',
                 sampling_rate: int = 100,
                 window_length: float = 2.0,
                 shift_after_stim: float = 0.5,
                 min_rt: float = 0.1,
                 max_rt: float = 3.0,
                 bandpass_low: float = 0.5,
                 bandpass_high: float = 45.0,
                 apply_filtering: bool = True,
                 debug: bool = False):
        """
        Initialize BIDS data processor

        Args:
            data_folder: Root folder containing BIDS datasets
            output_file: Output NPZ file path
            task: Task name to process (default: contrastChangeDetection)
            sampling_rate: Expected sampling rate (100 Hz for challenge)
            window_length: Epoch length in seconds (default: 2.0)
            shift_after_stim: Shift after stimulus onset in seconds (default: 0.5)
            min_rt: Minimum valid reaction time (default: 0.1)
            max_rt: Maximum valid reaction time (default: 3.0)
            bandpass_low: Low cutoff for bandpass filter (default: 0.5 Hz)
            bandpass_high: High cutoff for bandpass filter (default: 45 Hz)
            apply_filtering: Whether to apply bandpass filtering (default: True)
            debug: Debug mode - process fewer subjects (default: False)
        """
        self.data_folder = Path(data_folder)
        self.output_file = Path(output_file)
        self.task = task
        self.sampling_rate = sampling_rate
        self.window_length = window_length
        self.shift_after_stim = shift_after_stim
        self.min_rt = min_rt
        self.max_rt = max_rt
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.apply_filtering = apply_filtering
        self.debug = debug

        self.processed_subjects = []
        self.failed_subjects = []

    def discover_bids_datasets(self) -> List[Path]:
        """Discover all BIDS datasets in the data folder"""
        print(f"Scanning for BIDS datasets in: {self.data_folder}")
        bids_datasets = []

        # Check if data_folder itself is a BIDS dataset
        if self._is_bids_dataset(self.data_folder):
            bids_datasets.append(self.data_folder)
        else:
            # Look for BIDS datasets in subdirectories
            for path in sorted(self.data_folder.iterdir()):
                if path.is_dir() and self._is_bids_dataset(path):
                    bids_datasets.append(path)

        if not bids_datasets:
            raise ValueError(
                f"No BIDS datasets found in {self.data_folder}\n"
                f"Expected BIDS format with sub-XXX/ directories"
            )

        print(f"Found {len(bids_datasets)} BIDS dataset(s): {[d.name for d in bids_datasets]}")
        return bids_datasets

    def _is_bids_dataset(self, path: Path) -> bool:
        """Check if path is a BIDS dataset"""
        has_dataset_desc = (path / "dataset_description.json").exists()
        has_participants = (path / "participants.tsv").exists()
        has_subjects = any(p.is_dir() and p.name.startswith("sub-")
                          for p in path.iterdir() if p.is_dir())
        return (has_dataset_desc or has_participants) and has_subjects

    def discover_subjects(self, dataset_path: Path) -> List[str]:
        """Discover all subjects in a BIDS dataset"""
        subjects = []
        for path in sorted(dataset_path.iterdir()):
            if path.is_dir() and path.name.startswith("sub-"):
                subjects.append(path.name)
        return subjects

    def process_all_datasets(self) -> Dict[str, np.ndarray]:
        """Process all BIDS datasets and combine data"""
        bids_datasets = self.discover_bids_datasets()

        all_X = []
        all_y = []
        all_subjects = []
        all_metadata = []

        for dataset_path in bids_datasets:
            print(f"\nProcessing dataset: {dataset_path.name}")
            subjects = self.discover_subjects(dataset_path)
            print(f"Found {len(subjects)} subjects")

            if self.debug:
                subjects = subjects[:5]
                print(f"Debug mode: processing only {len(subjects)} subjects")

            # Process each subject
            for subject_id in tqdm(subjects, desc=f"Processing {dataset_path.name}"):
                try:
                    subject_data = self.process_subject(dataset_path, subject_id)
                    if subject_data is not None:
                        all_X.append(subject_data['X'])
                        all_y.append(subject_data['y'])
                        # Track dataset in subject ID
                        full_subject_id = f"{dataset_path.name}_{subject_id}"
                        all_subjects.extend([full_subject_id] * len(subject_data['X']))
                        all_metadata.extend(subject_data.get('metadata', []))
                        self.processed_subjects.append(full_subject_id)
                except Exception as e:
                    print(f"  Error processing {subject_id}: {e}")
                    self.failed_subjects.append((subject_id, str(e)))

        if not all_X:
            raise ValueError("No data was successfully processed!")

        # Concatenate all data
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        subjects = np.array(all_subjects)

        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total samples: {len(X):,}")
        print(f"Subjects processed: {len(self.processed_subjects)}")
        print(f"Subjects failed: {len(self.failed_subjects)}")
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Response time range: [{y.min():.3f}, {y.max():.3f}] seconds")
        print(f"{'='*70}\n")

        return {
            'X': X,
            'y': y,
            'subjects': subjects,
            'metadata': {
                'processed_subjects': self.processed_subjects,
                'failed_subjects': self.failed_subjects,
                'task': self.task,
                'sampling_rate': self.sampling_rate,
                'window_length': self.window_length
            }
        }

    def process_subject(self, dataset_path: Path, subject_id: str) -> Optional[Dict]:
        """Process a single subject's BDF/EDF files"""
        subject_dir = dataset_path / subject_id / "eeg"

        if not subject_dir.exists():
            return None

        # Find EEG files for the task (support .bdf, .edf, .set)
        task_pattern = f"*task-{self.task}*_eeg.*"
        eeg_files = []
        for ext in ['.bdf', '.edf', '.set']:
            eeg_files.extend(subject_dir.glob(f"*task-{self.task}*_eeg{ext}"))
        eeg_files = sorted(eeg_files)

        if not eeg_files:
            return None

        all_X = []
        all_y = []
        all_metadata = []

        for eeg_file in eeg_files:
            try:
                # Load EEG file
                ext = eeg_file.suffix.lower()
                if ext == '.bdf':
                    raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)
                elif ext == '.edf':
                    raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
                elif ext == '.set':
                    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
                else:
                    continue

                # Apply filtering if requested
                if self.apply_filtering:
                    raw.filter(
                        l_freq=self.bandpass_low,
                        h_freq=self.bandpass_high,
                        verbose=False
                    )

                # Check sampling rate
                if raw.info['sfreq'] != self.sampling_rate:
                    # Resample if needed
                    raw.resample(self.sampling_rate, verbose=False)

                # Find events file
                events_file = self._find_events_file(eeg_file)
                if events_file is None:
                    continue

                # Extract epochs
                X_run, y_run, metadata_run = self._extract_epochs(raw, events_file)
                if X_run is not None and len(X_run) > 0:
                    all_X.append(X_run)
                    all_y.append(y_run)
                    all_metadata.extend(metadata_run)

            except Exception as e:
                print(f"    Warning: Failed to process {eeg_file.name}: {e}")
                continue

        if all_X:
            X = np.concatenate(all_X, axis=0)
            y = np.concatenate(all_y, axis=0)

            # Ensure correct number of channels (129 for challenge)
            if X.shape[1] != 129:
                if X.shape[1] > 129:
                    X = X[:, :129, :]
                else:
                    # Pad with zeros
                    pad_width = ((0, 0), (0, 129 - X.shape[1]), (0, 0))
                    X = np.pad(X, pad_width, mode='constant')

            return {
                'X': X,
                'y': y,
                'metadata': all_metadata
            }

        return None

    def _find_events_file(self, eeg_file: Path) -> Optional[Path]:
        """Find corresponding events.tsv file"""
        # Replace _eeg.{ext} with _events.tsv
        base_name = eeg_file.name
        for ext in ['.bdf', '.edf', '.set']:
            base_name = base_name.replace(f'_eeg{ext}', '')
        events_file = eeg_file.parent / f"{base_name}_events.tsv"
        return events_file if events_file.exists() else None

    def _extract_epochs(self, raw: mne.io.Raw, events_file: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List]:
        """Extract epochs from raw data using events file"""
        # Load events
        events_df = pd.read_csv(events_file, sep='\t')

        # Build trial table
        trials = self._build_trial_table(events_df)

        # Filter valid trials
        trials = trials[
            (trials['stimulus_onset'].notna()) &
            (trials['response_onset'].notna()) &
            (trials['rt_from_stimulus'] >= self.min_rt) &
            (trials['rt_from_stimulus'] <= self.max_rt)
        ]

        if trials.empty:
            return None, None, []

        # Extract epochs
        X = []
        y = []
        metadata = []

        expected_samples = int(self.window_length * self.sampling_rate)

        for _, trial in trials.iterrows():
            try:
                # Calculate epoch timing
                epoch_start = trial['stimulus_onset'] + self.shift_after_stim
                epoch_end = epoch_start + self.window_length

                start_sample = int(epoch_start * self.sampling_rate)
                end_sample = int(epoch_end * self.sampling_rate)

                # Extract data segment
                data, times = raw[:, start_sample:end_sample]

                # Ensure correct length
                if data.shape[1] == expected_samples:
                    X.append(data)
                    y.append(trial['rt_from_stimulus'])
                    metadata.append({
                        'correct': trial.get('correct', None),
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

            # Find stimulus and response in this trial window
            trial_stimuli = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
            trial_responses = responses[(responses["onset"] >= start) & (responses["onset"] < end)]

            if not trial_stimuli.empty and not trial_responses.empty:
                stim = trial_stimuli.iloc[0]
                resp = trial_responses.iloc[0]

                stim_onset = float(stim["onset"])
                resp_onset = float(resp["onset"])
                rt = resp_onset - stim_onset

                # Check if correct response
                stim_side = "left" if "left" in str(stim.get("value", stim.get("trial_type", ""))) else "right"
                resp_side = "left" if "left" in str(resp.get("value", resp.get("trial_type", ""))) else "right"
                correct = stim_side == resp_side

                rows.append({
                    'trial_start_onset': start,
                    'stimulus_onset': stim_onset,
                    'response_onset': resp_onset,
                    'rt_from_stimulus': rt,
                    'correct': correct,
                    'response_type': resp_side
                })

        return pd.DataFrame(rows)

    def save_data(self, data: Dict[str, np.ndarray]):
        """Save processed data to NPZ file"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving data to: {self.output_file}")
        np.savez_compressed(
            self.output_file,
            X=data['X'],
            y=data['y'],
            subjects=data['subjects'],
            metadata=data['metadata']
        )
        print(f"Data saved successfully!")

        # Print file size
        file_size = self.output_file.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Process BIDS EEG data for training with train_optimal_model.py"
    )
    parser.add_argument(
        '--data-folder',
        type=str,
        required=True,
        help='Root folder containing BIDS datasets (e.g., data/)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Output NPZ file path (e.g., processed_data/r5_full_data.npz)'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='contrastChangeDetection',
        help='Task name to process (default: contrastChangeDetection)'
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Skip bandpass filtering'
    )
    parser.add_argument(
        '--bandpass-low',
        type=float,
        default=0.5,
        help='Low cutoff for bandpass filter in Hz (default: 0.5)'
    )
    parser.add_argument(
        '--bandpass-high',
        type=float,
        default=45.0,
        help='High cutoff for bandpass filter in Hz (default: 45.0)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode - process only 5 subjects per dataset'
    )

    args = parser.parse_args()

    # Create processor
    processor = BIDSDataProcessor(
        data_folder=args.data_folder,
        output_file=args.output_file,
        task=args.task,
        apply_filtering=not args.no_filter,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        debug=args.debug
    )

    try:
        # Process all datasets
        data = processor.process_all_datasets()

        # Save data
        processor.save_data(data)

        print("\nProcessing complete!")
        print(f"\nYou can now train with:")
        print(f"  python3 train_optimal_model.py --data-path {args.output_file}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
