#!/usr/bin/env python3
"""
Professional BDF to NPZ Converter for R5 L100 Dataset
=====================================================

This script processes the R5 L100 BDF format EEG data into the standardised
NPZ format required for the EEG Challenge 2025. It implements the exact
preprocessing pipeline from the challenge notebook while handling large-scale
data efficiently.

Author: Machine Learning Research Group
Date: October 2025
Version: 1.0.0
"""

import os
import sys
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
from typing import Tuple, List, Dict, Optional
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split

# Configure environment
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


class R5BDFProcessor:
    """
    Professional-grade processor for R5 BDF format EEG data.

    This class implements the complete processing pipeline for converting
    BDF files to the challenge-compliant NPZ format, including proper
    epoching, artifact handling, and quality control.
    """

    def __init__(self,
                 bdf_dir: str = "R5_L100_BDF",
                 output_file: str = "r5_l100_processed.npz",
                 sfreq: int = 100,
                 n_channels: int = 129,
                 epoch_length: float = 2.0,
                 max_subjects: Optional[int] = None):
        """
        Initialise the BDF processor.

        Args:
            bdf_dir: Directory containing BDF files in BIDS structure
            output_file: Output NPZ file path
            sfreq: Target sampling frequency (100 Hz for challenge)
            n_channels: Number of channels (129 for challenge)
            epoch_length: Length of epochs in seconds (2s for challenge)
            max_subjects: Maximum number of subjects to process (None for all)
        """
        self.bdf_dir = Path(bdf_dir)
        self.output_file = Path(output_file)
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.epoch_length = epoch_length
        self.max_subjects = max_subjects

        # Processing parameters from challenge notebook
        self.shift_after_stim = 0.5  # Shift after stimulus onset
        self.window_len = 2.0  # Window length

        # Quality control thresholds
        self.min_trials_per_subject = 10
        self.max_response_time = 3.0
        self.min_response_time = 0.1

        # Initialise counters
        self.processed_subjects = []
        self.failed_subjects = []
        self.total_trials = 0

    def process_dataset(self) -> Dict:
        """
        Main processing pipeline for the complete dataset.

        Returns:
            Dictionary containing processing statistics
        """
        print("="*70)
        print("R5 L100 BDF Dataset Processing Pipeline")
        print("="*70)
        print(f"Source directory: {self.bdf_dir}")
        print(f"Output file: {self.output_file}")

        # 1. Discover subjects
        subjects = self._discover_subjects()
        print(f"\n1. Subject Discovery")
        print(f"   Found {len(subjects)} subjects")

        if self.max_subjects:
            subjects = subjects[:self.max_subjects]
            print(f"   Processing first {len(subjects)} subjects")

        # 2. Process each subject
        print(f"\n2. Processing Subjects")
        all_data = []
        all_labels = []
        all_subjects = []
        all_metadata = []

        for subj_idx, subject_id in enumerate(tqdm(subjects, desc="Processing subjects")):
            try:
                X, y, metadata = self._process_subject(subject_id)

                if X is not None and len(X) >= self.min_trials_per_subject:
                    all_data.append(X)
                    all_labels.append(y)
                    all_subjects.extend([subject_id] * len(X))
                    all_metadata.extend(metadata)
                    self.processed_subjects.append(subject_id)
                    self.total_trials += len(X)
                else:
                    self.failed_subjects.append((subject_id, "Insufficient trials"))

            except Exception as e:
                self.failed_subjects.append((subject_id, str(e)))
                if len(self.failed_subjects) % 10 == 0:
                    print(f"\n   Warning: {len(self.failed_subjects)} subjects failed so far")

            # Memory management
            if (subj_idx + 1) % 50 == 0:
                gc.collect()

        # 3. Combine all data
        print(f"\n3. Combining Data")
        X_all = np.concatenate(all_data, axis=0)
        y_all = np.concatenate(all_labels, axis=0)
        subjects_all = np.array(all_subjects)

        print(f"   Total shape: X={X_all.shape}, y={y_all.shape}")
        print(f"   Response time range: [{y_all.min():.3f}, {y_all.max():.3f}] seconds")

        # 4. Create train/val/test splits
        print(f"\n4. Creating Data Splits")
        splits = self._create_splits(X_all, y_all, subjects_all)

        # 5. Save to NPZ
        print(f"\n5. Saving Processed Data")
        self._save_npz(X_all, y_all, subjects_all, splits, all_metadata)

        # 6. Generate report
        stats = self._generate_report()

        return stats

    def _discover_subjects(self) -> List[str]:
        """Discover all subject directories."""
        subjects = []
        for path in sorted(self.bdf_dir.iterdir()):
            if path.is_dir() and path.name.startswith("sub-"):
                subjects.append(path.name)
        return subjects

    def _process_subject(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Process a single subject's data.

        Args:
            subject_id: Subject identifier

        Returns:
            Tuple of (X, y, metadata) where X is EEG data, y is response times
        """
        subject_dir = self.bdf_dir / subject_id / "eeg"

        # Find all CCD task files
        ccd_files = sorted(subject_dir.glob("*task-contrastChangeDetection*.bdf"))

        if not ccd_files:
            return None, None, []

        all_X = []
        all_y = []
        all_metadata = []

        for bdf_file in ccd_files:
            # Load BDF and events
            raw = mne.io.read_raw_bdf(bdf_file, preload=False, verbose=False)

            # Resample if necessary
            if raw.info['sfreq'] != self.sfreq:
                raw.resample(self.sfreq, npad='auto', verbose=False)

            # Load events
            events_file = bdf_file.with_suffix('.tsv').name.replace('_eeg.tsv', '_events.tsv')
            events_path = subject_dir / events_file

            if not events_path.exists():
                continue

            # Process events and create epochs
            X_run, y_run, meta_run = self._process_run(raw, events_path)

            if X_run is not None:
                all_X.append(X_run)
                all_y.append(y_run)
                all_metadata.extend(meta_run)

        if all_X:
            X = np.concatenate(all_X, axis=0)
            y = np.concatenate(all_y, axis=0)

            # Ensure correct number of channels
            if X.shape[1] != self.n_channels:
                # Handle channel count mismatch
                if X.shape[1] > self.n_channels:
                    X = X[:, :self.n_channels, :]
                else:
                    # Pad with zeros if fewer channels
                    pad_width = ((0, 0), (0, self.n_channels - X.shape[1]), (0, 0))
                    X = np.pad(X, pad_width, mode='constant')

            return X.astype(np.float32), y.astype(np.float32), all_metadata

        return None, None, []

    def _process_run(self, raw: mne.io.Raw, events_path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Process a single run (recording) following the challenge notebook approach.

        Args:
            raw: MNE Raw object
            events_path: Path to events TSV file

        Returns:
            Tuple of (X, y, metadata)
        """
        # Load events
        events_df = pd.read_csv(events_path, sep='\t')

        # Build trial table (from challenge notebook)
        trials = self._build_trial_table(events_df)

        # Filter valid trials
        trials = trials[
            (trials['stimulus_onset'].notna()) &
            (trials['response_onset'].notna()) &
            (trials['rt_from_stimulus'] >= self.min_response_time) &
            (trials['rt_from_stimulus'] <= self.max_response_time)
        ]

        if trials.empty:
            return None, None, []

        # Create epochs
        X = []
        y = []
        metadata = []

        for _, trial in trials.iterrows():
            # Calculate epoch timing (stimulus + 0.5s shift)
            epoch_start = trial['stimulus_onset'] + self.shift_after_stim
            epoch_end = epoch_start + self.epoch_length

            # Extract epoch
            try:
                start_sample = int(epoch_start * self.sfreq)
                end_sample = int(epoch_end * self.sfreq)

                # Load data for this segment
                raw_copy = raw.copy()
                raw_copy.load_data()
                data, times = raw_copy[:, start_sample:end_sample]

                # Ensure correct length
                if data.shape[1] == int(self.epoch_length * self.sfreq):
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
        """
        Build trial table from events (from challenge notebook).

        Args:
            events_df: Events dataframe

        Returns:
            Trial table with stimulus/response information
        """
        events_df = events_df.copy()
        events_df["onset"] = pd.to_numeric(events_df["onset"], errors="coerce")
        events_df = events_df.sort_values("onset").reset_index(drop=True)

        # Extract trial starts
        trials = events_df[events_df["value"].eq("contrastTrial_start")].copy()

        if trials.empty:
            # Try alternative naming
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

    def _create_splits(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Dict:
        """
        Create train/val/test splits at the subject level.

        Args:
            X: EEG data
            y: Response times
            subjects: Subject IDs for each sample

        Returns:
            Dictionary containing split indices and data
        """
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)

        # Split ratios
        test_ratio = 0.15
        val_ratio = 0.15

        # Subject-level splits
        train_val_subj, test_subj = train_test_split(
            unique_subjects,
            test_size=test_ratio,
            random_state=42
        )

        train_subj, val_subj = train_test_split(
            train_val_subj,
            test_size=val_ratio / (1 - test_ratio),
            random_state=43
        )

        # Create masks
        train_mask = np.isin(subjects, train_subj)
        val_mask = np.isin(subjects, val_subj)
        test_mask = np.isin(subjects, test_subj)

        # Split data
        splits = {
            'X_train': X[train_mask],
            'y_train': y[train_mask],
            'X_val': X[val_mask],
            'y_val': y[val_mask],
            'X_test': X[test_mask],
            'y_test': y[test_mask],
            'train_subjects': train_subj,
            'val_subjects': val_subj,
            'test_subjects': test_subj,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }

        print(f"   Train: {len(splits['X_train']):,} samples from {len(train_subj)} subjects")
        print(f"   Val: {len(splits['X_val']):,} samples from {len(val_subj)} subjects")
        print(f"   Test: {len(splits['X_test']):,} samples from {len(test_subj)} subjects")

        return splits

    def _save_npz(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                  splits: Dict, metadata: List[Dict]):
        """
        Save processed data to NPZ file.

        Args:
            X: Full dataset
            y: Full labels
            subjects: Subject IDs
            splits: Train/val/test splits
            metadata: Additional metadata
        """
        # Prepare save dictionary
        save_dict = {
            'X': X,
            'y': y,
            'subjects': subjects,
            'X_train': splits['X_train'],
            'y_train': splits['y_train'],
            'X_val': splits['X_val'],
            'y_val': splits['y_val'],
            'X_test': splits['X_test'],
            'y_test': splits['y_test'],
            'train_subjects': splits['train_subjects'],
            'val_subjects': splits['val_subjects'],
            'test_subjects': splits['test_subjects']
        }

        # Save main data
        np.savez_compressed(self.output_file, **save_dict)
        print(f"   Saved to: {self.output_file}")
        print(f"   File size: {self.output_file.stat().st_size / (1024**2):.1f} MB")

        # Save metadata separately
        metadata_file = self.output_file.with_suffix('.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'metadata': metadata,
                'processed_subjects': self.processed_subjects,
                'failed_subjects': self.failed_subjects,
                'processing_date': datetime.now().isoformat(),
                'total_trials': self.total_trials,
                'parameters': {
                    'sfreq': self.sfreq,
                    'n_channels': self.n_channels,
                    'epoch_length': self.epoch_length,
                    'shift_after_stim': self.shift_after_stim
                }
            }, f)
        print(f"   Metadata saved to: {metadata_file}")

    def _generate_report(self) -> Dict:
        """Generate processing report."""
        stats = {
            'total_subjects_found': len(self._discover_subjects()),
            'subjects_processed': len(self.processed_subjects),
            'subjects_failed': len(self.failed_subjects),
            'total_trials': self.total_trials,
            'avg_trials_per_subject': self.total_trials / max(len(self.processed_subjects), 1),
            'output_file': str(self.output_file)
        }

        print("\n" + "="*70)
        print("Processing Report")
        print("="*70)
        print(f"Subjects processed: {stats['subjects_processed']}")
        print(f"Subjects failed: {stats['subjects_failed']}")
        print(f"Total trials: {stats['total_trials']:,}")
        print(f"Average trials per subject: {stats['avg_trials_per_subject']:.1f}")

        if self.failed_subjects:
            print(f"\nFirst 5 failed subjects:")
            for subj, reason in self.failed_subjects[:5]:
                print(f"  {subj}: {reason[:50]}")

        print("="*70)

        return stats


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process R5 L100 BDF files to NPZ format for EEG Challenge 2025",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--bdf-dir',
        type=str,
        default='R5_L100_BDF',
        help='Directory containing BDF files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='r5_l100_processed.npz',
        help='Output NPZ file path'
    )
    parser.add_argument(
        '--max-subjects',
        type=int,
        default=None,
        help='Maximum number of subjects to process (None for all)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only 5 subjects'
    )

    args = parser.parse_args()

    # Test mode
    if args.test:
        args.max_subjects = 5
        args.output = 'r5_l100_test.npz'
        print("Running in TEST mode (5 subjects only)")

    # Check if BDF directory exists
    if not Path(args.bdf_dir).exists():
        print(f"Error: BDF directory not found: {args.bdf_dir}")
        sys.exit(1)

    # Initialise processor
    processor = R5BDFProcessor(
        bdf_dir=args.bdf_dir,
        output_file=args.output,
        max_subjects=args.max_subjects
    )

    # Process dataset
    try:
        stats = processor.process_dataset()

        # Test loading the saved file
        print("\nTesting saved file...")
        data = np.load(args.output)
        print(f"Successfully loaded: {args.output}")
        print(f"  Keys: {list(data.keys())}")
        print(f"  Full dataset: X={data['X'].shape}, y={data['y'].shape}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()