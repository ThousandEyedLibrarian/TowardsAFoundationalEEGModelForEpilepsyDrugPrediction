#!/usr/bin/env python3
"""
Efficient R5 dataset downloader for EEG Challenge 2025
Downloads and processes CCD data in batches to avoid memory issues
Based on challenge_1.ipynb implementation
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
import torch
from tqdm import tqdm
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# Import EEGDash
from eegdash.dataset import EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from mne_bids import get_bids_path_from_fname


# ===== Epoching utilities from challenge notebook =====

def build_trial_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """One row per contrast trial with stimulus/response metrics."""
    events_df = events_df.copy()
    events_df["onset"] = pd.to_numeric(events_df["onset"], errors="raise")
    events_df = events_df.sort_values("onset", kind="mergesort").reset_index(drop=True)

    trials = events_df[events_df["value"].eq("contrastTrial_start")].copy()
    stimuli = events_df[events_df["value"].isin(["left_target", "right_target"])].copy()
    responses = events_df[events_df["value"].isin(["left_buttonPress", "right_buttonPress"])].copy()

    trials = trials.reset_index(drop=True)
    trials["next_onset"] = trials["onset"].shift(-1)
    trials = trials.dropna(subset=["next_onset"]).reset_index(drop=True)

    rows = []
    for _, tr in trials.iterrows():
        start = float(tr["onset"])
        end   = float(tr["next_onset"])

        stim_block = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
        stim_onset = np.nan if stim_block.empty else float(stim_block.iloc[0]["onset"])

        if not np.isnan(stim_onset):
            resp_block = responses[(responses["onset"] >= stim_onset) & (responses["onset"] < end)]
        else:
            resp_block = responses[(responses["onset"] >= start) & (responses["onset"] < end)]

        if resp_block.empty:
            resp_onset = np.nan
            resp_type  = None
            feedback   = None
        else:
            resp_onset = float(resp_block.iloc[0]["onset"])
            resp_type  = resp_block.iloc[0]["value"]
            feedback   = resp_block.iloc[0]["feedback"]

        rt_from_stim  = (resp_onset - stim_onset) if (not np.isnan(stim_onset) and not np.isnan(resp_onset)) else np.nan
        rt_from_trial = (resp_onset - start)       if not np.isnan(resp_onset) else np.nan

        correct = None
        if isinstance(feedback, str):
            if feedback == "smiley_face": correct = True
            elif feedback == "sad_face":  correct = False

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


def _to_float_or_none(x):
    return None if pd.isna(x) else float(x)

def _to_int_or_none(x):
    if pd.isna(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        return int(x)
    except Exception:
        return None

def _to_str_or_none(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)


def annotate_trials_with_target(raw, target_field="rt_from_stimulus", epoch_length=2.0,
                                require_stimulus=True, require_response=True):
    """Create 'contrast_trial_start' annotations with float target in extras."""
    fnames = raw.filenames
    assert len(fnames) == 1, "Expected a single filename"
    bids_path = get_bids_path_from_fname(fnames[0])
    events_file = bids_path.update(suffix="events", extension=".tsv").fpath

    events_df = (pd.read_csv(events_file, sep="\t")
                   .assign(onset=lambda d: pd.to_numeric(d["onset"], errors="raise"))
                   .sort_values("onset", kind="mergesort").reset_index(drop=True))

    trials = build_trial_table(events_df)

    if require_stimulus:
        trials = trials[trials["stimulus_onset"].notna()].copy()
    if require_response:
        trials = trials[trials["response_onset"].notna()].copy()

    if target_field not in trials.columns:
        raise KeyError(f"{target_field} not in computed trial table.")
    targets = trials[target_field].astype(float)

    onsets     = trials["trial_start_onset"].to_numpy(float)
    durations  = np.full(len(trials), float(epoch_length), dtype=float)
    descs      = ["contrast_trial_start"] * len(trials)

    extras = []
    for i, v in enumerate(targets):
        row = trials.iloc[i]

        extras.append({
            "target": _to_float_or_none(v),
            "rt_from_stimulus": _to_float_or_none(row["rt_from_stimulus"]),
            "rt_from_trialstart": _to_float_or_none(row["rt_from_trialstart"]),
            "stimulus_onset": _to_float_or_none(row["stimulus_onset"]),
            "response_onset": _to_float_or_none(row["response_onset"]),
            "correct": _to_int_or_none(row["correct"]),
            "response_type": _to_str_or_none(row["response_type"]),
        })

    new_ann = mne.Annotations(onset=onsets, duration=durations, description=descs,
                              orig_time=raw.info["meas_date"], extras=extras)
    raw.set_annotations(new_ann, verbose=False)
    return raw


def add_aux_anchors(raw, stim_desc="stimulus_anchor", resp_desc="response_anchor"):
    ann = raw.annotations
    mask = (ann.description == "contrast_trial_start")
    if not np.any(mask):
        return raw

    stim_onsets, resp_onsets = [], []
    stim_extras, resp_extras = [], []

    for idx in np.where(mask)[0]:
        ex = ann.extras[idx] if ann.extras is not None else {}
        t0 = float(ann.onset[idx])

        stim_t = ex.get("stimulus_onset")
        resp_t = ex.get("response_onset")

        if stim_t is None or (isinstance(stim_t, float) and np.isnan(stim_t)):
            rtt = ex.get("rt_from_trialstart")
            rts = ex.get("rt_from_stimulus")
            if rtt is not None and rts is not None:
                stim_t = t0 + float(rtt) - float(rts)

        if resp_t is None or (isinstance(resp_t, float) and np.isnan(resp_t)):
            rtt = ex.get("rt_from_trialstart")
            if rtt is not None:
                resp_t = t0 + float(rtt)

        if (stim_t is not None) and not (isinstance(stim_t, float) and np.isnan(stim_t)):
            stim_onsets.append(float(stim_t))
            stim_extras.append(dict(ex, anchor="stimulus"))
        if (resp_t is not None) and not (isinstance(resp_t, float) and np.isnan(resp_t)):
            resp_onsets.append(float(resp_t))
            resp_extras.append(dict(ex, anchor="response"))

    new_onsets = np.array(stim_onsets + resp_onsets, dtype=float)
    if len(new_onsets):
        aux = mne.Annotations(
            onset=new_onsets,
            duration=np.zeros_like(new_onsets, dtype=float),
            description=[stim_desc]*len(stim_onsets) + [resp_desc]*len(resp_onsets),
            orig_time=raw.info["meas_date"],
            extras=stim_extras + resp_extras,
        )
        raw.set_annotations(ann + aux, verbose=False)
    return raw


def keep_only_recordings_with(desc, concat_ds):
    kept = []
    for ds in concat_ds.datasets:
        if np.any(ds.raw.annotations.description == desc):
            kept.append(ds)
    return BaseConcatDataset(kept)


def process_single_recording(dataset, idx, save_dir):
    """Process a single recording and save to disk"""
    try:
        # Get the raw data
        raw = dataset.datasets[idx].raw
        subject_id = dataset.datasets[idx].description.get('subject', f'subj_{idx}')

        print(f"  Processing subject {subject_id}...")

        # Apply preprocessing
        EPOCH_LEN_S = 2.0
        SFREQ = 100

        # Annotate trials
        raw = annotate_trials_with_target(
            raw,
            target_field="rt_from_stimulus",
            epoch_length=EPOCH_LEN_S,
            require_stimulus=True,
            require_response=True
        )
        raw = add_aux_anchors(raw)

        # Check if stimulus anchors exist
        if not np.any(raw.annotations.description == "stimulus_anchor"):
            print(f"    No stimulus anchors found for {subject_id}, skipping...")
            return None

        # Create windows
        ANCHOR = "stimulus_anchor"
        SHIFT_AFTER_STIM = 0.5
        WINDOW_LEN = 2.0

        # Create temporary dataset for this recording
        temp_ds = BaseConcatDataset([dataset.datasets[idx]])

        # Create windows
        windows = create_windows_from_events(
            temp_ds,
            mapping={ANCHOR: 0},
            trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
            trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
            window_size_samples=int(EPOCH_LEN_S * SFREQ),
            window_stride_samples=SFREQ,
            preload=True,
        )

        # Extract data and labels
        X = []
        y = []

        for window_idx in range(len(windows)):
            window = windows[window_idx]
            x_data = window[0]  # Shape: (129, 200)

            # Extract response time from metadata
            if hasattr(windows.datasets[0], 'windows_events'):
                events = windows.datasets[0].windows_events[window_idx]
                if len(events) > 0 and hasattr(events[0], 'extras'):
                    rt = events[0].extras.get('rt_from_stimulus', None)
                    if rt is not None and not np.isnan(rt):
                        X.append(x_data)
                        y.append(rt)

        if len(X) > 0:
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32).reshape(-1, 1)

            # Save to disk
            subject_file = save_dir / f"{subject_id}.npz"
            np.savez_compressed(subject_file, X=X, y=y)

            print(f"    Saved {len(X)} trials for {subject_id}")
            return len(X)
        else:
            print(f"    No valid trials for {subject_id}")
            return 0

    except Exception as e:
        print(f"    Error processing recording {idx}: {e}")
        return None


def download_r5_dataset(save_dir="r5_data", mini=False, max_recordings=None):
    """
    Download and process R5 dataset efficiently

    Args:
        save_dir: Directory to save processed data
        mini: Use mini release for testing
        max_recordings: Maximum number of recordings to process (None for all)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("EEG Challenge 2025 - R5 Dataset Download")
    print("="*60)

    # Create cache directory
    cache_dir = Path.home() / "eegdash_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\n1. Loading {'mini ' if mini else ''}R5 release...")
    dataset_ccd = EEGChallengeDataset(
        task="contrastChangeDetection",
        release="R5",
        cache_dir=cache_dir,
        mini=mini
    )

    n_recordings = len(dataset_ccd.datasets)
    print(f"   Found {n_recordings} CCD recordings")

    if max_recordings:
        n_recordings = min(n_recordings, max_recordings)
        print(f"   Processing first {n_recordings} recordings")

    # Process recordings one by one
    print(f"\n2. Processing recordings and saving to {save_dir}/...")

    total_trials = 0
    processed_subjects = []

    for idx in tqdm(range(n_recordings), desc="Processing recordings"):
        n_trials = process_single_recording(dataset_ccd, idx, save_dir)

        if n_trials is not None and n_trials > 0:
            total_trials += n_trials
            subject_id = dataset_ccd.datasets[idx].description.get('subject', f'subj_{idx}')
            processed_subjects.append(subject_id)

        # Clear memory
        gc.collect()

    # Save metadata
    metadata = {
        'n_subjects': len(processed_subjects),
        'n_trials': total_trials,
        'subjects': processed_subjects,
        'task': 'contrastChangeDetection',
        'release': 'R5' + ('_mini' if mini else ''),
        'n_channels': 129,
        'n_times': 200,
        'sfreq': 100
    }

    metadata_file = save_dir / "metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"Processed subjects: {len(processed_subjects)}")
    print(f"Total trials: {total_trials}")
    if len(processed_subjects) > 0:
        print(f"Average trials per subject: {total_trials/len(processed_subjects):.1f}")
    else:
        print("No valid trials extracted from any subjects")
    print(f"Data saved to: {save_dir}/")

    return metadata


def load_local_r5_data(data_dir="r5_data", test_size=0.2, val_size=0.1, seed=42):
    """
    Load locally saved R5 data and create train/val/test splits

    Args:
        data_dir: Directory containing processed data
        test_size: Fraction for test set
        val_size: Fraction for validation set
        seed: Random seed for splitting
    """
    data_dir = Path(data_dir)

    # Load metadata
    metadata_file = data_dir / "metadata.pkl"
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    print(f"Loading R5 data from {data_dir}/")
    print(f"  Subjects: {metadata['n_subjects']}")
    print(f"  Total trials: {metadata['n_trials']}")

    # Load all subject data
    X_all = []
    y_all = []
    subjects = []

    for subject_file in sorted(data_dir.glob("*.npz")):
        if subject_file.stem == "metadata":
            continue

        data = np.load(subject_file)
        X = data['X']
        y = data['y']

        X_all.append(X)
        y_all.append(y)
        subjects.extend([subject_file.stem] * len(X))

    # Combine all data
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    subjects = np.array(subjects)

    # Create subject-based splits
    unique_subjects = np.unique(subjects)
    np.random.seed(seed)
    np.random.shuffle(unique_subjects)

    n_subjects = len(unique_subjects)
    n_test = int(n_subjects * test_size)
    n_val = int(n_subjects * val_size)
    n_train = n_subjects - n_test - n_val

    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train+n_val]
    test_subjects = unique_subjects[n_train+n_val:]

    # Create masks
    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = np.isin(subjects, test_subjects)

    # Split data
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} trials from {len(train_subjects)} subjects")
    print(f"  Val: {len(X_val)} trials from {len(val_subjects)} subjects")
    print(f"  Test: {len(X_test)} trials from {len(test_subjects)} subjects")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download R5 dataset for EEG Challenge")
    parser.add_argument('--mini', action='store_true', help='Use mini release (faster)')
    parser.add_argument('--max_recordings', type=int, default=None,
                       help='Maximum recordings to process')
    parser.add_argument('--save_dir', type=str, default='r5_data',
                       help='Directory to save processed data')

    args = parser.parse_args()

    # Download dataset
    metadata = download_r5_dataset(
        save_dir=args.save_dir,
        mini=args.mini,
        max_recordings=args.max_recordings
    )

    # Test loading
    print("\nTesting data loading...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_local_r5_data(args.save_dir)

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    print(f"\nResponse time ranges:")
    print(f"  Train: [{y_train.min():.3f}, {y_train.max():.3f}] seconds")
    print(f"  Val: [{y_val.min():.3f}, {y_val.max():.3f}] seconds")
    print(f"  Test: [{y_test.min():.3f}, {y_test.max():.3f}] seconds")