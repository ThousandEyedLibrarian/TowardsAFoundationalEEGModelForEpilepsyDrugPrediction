import os
import shutil
from argparse import ArgumentParser
import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, sosfilt, butter, iirnotch, lfilter
from multiprocessing import Pool
from collections import deque
from meegkit import dss
import contextlib
import random


EEG_CHANNELS = [
    "EEG FP1-REF",
    "EEG FP2-REF",
    "EEG F3-REF",
    "EEG F4-REF",
    "EEG C3-REF",
    "EEG C4-REF",
    "EEG P3-REF",
    "EEG P4-REF",
    "EEG O1-REF",
    "EEG O2-REF",
    "EEG F7-REF",
    "EEG F8-REF",
    "EEG T3-REF",
    "EEG T4-REF",
    "EEG T5-REF",
    "EEG T6-REF",
    "EEG FZ-REF",
    "EEG CZ-REF",
    "EEG PZ-REF",
]

# Channel mapping from standard 10-20 to EGI 129-channel system
EGI_CHANNEL_MAP = {
    'EEG FP1-REF': 'E22',
    'EEG FP2-REF': 'E9',
    'EEG F3-REF': 'E24',
    'EEG F4-REF': 'E124',
    'EEG C3-REF': 'E36',
    'EEG C4-REF': 'E104',
    'EEG P3-REF': 'E52',
    'EEG P4-REF': 'E92',
    'EEG O1-REF': 'E70',
    'EEG O2-REF': 'E83',
    'EEG F7-REF': 'E33',
    'EEG F8-REF': 'E122',
    'EEG T3-REF': 'E45',
    'EEG T4-REF': 'E108',
    'EEG T5-REF': 'E58',
    'EEG T6-REF': 'E96',
    'EEG FZ-REF': 'E11',
    'EEG CZ-REF': 'Cz',
    'EEG PZ-REF': 'E62'
}

NUM_EEG_CHANNELS = len(EEG_CHANNELS)

EEG_FREQ_BANDS = dict(
    delta=[0.5, 4], alpha=[4, 8], theta=[8, 13], beta=[13, 30], gamma=[30, 70]
)

NUM_EEG_FREQ_BANDS = len(list(EEG_FREQ_BANDS.keys()))
EEG_FREQ_BAND_NAMES = list(EEG_FREQ_BANDS.keys())


class NumpyRingBuffer:
    def __init__(self, capacity: int, dtype=np.float32):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=dtype)
        self.index = 0
        self.full = False

    def __len__(self):
        """Current number of elements in the buffer."""
        return self.capacity if self.full else self.index

    def __iter__(self):
        """Iterate over buffer contents in order."""
        return iter(self.get())

    def append(self, value: float):
        """Append a single value to the buffer."""
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.capacity
        if not self.full and self.index == 0:
            self.full = True

    def extend(self, values: np.ndarray):
        """Append multiple values (1D array) to the buffer."""
        n = len(values)
        if n >= self.capacity:
            # If chunk is larger than capacity, keep only last part
            self.buffer[:] = values[-self.capacity :]
            self.index = 0
            self.full = True
        else:
            end = self.index + n
            if end <= self.capacity:
                self.buffer[self.index : end] = values
            else:
                # wrap around
                first = self.capacity - self.index
                self.buffer[self.index :] = values[:first]
                self.buffer[: end % self.capacity] = values[first:]
            self.index = (self.index + n) % self.capacity
            if not self.full and self.index == 0:
                self.full = True

    def get(self):
        """Return buffer contents in correct order (oldest → newest)."""
        if not self.full:
            return self.buffer[: self.index]
        return np.concatenate([self.buffer[self.index :], self.buffer[: self.index]])


def butter_bandpass_filter(data, lowcut: float, highcut: float, fs: int, order=4):
    sos = butter(order, [lowcut, highcut], btype="bandpass", output="sos", fs=fs)
    y = sosfilt(sos, data, axis=-1)
    return y


def butter_lowpass_filter(
    data: np.ndarray, lowcut: float, fs, order: int = 4
) -> np.ndarray:
    sos = butter(order, lowcut, fs=fs, btype="lowpass", output="sos")
    y = sosfilt(sos, data, axis=-1)
    return y


def butter_highpass_filter(
    data: np.ndarray, highcut: float, fs, order: int = 4
) -> np.ndarray:
    sos = butter(order, highcut, fs=fs, btype="highpass", output="sos")
    y = sosfilt(sos, data, axis=-1)
    return y


def butter_bandstop_filter(data: np.ndarray, stop_freq, fs) -> np.ndarray:
    Q = stop_freq / (fs / 2) * 10
    b, a = iirnotch(stop_freq, Q, fs)
    y = lfilter(b, a, data, axis=-1)
    return y


def moving_zscore_buffered(x, buffers, w_samples, freq_band_name):
    """
    Vectorized causal moving z-score using persistent buffers.
    x: (n_channels, n_times_chunk)
    buffers: list of np.ndarray per channel, previous samples (max length w_samples)
    w_samples: window size in samples
    """
    n_channels, n_times = x.shape
    z = np.zeros_like(x, dtype=np.float32)

    # Prepend buffer to the current chunk for each channel
    extended = np.zeros((n_channels, len(buffers[0]) + n_times), dtype=np.float32)
    for ch in range(n_channels):
        if len(buffers[ch]) > 0:
            extended[ch, : len(buffers[ch])] = buffers[ch].get()
        extended[ch, len(buffers[ch]) :] = x[ch]

    # Compute cumulative sums for mean/std
    cumsum = np.cumsum(extended, axis=1)
    cumsum2 = np.cumsum(extended**2, axis=1)

    for t in range(n_times):
        start_idx = max(0, len(buffers[0]) + t - w_samples + 1)
        end_idx = len(buffers[0]) + t

        window_sum = cumsum[:, end_idx] - (
            cumsum[:, start_idx - 1] if start_idx > 0 else 0
        )
        window_sum2 = cumsum2[:, end_idx] - (
            cumsum2[:, start_idx - 1] if start_idx > 0 else 0
        )
        window_len = end_idx - start_idx + 1

        mean = window_sum / window_len
        std = np.sqrt(window_sum2 / window_len - mean**2 + 1e-8)
        z[:, t] = np.nan_to_num((x[:, t] - mean) / std)

    # Update buffers
    for ch in range(n_channels):
        buffers[ch].extend(x[ch])

    return z, buffers


def process_signal(signal, freq_band_name, sfreq, resampling_rate):
    """
    Process signal with frequency band filtering and resampling.
    Modified to handle lower sampling rates and Nyquist frequency constraints.
    """
    freq_band = EEG_FREQ_BANDS[freq_band_name]
    nyquist = sfreq / 2

    if freq_band_name == "gamma":
        signal = np.reshape(signal, (-1, 1, 1))
        with open("/dev/null", "w") as f, contextlib.redirect_stdout(f):
            # Only apply notch filters if below Nyquist frequency
            if 50 < nyquist:
                signal, artifact = dss.dss_line(
                    signal, nremove=1, sfreq=sfreq, nfft=signal.shape[0] // 2, fline=50
                )
            if 60 < nyquist:
                signal, artifact = dss.dss_line(
                    signal, nremove=1, sfreq=sfreq, nfft=signal.shape[0] // 2, fline=60
                )
            signal = signal.flatten()

    # Adjust frequency band to stay within Nyquist limit
    lowcut = freq_band[0]
    highcut = min(freq_band[1], nyquist - 1)

    # Only apply bandpass filter if valid frequency range
    if lowcut < highcut:
        signal = butter_bandpass_filter(
            signal,
            lowcut,
            highcut,
            fs=int(sfreq),
            order=4,
        )

    # Handle resampling - avoid division by zero
    if sfreq != resampling_rate and sfreq > 0 and resampling_rate > 0:
        # Use integer GCD for stable resampling
        from math import gcd
        g = gcd(int(sfreq), int(resampling_rate))
        up_factor = int(resampling_rate) // g
        down_factor = int(sfreq) // g
        resampled_signal = resample_poly(signal, up=up_factor, down=down_factor, axis=-1)
    else:
        resampled_signal = signal

    return resampled_signal


def random_interval(min_val, max_val, range_length):
    """
    Returns a random interval [start, end] within [min_val, max_val]
    with a specific range_length.
    """
    if range_length > (max_val - min_val):
        raise ValueError("The range length is too large for the given bounds.")

    start = random.randint(min_val, max_val - range_length)
    end = start + range_length
    return start, end


def zscore_epochs_buffered(
    edf_path,
    output_file,
    window_sizes_sec,
    epoch_length=2.0,
    resampling_rate=256,
    preload=False,
):
    """
    Stream EDF/BDF/SET files, compute causal moving z-scores with persistent buffers across chunks,
    and save epochs to NPZ file.
    """

    # Support multiple file formats
    ext = os.path.splitext(edf_path)[1].lower()
    if ext == '.edf':
        raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=False)
    elif ext == '.bdf':
        raw = mne.io.read_raw_bdf(edf_path, preload=preload, verbose=False)
    elif ext == '.set':
        raw = mne.io.read_raw_eeglab(edf_path, preload=preload, verbose=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    sfreq = int(raw.info["sfreq"])
    print(edf_path, sfreq, raw.n_times // sfreq)
    n_samples = raw.n_times
    samples_per_epoch = int(epoch_length * sfreq)
    resampled_samples_per_epoch = int(epoch_length * resampling_rate)

    # Adjust start and duration for different sampling rates
    if sfreq <= 100:
        # For 100Hz or lower, adjust minimum start time and duration
        start = max(0, sfreq * 60)  # Start after 1 minute instead of 5
        max_duration = min(600 * sfreq, n_samples - start)  # 10 minutes max instead of 1 hour
    else:
        # Original settings for higher sampling rates
        start = sfreq * 300  # Start after 5 minutes
        max_duration = min(3600 * sfreq, n_samples - start)  # 1 hour max

    # Ensure we have enough data to process
    if max_duration < samples_per_epoch:
        print(f"Warning: Not enough data in {edf_path}, skipping...")
        return

    cropped_start, cropped_end = random_interval(
        start, n_samples, max_duration
    )

    # Map standard channel names to actual channel names in the data
    clean_ch_names = []
    for std_name in EEG_CHANNELS:
        egi_name = EGI_CHANNEL_MAP.get(std_name)
        if egi_name and egi_name in raw.ch_names:
            clean_ch_names.append(egi_name)
        elif std_name.replace('EEG ', '').replace('-REF', '') in raw.ch_names:
            # Try simplified name (e.g., "FP1", "F3", etc.)
            clean_ch_names.append(std_name.replace('EEG ', '').replace('-REF', ''))
        elif std_name in raw.ch_names:
            # Try exact match
            clean_ch_names.append(std_name)

    if len(clean_ch_names) == 0:
        print(f"Warning: No matching channels found in {edf_path}, skipping...")
        return

    print(f"Found {len(clean_ch_names)} matching channels out of {len(EEG_CHANNELS)} expected")

    # Adjust NUM_EEG_CHANNELS for this file
    actual_num_channels = len(clean_ch_names)

    # Initialize buffers for each window size
    max_w_samples = max(window_sizes_sec) * sfreq
    buffer_freq_band_dict = dict()
    for freq_band_name in EEG_FREQ_BANDS:
        buffer_freq_band_dict[freq_band_name] = [
            NumpyRingBuffer(max_w_samples, dtype=np.float32)
            for _ in range(actual_num_channels)
        ]

        for ch_idx, ch in enumerate(clean_ch_names):
            # keep only the last w_samples from warmup
            warmup_start = max(0, start - max_w_samples)
            warmup_data = raw.get_data(
                picks=[ch], start=warmup_start, stop=start
            ).flatten()
            buffer_freq_band_dict[freq_band_name][ch_idx].extend(warmup_data)

    # Initialize data storage for NPZ (accumulate in memory)
    data_dict = {}
    for window_size in window_sizes_sec:
        data_dict[f'data_{window_size}s'] = []

    # Process epochs
    for start in range(cropped_start, cropped_end, samples_per_epoch):
        stop = min(start + samples_per_epoch, cropped_end)
        window = {
            fb: np.zeros((actual_num_channels, resampled_samples_per_epoch))
            for fb in EEG_FREQ_BANDS.keys()
        }

        skip_block = False
        for ch_idx, ch in enumerate(clean_ch_names):
            data = raw.get_data(
                picks=[ch], start=start, stop=stop
            ).flatten()  # (n_channels, chunk_len)

            if data.shape[-1] < samples_per_epoch:
                skip_block = True
                break

            for freq_band_name in EEG_FREQ_BANDS:
                try:
                    signal = process_signal(data, freq_band_name, sfreq, resampling_rate)
                    window[freq_band_name][ch_idx] = signal
                except Exception as e:
                    print(f"Warning: Error processing {freq_band_name} for channel {ch}: {e}")
                    # Use zeros if processing fails
                    window[freq_band_name][ch_idx] = np.zeros(resampled_samples_per_epoch)

        if skip_block:
            continue

        # Compute z-scores for all window sizes using buffers
        for w_sec in window_sizes_sec:
            w_samples = max(1, int(w_sec * sfreq))
            norm_signal = np.zeros(
                (NUM_EEG_FREQ_BANDS, actual_num_channels, resampled_samples_per_epoch),
                dtype=np.float32
            )
            for freq_idx, freq_band_name in enumerate(EEG_FREQ_BAND_NAMES):
                z, buffer_freq_band_dict[freq_band_name] = moving_zscore_buffered(
                    window[freq_band_name],
                    buffer_freq_band_dict[freq_band_name],
                    w_samples,
                    freq_band_name,
                )
                norm_signal[freq_idx] = z

            data_dict[f'data_{w_sec}s'].append(norm_signal)

    # Convert lists to arrays and save as NPZ
    save_dict = {}
    for key in data_dict:
        if len(data_dict[key]) > 0:
            # Stack along first dimension to create (n_epochs, n_bands, n_channels, n_samples)
            save_dict[key] = np.array(data_dict[key], dtype=np.float32)

    # Add metadata
    save_dict['sampling_rate'] = resampling_rate
    save_dict['epoch_length'] = epoch_length
    save_dict['channel_names'] = clean_ch_names
    save_dict['freq_band_names'] = EEG_FREQ_BAND_NAMES
    save_dict['freq_bands'] = {name: band for name, band in EEG_FREQ_BANDS.items()}
    save_dict['window_sizes'] = window_sizes_sec

    # Save as compressed NPZ
    np.savez_compressed(output_file, **save_dict)

    # Print summary
    for key in data_dict:
        if len(data_dict[key]) > 0:
            window_size = key.replace('data_', '').replace('s', '')
            n_epochs = len(data_dict[key])
            print(f"  Window {window_size}s: {n_epochs} epochs saved")

    return


def process_edf(params):
    """
    Process a single EEG file (EDF/BDF/SET format) and save as NPZ.
    """
    f, save_folder, window_size = params
    file_path = f["edf_fpath"]  # Can be .edf, .bdf, or .set

    # Extract filename without extension for save path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    save_file = os.path.join(save_folder, base_name + ".npz")

    window_sizes_sec = [60, 180, 300]

    # Check if already processed
    if os.path.exists(save_file):
        try:
            data = np.load(save_file)
            if 'data_60s' in data and data['data_60s'].shape[0] > 0:
                print(f"Skipping {save_file} - already processed")
                return
        except:
            pass  # If file is corrupted, reprocess

    try:
        zscore_epochs_buffered(
            file_path,
            save_file,
            window_sizes_sec=window_sizes_sec,
            epoch_length=window_size,
            resampling_rate=256,
            preload=False,
        )
        print(f"Successfully processed {file_path} -> {save_file}")
    except Exception as e:
        print(f"Error processing {save_file}: {e}")
        if os.path.exists(save_file):
            os.remove(save_file)


def parse_bids_directory(data_folder, output_csv=None, extensions=None):
    """
    Parse a BIDS-formatted directory structure and create a CSV file with EEG file paths.

    Args:
        data_folder: Root folder containing BIDS datasets (e.g., 'data/')
        output_csv: Path to output CSV file (optional, returns DataFrame if None)
        extensions: List of file extensions to search for (default: ['.bdf', '.set', '.edf'])

    Returns:
        DataFrame with 'edf_fpath' column containing absolute paths to EEG files
    """
    if extensions is None:
        extensions = ['.bdf', '.set', '.edf']

    eeg_files = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            # Check if file has one of the target extensions and follows BIDS naming
            if any(file.endswith(f'_eeg{ext}') for ext in extensions):
                file_path = os.path.abspath(os.path.join(root, file))
                eeg_files.append({'edf_fpath': file_path})

    # Create DataFrame
    df = pd.DataFrame(eeg_files)

    # Sort by file path for consistency
    if len(df) > 0:
        df = df.sort_values('edf_fpath').reset_index(drop=True)

    # Save to CSV if output path provided
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        print(f"Created CSV with {len(df)} EEG files at: {output_csv}")

    return df


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--file-list", required=False,
                           help="CSV file with column 'edf_fpath' containing paths to EEG files")
    arg_parser.add_argument("--data-folder", required=False,
                           help="Parse BIDS directory and create file list automatically (alternative to --file-list)")
    arg_parser.add_argument("--save-folder", required=True,
                           help="Output folder for processed NPZ files")
    arg_parser.add_argument("--window-size", required=False, type=int, default=10,
                           help="Epoch length in seconds (default: 10)")
    arg_parser.add_argument("--root-folder", required=False, type=str, default=None,
                           help="Root folder to prepend to file paths")
    arg_parser.add_argument("--key", required=False, type=str, default="edf_fpath",
                           help="Column name in CSV file containing file paths")
    arg_parser.add_argument("--num-workers", required=False, type=int, default=2,
                           help="Number of parallel workers (default: 2)")
    arg_parser.add_argument("--extensions", required=False, type=str, default=".bdf,.set,.edf",
                           help="Comma-separated list of file extensions to search for (default: .bdf,.set,.edf)")

    args = arg_parser.parse_args()

    # Check that either --file-list or --data-folder is provided
    if args.file_list is None and args.data_folder is None:
        arg_parser.error("Either --file-list or --data-folder must be provided")

    if args.file_list is not None and args.data_folder is not None:
        arg_parser.error("Cannot specify both --file-list and --data-folder")

    # Parse data folder if specified
    if args.data_folder is not None:
        print(f"Parsing BIDS directory: {args.data_folder}")
        extensions = [ext.strip() for ext in args.extensions.split(',')]
        files = parse_bids_directory(args.data_folder, extensions=extensions)
        print(f"Found {len(files)} EEG files")
    else:
        # Load from CSV file
        files = pd.read_csv(args.file_list)

        # Handle root folder if specified
        if args.root_folder is not None:
            files["edf_fpath"] = files[args.key].map(lambda x: os.path.join(args.root_folder, x))
        elif args.key != "edf_fpath":
            files["edf_fpath"] = files[args.key]

    files = files.to_dict("records")

    print(f"Processing {len(files)} files with {args.num_workers} workers...")

    pool = Pool(args.num_workers)
    for result in pool.imap_unordered(
        process_edf,
        list(
            zip(
                files,
                [args.save_folder] * len(files),
                [args.window_size] * len(files),
            )
        ),
    ):
        if result is not None:
            print(result)

    pool.close()
    pool.join()
    pool.terminate()

    print("Processing complete!")