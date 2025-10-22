import os
import shutil
from argparse import ArgumentParser
import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, sosfilt, butter, iirnotch, lfilter
from multiprocessing import Pool
import os
from collections import deque
from ml_utils.data_generator import helpers
import zarr
from numcodecs import Blosc
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
        buffers.extend(x[ch])

    return z, buffers


def process_signal(signal, freq_band_name, sfreq, resampling_rate):
    freq_band = EEG_FREQ_BANDS[freq_band_name]
    if freq_band_name == "gamma":
        signal = np.reshape(signal, (-1, 1, 1))
        with open("/dev/null", "w") as f, contextlib.redirect_stdout(f):
            signal, artifact = dss.dss_line(
                signal, nremove=1, sfreq=sfreq, nfft=signal.shape[0] // 2, fline=50
            )
            signal, artifact = dss.dss_line(
                signal, nremove=1, sfreq=sfreq, nfft=signal.shape[0] // 2, fline=60
            )
            signal = signal.flatten()

    signal = butter_bandpass_filter(
        signal,
        freq_band[0],
        freq_band[1],
        fs=int(sfreq),
        order=4,
    )
    resampled_signal = resample_poly(signal, down=sfreq, up=resampling_rate, axis=-1)

    return resampled_signal


def append_epochs(zarr_dataset, epochs):
    """
    Append a batch of epochs to the dataset.

    Parameters
    ----------
    zarr_dataset : zarr.core.Array
        The target resizable Zarr array.
    epochs : np.ndarray, shape (batch_size, n_channels, n_times)
        Batch of EEG epochs.
    """
    n_epochs = zarr_dataset.shape[0]
    batch_size = epochs.shape[0]
    zarr_dataset.resize((n_epochs + batch_size,) + zarr_dataset.shape[1:])
    zarr_dataset[n_epochs : n_epochs + batch_size] = epochs.astype("float32")
    return zarr_dataset


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
    output_dir,
    window_sizes_sec,
    epoch_length=2.0,
    resampling_rate=256,
    preload=False,
):
    """
    Stream EDF, compute causal moving z-scores with persistent buffers across chunks,
    and write epochs to Petastorm with:
      - window_sizes_sec array
      - one field per window size
    """

    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=False)
    sfreq = int(raw.info["sfreq"])
    print(edf_path, sfreq, raw.n_times // sfreq)
    n_samples = raw.n_times
    samples_per_epoch = int(epoch_length * sfreq)
    resampled_samples_per_epoch = int(epoch_length * resampling_rate)

    start = sfreq * 300  # ignore the first 5 minutes
    cropped_start, cropped_end = random_interval(
        start, n_samples, min(3600 * sfreq, n_samples - start)
    )

    clean_ch_names = [
        helpers.is_channel_in_list(ch, raw.ch_names) for ch in EEG_CHANNELS
    ]

    # Initialize buffers for each window size
    max_w_samples = max(window_sizes_sec) * sfreq
    buffer_freq_band_dict = dict()
    for freq_band_name in EEG_FREQ_BANDS:
        buffer_freq_band_dict[freq_band_name] = [
            NumpyRingBuffer(max_w_samples, dtype=np.float32)
            for _ in range(NUM_EEG_CHANNELS)
        ]

        for ch_idx, ch in enumerate(clean_ch_names):
            # keep only the last w_samples from warmup
            warmup_data = raw.get_data(
                picks=[ch], start=min(start - max_w_samples, 0), stop=start
            ).flatten()
            buffer_freq_band_dict[freq_band_name][ch_idx].extend(warmup_data)

    # create zarr datasets
    store = zarr.DirectoryStore(output_dir)
    zarr_datasets = zarr.open_group(store, mode="w")
    for window_size in window_sizes_sec:
        zarr_datasets.create_dataset(
            f"{window_size}",
            shape=(
                0,
                len(EEG_FREQ_BANDS.keys()),
                NUM_EEG_CHANNELS,
                resampled_samples_per_epoch,
            ),
            chunks=(
                1,
                len(EEG_FREQ_BANDS.keys()),
                NUM_EEG_CHANNELS,
                resampled_samples_per_epoch,
            ),  # chunk along epochs
            dtype="float32",
            compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
        )

    for start in range(cropped_start, cropped_end, samples_per_epoch):
        stop = min(start + samples_per_epoch, cropped_end)
        window = {
            fb: np.zeros((NUM_EEG_CHANNELS, resampled_samples_per_epoch))
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
                signal = process_signal(data, freq_band_name, sfreq, resampling_rate)
                window[freq_band_name][ch_idx] = signal

        if skip_block:
            continue

        # Compute z-scores for all window sizes using buffers
        for w_sec in window_sizes_sec:
            w_samples = max(1, int(w_sec * sfreq))
            norm_signal = np.zeros(
                (1, NUM_EEG_FREQ_BANDS, NUM_EEG_CHANNELS, resampled_samples_per_epoch)
            )
            for freq_idx, freq_band_name in enumerate(EEG_FREQ_BAND_NAMES):
                z, buffer_freq_band_dict[freq_band_name] = moving_zscore_buffered(
                    window[freq_band_name],
                    buffer_freq_band_dict[freq_band_name],
                    w_samples,
                    freq_band_name,
                )
                norm_signal[:, freq_idx] = z

            append_epochs(zarr_datasets[f"{w_sec}"], norm_signal)
    return


def process_edf(params):
    f, save_folder, window_size = params
    edf_fpath = f["edf_fpath"]
    edf_fname = edf_fpath.split("/")[-1]

    save_file = os.path.join(save_folder, edf_fname + ".zarr")

    window_sizes_sec = [60, 180, 300]

    if os.path.exists(save_file):
        dt = zarr.open(save_file, mode="r")
        if dt[window_sizes_sec[0]].shape[0] > 0:
            return

    try:
        zscore_epochs_buffered(
            edf_fpath,
            save_file,
            window_sizes_sec=window_sizes_sec,
            epoch_length=window_size,
            resampling_rate=256,
            preload=False,
        )
    except Exception as e:
        print(save_file, e)
        if os.path.exists(save_file):
            shutil.rmtree(save_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--file-list", required=True)
    arg_parser.add_argument("--save-folder", required=True)
    arg_parser.add_argument("--window-size", required=False, type=int, default=10)
    arg_parser.add_argument("--root-folder", required=False, type=str, default=None)
    arg_parser.add_argument("--key", required=False, type=str)

    args = arg_parser.parse_args()

    files = pd.read_csv(args.file_list)
    if args.root_folder is not None:
        files["edf_fpath"] = files[args.key].map(lambda x: os.path.join(args.root_folder, x))

    files = files.to_dict("records")

    pool = Pool(2)
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
        print(result)

    pool.close()
    pool.join()
    pool.terminate()
