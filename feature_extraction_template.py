
from typing import List, Optional

import pathlib
import os
import numpy as np
import librosa

from utils import get_audio_files_from_subdirs, get_audio_file_data


def extract_spectrogram(audio_signal: np.ndarray,
                        n_fft: Optional[int] = 2048,
                        hop_length: Optional[int] = 1024,
                        window: Optional[str] = 'hamm') \
        -> np.ndarray:
    """Extracts and returns the magnitude spectrogram from the `audio_signal` signal.

    :param audio_signal: Audio signal.
    :type audio_signal: numpy.ndarray
    :param n_fft: STFT window length (in samples), defaults to 2048.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 1024.
    :type hop_length: Optional[int]
    :param window: Window type, defaults 'hamm'.
    :type window: Optional[str]
    :return: Magnitude of the short-time Fourier transform of the audio signal [shape=(n_frames, n_bins)].
    :rtype: numpy.ndarray
    """

    stft = librosa.stft(audio_signal, n_fft= n_fft, hop_length= hop_length, window= window)
    return stft.T


def split_into_sequences(spec: np.ndarray, seq_len: int) \
        -> List[np.ndarray]:
    """Splits the spectrum `spec` into sequences of length `seq_len`.

    :param spec: Spectrum to be split into sequences.
    :type spec: numpy.ndarray
    :param seq_len: Length of the sequences.
    :type seq_len: int
    :return: List of sequences.
    :rtype: list[numpy.ndarray]
    """
    num_sequences = spec.shape[0] // seq_len
    stft = spec[:num_sequences * seq_len]

    return stft.reshape(num_sequences, seq_len, -1)


def main(dataset_paths: List[pathlib.Path]):
    splits = ["training", "testing"]
    seq_len = 60
    for dataset_path in dataset_paths:
        for split in splits:
            audio_paths = get_audio_files_from_subdirs(dataset_path / split)
            output_dir = dataset_path / (split + '_features')
            try:
                os.mkdir(output_dir)
            except OSError as error:
                print("File already exists. Continuing...")
            print(f'Extracting features from {dataset_path / split} to {output_dir}')
            for audio_path in audio_paths:
                audio_data, fs = get_audio_file_data(audio_path)
                stft = extract_spectrogram(audio_data)
                for seq_idx, seq in enumerate(split_into_sequences(stft, seq_len)):
                    file_name = f'{audio_path.stem[:3]}_{audio_path.parent.stem}_seq_{seq_idx:03}.npy'
                    np.save(f'{output_dir / file_name}', seq)


if __name__ == '__main__':
    dataset_root_path = pathlib.Path("dataset")
    dataset_paths = [dataset_root_path / "Mixtures", dataset_root_path / "Sources"]
    main(dataset_paths)
