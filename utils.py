#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Union, Tuple
from pathlib import Path
import pathlib
import os
import numpy as np
import librosa


__docformat__ = 'reStructuredText'
__all__ = [ 'get_files_from_dir_with_pathlib',
            'get_audio_files_from_subdirs',
            'get_audio_file_data',
            'to_audio'
           ]


def get_files_from_dir_with_pathlib(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the files in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return list(pathlib.Path(dir_name).iterdir())


def get_audio_files_from_subdirs(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the audio files in the subdirectories of `dir_name`.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the audio files in the subdirectories `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return [Path(dirpath) / Path(filename) for dirpath, _, filenames in os.walk(dir_name)
                                           for filename in filenames
                                           if filename[-4:] == '.wav']


def get_audio_file_data(audio_file: Union[str, pathlib.Path]) \
        -> Tuple[np.ndarray, float]:
    """Loads and returns the audio data from the `audio_file`.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str
    :return: Data of the `audio_file` audio file.
    :rtype: Tuple[numpy.ndarray, float]
    """
    return librosa.core.load(path=audio_file, sr=None, mono=True)


def to_audio(mix_waveform: np.ndarray,
             predicted_vectors: np.ndarray) \
    -> np.ndarray:
    """
    :param mix_waveform: The waveform of the monaural mixture. Expected shape (n_samples,)
    :type mix_waveform: numpy.ndarray
    :param predicted_vectors: A numpy array of shape: (chunks, time_frames, frequency_bins)
    :type predicted_vectors: numpy.ndarray
    :return: predicted_waveform: The waveform of the predicted signal: (~n_samples,)
    :rtype: numpy.ndarray
    """
    # Pre-defined (I)STFT parameters
    win_size = 2048
    hop_size = win_size // 2
    win_type = 'hamm'

    # STFT analysis of waveform
    c_x = librosa.stft(mix_waveform, n_fft=win_size, win_length=win_size, hop_length=hop_size, window=win_type)
    # Phase computation
    phs_x = np.angle(c_x)
    # Get the number of time-frames
    tf = phs_x.shape[1]

    # Number of chunks/sequences
    n_chunks, seq_len, fb = predicted_vectors.shape
    p_end = seq_len*n_chunks
    # Reshaping
    #rs_vectors = np.reshape(np.moveaxis(predicted_vectors, 0, 1), (fb, p_end))
    rs_vectors = np.reshape(np.transpose(predicted_vectors, (2, 0, 1)), (fb, p_end))
    # Reconstruction
    if p_end > tf:
        # Appending zeros to phase
        c_vectors = np.hstack((phs_x, np.zeros_like(phs_x[:, :p_end-seq_len])))
    else:
        c_vectors = rs_vectors * np.exp(1j * phs_x[:, :p_end])
    # ISTFT
    predicted_waveform = librosa.istft(c_vectors, win_length=win_size, hop_length=hop_size, window=win_type)

    return predicted_waveform

# EOF
