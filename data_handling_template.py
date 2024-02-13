
from typing import Union, List, Tuple

import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from utils import get_files_from_dir_with_pathlib, get_audio_file_data


class MyDSD100Dataset(Dataset):
    def __init__(self, root_dir: Union[str, Path],
                 split: str = 'training') \
            -> None:
        """Pytorch Dataset class for doing voice separation with the DSD100 dataset.

        :param root_dir: Root directory of the dataset (should contain the Mixtures and Sources folders).
        :type root_dir: str or pathlib.Path
        :param split: Split to use (training or testing), defaults to 'training'.
        :type split: str
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.mixtures = []
        self.vocals = []
        self.load_data()

    def load_data(self) -> None:
        """Loads the data into memory."""
        mix_features_dir = ?
        self.mixtures = ?
        voc_features_dir = ?
        self.vocals = ?

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return ?

    def __getitem__(self, idx):
        """Returns the item at index `idx`."""
        return ?


class MyDSD100DatasetFullSongs(Dataset):
    def __init__(self, root_dir: Union[str, Path],
                 split: str = 'training') \
            -> None:
        """Pytorch Dataset class for doing voice separation with the DSD100 dataset.

        :param root_dir: Root directory of the dataset (should contain the Mixtures and Sources folders).
        :type root_dir: str or pathlib.Path
        :param split: Split to use (training or testing), defaults to 'training'.
        :type split: str
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.mixtures = []
        self.mixtures_raw = []
        self.vocals_raw = []
        self.load_data()

    def load_data(self) -> None:
        """Loads the data into memory."""
        songs = [f.stem for f in get_files_from_dir_with_pathlib(self.root_dir / 'Mixtures' / self.split)]
        mix_features_files = get_files_from_dir_with_pathlib(self.root_dir / 'Mixtures' / (self.split + '_features'))
        for song in songs:
            mix_features = [np.load(str(f)) for f in mix_features_files if song in f.stem]
            mix_raw = ?
            voc_raw = ?
            self.mixtures.append(mix_features)
            self.mixtures_raw.append(mix_raw)
            self.vocals_raw.append(voc_raw)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return ?

    def __getitem__(self, item: int) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Returns an item from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: List of the features of the mix, raw mix, and raw ground-truth vocals of the item.
        :rtype: (list[numpy.ndarray], numpy.ndarray, numpy.ndarray)
        """
        return ?
