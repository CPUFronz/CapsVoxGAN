import torch
import h5py

from torch.utils.data import Dataset


class VoxelData(Dataset):
    def __init__(self, path, models=None):
        self.path = path
        self.models = models
        self.index = self._create_index()
        print('Using {} training examples'.format(len(self.index)))

    def __getitem__(self, index):
        idx_c, idx_d = self.index[index]
        with h5py.File(self.path) as hdf:
            tensor = torch.Tensor(hdf[idx_c][idx_d])
        return tensor

    def _create_index(self):
        with h5py.File(self.path) as hdf:
            pairs = []
            if self.models:
                categories = self.models
            else:
                categories = list(hdf.keys())

            for c in categories:
                datasets = list(hdf[c].keys())
                for d in datasets:
                    pairs.append((c, d))
        return pairs

    def __len__(self):
        return len(self.index)
