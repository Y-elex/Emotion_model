# dataset.py
import numpy as np
from torch.utils.data import Dataset
import torch

class FeatureDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.bow = data['bow'].astype('float32')
        self.cnn = data['cnn'].astype('float32')
        self.y = data['y'].astype('int64')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        bow = torch.from_numpy(self.bow[idx])
        cnn = torch.from_numpy(self.cnn[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return bow, cnn, y
