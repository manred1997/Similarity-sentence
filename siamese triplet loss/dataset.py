from numpy.lib.npyio import load
import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from config import config
from utils import load_file_npy

class SiameseLSTMDataset(Dataset):
    def __init__(self, config):
        super(SiameseLSTMDataset, self).__init__()
        X_left_seq = load_file_npy(config["source"]["X_left_seq.npy"])
        X_right_seq = load_file_npy(config["source"]["X_right_seq.npy"])

        Y = load_file_npy(config["source"]["label"])

        positive_index = list(np.where(Y == 1)[0])
        negative_index = list(np.where(Y == 1)[0])

        X_anchor = X_left_seq[positive_index]
        self.X_anchor = np.concatenate((X_anchor, X_right_seq[positive_index]), axis=0)

        X_positive = X_right_seq[positive_index]
        self.X_positive = np.concatenate((X_positive, X_left_seq[positive_index]), axis=0)

        X_negative = X_left_seq[negative_index] # target: use all dataset
        X_negative = np.concatenate((X_negative, X_right_seq[negative_index]), axis=0) # target: use all dataset

        self.X_negative = X_negative[np.random.choice(X_negative.shape[0], X_anchor.shape[0], replace=False), :]

        assert len(X_anchor) == len(X_positive) == len(X_negative)

    def __len__(self):
        return len(self.X_anchor)
    
    def __getitem__(self, index):
        # Anchor, Positive, Negative
        Anchor = torch.tensor(self.X_anchor[index], dtype=torch.int32)
        Positive = torch.tensor(self.X_positive[index], dtype=torch.int32)
        Negative = torch.tensor(self.X_negative[index], dtype=torch.int32)
        return Anchor, Positive, Negative
