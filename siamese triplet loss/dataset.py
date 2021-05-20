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
        

    def __len__(self):

    
    def __getitem__(self, index):
