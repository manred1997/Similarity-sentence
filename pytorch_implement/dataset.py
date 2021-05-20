import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import config
from utils import load_file_npy


class SiameseLSTMDataset(Dataset):
    def __init__(self, config, model):
        print(f"Mode {model}")
        super(SiameseLSTMDataset, self).__init__()
        if model == "train":

            self.X_left = load_file_npy(config["source"]["train"]["sentence_1"])
            self.X_right = load_file_npy(config["source"]["train"]["sentence_2"])

            self.Y = load_file_npy(config["source"]["train"]["label"])

        elif model == "dev":

            self.X_left = load_file_npy(config["source"]["dev"]["sentence_1"])
            self.X_right = load_file_npy(config["source"]["dev"]["sentence_2"])

            self.Y = load_file_npy(config["source"]["dev"]["label"])
        
        else:

            self.X_left = load_file(config["source"]["test"]["sentence_1"])
            self.X_right = load_file(config["source"]["test"]["sentence_2"])

            self.Y = load_file(config["source"]["test"]["label"])


    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):

        sentence_1 = torch.tensor(self.X_left[index], dtype=torch.int32)
        sentence_2 = torch.tensor(self.X_right[index], dtype=torch.int32)

        label = torch.tensor(self.Y[index], dtype=torch.float)
        return sentence_1, sentence_2, label
