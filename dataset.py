import torch
from torch.utils.data import Dataset

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_data, load_word2vec, make_w2v_embeddings, split_and_zero_padding


class SiameseLSTMDataset(Dataset):
    def __init__(self, config):
        super(SiameseLSTMDataset, self).__init__()

        self.embedding_dict = load_word2vec(path_file=config["source"]["word2vec"])
        self.df = load_data(path_file=config["source"]["data"])

        self.df, self.embeddings, self.vocabs = make_w2v_embeddings(self.embedding_dict, self.df, config["model"]["embedded_size"])

        del self.embedding_dict

        self.X = self.df[["question1_n", "question2_n"]]
        self.Y = self.df["is_duplicate"]

        self.X = split_and_zero_padding(self.X, config["model"]["max_seq_length"])
        self.Y = self.Y.values

        assert self.X["left"].shape == self.X["right"].shape
        assert len(self.X["left"]) == len(self.Y)   

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):

        sentence_1 = torch.tensor(self.X["left"][index], dtype=torch.int32)
        sentence_2 = torch.tensor(self.X["right"][index], dtype=torch.int32)

        label = torch.tensor(self.Y[index], dtype=torch.float)
        return sentence_1, sentence_2, label
