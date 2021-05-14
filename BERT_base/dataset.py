import torch
from torch.utils.data import Dataset


class SentenceSimlarity_BertDataset(Dataset):
    def __init__(self, X, Y) -> None:
        super().__init__()

        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y[0])

    def __getitem__(self, index):
        
        input_ids = torch.tensor(self.X[0][index], dtype=torch.int64)
        segment_ids = torch.tensor(self.X[2][index], dtype=torch.int64)
        input_mask = torch.tensor(self.X[1][index], dtype=torch.float)

        label = torch.tensor(self.Y[0][index], dtype=torch.float)

        return input_ids, segment_ids, input_mask, label

