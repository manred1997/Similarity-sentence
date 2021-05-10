from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from config import config
from dataset import SiameseLSTMDataset
from model import SiameseLSTM

def train(data_loader, model, optimizer, loss_fn):

    model.train()

    for step, batch in enumerate(data_loader):
        batch = tuple(t.to(device) for t in batch)
        sentence_left, sentence_right, label = batch
        optimizer.zero_grad()

        output = model(sentence_left, sentence_right)
        
        

def eval(data_loader, model):

    pass

def main(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = SiameseLSTMDataset(config)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], sampler=train_sampler)

    dev_dataset = SiameseLSTMDataset(config)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset)

    model = SiameseLSTM(config).to(device=device)

    optimizer = torch.optim.Adam(lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.BCELoss()

    for epoch in range(1, config["model"]["epoch"]):
        print(f"Training epoch {str(epoch)}")

        train(train_loader, model)
        
        print(f"Evaluate model.............")

        eval(dev_loader, model)
