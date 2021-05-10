from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from config import config
from dataset import SiameseLSTMDataset
from model import SiameseLSTM

# from sklearn.metrics import accuracy_score


def train(data_loader, model, optimizer, loss_fn, loss_previous, epoch, device):

    loss = loss_previous

    model.train()

    for step, batch in enumerate(data_loader):
        batch = tuple(t.to(device) for t in batch)
        sentence_left, sentence_right, label = batch
        optimizer.zero_grad()

        output = model(sentence_left, sentence_right)

        loss_train = loss_fn(output.double(), label.double())
        if loss_train <  loss:
            loss = loss_train
            print(f"Loss train: {loss:.4f} at epoch {epoch}")
            torch.save(model.state_dict(), f"./best_model_train/model.pth")

        loss_train.backward()
        optimizer.step()
    return loss

def evaluate(data_loader, model, loss_fn, acc_previous, loss_previous, epoch, len_dataset, device):

    model.eval()

    loss_dev = loss_previous
    acc_dev = 0

    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        sentence_left, sentence_right, label = batch

        with torch.no_grad():
            outputs = model(sentence_left, sentence_right)

            loss = loss_fn(outputs.double(), label.double())

            if loss < loss_dev:
                loss_dev = loss
            for output in outputs:
                if output > 0.5:
                    acc_dev += 1

    acc_dev = acc_dev/len_dataset
    print(f"Accuracy score: {acc_dev:.4f} at epoch {epoch}")
    print(f"Loss: {loss_dev:.4f} at epoch {epoch}")

    if acc_dev > acc_previous:
        torch.save(model.state_dict(), f"./best_model_eval/model_acc_{round(acc_dev, 4)}.pth")
    if loss_dev < loss_previous:
        torch.save(model.state_dict(), f"./best_model_eval/model_loss_{round(int(loss_dev), 4)}.pth")
    return acc_dev, loss_dev

def main(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = SiameseLSTMDataset(config)
    print(len(train_dataset))
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], sampler=train_sampler)

    dev_dataset = SiameseLSTMDataset(config)
    print(len(dev_dataset))
    dev_sampler = SequentialSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset)

    model = SiameseLSTM(config).double().to(device=device)

    # optimizer = torch.optim.Adam(lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]

    optimizer = torch.optim.Adam(lr=1e-5, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)
    loss_fn = nn.BCELoss()
    
    acc_dev = 0
    loss_dev = 1000
    loss_train = 1000

    for epoch in range(1, config["model"]["epoch"]):
        print(f"Training epoch {str(epoch)}")

        loss_train = train(train_loader, model, optimizer, loss_fn, loss_train, epoch, device)
        
        print(f"Evaluate model.............")

        acc_dev, loss_dev = evaluate(dev_loader, model, loss_fn, acc_dev, loss_dev, epoch, len(dev_dataset), device)

if __name__ == "__main__":
    main(config)
