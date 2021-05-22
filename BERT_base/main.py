from genericpath import samefile
import json
import random
import sys

from tqdm import tqdm
from colorama import Fore

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sample import Processor, Sample
from config import config
from dataset import SentenceSimlarity_BertDataset
from model import SimilaritySentenceBert

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AutoTokenizer

_PRETRAINED_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
_PRETRAINED_TOKENIZER = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

_FOLDER_DATA_TRAIN = "../data"
_FOLDER_DATA_TEST = "../data"

tokenizer = AutoTokenizer.from_pretrained(_PRETRAINED_TOKENIZER)

processor = Processor()

examples = processor.get_train_examples(_FOLDER_DATA_TRAIN)
label_list = processor.get_labels()

sample = Sample(examples, label_list, tokenizer, config)
X, Y = sample.preprocessing()

train_data = SentenceSimlarity_BertDataset(X, Y)
train_sampler = RandomSampler(train_data)
train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=128)

examples = processor.get_test_examples(_FOLDER_DATA_TEST)
sample = Sample(examples, label_list, tokenizer, config)
X_test, Y_test = sample.preprocessing()

test_data = SentenceSimlarity_BertDataset(X_test, Y_test)
test_sampler = SequentialSampler(test_data)
test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimilaritySentenceBert(_PRETRAINED_MODEL)

model = model.to(device=device)


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = torch.optim.Adam(lr=1e-3, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)

# loss_fn = nn.BCELoss()

loss_fn = nn.BCELoss()

for epoch in range(1, 10):
    print("Training epoch ", str(epoch))
    training_pbar = tqdm(total=len(train_data),
                         position=0, leave=True,
                         file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    
    
    for step, batch in enumerate(train_data_loader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, segment_ids, input_mask, label = batch
        optimizer.zero_grad()
        
        score = model(input_ids=input_ids,
                        segment_ids=segment_ids,
                        input_mask=input_mask)
        # print(loss)
        loss = loss_fn(score, label)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1
        if step % 1000 == 0:
          print(f"Loss = {loss} / step {step}")
        training_pbar.update(input_ids.size(0))
    training_pbar.close()
    print(f"\n Binary Cross Entropy loss = {tr_loss/nb_tr_steps:.8f}/ epoch {epoch}")
    torch.save(model.state_dict(), "./weights_" + str(epoch) + ".pth")

    validation_pbar = tqdm(total=len(test_data),
                            position=0, leave=True,
                            file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    model.eval()
    tr_loss = 0
    nb_tr_steps = 0 
    correct = 0

    for batch in test_data_loader:
        batch = tuple(t for t in batch)
        input_ids, segment_ids, input_mask, label = batch
        with torch.no_grad():
            score = model(input_ids=input_ids,
                        segment_ids=segment_ids,
                        input_mask=input_mask)
            tr_loss += loss.item()
            nb_tr_steps += 1
            correct += ((score > 0.5).float() == label).float().sum()
        validation_pbar.update(input_ids.size(0))
    acc = correct / len(test_data)
    validation_pbar.close()
    print(f"\nEpoch={epoch}, exact match score={acc:.2f}")