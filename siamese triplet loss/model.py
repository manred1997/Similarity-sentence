import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from config import config
from utils import load_file_npy

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        # Convolution neural network
        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=config["model"]["embedded_size"], # 300
                        out_channels=config["model"]["embedded_size"], # 300
                        kernel_size=config["model"]["kernel_size"],
                        stride=config["model"]["stride"],
                        padding=config["model"]["padding"]), # B X H x L
                nn.ReLU()
                )
        self.maxpooling = nn.MaxPool1d(config["model"]["kernel_size"], stride=config["model"]["stride"]*2)
        self.conv2 = nn.Sequential(
                nn.Conv1d(in_channels=config["model"]["embedded_size"],
                        out_channels=config["model"]["embedded_size"],
                        kernel_size=config["model"]["kernel_size"],
                        stride=config["model"]["stride"],
                        padding=config["model"]["padding"]), # B X H x L
                nn.ReLU()
                )

        self.dropout = nn.Dropout(config["model"]["p_drop"])
    
    def forward(self, embedding):
        # Shape embedding: B X L X H
        embedding = embedding.permute(0, 2, 1)
        embedding = self.conv1(embedding) # B X H X L'
        embedding = self.maxpooling(embedding) # B X H X L''
        embedding = self.conv2(embedding) # B X H X L''
        logit = self.dropout(embedding)
        return logit # B X H X L''


class BiLSTMEncoder(nn.Module):
    def __init__(self, config):
        super(BiLSTMEncoder, self).__init__()

        self.embedded_size = config['model']['embedded_size']
        self.batch_size = config['model']['batch_size']
        self.hidden_size = config['model']['hidden_size']
        self.num_layers = config['model']['num_layers']
        self.bidir = config['model']['bidirectional']

        if self.bidir:
            self.direction = 2
        else: self.direction = 1

        self.dropout = config['model']['dropout']

        self.lstm = nn.LSTM(input_size=self.embedded_size, hidden_size=self.hidden_size,\
                            num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidir)

    def initHiddenCell(self):
        hidden_rand = Variable(torch.randn(self.direction*self.num_layers, self.batch_size, self.hidden_size))
        cell_rand = Variable(torch.randn(self.direction*self.num_layers, self.batch_size, self.hidden_size))
        return hidden_rand, cell_rand
    
    def forward(self, input_embeddings):

        input_embeddings = input_embeddings.permute(2, 0, 1)
        output, (_, _) = self.lstm(input_embeddings)
        return output
    
class WordAttention(nn.Module):
    def __init__(self, config):
        super(WordAttention, self).__init__()

        self.bidir = config['model']['bidirectional']
        if self.bidir:
            self.direction = 2
        else: self.direction = 1

        self.hidden_size = config['model']['hidden_size'] * self.direction
        
        self.W = nn.Linear(self.hidden_size , config["model"]['attention_size']) # 700
        self.V = nn.Linear( config["model"]['attention_size'], 1, bias=False)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstmencoder):

        # B x len x dim 
        lstmencoder = lstmencoder.permute(1,0,2)
        
        energy = self.W(lstmencoder) # B x len x attn_size
        attention = self.V(self.tanh(energy)) # B x len x 1

        attention = attention.squeeze(-1) # Batch, Length
        attention = self.softmax(attention) # Batch, Length

        sent_representation = torch.bmm(attention.unsqueeze(1),lstmencoder)
        # B x 1 x d
        sent_representation = sent_representation.squeeze(1)

        return sent_representation


class CNN_LSTM(nn.Module):
    def __init__(self, config):
        super(CNN_LSTM, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(load_file_npy(config["model"]["embeddings"])), freeze=True)
        # Shape [B, lenght, hidden_size]
        # CNN Layer
        self.cnn = CNN(config)
        # Shape [B, hidden_size, length]
        # LSTM Layer
        self.lstm = BiLSTMEncoder(config)
        # Shape [length, B, hidden_size]
        # Self - Attention Layer
        self.attention = WordAttention(config)
    
    def forward(self, anchor, positive, negative):

        # Compute embedding
        anchor = self.embedding(anchor)  # B X L X H
        positive = self.embedding(positive) # B X L X H
        negative = self.embedding(negative) # B X L X H

        # Compute cnn
        anchor = anchor.permute(0, 2, 1) # B X H X L
        positive = positive.permute(0, 2, 1) # B X H X L
        negative = negative.permute(0, 2, 1) # B X H' X L

        anchor = self.cnn(anchor)
        positive = self.cnn(positive)
        negative = self.cnn(negative)

        # Compute lstm
        anchor = self.lstm(anchor)
        positive = self.lstm(positive)
        negative = self.lstm(negative)

        # Compute attention
        anchor = self.attention(anchor)
        positive = self.attention(positive)
        negative = self.attention(negative)

        return anchor, positive, negative