import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import config

from utils import load_file_npy

import numpy as np


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

        input_embeddings = input_embeddings.permute(1,0,2)
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
        # print(attention)
        # print(attention[0])
        # attention = attention.unsqueeze(2).repeat(1, 1, self.hidden_size)
        # print(attention.shape)
        # attention : B x 1 x len , lstmencoder : B x len x d
        sent_representation = torch.bmm(attention.unsqueeze(1),lstmencoder)
        # B x 1 x d
        sent_representation = sent_representation.squeeze(1)

        # sent_representation = torch.mul(lstmencoder, attention)
        
        # print(sent_representation.shape)
        # sent_representation = torch.sum(sent_representation, dim=1)
        # print(sent_representation)
        # print(sent_representation.shape)
        # print("----------------")
        return sent_representation


class SiameseLSTM(nn.Module):
    def __init__(self, config):
        super(SiameseLSTM, self).__init__()

        self.hidden_size = config['model']['hidden_size']

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(load_file_npy(config["model"]["embeddings"])), freeze=True)

        self.encoder = BiLSTMEncoder(config)
        self.attention = WordAttention(config)

        self.linear_1 = nn.Linear(self.hidden_size*6 , 16)
        self.linear_2 = nn.Linear(16, 4)
        self.linear_3 = nn.Linear(4, 2)
        self.linear_4 = nn.Linear(2, 1)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, sentence_1, sentence_2):

        embedding_1 = self.embedding(sentence_1)
        embedding_2 = self.embedding(sentence_2)


        output1= self.encoder(embedding_1)
        output2= self.encoder(embedding_2)
        # print(output1.shape)
        # print("=========")
        left_sen_representation = self.attention(output1)
        right_sen_representation = self.attention(output2)
        # print("****************")

        man_distance = torch.abs(left_sen_representation - right_sen_representation)

        sen_representation = torch.cat((left_sen_representation, right_sen_representation, man_distance), 1)

        similarity = self.sigmoid(self.linear_4(self.linear_3(self.linear_2(self.linear_1(sen_representation))))).view(-1)

        return similarity