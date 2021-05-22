import torch 
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel



class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.linear_1 = nn.Linear(hidden_size, hidden_size//2)
        self.linear_2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.linear_3 = nn.Linear(hidden_size//4, hidden_size//8)

        self.seq_relationship = nn.Linear(hidden_size//8, 1)

        self.relu = nn.ReLU()

    def forward(self, pooled_output):


        seq_relationship_score = self.linear_1(pooled_output)
        seq_relationship_score = self.relu(seq_relationship_score)
        seq_relationship_score = self.linear_2(seq_relationship_score)
        seq_relationship_score = self.relu(seq_relationship_score)
        seq_relationship_score = self.linear_3(seq_relationship_score)
        seq_relationship_score = self.relu(seq_relationship_score)
        seq_relationship_score = self.seq_relationship(pooled_output)

        return seq_relationship_score

class SimilaritySentenceBert(nn.Module):
    def __init__(self, PRETRAINED_NAME):
        super().__init__()
        self.hidden_size = 768
        self.bert = BertModel.from_pretrained(PRETRAINED_NAME)
        self.cls  = Classifier(self.hidden_size)

        self.sigmoid = nn.Sigmoid()
        # self.init_weights()
    
    def forward(
        self,
        input_ids = None,
        segment_ids = None,
        input_mask = None):

        pooled_output = self.bert(input_ids = input_ids,
                                        attention_mask = input_mask,
                                        token_type_ids = segment_ids)[0][:, 0, :] ### B X D


        seq_relationship_score = self.cls(pooled_output) ### B X 1
        
        seq_relationship_score = self.sigmoid(seq_relationship_score)

        return seq_relationship_score