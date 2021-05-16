import torch 
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel



class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size//2)
        self.linear_2 = nn.Linear(config.hidden_size//2, config.hidden_size//4)
        self.linear_3 = nn.Linear(config.hidden_size//4, config.hidden_size//8)

        self.seq_relationship = nn.Linear(config.hidden_size//8, 1)

    def forward(self, pooled_output):


        seq_relationship_score = self.linear_1(pooled_output)
        seq_relationship_score = self.linear_2(seq_relationship_score)
        seq_relationship_score = self.linear_3(seq_relationship_score)
        seq_relationship_score = self.seq_relationship(seq_relationship_score)

        return seq_relationship_score

class SimilaritySentenceBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls  = Classifier(config)

        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self,
        input_ids = None,
        segment_ids = None,
        input_mask = None):

        pooled_output = self.bert(input_ids = input_ids,
                                        attention_mask = input_mask,
                                        token_type_ids = segment_ids)[1] ### B X D


        # seq_relationship_score = self.cls(pooled_output) ### B X 1

        # seq_relationship_score = self.sigmoid(seq_relationship_score)

        return seq_relationship_score