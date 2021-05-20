
import torch
from keras.preprocessing.sequence import pad_sequences

from model_attention import SiameseLSTM

from utils import load_file_npy, load_json, text_to_word_list


class SentencePair(object):
    def __init__(self, config) -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.model = SiameseLSTM(config).double().to(device=self.device)
        print("Loading model ...................................")
        self.model.load_state_dict(torch.load(config["model"]["pretrained"], map_location=torch.device('cpu')))
        print("Loaded model")
        self.model = self.model.to(device = self.device)
        
        self.embeddings = load_file_npy(config["model"]["embeddings"])

        self.vocab = load_json(config["model"]["vocab"])

        self.config = config
        

    def preprocessing(self, sentence_1, sentence_2):

        sentence_1 = text_to_word_list(sentence_1)
        sentence_2 = text_to_word_list(sentence_2)

        sentence_1 = [self.vocab[x] for x in sentence_1]
        sentence_2 = [self.vocab[x] for x in sentence_2]

        sentence_1 = pad_sequences([sentence_1], maxlen=self.config["model"]["max_seq_length"], padding="pre", truncating="post")
        sentence_2 = pad_sequences([sentence_2], maxlen=self.config["model"]["max_seq_length"], padding="pre", truncating="post")

        return torch.tensor(sentence_1, dtype=torch.int32), torch.tensor(sentence_2, dtype=torch.int32)

    def predict(self, sentence_1, sentence_2):

        sentence_1, sentence_2 = self.preprocessing(sentence_1, sentence_2)

        sentence_1 = sentence_1.to(device=self.device)
        sentence_2 = sentence_2.to(device=self.device)

        score = self.model(sentence_1, sentence_2)
        
        if score > 0.5:
            result = "Similarity"
        else:
            result = "Not Similarity"
        
        return {
            "score": float(score),
            "result": result
        }



        