from time import time
import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from gensim.models import KeyedVectors
from utils import make_w2v_embeddings, split_and_zero_padding, ManDist, load_word2vec, load_data, load_model
# from config import EMBEDDING_DIM, MAX_SEQ_LENGTH, FLAG, SPLIT_RATIO, BATCH_SIZE, N_EPOCH, N_HIDDEN
from config import config

def load_file_npy(file_path):
    import numpy as np
    with open(file_path, "rb") as f:
        return np.load(f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", default="../../data_quora/quora_test_new.csv", type=str, help="Path of test data")
    parser.add_argument("--embedding_path", default="./GoogleNews-vectors-negative300.bin.gz", type=str, help="Path of word to vector")
    parser.add_argument("--model_path", default="./en_SiameseLSTM_no_time.h5", type=str, help="Model path")
    args = parser.parse_args()

    # embedding_dict = load_word2vec(path_file=args.embedding_path)

    # test_df = load_data(args.test_data)

    # test_df, embeddings = make_w2v_embeddings(FLAG, embedding_dict, test_df, embedding_dim=EMBEDDING_DIM)

    # X_test = split_and_zero_padding(test_df, MAX_SEQ_LENGTH)
    # Y_test = test_df['is_duplicate'].values

    # assert X_test['left'].shape == X_test['right'].shape
    # assert len(X_test['left']) == len(Y_test)

    #load data

    X_test_left = load_file_npy(config["source"]["test"]["sentence_1"])
    X_test_right = load_file_npy(config["source"]["test"]["sentence_2"])
    Y_test = load_file_npy(config["source"]["test"]["label"])

    # load model
    model = load_model(args.model_path)

    prediction = model.predict([X_test_left, X_test_right])
    # print(prediction)
    prediction_list = prediction.tolist()

    # Evaluate
    accuracy = 0
    for i in range(len(prediction_list)):
        if prediction_list[i][0] < 0.5:
            predict_pro = 0
        else:
            predict_pro = 1
        if predict_pro == Y_test[i]:
            accuracy += 1
    print(f"Accuracy: {accuracy / len(Y_test)}")