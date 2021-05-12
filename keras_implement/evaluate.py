from time import time
import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from gensim.models import KeyedVectors
from utils import load_model, load_file_npy
from config import config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./en_SiameseLSTM_no_time.h5", type=str, help="Model path")
    args = parser.parse_args()

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