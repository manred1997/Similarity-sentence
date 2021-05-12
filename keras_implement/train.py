from time import time
import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from gensim.models import KeyedVectors
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, \
    Bidirectional, TimeDistributed, Dropout, Conv1D, GlobalMaxPool1D
from keras.layers.merge import multiply, concatenate
import keras.backend as K

from model.model import shared_model

from utils import ManDist, load_file_npy
from config import config


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model", default="./en_SiameseLSTM_no_time.h5", type=str, help="Save model")
    parser.add_argument("--plot_training", action="store_true", help="Plot history training")
    args = parser.parse_args()

    X_train_left = load_file_npy(config["source"]["train"]["sentence_1"])
    X_train_right = load_file_npy(config["source"]["train"]["sentence_2"])
    Y_train = load_file_npy(config["source"]["train"]["label"])

    X_dev_left = load_file_npy(config["source"]["dev"]["sentence_1"])
    X_dev_right = load_file_npy(config["source"]["dev"]["sentence_2"])
    Y_dev = load_file_npy(config["source"]["dev"]["label"])

    embeddings = load_file_npy(config["model"]["embeddings"])

    left_input = Input(shape=(config["model"]["max_seq_length"],), dtype='float32')
    right_input = Input(shape=(config["model"]["max_seq_length"],), dtype='float32')
    left_sen_representation = shared_model(left_input, embeddings, config["model"]["embedded_size"], config["model"]["max_seq_length"], config["model"]["hidden_size"])
    right_sen_representation = shared_model(right_input, embeddings, config["model"]["embedded_size"], config["model"]["max_seq_length"], config["model"]["hidden_size"])


    man_distance = ManDist()([left_sen_representation, right_sen_representation])
    sen_representation = concatenate([left_sen_representation, right_sen_representation, man_distance])
    similarity = Dense(1, activation='sigmoid')(Dense(2)(Dense(4)(Dense(16)(sen_representation))))
    
    model = Model(inputs=[left_input, right_input], outputs=[similarity])

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    # training model
    training_start_time = time()
    malstm_trained = model.fit([X_train_left, X_train_right], Y_train,
                               batch_size=config["model"]["batch_size"], epochs=config["model"]["epoch"],
                               validation_data=([X_dev_left, X_dev_right], Y_dev))
    training_end_time = time()
    
    print("Training time finished.\n%d epochs in %12.2f" % (config["model"]["epoch"], training_end_time - training_start_time))

    # save model
    model.save(args.save_model)

    # Plot accuracy
    if args.plot_training:
        plt.subplot(211)
        plt.plot(malstm_trained.history['accuracy'])
        plt.plot(malstm_trained.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot loss
        plt.subplot(212)
        plt.plot(malstm_trained.history['loss'])
        plt.plot(malstm_trained.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout(h_pad=1.0)
        plt.savefig('./history-graph_no_time.png')

        
        print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
            "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
        print("Done.")
