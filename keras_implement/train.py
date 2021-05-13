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

from model.model import shared_model, shared_model_cnn

from utils import make_w2v_embeddings, split_and_zero_padding, ManDist, load_data, load_word2vec
from config import config


# def preprocessing_data(data, embedding_dict, flag, embedding_dim, max_seq_length=MAX_SEQ_LENGTH, split_ratio=SPLIT_RATIO):
#     train_df, embeddings = make_w2v_embeddings(flag, embedding_dict, data, embedding_dim=embedding_dim)
#     X = train_df[['question1_n', 'question2_n']]
#     Y = train_df['is_duplicate']

#     X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=split_ratio)
#     X_train = split_and_zero_padding(X_train, max_seq_length)
#     X_validation = split_and_zero_padding(X_validation, max_seq_length)

#     Y_train = Y_train.values
#     Y_validation = Y_validation.values

#     assert X_train['left'].shape == X_train['right'].shape
#     assert len(X_train['left']) == len(Y_train)

#     return X_train, Y_train, X_validation, Y_validation, embeddings

def load_file_npy(file_path):
    import numpy as np
    with open(file_path, "rb") as f:
        return np.load(f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_data", default="../../data_quora/quora_train_10000.csv", type=str, help="Path of training and dev data")
    # parser.add_argument("--embedding_path", default="./GoogleNews-vectors-negative300.bin.gz", type=str, help="Path of word to vector")
    parser.add_argument("--save_model", default="./en_SiameseLSTM.h5", type=str, help="Save model")
    parser.add_argument("--plot_training", action="store_true", help="Plot history training")
    args = parser.parse_args()

    # embedding_dict = load_word2vec(path_file=args.embedding_path)

    # train_df = load_data(path_file=args.train_data)

    # X_train, Y_train, X_validation, Y_validation, embeddings = preprocessing_data(data=train_df, embedding_dict=embedding_dict, flag=FLAG, embedding_dim=EMBEDDING_DIM)

    X_train_left = load_file_npy(config["source"]["train"]["sentence_1"])
    X_train_right = load_file_npy(config["source"]["train"]["sentence_2"])
    Y_train = load_file_npy(config["source"]["train"]["label"])
    print(len(Y_train))
    assert X_train_left.shape[0] == X_train_right.shape[0] == len(Y_train) 

    X_dev_left = load_file_npy(config["source"]["test"]["sentence_1"])
    X_dev_right = load_file_npy(config["source"]["test"]["sentence_2"])
    Y_dev = load_file_npy(config["source"]["test"]["label"])
    print(len(Y_dev))

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
        plt.savefig('./history-graph.png')

        
        print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
            "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
        print("Done.")
