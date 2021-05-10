from time import time
import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import keras
from gensim.models import KeyedVectors
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, \
    Bidirectional, TimeDistributed, Dropout, Conv1D, GlobalMaxPool1D
from keras.layers.merge import multiply, concatenate
import keras.backend as K

from model.model import shared_model, shared_model_cnn

from util import make_w2v_embeddings, split_and_zero_padding, ManDist, load_data, load_word2vec
from config import EMBEDDING_DIM, MAX_SEQ_LENGTH, FLAG, SPLIT_RATIO, BATCH_SIZE, N_EPOCH, N_HIDDEN


def preprocessing_data(data, embedding_dict, flag, embedding_dim, max_seq_length=MAX_SEQ_LENGTH, split_ratio=SPLIT_RATIO):
    train_df, embeddings = make_w2v_embeddings(flag, embedding_dict, data, embedding_dim=embedding_dim)
    X = train_df[['question1_n', 'question2_n']]
    Y = train_df['is_duplicate']

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=split_ratio)
    X_train = split_and_zero_padding(X_train, max_seq_length)
    X_validation = split_and_zero_padding(X_validation, max_seq_length)

    Y_train = Y_train.values
    Y_validation = Y_validation.values

    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)

    return X_train, Y_train, X_validation, Y_validation, embeddings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="../../data_quora/quora_train_new.csv", type=str, help="Path of training and dev data")
    parser.add_argument("--embedding_path", default="./GoogleNews-vectors-negative300.bin.gz", type=str, help="Path of word to vector")
    parser.add_argument("--save_model", default="./en_SiameseLSTM_new.h5", type=str, help="Save model")
    parser.add_argument("--plot_training", action="store_true", help="Plot history training")
    args = parser.parse_args()

    embedding_dict = load_word2vec(path_file=args.embedding_path)

    train_df = load_data(path_file=args.train_data)

    X_train, Y_train, X_validation, Y_validation, embeddings = preprocessing_data(data=train_df, embedding_dict=embedding_dict, flag=FLAG, embedding_dim=EMBEDDING_DIM)

    left_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='float32')
    right_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='float32')
    left_sen_representation = shared_model(left_input, embeddings, EMBEDDING_DIM, MAX_SEQ_LENGTH, N_HIDDEN)
    right_sen_representation = shared_model(right_input, embeddings, EMBEDDING_DIM, MAX_SEQ_LENGTH, N_HIDDEN)


    man_distance = ManDist()([left_sen_representation, right_sen_representation])
    sen_representation = concatenate([left_sen_representation, right_sen_representation, man_distance])
    similarity = Dense(1, activation='sigmoid')(Dense(2)(Dense(4)(Dense(16)(sen_representation))))
    
    model = Model(inputs=[left_input, right_input], outputs=[similarity])

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    # training model
    training_start_time = time()
    malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                               batch_size=BATCH_SIZE, epochs=N_EPOCH,
                               validation_data=([X_validation['left'], X_validation['right']], Y_validation))
    training_end_time = time()
    
    print("Training time finished.\n%d epochs in %12.2f" % (N_EPOCH, training_end_time - training_start_time))

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
        plt.savefig('./history-graph_new.png')

        
        print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
            "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
        print("Done.")
