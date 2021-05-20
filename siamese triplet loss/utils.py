from keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors

import numpy as np
import pandas as pd
import itertools

import json



def text_to_word_list(text):
    text = str(text)
    text = text.lower()

    import re
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def make_w2v_embeddings(word2vec, df, embedding_dim):
    vocabs = {}
    vocabs_cnt = 0

    vocabs_not_w2v = {}
    vocabs_not_w2v_cnt = 0

    
    for index, row in df.iterrows():
        # 打印处理进度
        if index != 0 and index % 100000 == 0:
            print(str(index) + " sentences embedded.")

        for question in ['question1', 'question2']:
            q2n = []  # q2n -> question to numbers representation
            words = text_to_word_list(row[question])

            for word in words:
                # if word in stops:  # 去停用词
                    # continue
                if word not in word2vec and word not in vocabs_not_w2v:  
                    vocabs_not_w2v_cnt += 1
                    vocabs_not_w2v[word] = vocabs_not_w2v_cnt
                if word not in vocabs:  
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])
            df.at[index, question + '_n'] = q2n

    embeddings = 1 * np.random.randn(len(vocabs) + 2, embedding_dim)
    '''
    词1 [a1, a2, a3, ..., a60]
    词2 [b1, b2, b3, ..., b60]
    词3 [c1, c2, c3, ..., c60]
    '''
    embeddings[0] = 0 

    for index in vocabs:
        vocab_word = vocabs[index]
        if vocab_word in word2vec:
            embeddings[index] = word2vec[vocab_word]
    embeddings[-1] = word2vec["UNK"]
    vocabs["UNK"] = vocabs_cnt + 1
    del word2vec

    return df, embeddings, vocabs, vocabs_not_w2v


def split_and_zero_padding(df, max_seq_length):

    X = {'left': df['question1_n'], 'right': df['question2_n']}

    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='post', truncating='post', maxlen=max_seq_length)

    return dataset


def load_word2vec(file_path):
    print("Loading word2vec model(it may takes 2-3 mins) ...")
    embedding_dict = KeyedVectors.load_word2vec_format(file_path, binary=True)
    return embedding_dict

def load_data(file_path):
    df = pd.read_csv(file_path)
    for q in ['question1', 'question2']:
        df[q + '_n'] = df[q]
    return df

def load_file_npy(file_path):
    import numpy as np
    with open(file_path, "rb") as f:
        return np.load(f)

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data