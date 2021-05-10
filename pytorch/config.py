config = {
    "source": {
        "word2vec": "../GoogleNews-vectors-negative300.bin.gz",
        "data": "../../../data_quora/quora_train_10000.csv"
    },
    "model": {
        "embedded_size": 300,
        "max_seq_length": 30,
        "batch_size": 32,
        "hidden_size": 50,
        "num_layers": 1,
        "bidirectional": True,
        "dropout": 0.2,
        "embeddings": "embeddings.npy"
    },
    "X_left": "sen_1.npy",
    "X_right": "sen_2.npy",
    "label": "label.npy"
}