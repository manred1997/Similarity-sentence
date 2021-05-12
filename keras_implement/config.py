config = {
    "source": {
        "word2vec": "../GoogleNews-vectors-negative300.bin.gz",
        "train": {
          "sentence_1": "./data/train/sentence_1.npy",
          "sentence_2": "./data/train/sentence_2.npy",
          "label": "./data/train/label.npy"
        },
        "dev": {
          "sentence_1": "./data/dev/sentence_1.npy",
          "sentence_2": "./data/dev/sentence_2.npy",
          "label": "./data/dev/label.npy"
        },
        "test": {
          "sentence_1": "./data/test/sentence_1.npy",
          "sentence_2": "./data/test/sentence_2.npy",
          "label": "./data/test/label.npy"
        }
    },
    "model": {
        "embedded_size": 300,
        "max_seq_length": 30,
        "batch_size": 1024,
        "hidden_size": 50,
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.2,
        "embeddings": "./data/embeddings.npy",
        "epoch": 20
    }
}