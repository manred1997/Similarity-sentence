config = {
    "source": {
        "word2vec": "../GoogleNews-vectors-negative300.bin.gz",
        "train": {
          "sentence_1": "../../../data_quora/data_mini/numpy/train/sentence_1.npy",
          "sentence_2": "../../../data_quora/data_mini/numpy/train/sentence_2.npy",
          "label": "../../../data_quora/data_mini/numpy/train/label.npy"
        },
        "dev": {
          "sentence_1": "../../../data_quora/data_mini/numpy/dev/sentence_1.npy",
          "sentence_2": "../../../data_quora/data_mini/numpy/dev/sentence_2.npy",
          "label": "../../../data_quora/data_mini/numpy/dev/label.npy"
        },
        "test": {
          "sentence_1": "../../../data_quora/data_mini/numpy/test/sentence_1.npy",
          "sentence_2": "../../../data_quora/data_mini/numpy/test/sentence_2.npy",
          "label": "../../../data_quora/data_mini/numpy/test/label.npy"
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
        "embeddings": "../data/embeddings.npy",
        "epoch": 30,
        "attention_size" : 700,
        "pretrained": "model.pth",
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "p_drop": 0.2
    }
}