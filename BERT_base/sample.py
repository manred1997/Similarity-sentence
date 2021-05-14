import os
import re

from tqdm import tqdm

import numpy as np
import pandas as pd

from tokenizers import BertWordPieceTokenizer

from utils import convert_to_unicode, _truncate_seq_pair


def convert_example_feature(index, example, label_list, tokenizer, config):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    
    tokens_text_left = tokenizer.tokenize(example.text_left)

    tokens_text_right = None
    if example.text_right:
        tokens_text_right = tokenizer.tokenize(example.text_right)
    
    if tokens_text_right:
        _truncate_seq_pair(tokens_text_left, tokens_text_right, config)
    
    else:
        if len(tokens_text_left) > config["model"]["max_seq_leght"] - 2:
            tokens_text_left = tokens_text_left[0: (config["model"]["max_seq_leght"] - 2)]
    
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:                      [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  segment_ids/token_ids:       0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:                  [CLS] the dog is hairy . [SEP]
    #  segment_ids/token_ids:   0     0   0   0  0     0 0

    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    tokens.extend(tokens_text_left)
    tokens.append("[SEP]")

    segment_ids.extend([0]*len(tokens))

    if tokens_text_right:
        tokens.extend(tokens_text_right)
        tokens.append("[SEP]")
    
        segment_ids.extend([1]*(len(tokens_text_right)+1))
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1]*len(input_ids)

    padding_length = config["model"]["max_seq_length"] - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
    
    assert len(input_ids) == len(segment_ids) == len(input_mask) == config["model"]["max_seq_length"]

    label_id = label_map[example.label]

    if index < 5:
        pass #write log

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id)
    return feature

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_left, text_right=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_left: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_right: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_left = text_left
    self.text_right = text_right
    self.label = label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, "train.csv")
        data = pd.read_csv(file_path)
        examples = []
        for index, row in enumerate(data.iterrows()):
            guid = 'train-%d' %index
            
            text_left = convert_to_unicode(row["question1"])
            text_right = convert_to_unicode(row["question2"])

            label = convert_to_unicode(row["is_duplicate"])

            examples.append(InputExample(guid=guid,
                                        text_left=text_left,
                                        text_right=text_right,
                                        label=label))
            print(f"the length of train data : {len(examples)}")
        return examples
    
    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, "dev.csv")
        data = pd.read_csv(file_path)
        examples = []
        for index, row in enumerate(data.iterrows()):
            guid = 'dev-%d' %index
            
            text_left = convert_to_unicode(row["question1"])
            text_right = convert_to_unicode(row["question2"])

            label = convert_to_unicode(row["is_duplicate"])

            examples.append(InputExample(guid=guid,
                                        text_left=text_left,
                                        text_right=text_right,
                                        label=label))
        print(f"the length of dev data : {len(examples)}")
        return examples

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, "test.csv")
        data = pd.read_csv(file_path)
        examples = []
        for index, row in enumerate(data.iterrows()):
            guid = 'test-%d' %index
            
            text_left = convert_to_unicode(row["question1"])
            text_right = convert_to_unicode(row["question2"])

            label = convert_to_unicode(row["is_duplicate"])

            examples.append(InputExample(guid=guid,
                                        text_left=text_left,
                                        text_right=text_right,
                                        label=label))
        print(f"the length of test data : {len(examples)}")
        return examples

    def get_labels(self):
        return ["0", "1"]

class Sample:
    def __init__(self, examples, label_list, tokenizer, config) -> None:
        
        self.examples = examples
        self.label_list = label_list

        self.tokenizer = tokenizer
        
        self.config = config

    def preprocessing(self):

        dataset_dict = {
            "input_ids": [],
            "input_mask": [],
            "segment_ids": [],
            "label_id": [],
            "is_real_example": []
        }

        for index, example in enumerate(self.examples):
            if index % 1000 == 0:   print("Writing example %d of %d" % (index, len(example)))

            feature = convert_example_feature(index, example, self.label_list, self.tokenizer, self.config)

            for key in dataset_dict:
                dataset_dict[key].append(getattr(feature, key))
        
        for key in dataset_dict:
            dataset_dict[key] = np.array(dataset_dict[key])
        
        X = [dataset_dict["input_ids"], dataset_dict["input_mask"], dataset_dict["segment_ids"]]
        Y = [dataset_dict["label_id"]]

        return X, Y