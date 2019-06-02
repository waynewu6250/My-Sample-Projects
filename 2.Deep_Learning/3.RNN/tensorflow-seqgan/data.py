import numpy as np
import json
import re
import pickle
from eval_model.config import opt

class CaptionData:

    def __init__(self, opt):
        # Load dictionary
        self.word2id = self.load_pickle(opt.data_path)["word2id"]
        self.id2word = self.load_pickle(opt.data_path)["id2word"]
        self.pad = len(self.word2id) #Last token

        # Load data
        self.data, self.labbel = self.load_data(opt)

        # Current batch indicator
        self.pointer = 0

    def load_data(self, opt):

        # Load data
        training_data = self.load_pickle(opt.pred_path)
        ground_truths = [[self.word2id[tok] for tok in self.text_prepare(string[0])] \
                        for img_path, string in training_data["train_gts"].items()]
        predict_results = [[self.word2id[tok] for tok in self.text_prepare(string[0])] \
                        for img_path, string in training_data["train_res"].items()]

        data = np.array(ground_truths+predict_results)
        
        # Generate labels
        positive_labels = [[0, 1] for _ in ground_truths]
        negative_labels = [[1, 0] for _ in predict_results]
        labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]

        # Split batches
        self.self.num_batch = int(len(labels) / opt.batch_size)
        data = data[:self.num_batch * opt.batch_size]
        labels = labels[:self.num_batch * opt.batch_size]
        data = np.split(data, self.num_batch, 0)
        labels = np.split(labels, self.num_batch, 0)

        return data, labels
    
    def next_batch(self):
        ret = self.data[self.pointer], self.labels[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

    # Load pickle files
    @staticmethod
    def load_pickle(file):
        with open(file,'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_pickle(data, file):
        with open(file,'wb') as f:
            pickle.dump(data, f)
    
    # Prepare text
    @staticmethod
    def text_prepare(text):
        """
            text: a string
            
            return: modified string tokens 
                    [tok1, tok2 , ...] which is a single sentence from one character
        """
        REPLACE_BY_SPACE_RE = re.compile(r'[-(){}\[\]\|@;]')
        BAD_SYMBOLS_RE = re.compile(r'[#+_]')
        mxlen = 50

        tok = ["<START>"] # add START token to represent sentence start
        text = text.lower() # lowercase text
        text = re.sub(REPLACE_BY_SPACE_RE, ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = re.sub(BAD_SYMBOLS_RE, '', text) # delete symbols which are in BAD_SYMBOLS_RE from text
        tok += (text.split(" ")+["<EOS>"]) # add EOS token to represent sentence end
        if len(tok) > mxlen:
            tok = tok[:mxlen]
        
        return tok


