from tflearn.data_utils import to_categorical
from tflearn.datasets import imdb
import numpy as np

class IMDBDataset:
    def __init__(self, X, Y, maxlen):
        self.xdata = self.pad_sequences(X, maxlen)
        self.ydata = to_categorical(Y, nb_classes=2)
        self.num_examples = len(X)
        self.ptr = 0
    
    def next_batch(self, size):
        if self.ptr+size < self.num_examples:
            xbatch = self.xdata[self.ptr:self.ptr+size]
            ybatch = self.ydata[self.ptr:self.ptr+size]
        else:
            xbatch = np.concatenate((self.xdata[:self.ptr],
                                     self.xdata[:size-len(self.xdata[self.ptr:])]))
            ybatch = np.concatenate((self.ydata[:self.ptr],
                                     self.ydata[:size-len(self.ydata[self.ptr:])]))
        self.ptr = (self.ptr+size) % self.num_examples

        return xbatch, ybatch
    
    @staticmethod
    def pad_sequences(X, maxlen):
        new_seqs = np.zeros((len(X), maxlen))
        for i,seq in enumerate(X):
            if len(seq) <= maxlen:
                new_seqs[i, :len(seq)] = seq
            else:
                new_seqs[i, :] = seq[:maxlen]
        return new_seqs


(x_train, y_train), (x_val, y_val), _ = imdb.load_data('imdb.pkl', n_words=10000, valid_portion=0.1)
train_data = IMDBDataset(x_train, y_train, 100)
val_data = IMDBDataset(x_val, y_val, 100)



