from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import numpy as np

class IMDBDataset:
    def __init__(self, X, Y):
        self.xdata = X
        self.ydata = Y
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


(x_train, y_train), (x_test, y_test), _ = imdb.load_data('imdb.pkl', n_words=10000, valid_portion=0.1)
x_train = pad_sequences(x_train, maxlen=500, value=0.)
x_test = pad_sequences(x_train, maxlen=500, value=0.)
y_train = to_categorical(y_train, nb_classes=2)
y_test = to_categorical(y_test, nb_classes=2)

train_data = IMDBDataset(x_train, y_train)
test_data = IMDBDataset(x_test, y_test)



