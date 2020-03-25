import keras
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model, load_model
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

from utils import *
from model import CAE_model
from model import DCEC, AttentionModel
from config import opt

def pretrain():

    (x_train, x_test, test_data) = data_preprocess('single', 'G2')

    autoencoder = CAE_model()
    autoencoder.summary()

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    #autoencoder.compile(optimizer='sgd', loss='mean_squared_error')

    #autoencoder = load_model('model.h5')

    # Pretrain the model
    history = autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test, x_test))
    
    # Extract features
    feature_extractor = Model(autoencoder.input, autoencoder.get_layer('max_pooling1d_3').output)
    features = feature_extractor.predict(test_data)
    features = features.squeeze(-1)

def train(mode):
    
    data = data_preprocess('single', 'G2')
    print("1. Get data ready!")

    if mode == 'dcec':
        model = DCEC(opt.input_shape, opt.filters, opt.kernel_size, opt.n_clusters, opt.weights, data, opt.alpha, pretrain=True)
        model.compile(loss=['kld', 'binary_crossentropy'], optimizer='adam')
        print("3. Compile model!")
        
        model.fit(data, opt)
    
    elif mode == 'attention':
        model = AttentionModel(opt.input_shape, opt.filters, opt.kernel_size, opt.n_clusters, opt.weights, data, opt.alpha, pretrain=True)
        model.compile(optimizer='adam')
        print("3. Compile model!")

        model.fit(data, opt)

        model.predict(data)


    # labels = model.cur_label.reshape(data[3].shape[0], data[3].shape[1])
    # plt.title('Final Output cluster:')
    # plt.imshow(labels)
    # plt.savefig('final_2.png')


if __name__ == '__main__':
    mode = 'dcec'
    train(mode)








