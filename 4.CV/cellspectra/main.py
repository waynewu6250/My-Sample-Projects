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

if __name__ == '__main__':
    pretrain()








