import keras
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model, load_model
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import pickle

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

    """
    image1data.pkl: images_G1 for image (4, 172, 196)
    image2data.pkl: images_G2 for image (6, 140, 278)
    """
    
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

def test():
    
    all_labels = np.zeros((6, 140, 278)) #(4, 172, 196)
    data = data_preprocess('single', 'G2', True)
    _, _, _, original_image, images_G = data
    model = DCEC(opt.input_shape, opt.filters, opt.kernel_size, opt.n_clusters, opt.weights, data, opt.alpha, pretrain=True)
    model.compile(loss=['kld', 'binary_crossentropy'], optimizer='adam')
    
    for i in range(6):
        print("Current image", i)
        test_data = images_G[i].reshape(np.prod(images_G[i].shape[:2]), -1)[:,:,np.newaxis]

        q, _ = model.model.predict(test_data)
        cur_label = np.argmax(q, axis = 1)
        labels = cur_label.reshape(original_image.shape[0], original_image.shape[1])
        plt.title('Image no. {}'.format(i))
        plt.imshow(labels)
        plt.savefig('graph/results-dcec/image2_{}.png'.format(i))

        all_labels[i,:,:] = labels

        # for j in range(5, 8):
        #     model = DCEC(opt.input_shape, opt.filters, opt.kernel_size, opt.clusters, opt.weights, data, opt.alpha, pretrain=True)
        #     model.compile(loss=['kld', 'binary_crossentropy'], optimizer='adam')

        #     q, _ = model.predict(test_data)
        #     cur_label = np.argmax(q, axis = 1)
        #     labels = cur_label.reshape(original_image.shape[0], original_image.shape[1])
        #     all_labels[i,j-5,:,:] = labels

        # plt.figure(figsize=(30,10))
        # for j in range(3):
        #     plt.subplot(1,3,j+1)
        #     plt.title('Current cluster number: {}'.format(j+5))
        #     plt.imshow(all_labels[i,j,:,:])
        # plt.savefig('graph/results-dcec/image1_{}.png'.format(i))
    
    with open('graph/results-dcec/image2.pkl', 'wb') as f:
        pickle.dump(all_labels, f)


if __name__ == '__main__':
    mode = 'dcec'
    #train(mode)
    test()








