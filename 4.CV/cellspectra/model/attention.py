from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, UpSampling1D, BatchNormalization, LSTM, RepeatVector, Lambda, Dense, Activation, Reshape
from keras.models import Model, load_model

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from model.cae import CAE_model



class Attention:

    def __init__(self, input_shape, filters, kernel_size, n_clusters, weights, data, alpha=1.0, pretrain=True):
        
        if pretrain:
            self.autoencoder = load_model('model.h5')
        else:
            print("Start Pretraining...")
            self.autoencoder = CAE_model(input_shape, filters, kernel_size)
            (x_train, x_test, test_data, _) = data
            self.pretrain_model(x_train, x_test)
            print("Pretraining Complete")
        
        features = self.autoencoder.get_layer('max_pooling1d_6').output
        
        self.feature_extractor = Model(self.autoencoder.input, features)
        
        features_s = Lambda(lambda x: K.squeeze(x, axis=2))(features)
        probs = ClusterLayer(n_clusters, alpha, name='cluster')(features_s)
        self.model = Model(inputs = self.autoencoder.input, outputs=[probs, self.autoencoder.output])

        if weights:
            self.model.load_weights(weights)
            print("2. Successfully loading weights!!")
    
    def pretrain_model(self, x_train, x_test):

        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        self.autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test, x_test))
        
        self.autoencoder.save('checkpoints-dcec/model.h5')
        self.pretrain = True
    
    @staticmethod
    def target(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
    
    def compile(self, loss, optimizer):
        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, data, opt):

        (x_train, x_test, test_data, _) = data
        
        # Initialize cluster centers by K-Means
        features = self.feature_extractor.predict(test_data)
        features = features.squeeze(-1)

        kmeans_model = KMeans(n_clusters=opt.n_clusters, n_init = 20, random_state=1)
        prev_label = kmeans_model.fit_predict(features)
        self.model.get_layer(name='cluster').set_weights([kmeans_model.cluster_centers_])

        # Start deep clustering training
        index = 0
        for iter in range(opt.max_iter):

            # Update our target distribution
            if iter % opt.update_interval == 0:

                q, _ = self.model.predict(test_data)
                p = self.target(q)
                self.cur_label = np.argmax(q, axis = 1)

                # Check when to stop
                diff = np.sum(self.cur_label != prev_label).astype(np.float32) / self.cur_label.shape[0]
                prev_label = np.copy(self.cur_label)
                if iter > 0 and diff < opt.tol:
                    print('Difference ', diff, 'is smaller than tol ', opt.tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            
            # train on batch
            if (index + 1) * opt.batch_size > test_data.shape[0]:
                loss = self.model.train_on_batch(x=test_data[index * opt.batch_size::],
                                                 y=[p[index * opt.batch_size::], test_data[index * opt.batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=test_data[index * opt.batch_size:(index + 1) * opt.batch_size],
                                                 y=[p[index * opt.batch_size:(index + 1) * opt.batch_size],
                                                    test_data[index * opt.batch_size:(index + 1) * opt.batch_size]])
                index += 1

            # save intermediate model
            if (iter+1) % opt.save_interval == 0:
                # save DCEC model checkpoints
                print('Saving model no.', iter)
                self.model.save_weights('checkpoints-dcec/dcec_model_' + str(iter) + '.h5')
                
                labels = self.cur_label.reshape(data[3].shape[0], data[3].shape[1])
                plt.title('Final Image')
                plt.imshow(labels)
                plt.savefig('graph/dcec/final_{}.png'.format(iter))
            


            








        


        
        



        
        









