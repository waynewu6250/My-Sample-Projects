from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, UpSampling1D, BatchNormalization, LSTM, RepeatVector, Lambda, Dense, Activation, Reshape
from keras.models import Model, load_model

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from model.cae import CAE_model

class WeightedEmbedding(Layer):
    """
    Compute weighted sum between cluster embeddings and scores
    return shape (N, H)
    """
    def __init__(self, hidden_dim, **kwargs):
        super(WeightedEmbedding, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
    
    def compute_mask(self, input_tensor, mask=None):
        return None
    
    def build(self, input_shape):
        
        n_clusters = input_shape[1]
        self.W = self.add_weight(shape=(n_clusters, self.hidden_dim),
                                 initializer='glorot_uniform', name='W_embedding')
        self.built = True

    def call(self, input_tensor):
        return K.dot(input_tensor, self.W)

class CustomLoss(Layer):
    """
    Compute losses: reconstruction loss + negative loss
    return loss
    """

    def __init__(self, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)

    def norm(self, tensor):
        return K.cast(K.epsilon() + K.sqrt(K.sum(K.square(tensor), axis=-1, keepdims=True)), K.floatx())
    
    def call(self, input_tensor):

        z_s, r_s = input_tensor[0], input_tensor[1]

        z_s = z_s / self.norm(z_s)
        r_s = r_s / self.norm(r_s)

        pos = K.sum(z_s*r_s, axis=-1, keepdims=False)
        loss = K.cast(K.sum(K.maximum(0., (1. - pos ))), K.floatx())
        
        self.add_loss(loss, inputs = input_tensor)
        return loss

    def compute_mask(self, input_tensor, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return 1

class AttentionModel:

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
        preds = Dense(n_clusters)(features_s)
        scores = Activation('softmax', name='scores')(preds)
        r_s = WeightedEmbedding(features.shape[1], name='cluster')(scores) # NxH
        
        # Calculate loss
        loss = CustomLoss()([features_s, r_s])

        self.model = Model(inputs = self.autoencoder.input, outputs=loss)

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

    
    def compile(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=None)

    def fit(self, data, opt):
        
        (x_train, x_test, test_data, _) = data
        
        # Initialize cluster centers by K-Means
        features = self.feature_extractor.predict(test_data)
        features = features.squeeze(-1)

        kmeans_model = KMeans(n_clusters=opt.n_clusters, n_init = 20, random_state=1)
        prev_label = kmeans_model.fit_predict(features)
        self.model.get_layer(name='cluster').set_weights([kmeans_model.cluster_centers_])
        self.model.get_layer(name='cluster').trainable=False

        # Train with attention
        self.model.fit(x = x_train,
              epochs=1,
              batch_size=100,
              shuffle=True,
              validation_split=0.1)
        self.model.save_weights('checkpoints/attention_model.h5')

    def predict(self, data):
        
        (x_train, x_test, test_data, _) = data
        
        test_fn = K.function(self.model.input, self.model.get_layer('scores').output)
        probs = test_fn(test_data)
        self.cur_label = np.argmax(probs, axis = 1)
        
        labels = self.cur_label.reshape(data[3].shape[0], data[3].shape[1])
        plt.title('Final Image')
        plt.imshow(labels)
        plt.savefig('graph/attention/final.png')
            


            








        


        
        



        
        









