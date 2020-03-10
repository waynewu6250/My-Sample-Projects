from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from cae import CAE_model

class ClusterLayer(Layer):
    """
    Input: features | shape: (N, feature_dim)
    Ouptut: Probability of features belonging each cluster | shape: (N, C)
    Weights: Cluster Centers | shape: (C, feature_dim)
    """

    def __init__(self, n_clusters, **kwargs):
        
        super(ClusterLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
    
    def build(self, input_shape):

        D = input_shape[1]
        
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, D))
        self.clusters = self.add_weight((self.n_clusters, D), initializer='glorot_uniform', name='clusters')
        self.built = True



class DCEC:

    def __init__(self):
        pass