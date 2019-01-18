import os
class Config:
    # Data
    data_dir = os.path.join('data/cifar10_data', 'cifar-10-batches-bin')

    # Architecture
    n_hidden_1 = 256
    n_hidden_2 = 256

    # Parameters
    lr = 0.01
    training_epochs = 10
    batch_size = 128
    display_step = 1

opt = Config()