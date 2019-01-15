class Config:
    # Data
    data_dir = 'data/cifar-10-data/'

    # Architecture
    n_hidden_1 = 256
    n_hidden_2 = 256

    # Parameters
    learning_rate = 0.01
    training_epochs = 10
    batch_size = 128
    display_step = 1

opt = Config()