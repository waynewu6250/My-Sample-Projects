class Config:

    # Model
    input_shape = (1600, 1)
    filters = [16, 8, 1]
    kernel_size = 3
    alpha = 1

    # Training
    batch_size = 1024
    n_clusters = 6
    max_iter = 100
    update_interval = 10
    save_interval = 10
    tol = 1e-3

    weights = 'checkpoints/dcec_model_2_9.h5'

opt = Config()