class Config:

    # split images
    slices = 2

    # images
    train_path = './imgs_to_use/train_imgs/'
    label_path = './imgs_to_use/label_imgs/'
    h5_path = './imgs_to_use/h5_files/'
    color = 'red'

    # size
    train_size = 3
    val_size = 1
    batch_size = 1

    # hyperparameters
    model = "UNet" #"FCRN_A"
    h_flip = 0.9
    v_flip = 0.9
    unet_filters = 64
    conv = 2
    learning_rate = 0.2
    epochs = 150

    plot = True




opt = Config()