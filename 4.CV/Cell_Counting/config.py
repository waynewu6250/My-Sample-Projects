class Config:

    # split images
    slices = 2

    # images
    train_path = './imgs_to_use/train_imgs/'
    label_path = './imgs_to_use/label_imgs/'
    h5_path = './imgs_to_use/h5_files/'

    # size
    train_size = 1
    val_size = 1
    batch_size = 4


opt = Config()