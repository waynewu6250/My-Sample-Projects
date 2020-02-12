class Config:

    # split images
    slices = 2

    # images
    train_path = './imgs_to_use/train_imgs/'
    label_path = './imgs_to_use/label_imgs/'
    h5_path = './imgs_to_use/h5_files/'
    color = 'red'

    # size
    train_size = 2
    val_size = 2
    batch_size = 1

    # hyperparameters
    model = "UNet" #"FCRN_A"
    h_flip = 0.9
    v_flip = 0.9
    unet_filters = 64
    conv = 2
    learning_rate = 0.002
    epochs = 10

    plot = False

    # test_img
    test_path = "imgs_to_use/train_imgs/62x_Salac_Pa14wt_SaPa14wt1-11-10100-110-1_co_SCFM2_tile2x2_4-17-19_z19_t01_p3_m3.tif-2.jpg"




opt = Config()