class Config:

    # split images
    slices = 2

    # data type
    data_type = 'bacteria' #'cell'

    # bacteria images
    train_path = './imgs_to_use/train_imgs/'
    label_path = './imgs_to_use/label_imgs/'
    h5_path = './imgs_to_use/h5_files/'
    color = 'green'

    # cell images
    cell_h5_path = 'cell_h5_files/'

    # size
    train_size = 150 if data_type == 'cell' else 20
    val_size = 50 if data_type == 'cell' else 8
    batch_size = 8 if data_type == 'cell' else 2

    # hyperparameters
    model = "UNet_cell" if data_type == 'cell' else "UNet" #"FCRN_A"
    h_flip = 0.0
    v_flip = 0.0
    unet_filters = 64
    conv = 2
    learning_rate = 0.01 if data_type == 'cell' else 0.002
    epochs = 50

    plot = False

    # test_img
    test_path = "imgs_to_use/train_imgs/62x_Salac_Pa14wt_SaPa14wt=1-1,1-10,100-1,10-1_co_SCFM2_tile2x2_3-13-19_z25_t00_p3_m3.tif-1.jpg"
    cell_test_path = "001cell.png"




opt = Config()