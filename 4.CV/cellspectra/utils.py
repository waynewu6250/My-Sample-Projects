import numpy as np
import matplotlib.pyplot as plt
import h5py

def plot_history(history):
    
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")

def crop_image(data, N_H, N_W):

    images = []
    H, W, C = data.shape
    mid_H = H//N_H
    mid_W = W//N_W
    images.append(data[:mid_H, :mid_W, :])
    images.append(data[mid_H:, :mid_W, :])
    images.append(data[:mid_H, mid_W:, :])
    images.append(data[mid_H:, mid_W:, :])

    return images

def data_preprocess(mode, data_to_use):

    # Data Preprocessing

    if data_to_use == "G1":

        with h5py.File('MOSAIC_Slices_8 9 10 11 12 13 _PROCESS_same.h5', 'r') as f:
            base_items = list(f.items())
            G1 = f.get('/BCARSImage').get("/BCARSImage/Z97 to 105_58").get("Z97 to 105_58_z13_SubDark_MergeNRBs_Anscombe_SVD_InvAnscombe_KK_PhaseErrorCorrectALS_ScaleErrorCorrectSG_SubtractROI")
            G1_items = np.array(G1)
        
        data_G1 = (G1_items['Re']**2+G1_items['Im']**2)**0.5
        images_G1 = crop_image(data_G1, 3, 2)
    
    elif data_to_use == "G2":

        with h5py.File('MOSAIC_HCMV01_2018925_15_18_27_40030_PROCESS_2018925_16_29_5_508729_(3)_PROCESS_2019122_2_0_59_82982_December_2019.h5', 'r') as f:
            base_items = list(f.items())
            G2 = f.get('/BCARSImage').get('/BCARSImage/AlgaeI_3_5ms_Pos_0_11').get('AlgaeI_3_5ms_Pos_0_11_z16-19_SubDark_Anscombe_SVD_InvAnscombe_MergeNRBs_KK_PhaseErrorCorrectALS_ScaleErrorCorrectSG_Continue_SubtractROI')
            G2_items = np.array(G2)
    
        data_G2 = (G2_items['Re']**2+G2_items['Im']**2)**0.5
        images_G2 = crop_image(data_G2, 2, 2)

    # Take the first N image
    if mode == "all":
        
        N = 3

        images = np.stack(images_G2[:N])
        images = images.reshape(np.prod(images.shape[:3]), -1)[:,:,np.newaxis]

        x_train = images
        x_test = images_G2[3].reshape(np.prod(images_G2[3].shape[:2]), -1)[:,:,np.newaxis]

        test_data = images
    
    # Take the first image
    elif mode == "single":
        
        data_G2 = images_G2[0].reshape(np.prod(images_G2[0].shape[:2]), -1)[:,:,np.newaxis]

        N = len(data_G2)
        train_size = int(np.floor(0.8*N))
        indices = np.random.permutation(N)

        x_train = data_G2[indices[:train_size], :]
        x_test = data_G2[indices[train_size:], :]

        test_data = data_G2
    
    return (x_train, x_test, test_data, images_G2[0])




