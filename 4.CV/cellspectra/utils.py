import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle

def plot_history(history):
    
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")

def crop_image2(data, N_H, N_W):

    images = []
    H, W, C = data.shape
    mid_H = H//N_H
    mid_W = W//N_W
    images.append(data[:mid_H, :mid_W, :])
    images.append(data[mid_H:, :mid_W, :])
    images.append(data[:mid_H, mid_W:, :])
    images.append(data[mid_H:, mid_W:, :])

    return images

def crop_image3(data, N_H, N_W):
    
    images = []
    H, W, C = data.shape
    mid_H = H//N_H
    mid_W = W//N_W
    images.append(data[:mid_H, :mid_W, :])
    images.append(data[mid_H:2*mid_H, :mid_W, :])
    images.append(data[2*mid_H:, :mid_W, :])
    images.append(data[:mid_H, mid_W:, :])
    images.append(data[mid_H:2*mid_H, mid_W:, :])
    images.append(data[2*mid_H:, mid_W:, :])

    return images

def data_preprocess(mode, data_to_use, return_all=False):
    
    # Data Preprocessing
    if data_to_use == "G1":

        with open('image1data.pkl', 'rb') as f:
            images_G = pickle.load(f)
    
    elif data_to_use == "G2":

        with open('image2data.pkl', 'rb') as f:
            images_G = pickle.load(f)

    # Take the first N image
    if mode == "all":
        
        N = 3

        images = np.stack(images_G[:N])
        images = images.reshape(np.prod(images.shape[:3]), -1)[:,:,np.newaxis]

        x_train = images
        x_test = images_G[3].reshape(np.prod(images_G[3].shape[:2]), -1)[:,:,np.newaxis]

        test_data = images
    
    # Take the first image
    elif mode == "single":
        
        data_G = images_G[0].reshape(np.prod(images_G[0].shape[:2]), -1)[:,:,np.newaxis]

        N = len(data_G)
        train_size = int(np.floor(0.8*N))
        indices = np.random.permutation(N)

        x_train = data_G[indices[:train_size], :]
        x_test = data_G[indices[train_size:], :]

        test_data = data_G
    
    if return_all:
        return (x_train, x_test, test_data, images_G[0], images_G)
    else:
        return (x_train, x_test, test_data, images_G[0])



