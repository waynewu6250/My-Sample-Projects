import h5py
import glob
from config import opt
from scipy import ndimage
from PIL import Image
import numpy as np
import os

def create_hdf5(train_size, valid_size, img_size, in_channels):

    train_h5 = h5py.File(opt.h5_path+'train.h5', 'w')
    valid_h5 = h5py.File(opt.h5_path+'valid.h5', 'w')

    for h5, size in ((train_h5, train_size), (valid_h5, valid_size)):
        h5.create_dataset('images', (size, in_channels, *img_size))
        h5.create_dataset('label_red', (size, 1, *img_size))
        h5.create_dataset('label_green', (size, 1, *img_size))
    
    return train_h5, valid_h5

def fill_hdf5(h5_file, img_paths):

    counter = 0
    for img_path in img_paths:

        header = ".".join(img_path.split('/')[-1].split('.')[:2])
        label_path = opt.label_path+header+'.label.png'
        
        
        if not os.path.exists(label_path):
            continue

        image = np.array(Image.open(img_path), dtype=np.float32) / 255
        image = np.transpose(image, (2,0,1))

        label = np.array(Image.open(label_path))
        red = 100.0 * (label[:,:,0] > 0)
        red = ndimage.gaussian_filter(red, sigma=(2, 2), order=0)
        green = 100.0 * (label[:,:,1] > 0)
        green = ndimage.gaussian_filter(green, sigma=(2, 2), order=0)

        # save data to HDF5 file
        h5_file['images'][counter] = image
        h5_file['label_red'][counter, 0] = red
        h5_file['label_green'][counter, 0] = green
        counter += 1

def get_data():

    train_h5, valid_h5 = create_hdf5(train_size=opt.train_size, 
                                     valid_size=opt.val_size, 
                                     img_size=(256,256), 
                                     in_channels=3)
    
    img_paths = []
    raw_paths = glob.glob(opt.train_path+'*.jpg')
    
    for img_path in raw_paths:
        header = ".".join(img_path.split('/')[-1].split('.')[:2])
        label_path = opt.label_path+header+'.label.png'
        if os.path.exists(label_path):
            img_paths.append(img_path)

    fill_hdf5(train_h5, img_paths[:opt.train_size])
    fill_hdf5(valid_h5, img_paths[opt.train_size:])

    # close HDF5 files
    train_h5.close()
    valid_h5.close()

if __name__ == "__main__":
    get_data()

    # To use
    # h5 = h5py.File(opt.h5_path+'train.h5', 'r')
    # images = h5['images']
    # red = h5['label_red']
    # green = h5['label_green']
    


