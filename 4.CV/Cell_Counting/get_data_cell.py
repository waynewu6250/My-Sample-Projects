import h5py
import glob
from config import opt
from scipy import ndimage
from PIL import Image
import numpy as np
import os

def create_hdf5(train_size, valid_size, img_size, in_channels):

    train_h5 = h5py.File(opt.cell_h5_path+'train.h5', 'w')
    valid_h5 = h5py.File(opt.cell_h5_path+'valid.h5', 'w')

    for h5, size in ((train_h5, train_size), (valid_h5, valid_size)):
        h5.create_dataset('images', (size, in_channels, *img_size))
        h5.create_dataset('labels', (size, 1, *img_size))
    
    return train_h5, valid_h5

def fill_hdf5(h5_file, img_paths):

    for i, img_path in enumerate(img_paths):
        
        label_path = img_path.replace('cell.png', 'dots.png')
        
        image = np.array(Image.open(img_path), dtype=np.float32) / 255
        image = np.transpose(image, (2, 0, 1))

        label = np.array(Image.open(label_path))
        label = 100.0 * (label[:, :, 0] > 0)
        label = ndimage.gaussian_filter(label, sigma=(1, 1), order=0)

        # save data to HDF5 file
        h5_file['images'][i] = image
        h5_file['labels'][i, 0] = label


def get_data():

    train_h5, valid_h5 = create_hdf5(train_size=opt.train_size, 
                                     valid_size=opt.val_size, 
                                     img_size=(256,256), 
                                     in_channels=3)
    
    img_paths = glob.glob(os.path.join('cells', '*cell.*'))
    img_paths.sort()

    fill_hdf5(train_h5, img_paths[:opt.train_size])
    fill_hdf5(valid_h5, img_paths[opt.train_size:])

    # close HDF5 files
    train_h5.close()
    valid_h5.close()

if __name__ == "__main__":
    get_data()

    # To use
    # h5 = h5py.File(opt.cell_h5_path+'train.h5', 'r')
    # images = h5['images']
    # labels = h5['labels']
    


