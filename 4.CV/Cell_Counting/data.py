import os
from random import random
from typing import Optional

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import opt

class FluoData(Dataset):

    def __init__(self, dataset_path, data_type, color, horizontal_flip=0.0, vertical_flip=0.0):
        """mode specifies which bacteria to count: red or green"""

        super(FluoData, self).__init__()
        self.h5 = h5py.File(dataset_path, 'r')
        self.images = self.h5['images']

        if data_type == 'cell':
            self.labels = self.h5['labels']
        elif data_type == 'bacteria':
            self.labels = self.h5['label_red'] if color == 'red' else self.h5['label_green']

        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def __getitem__(self, index):
        """Return next sample (randomly flipped)."""
        
        if not (self.horizontal_flip or self.vertical_flip):
            return self.images[index], self.labels[index]

        # axis = 1 (vertical flip), axis = 2 (horizontal flip)
        axis_to_flip = []

        if random() < self.vertical_flip:
            axis_to_flip.append(1)

        if random() < self.horizontal_flip:
            axis_to_flip.append(2)

        return (np.flip(self.images[index], axis=axis_to_flip).copy(),
                np.flip(self.labels[index], axis=axis_to_flip).copy())
    
    def __len__(self):
        """Return no. of samples in HDF5 file."""
        return len(self.images)


def run_batch(flip):
    """Sanity check for HDF5 dataloader checks for shapes and empty arrays."""

    for h5 in (opt.h5_path+'train.h5', opt.h5_path+'valid.h5'):
        
        data = FluoData(h5, horizontal_flip=1.0 * flip, vertical_flip=1.0 * flip)

        data_loader = DataLoader(data, batch_size=opt.batch_size)

        # take one batch, check samples, and go to the next file
        for img, label in data_loader:
            # image batch shape (#workers, #channels, resolution)
            assert img.shape == (opt.batch_size, 3, 256, 256)
            # label batch shape (#workers, 1, resolution)
            assert label.shape == (opt.batch_size, 1, 256, 256)

            assert torch.sum(img) > 0
            assert torch.sum(label) > 0

            break