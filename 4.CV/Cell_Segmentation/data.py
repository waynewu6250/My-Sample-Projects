import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import glob
from skimage import segmentation
import numpy as np
from PIL import Image

class segData(Dataset):

    def __init__(self, paths, opt, device):
        self.img_paths = glob.glob(paths)
        self.device = device
        self.opt = opt

    def __getitem__(self, index):

        # Image input
        im = Image.open(self.img_paths[index])
        im = np.array(im, dtype=np.float32) / 255
        image = np.transpose(im, (2,0,1))
        data = torch.from_numpy(image)
        data = Variable(data).to(self.device)

        labels = segmentation.slic(im, compactness=self.opt.compactness, n_segments=self.opt.num_superpixels)
        labels = labels.reshape(-1)
        label_nums = np.unique(labels)
        label_indices = [np.where(labels==label_nums[i])[0] for i in range(len(label_nums))]
        
        return data, label_indices

    def __len__(self):
        return len(self.img_paths)

def get_dataloader(paths, opt, device):
    dataset = segData(paths, opt, device)
    return DataLoader(dataset, 
                      batch_size=opt.batch_size, 
                      shuffle=False)

