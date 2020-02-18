import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from data import FluoData
from torch.utils.data import DataLoader
from model import UNet, FCRN_A
from config import opt
from scipy import ndimage
import imageio

def test():
    """Count the input image"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Image
    image = np.array(Image.open(opt.test_path), dtype=np.float32) / 255
    image = torch.Tensor(np.transpose(image, (2,0,1))).unsqueeze(0)
    
    # Ground Truth
    header = ".".join(opt.test_path.split('/')[-1].split('.')[:2])
    label_path = opt.label_path+header+'.label.png'
    label = np.array(Image.open(label_path))
    if opt.color == 'red':
        labels = 100.0 * (label[:,:,0] > 0)
    else:
        labels = 100.0 * (label[:,:,1] > 0)
    labels = ndimage.gaussian_filter(labels, sigma=(1, 1), order=0)
    labels = torch.Tensor(labels).unsqueeze(0)

    if opt.model.find("UNet") != -1:
        model = UNet(input_filters=3, filters=opt.unet_filters, N=opt.conv).to(device)
    else:
        model = FCRN_A(input_filters=3, filters=opt.unet_filters, N=opt.conv).to(device)
    model = torch.nn.DataParallel(model)
    
    if os.path.exists('{}.pth'.format(opt.model)):
        model.load_state_dict(torch.load('{}.pth'.format(opt.model)))

    model.eval()
    image = image.to(device)
    labels = labels.to(device)
    
    out = model(image)
    predicted_counts = torch.sum(out).item() / 100
    real_counts = torch.sum(labels).item() / 100
    print(predicted_counts, real_counts)
    
    label = np.zeros((image.shape[2], image.shape[2], 3))
    label[:,:,0] = out[0][0].cpu().detach().numpy()
    imageio.imwrite('test_density_map_{}.png'.format(opt.model), label)

def test_cell():
    """Count the input image"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Image
    image = np.array(Image.open('cells/'+opt.cell_test_path), dtype=np.float32) / 255
    image = torch.Tensor(np.transpose(image, (2,0,1))).unsqueeze(0)
    
    # Ground Truth
    label_path = opt.cell_test_path.replace('cell.png', 'dots.png')
    name = opt.cell_test_path.replace('cell.png', '')

    labels = np.array(Image.open('cells/'+label_path))
    labels = 100.0 * (labels[:, :, 0] > 0)
    labels = ndimage.gaussian_filter(labels, sigma=(1, 1), order=0)
    imageio.imwrite('cells/'+name+'cell_density.png', labels)
    
    labels = torch.Tensor(labels).unsqueeze(0)
    

    if opt.model.find("UNet") != -1:
        model = UNet(input_filters=3, filters=opt.unet_filters, N=opt.conv).to(device)
    else:
        model = FCRN_A(input_filters=3, filters=opt.unet_filters, N=opt.conv).to(device)
    model = torch.nn.DataParallel(model)
    
    if os.path.exists('{}.pth'.format(opt.model)):
        model.load_state_dict(torch.load('{}.pth'.format(opt.model)))

    model.eval()
    image = image.to(device)
    labels = labels.to(device)
    
    out = model(image)
    predicted_counts = torch.sum(out).item() / 100
    real_counts = torch.sum(labels).item() / 100
    print(predicted_counts, real_counts)
    
    label = np.zeros((image.shape[2], image.shape[2], 3))
    label[:,:,0] = out[0][0].cpu().detach().numpy()
    imageio.imwrite('test_density_map_{}.png'.format(opt.model), label)

if __name__ == '__main__':
    if opt.data_type == 'cell':
        test_cell()
    elif opt.data_type == 'bacteria':
        test()
    