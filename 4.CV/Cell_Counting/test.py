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
import glob

def test(path):
    """Count the input image"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Image
    image = np.array(Image.open(path), dtype=np.float32) / 255
    image = torch.Tensor(np.transpose(image, (2,0,1))).unsqueeze(0)
    
    # Ground Truth
    header = ".".join(path.split('/')[-1].split('.')[:2])
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
    if opt.color == 'red':
        label[:,:,0] = out[0][0].cpu().detach().numpy()
    else:
        label[:,:,1] = out[0][0].cpu().detach().numpy()
    
    imageio.imwrite('example/test_results/density_map_{}.png'.format(header), label)

    return header, predicted_counts, real_counts

def test_cell(path):
    """Count the input image"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Image
    print(path)
    image = np.array(Image.open(path), dtype=np.float32) / 255
    image = torch.Tensor(np.transpose(image, (2,0,1))).unsqueeze(0)
    path = path.split('/')[1]
    
    # Ground Truth
    label_path = path.replace('cell.png', 'dots.png')
    name = path.replace('cell.png', '')

    labels = np.array(Image.open('cells/'+label_path))
    labels = 100.0 * (labels[:, :, 0] > 0)
    labels = ndimage.gaussian_filter(labels, sigma=(1, 1), order=0)
    #imageio.imwrite('cells/'+name+'cell_density.png', labels)
    
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
    imageio.imwrite('example/test_results_cells/density_map_{}'.format(path), label)

    return path, predicted_counts, real_counts

if __name__ == '__main__':
    
    if opt.data_type == 'cell':
        mae = 0
        raw_paths = glob.glob('cells/*.png')
        raw_paths = [path for path in raw_paths if path.find('dots') == -1]
        with open('example/test_results_cells/results_cells.txt', 'w') as f:
            f.write("Count Results: \nName \t Predicted Counts \t Real Counts \n")
            for path in raw_paths:
                header, predicted_counts, real_counts = test_cell(path)
                mae += np.abs(predicted_counts-real_counts)
                f.write('{} : {:.2f}, {:.2f} \n'.format(header, predicted_counts, real_counts))
            f.write('Mean Absolute Error: {:.2f}'.format(mae/len(raw_paths)))
        print('Mean Absolute Error: {:.2f}'.format(mae/len(raw_paths)))
    
    elif opt.data_type == 'bacteria':
        mae = 0
        raw_paths = glob.glob(opt.train_path+'*.png')
        with open('example/test_results/results.txt', 'w') as f:
            f.write("Count Results: \nName \t Predicted Counts \t Real Counts \n")
            for path in raw_paths:
                header, predicted_counts, real_counts = test(path)
                mae += np.abs(predicted_counts-real_counts)
                f.write('{} : {:.2f}, {:.2f} \n'.format(header, predicted_counts, real_counts))
            f.write('Mean Absolute Error: {:.2f}'.format(mae/len(raw_paths)))
        print('Mean Absolute Error: {:.2f}'.format(mae/len(raw_paths)))
        
    