import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.autograd import Variable
import torchvision
import cv2
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init
from PIL import Image

from model import SegNet
from config import opt


def train():

    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

    # Image input
    im = Image.open(opt.img_path)
    im = np.array(im, dtype=np.float32) / 255
    image = np.transpose(im, (2,0,1))
    data = torch.from_numpy(image).unsqueeze(0)
    data = Variable(data).to(device)

    labels = segmentation.slic(im, compactness=opt.compactness, n_segments=opt.num_superpixels)
    labels = labels.reshape(-1)
    label_nums = np.unique(labels)
    label_indices = [np.where(labels==label_nums[i])[0] for i in range(len(label_nums))]

    # Model
    model = SegNet(opt, data.shape[1])
    if opt.model_path:
        model.load_state_dict(torch.load(model))
    model = model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum)

    for batch_idx in range(opt.maxIter):
        # forwarding
        optimizer.zero_grad()
        
        output = model(data)[0]
        output = output.permute(1,2,0).contiguous().view(-1, opt.nChannel)
        
        _, pred_clusters = torch.max(output, 1)
        
        n_clusters = len(np.unique(pred_clusters.data.cpu().numpy()))

        # superpixel
        # for i in range(len(label_indices)):
        #     labels_per_sp = pred_clusters[label_indices[i]]
        #     pred_clusters[label_indices[i]] = torch.mode(labels_per_sp)[0]
        
        pred_clusters = pred_clusters.data.cpu().numpy()
        for i in range(len(label_indices)):
            labels_per_sp = pred_clusters[label_indices[i]]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0] )
            pred_clusters[label_indices[i]] = u_labels_per_sp[np.argmax(hist)]
        target = torch.from_numpy(pred_clusters)
        target = target.to(device)
        target = Variable(target)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print (batch_idx, '/', opt.maxIter, ':', n_clusters, loss.item())

        if n_clusters <= opt.min_labels:
            print ("nLabels", n_clusters, "reached minLabels", opt.min_labels, ".")
            break
    
    # Save output
    output = model(data)[0]
    output = output.permute(1,2,0).contiguous().view(-1, opt.nChannel)
    _, pred_clusters = torch.max(output, 1)
    pred_clusters = pred_clusters.data.cpu().numpy()
    label_colors = np.random.randint(255,size=(100,3))
    im_target_rgb = np.array([label_colors[ c % 100 ] for c in pred_clusters])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
    
    path = opt.img_path.split('/')[1].split('.')[0]
    cv2.imwrite( "outputs/{}_out.png".format(path), im_target_rgb)


if __name__ == '__main__':
    train()




        

        






