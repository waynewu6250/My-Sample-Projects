import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.autograd import Variable
import torchvision
import cv2
import os
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init
from PIL import Image
import glob

from model import SegNet, DiscriminativeLoss
from data import get_dataloader
from config import opt

def batch_step(opt, optimizer, model, dataloader, criterion, criterion_d, device):
    prev_loss = 5
    for data, label_indices in dataloader:
    
        for batch_idx in range(opt.maxIter):
            # forwarding
            optimizer.zero_grad()
            
            feats, output = model(data)
            output = output[0].permute(1,2,0).contiguous().view(-1, opt.nClass)
            feats = feats[0].permute(1,2,0).contiguous().view(-1, opt.nChannel)
            
            _, pred_clusters = torch.max(output, 1)
            
            n_clusters = len(np.unique(pred_clusters.data.cpu().numpy()))
            
            d_loss = criterion_d(feats, pred_clusters, n_clusters)
            
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
            
            loss = criterion(output, target) + opt.lamda*d_loss
            loss.backward()
            optimizer.step()

            print (batch_idx, '/', opt.maxIter, ':', n_clusters, loss.item())

            if n_clusters <= opt.min_labels or loss.item() < prev_loss*0.7:
                prev_loss = loss.item()
                break
        
        if n_clusters <= opt.min_labels:
            print("nLabels", n_clusters, "reached minLabels", opt.min_labels, ".")
            break
    
    return model

def one_step(opt, optimizer, model, data, label_indices, criterion, criterion_d, device):
    
    for batch_idx in range(opt.maxIter):
        # forwarding
        optimizer.zero_grad()
        
        feats, output = model(data)
        output = output[0].permute(1,2,0).contiguous().view(-1, opt.nClass)
        feats = feats[0].permute(1,2,0).contiguous().view(-1, opt.nChannel)
        
        _, pred_clusters = torch.max(output, 1)
        
        n_clusters = len(np.unique(pred_clusters.data.cpu().numpy()))
        
        d_loss = criterion_d(feats, pred_clusters, n_clusters)
        
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
        
        loss = criterion(output, target) + opt.lamda*d_loss
        loss.backward()
        optimizer.step()

        print (batch_idx, '/', opt.maxIter, ':', n_clusters, loss.item())

        if n_clusters <= opt.min_labels or loss.item() < 0.2:
            print("nLabels", n_clusters, "reached minLabels", opt.min_labels, ".")
            break
    
    return model

def train(data_size='all'):

    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

    # Image input
    model = SegNet(opt, 3)
    model = model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    criterion_d = DiscriminativeLoss()
    optimizer = SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum)

    if data_size == 'all':
        dataloader = get_dataloader(opt.paths, opt, device)
        model = batch_step(opt, optimizer, model, dataloader, criterion, criterion_d, device)
        torch.save(model.state_dict(), 'model_all.pth')
    else:
        im = Image.open(opt.img_path)
        im = np.array(im, dtype=np.float32) / 255
        image = np.transpose(im, (2,0,1))
        data = torch.from_numpy(image).unsqueeze(0)
        data = Variable(data).to(device)

        labels = segmentation.slic(im, compactness=opt.compactness, n_segments=opt.num_superpixels)
        labels = labels.reshape(-1)
        label_nums = np.unique(labels)
        label_indices = [np.where(labels==label_nums[i])[0] for i in range(len(label_nums))]
        
        model = one_step(opt, optimizer, model, data, label_indices, criterion, criterion_d, device)
        torch.save(model.state_dict(), 'model_single.pth')
        
    
def test(img_path, data_size='single'):
    
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

    # Image input
    im = Image.open(img_path)
    im = np.array(im, dtype=np.float32) / 255
    image = np.transpose(im, (2,0,1))
    data = torch.from_numpy(image).unsqueeze(0)
    data = Variable(data).to(device)

    model = SegNet(opt, data.shape[1])
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
    model = model.to(device)
    model.train()

    feats, output = model(data)
    output = output[0].permute(1,2,0).contiguous().view(-1, opt.nClass)
    feats = feats[0].permute(1,2,0).contiguous().view(-1, opt.nChannel)
    _, pred_clusters = torch.max(output, 1)
    pred_clusters = pred_clusters.data.cpu().numpy()
    
    # Post processing
    labels = np.unique(pred_clusters)
    counts = {}
    for i in pred_clusters:
        counts[i] = counts.get(i, 0)+1
    sorts = sorted(counts.items(), key=lambda x: x[1])
    cache = {}
    cache[sorts[-1][0]] = 0
    n = 1
    for (num, _) in sorts[:-1]:
        cache[num] = n
        n+=1

    label_colors = [[10,10,10],[0,0,255],[0,255,0],[255,0,0],[255,255,0],[0,255,255],[255,0,255]]
    
    im_target_rgb = np.array([label_colors[cache[c]] for c in pred_clusters])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
    
    # change path
    path = ".".join(img_path.split('/')[1].split('.')[:2])
    #path = img_path.split('/')[1].split('.')[0]
    if data_size == 'single':
        cv2.imwrite("outputs_single/{}_out.png".format(path), im_target_rgb)
    elif data_size == 'all':
        cv2.imwrite("outputs_all/{}_out.png".format(path), im_target_rgb)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", dest="mode", default="train", type=str, metavar='<str>', help="Type the mode for train or test")
    parser.add_argument("-img", "--image", dest="img_path", default="images/image1.jpg", type=str, metavar='<str>', help="Image path")
    args = parser.parse_args()
    
    if args.mode == 'train':
        train('single')
    elif args.mode == 'test':
        paths = glob.glob(opt.paths)
        for path in paths:
            print("Processing: ", path)
            test(path, opt.model_mode)




        

        






