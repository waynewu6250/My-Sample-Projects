import cv2
import numpy as np
import os
import json
import collections
import matplotlib.pyplot as plt
import imageio
import scipy.ndimage as ndimage
import argparse
from config import opt

def split(slices, path):
    
    for img_path in os.listdir(path):
        if img_path == '.DS_Store' or img_path.find('json') != -1:
            continue
        img = cv2.imread(os.path.join(path, img_path))
        original_size = img.shape[0]
        edge = original_size // slices
        num = original_size / edge + 1
        
        filename = img_path.split('.')[0]

        count = 1
        for x in range(slices):
            for y in range(slices):
                sub_img = img[x*edge:x*edge+edge,y*edge:y*edge+edge,:]
                # for train
                # cv2.imwrite(os.path.join(path, filename)+'.tif-{}.png'.format(count), sub_img)
                # for label
                cv2.imwrite(os.path.join(path, filename)+'.tif-{}.label.png'.format(count), sub_img)
                count += 1
        os.remove(os.path.join(path, img_path))

def create_label(train_path, filename):

    if not os.path.exists(train_path+filename+'.json'):
        return

    with open(train_path+filename+'.json', 'r') as f:
        a = json.load(f)

    cache = collections.defaultdict(list)
    for item in a["shapes"]:
        point = item['points'][0]
        cache[item['label']].append(point)

    img = cv2.imread(train_path+filename+'.tif')
    size = img.shape[0]
    label = np.zeros((size, size, 3))

    for (y, x) in cache['red']:
        label[int(x)][int(y)][0] = 255.0
    for (y, x) in cache['green']:
        label[int(x)][int(y)][1] = 255.0
        
    imageio.imwrite(os.path.join('imgs_to_use_sub/label_imgs', filename+'.png'), label)

    # Gaussian Kernel
    red = 100.0 * (label[:,:,0] > 0)
    red = ndimage.gaussian_filter(red, sigma=(2, 2), order=0)
    green = 100.0 * (label[:,:,1] > 0)
    green = ndimage.gaussian_filter(green, sigma=(2, 2), order=0)

    label[:,:,0] = red
    label[:,:,1] = green

    imageio.imwrite(os.path.join('imgs_to_use_sub/density_maps', filename+'.density_map.png'), label)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='mode', default='label', help='Two modes: split, label')

    args = parser.parse_args()

    if args.mode == 'split':
        split(opt.slices, './imgs_to_use_sub/label_imgs')
    elif args.mode == 'label':
        filenames = set([filename.split('.')[0] for filename in os.listdir(opt.train_path)])
        for filename in filenames:
            create_label(opt.train_path, filename)



