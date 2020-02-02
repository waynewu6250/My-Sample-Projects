# Bacterial Counting

The repository includes pipeline for counting bacteria and the code for training counting applications. (Keras + Tensorflow)

## 1. Annotation

Two common ways of counting objects using Deep Learning are illustrated as belows: <br>
1) First detect them using convolutional neural network and then count all found instances: <br>
this approach is often done by following the object detection pipeline (Faster RCNN, Single Shot Detection); however it requires bounding box annotations which are hard to obtain.

2) We could do point-like annotations of objects positions instead, which reduces much computational efforts

Here we will use to annotate.



## 2. Method

In this project we will mainly perform our tasks on two fully-connected neural networks: U-Net and Fully Convolutional Regression Networks. The key idea is first to obtain a density map by applying a convolution with a Gaussian kernel of annotated images. Then we train a fully convolutional network to map an image to a density map, which can be later integrated to get the number of objects.

The two FCN papers are as follows:

**[1] U-Net: Convolutional Networks for Biomedical Image Segmentation.**

https://arxiv.org/abs/1505.04597

U-Net is a widely used FCN for image segmentation, very often applied to biomedical data. It has autoencoder-like structure. By downsampling images using convolutiona and pooling layers, it encodes the key features. Then second half of network is symmetric with pooling layers replaced by upsampling layers. The output will be the image segmentation map with same dimension as input image.

**[2] Microscopy Cell Counting with Fully Convolutional Regression Networks.**

http://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf

The architecture is similar to U-Net, without information from downsampling part directly sent to upsampling part. There are two proposed networks: FCRN-A and FCRN-B, which differ in downsampling intensity. Here we use FCRN-A performing pooling every convolution layer.


And the code on this project is based on [cell_counting_v2 respository](https://github.com/WeidiXie/cell_counting_v2) from UK Cambridge VGG group and [Object_counting_dmap](https://github.com/NeuroSYS-pl/objects_counting_dmap).


<br>

### Comments from [cell_counting_v2 respository](https://github.com/WeidiXie/cell_counting_v2)
In all architectures, they follow the fully convolutional idea, 
each architecture consists of a down-sampling path, followed by an up-sampling path. 
During the first several layers, the structure resembles the cannonical classification CNN, as convolution,
ReLU, and max pooling are repeatedly applied to the input image and feature maps. 
In the second half of the architecture, spatial resolution is recovered by performing up-sampling, convolution, eventually mapping the intermediate feature representation back to the original resolution. 

In the U-net version, low-level feature representations are fused during upsampling, aiming to compensate the information loss due to max pooling. Here, I only gave a very simple example here (64 kernels for all layers), not tuned for any dataset.



