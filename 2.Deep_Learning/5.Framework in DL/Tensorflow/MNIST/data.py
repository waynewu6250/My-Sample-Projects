from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import scipy.misc

os_path = "MNIST_data/raw/"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

if os.path.exists(os_path) is False:
    os.makedirs(os_path)

for i in range(50):
    image_array = mnist.train.images[i,:]
    image_array = image_array.reshape(28,28)

    filename = os_path + "mnist_train_{}.jpg".format(i)
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)
