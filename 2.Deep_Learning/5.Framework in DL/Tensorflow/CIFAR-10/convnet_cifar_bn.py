import cifar10_input
import tensorflow as tf
import time, os
from config import opt



#======== MODEL ==========#
def conv2d(input, weight_shape, bias_shape, phase_train, visualize=False):

    incoming = weight_shape[0]*weight_shape[1]*weight_shape[2]
    W = tf.get_variable("W", 
                        weight_shape, 
                        initializer=tf.random_normal_initializer(stddev=(2.0/incoming)**0.5))
    b = tf.get_variable("b",
                        bias_shape,
                        initializer=tf.constant_initializer(0))
    logits = tf.nn.bias_add(tf.nn)


def inference(x, keep_prob, phase_train):

    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 3, 64], [64], phase_train, visualize=True)
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 64, 64], [64], phase_train)
        pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("fc_1"):

        dim = 1
        for d in pool_2.get_shape()[1:].as_list():
            dim *= d

        pool_2_flat = tf.reshape(pool_2, [-1, dim])
        fc_1 = layer(pool_2_flat, [dim, 384], [384], phase_train)
        
        # apply dropout
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    with tf.variable_scope("fc_2"):

        fc_2 = layer(fc_1_drop, [384, 192], [192], phase_train)
        
        # apply dropout
        fc_2_drop = tf.nn.dropout(fc_2, keep_prob)

    with tf.variable_scope("output"):
        output = layer(fc_2_drop, [192, 10], [10], phase_train)

    return output






with tf.device("/cpu:0"):

        with tf.Graph().as_default():

            with tf.variable_scope("cifar_conv_bn_model"):

                # Load dataset
                distorted_images, distorted_labels = cifar10_input.distorted_inputs(data_dir=opt.data_dir,
                                                                                    batch_size=opt.batch_size)
                val_images, val_labels = cifar10_input.inputs(eval_data=True, data_dir=opt.data_dir,
                                                              batch_size=opt.batch_size)
                
                # Computation
                x = tf.placeholder("float", [None, 24, 24, 3])
                y = tf.placeholder("int32", [None])
                keep_prob = tf.placeholder(tf.float32) # dropout probability
                phase_train = tf.placeholder(tf.bool) # training or testing





