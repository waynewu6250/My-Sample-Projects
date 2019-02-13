import tensorflow as tf
from data import mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 1. Define variables
def conv2d(input, w_shape, b_shape):
    W = tf.get_variable("W", w_shape, initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1))
    b = tf.get_variable("b", b_shape, initializer=tf.constant_initializer(0.1))
    h_conv = tf.nn.conv2d(x_image, W, strides=[1,1,1,1], padding='SAME')+b
    return tf.nn.relu(h_conv)

def layer(input, w_shape, b_shape):
    W = tf.get_variable("W", w_shape, initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1))
    b = tf.get_variable("b", b_shape, initializer=tf.constant_initializer(0.1))
    fc = tf.matmul(input, W)+b
    return fc

def inference(x, keep_prob):
    with tf.variable_scope("conv_1"):
        conv1 = conv2d(x, [5, 5, 1, 32], [32])
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    with tf.variable_scope("conv_2"):
        conv2 = conv2d(x, [5, 5, 32, 64], [64])
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    with tf.variable_scope("fc_1"):
        pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(layer(pool2_flat, [7*7*64, 1024], [1024]))
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    with tf.variable_scope("fc_2"):
        y_conv = layer(h_fc1_drop, [1024, 10], [10])
    
    return y_conv

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 2. Compute
y_conv = inference(x_image, keep_prob)
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y))

# 3. Intialize
init = tf.global_variables_initializer()
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

# 4. Run
with tf.Session() as sess:
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_conv,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    for epoch in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        _, train_accuracy = sess.run([train_step,accuracy], feed_dict= {x: batch_xs, y: batch_ys, keep_prob: 0.5})
        if epoch % 100 == 0:
            print("Epoch %d, training accuracy = %g" % (epoch,train_accuracy))
    
    print("test accuracy = %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))