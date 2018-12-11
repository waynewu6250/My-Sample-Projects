import tensorflow as tf
from data import mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 1. Define variables
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[32]))
W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[64]))
W3 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[1024]))
W4 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[10]))

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 2. Compute
h_conv1 = tf.nn.conv2d(x_image, W1, strides=[1,1,1,1], padding='SAME')+b1
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

h_conv2 = tf.nn.conv2d(h_pool1, W2, strides=[1,1,1,1], padding='SAME')+b2
h_conv2 = tf.nn.relu(h_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W3)+b3)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv = tf.matmul(h_fc1_drop, W4)+b4

#-------------#
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
    

