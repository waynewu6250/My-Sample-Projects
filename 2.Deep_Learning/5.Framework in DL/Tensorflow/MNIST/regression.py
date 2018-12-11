import tensorflow as tf
from data import mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 1. Define variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 2. Compute
y_pred = tf.nn.softmax(tf.matmul(x,W)+b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred)))

# 3. Intialize
init = tf.global_variables_initializer()
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

# 4. Run
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _,cost = sess.run([train_step,cross_entropy], feed_dict= {x: batch_xs, y: batch_ys})
        if epoch % 100 == 0:
            print("Epoch %d Cost: " % epoch,cost)
        

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy: ",sess.run(accuracy,feed_dict = {x:mnist.test.images, y:mnist.test.labels}))
