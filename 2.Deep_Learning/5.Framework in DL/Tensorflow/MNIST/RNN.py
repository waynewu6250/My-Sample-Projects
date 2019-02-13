import tensorflow as tf
from data import mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

seq_len = 28
input_dim = 28
hidden_dim = 128
num_classes = 10
batch_size = 128

# 1. Define variables
W1 = tf.Variable(tf.truncated_normal([input_dim, hidden_dim], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden_dim,]))
W2 = tf.Variable(tf.truncated_normal([hidden_dim, num_classes], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[num_classes,]))

x = tf.placeholder(tf.float32, [None, seq_len, input_dim])
y = tf.placeholder(tf.float32, [None, num_classes])

# 2. Compute
x = tf.reshape(x, [-1, input_dim])
x_in = tf.matmul(x,W1)+b1
x_in = tf.reshape(x_in, [-1, seq_len, hidden_dim])
lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, forget_bias = 1.0, state_is_tuple=True)
initial_sate = lstm_cell.zero_state(batch_size, dtype=tf.float32)

outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state = initial_sate, time_major=False)

y_pred = tf.matmul(outputs,W2)+b2

#-------------#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

# 3. Intialize
init = tf.global_variables_initializer()
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

# 4. Run
with tf.Session() as sess:
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    for epoch in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, train_accuracy = sess.run([train_step,accuracy], feed_dict= {x: batch_xs, y: batch_ys})
        if epoch % 100 == 0:
            print("Epoch %d, training accuracy = %g" % (epoch,train_accuracy))
    
    print("test accuracy = %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    

