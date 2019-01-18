import cifar10_input
cifar10_input.maybe_download_and_extract()

import tensorflow as tf
import time, os
from config import opt
from cifar_model import *
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

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

                # Model
                output = inference(x,keep_prob,phase_train)
                # Loss
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.cast(y, tf.int64)))
                # Optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr).minimize(loss)
                init = tf.global_variables_initializer()
                # Accuracy
                correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1), dtype=tf.int32), y)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
                tf.summary.scalar("loss", loss)
                tf.summary.scalar("validation error", (1.0 - accuracy))
                
                # Visualization
                summary_op = tf.summary.merge_all()
                saver = tf.train.Saver()

                with tf.Session() as sess:
                    
                    summary_writer = tf.summary.FileWriter("conv_cifar_bn_logs/",
                                                        graph=sess.graph)
                    sess.run(init)
                    
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(coord=coord)

                    for epoch in range(opt.training_epochs):

                        avg_cost = 0.
                        total_batch = int(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/opt.batch_size)
                        
                        # Loop over all batches
                        for i in range(total_batch):

                            train_x, train_y = sess.run([distorted_images, distorted_labels])
                            _, cost = sess.run([optimizer, loss], feed_dict={x: train_x, y: train_y, keep_prob: 1, phase_train: True})
                            avg_cost += cost/total_batch
                            print("Epoch %d, minibatch %d of %d. Cost = %0.4f." %(epoch, i, total_batch, cost))
                    
                        # Display logs per epoch step
                        if epoch % opt.display_step == 0:
                            print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))


                            # Validation
                            val_x, val_y = sess.run([val_images, val_labels])
                            acc = sess.run(accuracy, feed_dict={x: val_x, y: val_y, keep_prob: 1, phase_train: False})

                            print("Validation Error:", (1 - acc))

                            #summary_str = sess.run(summary_op, feed_dict={x: train_x, y: train_y, keep_prob: 1, phase_train: False})
                            #summary_writer.add_summary(summary_str)

                            saver.save(sess, "conv_cifar_bn_logs/model-checkpoint",global_step=epoch)
                    
                    print("Optimization Finished!")

                    val_x, val_y = sess.run([val_images, val_labels])
                    acc = sess.run(accuracy, feed_dict={x: val_x, y: val_y, keep_prob: 1, phase_train: False})

                    print("Test Accuracy:", acc)

                    coord.request_stop()
                    coord.join(threads)



