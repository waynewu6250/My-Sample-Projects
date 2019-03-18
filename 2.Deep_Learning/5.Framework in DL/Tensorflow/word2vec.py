import tensorflow as tf

learning_rate = 0.03
batch_size = 64
vocab_size = 128
embedding_size = 64
val_examples = [1,2,3]
display_step = 5
val_step = 10

# Set embedding matrix and get its embedding vector
def embedding_layer(x, embedding_shape):
    with tf.variable_scope("embedding"):
        embedding_matrix = tf.get_variable("E",initializer=tf.random_uniform(embedding_shape,-1,1))
        return tf.nn.embedding_lookup(embedding_matrix, x), embedding_matrix

# Define Loss
def noise_contrastive_loss(embedding_lookup, weight_shape, bias_shape, y, vocab_size):
    with tf.variable_scope("nce"):
        nce_W = tf.get_variable("W", 
                                weight_shape, 
                                initializer=tf.truncated_normal_initializer(stddev=1.0/(weight_shape[1]**0.5)))
        nce_b = tf.get_variable("b", 
                                bias_shape, 
                                initializer=tf.zeros_initializer)
        total_loss = tf.nn.nce_loss(nce_W, nce_b, embedding_lookup, y, 64, vocab_size)
        return tf.reduce_mean(total_loss)

# Define optimizer
def training(cost, global_step):  
    with tf.variable_scope("training"):
        summary_op = tf.summary.scalar("cost", cost)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        return train_op, summary_op

# Evaluation
def validation(embedding_matrix, x_val):
    norm = tf.reduce_sum(embedding_matrix**2, 1, keep_dims=True)**0.5
    normalized = embedding_matrix / norm
    val_embeddings = tf.nn.embedding_lookup(normalized, x_val)
    cosine_similarity = tf.matmul(val_embeddings, normalized, transpose_b=True)
    return normalized, cosine_similarity

#########################################
if __name__ == "__main__":
    with tf.Graph().as_default():
        with tf.variable_scope("word2vec_model"):
            x = tf.placeholder(tf.int32, shape=[batch_size])
            y = tf.placeholder(tf.int32, shape=[batch_size,1])
            x_val = tf.constant(val_examples, dtype=tf.int32)
            global_step = tf.Variable(0, name='global_step', trainable=False)

            e_lookup, e_matrix = embedding_layer(x, [vocab_size, embedding_size])
            cost = noise_contrastive_loss(e_lookup, [vocab_size, embedding_size], [vocab_size], y, vocab_size)
            train_op, summary_op = training(cost, global_step)
            val_op = validation(e_matrix, x_val)

            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                
                sess.run(init)
                train_writer = tf.train.SummaryWriter("skipgram_logs/", graph=sess.graph)

                avg_cost = 0

                for epoch in range(100):
                    for minibatch in range(batch_size):
                        mbatch_x, mbatch_y = data.generate_batch(batch_size, num_skips, skip_window)

                        _, new_cost, train_summary = sess.run([train_op, cost, summary_op], feed_dict={x: mbatch_x, y: mbatch_y})
                        train_writer.add_summary(train_summary, sess.run(global_step))
                        avg_cost += new_cost/display_step

                        if epoch % display_step == 0:
                            print(avg_cost)
                        
                        if epoch % val_step == 0:
                            _, similarity = sess.run(val_op)
                            for i in range(val_size):
                                val_word = data.reverse_dictionary[val_examples[i]]
                                neighbors = (-similarity[i,].argsort()[1:4])
                                print_str = "Nearest Neigbor of %s" % val_word
                                for k in range(3):
                                    print_str += data.reverse_dictionary[neighbors[k]]
                                print(print_str[:-1])








