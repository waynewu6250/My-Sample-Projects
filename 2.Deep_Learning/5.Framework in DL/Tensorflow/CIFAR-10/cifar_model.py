import tensorflow as tf

def conv_batch_norm(x, n_out, phase_train):
    beta = tf.get_variable("beta", 
                           [n_out], 
                            initializer = tf.constant_initializer(value=0.0, dtype=tf.float32))
    gamma = tf.get_variable("gamma", 
                            [n_out], 
                            initializer = tf.constant_initializer(value=1.0, dtype=tf.float32))
    
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
        beta, gamma, 1e-3, True)
    return normed

def layer_batch_norm(x, n_out, phase_train):
    beta = tf.get_variable("beta", 
                           [n_out], 
                            initializer = tf.constant_initializer(value=0.0, dtype=tf.float32))
    gamma = tf.get_variable("gamma", 
                            [n_out], 
                            initializer = tf.constant_initializer(value=1.0, dtype=tf.float32))

    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var,
        beta, gamma, 1e-3, True)
    return tf.reshape(normed, [-1, n_out])


def conv2d(input, weight_shape, bias_shape, phase_train, visualize=False):

    incoming = weight_shape[0]*weight_shape[1]*weight_shape[2]
    W = tf.get_variable("W", 
                        weight_shape, 
                        initializer=tf.random_normal_initializer(stddev=(2.0/incoming)**0.5))
    b = tf.get_variable("b",
                        bias_shape,
                        initializer=tf.constant_initializer(0))
    logits = tf.nn.bias_add(tf.nn.conv2d(input, W, [1,1,1,1], padding='SAME'), b)

    W_T = tf.transpose(W, (3, 0, 1, 2))
    tf.summary.image("filters", W_T)

    return tf.nn.relu(conv_batch_norm(logits, weight_shape[3], phase_train))


def layer(input, weight_shape, bias_shape, phase_train):

    W = tf.get_variable("W", 
                        weight_shape, 
                        initializer=tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5))
    b = tf.get_variable("b",
                        bias_shape,
                        initializer=tf.constant_initializer(0))
    logits = tf.matmul(input,W)+b
    return tf.nn.relu(layer_batch_norm(logits, weight_shape[1], phase_train))

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