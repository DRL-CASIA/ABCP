#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np



def tools_pruning(w_weight, w_biases, layer_sparsity):
    w_and_b = tf.add(w_weight, tf.reshape(w_biases, [1, 1, 1, tf.shape(w_biases)[0]]))
    l1_w = tf.reduce_sum(tf.abs(w_and_b), axis=(0, 1, 2))
    l1_w_sort = tf.nn.top_k(l1_w, k=tf.shape(l1_w)[0])[0]
    real_layer_sparsity = 1.0 - layer_sparsity
    sparsity = tf.Variable([real_layer_sparsity], trainable=False)
    l1_w_sort_f = tf.to_float(tf.shape(l1_w_sort)[0])
    thre_index = tf.to_int32(tf.multiply(l1_w_sort_f, sparsity))[0]
    if layer_sparsity == 0.0:
        thre = l1_w_sort[thre_index - 1]
    else:
        thre = l1_w_sort[thre_index]

    # thre = l1_w_sort[thre_index - 1]

    one_w_and_b = tf.ones_like(l1_w)
    zero_w_and_b = tf.zeros_like(l1_w)
    mask_w_and_b = tf.where(l1_w <= thre, x=zero_w_and_b, y=one_w_and_b)
    mask_ini_w = tf.ones_like(w_weight)
    mask_w_fin = tf.multiply(tf.reshape(mask_w_and_b, [1, 1, 1, tf.shape(mask_w_and_b)[0]]), mask_ini_w)
    w = tf.multiply(w_weight, mask_w_fin)

    mask_ini_b = tf.ones_like(w_biases)
    mask_b_fin = tf.multiply(mask_w_and_b, mask_ini_b)
    b = tf.multiply(w_biases, mask_b_fin)
    return w, b

def maintain_tools_pruning(ini_w_weight, ini_w_biases, w_weight, w_biases, layer_sparsity):
    w_and_b = tf.add(ini_w_weight, tf.reshape(ini_w_biases, [1, 1, 1, tf.shape(ini_w_biases)[0]]))
    l1_w = tf.reduce_sum(tf.abs(w_and_b), axis=(0, 1, 2))
    l1_w_sort = tf.nn.top_k(l1_w, k=tf.shape(l1_w)[0])[0]
    real_layer_sparsity = 1.0 - layer_sparsity
    sparsity = tf.Variable([real_layer_sparsity], trainable=False)
    l1_w_sort_f = tf.to_float(tf.shape(l1_w_sort)[0])
    thre_index = tf.to_int32(tf.multiply(l1_w_sort_f, sparsity))[0]
    if layer_sparsity == 0.0:
        thre = l1_w_sort[thre_index - 1]
    else:
        thre = l1_w_sort[thre_index]

    # thre = l1_w_sort[thre_index - 1]

    one_w_and_b = tf.ones_like(l1_w)
    zero_w_and_b = tf.zeros_like(l1_w)
    mask_w_and_b = tf.where(l1_w <= thre, x=zero_w_and_b, y=one_w_and_b)
    mask_ini_w = tf.ones_like(w_weight)
    mask_w_fin = tf.multiply(tf.reshape(mask_w_and_b, [1, 1, 1, tf.shape(mask_w_and_b)[0]]), mask_ini_w)
    w = tf.multiply(w_weight, mask_w_fin)

    mask_ini_b = tf.ones_like(w_biases)
    mask_b_fin = tf.multiply(mask_w_and_b, mask_ini_b)
    b = tf.multiply(w_biases, mask_b_fin)
    return w, b

def tools_pruning_continuous(w_weight, w_biases, layer_sparsity):
    w_and_b = tf.add(w_weight, tf.reshape(w_biases, [1, 1, 1, tf.shape(w_biases)[0]]))
    l1_w = tf.reduce_sum(tf.abs(w_and_b), axis=(0, 1, 2))
    l1_w_sort = tf.nn.top_k(l1_w, k=tf.shape(l1_w)[0])[0]
    sparsity = tf.subtract(1.0, layer_sparsity)
    l1_w_sort_f = tf.to_float(tf.shape(l1_w_sort)[0])
    thre_index = tf.cond(tf.equal(sparsity, 1.0), lambda: tf.to_int32(tf.subtract(tf.multiply(l1_w_sort_f, sparsity), 1.0)), lambda: tf.to_int32(tf.multiply(l1_w_sort_f, sparsity)))
    thre = l1_w_sort[thre_index]

    one_w_and_b = tf.ones_like(l1_w)
    zero_w_and_b = tf.zeros_like(l1_w)
    mask_w_and_b = tf.cond(tf.equal(sparsity, 0.0), lambda: tf.where(l1_w <= thre, x=zero_w_and_b, y=one_w_and_b), lambda: tf.where(l1_w <= thre, x=zero_w_and_b, y=one_w_and_b))
    mask_ini_w = tf.ones_like(w_weight)
    mask_w_fin = tf.multiply(tf.reshape(mask_w_and_b, [1, 1, 1, tf.shape(mask_w_and_b)[0]]), mask_ini_w)
    w = tf.multiply(w_weight, mask_w_fin)

    mask_ini_b = tf.ones_like(w_biases)
    mask_b_fin = tf.multiply(mask_w_and_b, mask_ini_b)
    b = tf.multiply(w_biases, mask_b_fin)
    return w, b


def maintain_tools_pruning_continuous(ini_w_weight, ini_w_biases, w_weight, w_biases, layer_sparsity):
    w_and_b = tf.add(ini_w_weight, tf.reshape(ini_w_biases, [1, 1, 1, tf.shape(w_biases)[0]]))
    l1_w = tf.reduce_sum(tf.abs(w_and_b), axis=(0, 1, 2))
    l1_w_sort = tf.nn.top_k(l1_w, k=tf.shape(l1_w)[0])[0]
    sparsity = tf.subtract(1.0, layer_sparsity)
    l1_w_sort_f = tf.to_float(tf.shape(l1_w_sort)[0])
    # thre_index = tf.to_int32(tf.multiply(l1_w_sort_f, sparsity))
    thre_index = tf.cond(tf.equal(sparsity, 1.0),
                         lambda: tf.to_int32(tf.subtract(tf.multiply(l1_w_sort_f, sparsity), 1.0)),
                         lambda: tf.to_int32(tf.multiply(l1_w_sort_f, sparsity)))
    thre = l1_w_sort[thre_index]

    one_w_and_b = tf.ones_like(l1_w)
    zero_w_and_b = tf.zeros_like(l1_w)
    mask_w_and_b = tf.cond(tf.equal(sparsity, 0.0), lambda: tf.where(l1_w <= thre, x=zero_w_and_b, y=one_w_and_b), lambda: tf.where(l1_w <= thre, x=zero_w_and_b, y=one_w_and_b))  # 剪枝对象是权重中的通道维度  # 剪枝对象是权重中的通道维度
    mask_ini_w = tf.ones_like(w_weight)
    mask_w_fin = tf.multiply(tf.reshape(mask_w_and_b, [1, 1, 1, tf.shape(mask_w_and_b)[0]]), mask_ini_w)
    w = tf.multiply(w_weight, mask_w_fin)

    mask_ini_b = tf.ones_like(w_biases)
    mask_b_fin = tf.multiply(mask_w_and_b, mask_ini_b)
    b = tf.multiply(w_biases, mask_b_fin)
    return w, b


# %%
def conv(layer_name, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=True):
    """Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers. 
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    """

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name, reuse=True):
        if "fc" in layer_name:
          w = tf.get_variable(name='weights', trainable=True, shape=[kernel_size[0], kernel_size[1], in_channels, out_channels])
          b = tf.get_variable(name='biases', trainable=True, shape=[out_channels])
        else:
          w = tf.get_variable(name='weights', trainable=True, shape=[kernel_size[0], kernel_size[1], in_channels, out_channels])
          b = tf.get_variable(name='biases', trainable=True, shape=[out_channels])
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x

def conv_maintain(layer_name, prune_ratio, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=True):
    """Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers.
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    """

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name, reuse=True):
        ini_w = tf.get_variable(name='ini_weights',
                            trainable=False,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype='float')
        ini_b = tf.get_variable(name='ini_biases',
                            trainable=False,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0),
                            dtype='float')
        if "fc" in layer_name:
          w = tf.get_variable(name='weights',
                              trainable=True,
                              shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                              initializer=tf.contrib.layers.xavier_initializer(),
                              dtype='float')
          b = tf.get_variable(name='biases',
                              trainable=True,
                              shape=[out_channels],
                              initializer=tf.constant_initializer(0.0),
                              dtype='float')

        else:
          w = tf.get_variable(name='weights',
                              trainable=True,
                              shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                              initializer=tf.contrib.layers.xavier_initializer(),
                              dtype='float')
          b = tf.get_variable(name='biases',
                              trainable=True,
                              shape=[out_channels],
                              initializer=tf.constant_initializer(0.0),
                              dtype='float')
        pw, pb = maintain_tools_pruning(ini_w, ini_b, w, b, prune_ratio)

        if "fc" in layer_name:
            new_w = tf.get_variable(name='weights',
                                    trainable=True,
                                    shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype='float').assign(pw)
            new_b = tf.get_variable(name='biases',
                                    trainable=True,
                                    shape=[out_channels],
                                    initializer=tf.constant_initializer(0.0),
                                    dtype='float').assign(pb)
        else:
            new_w = tf.get_variable(name='weights',
                                    trainable=True,
                                    shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype='float').assign(pw)
            new_b = tf.get_variable(name='biases',
                                    trainable=True,
                                    shape=[out_channels],
                                    initializer=tf.constant_initializer(0.0),
                                    dtype='float').assign(pb)

        x = tf.nn.conv2d(x, new_w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, new_b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x

def conv_maintain_continuous(layer_name, prune_ratio, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=True):
    """Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers.
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    """

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name, reuse=True):
        ini_w = tf.get_variable(name='ini_weights',
                            trainable=False,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype='float')
        ini_b = tf.get_variable(name='ini_biases',
                            trainable=False,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0),
                            dtype='float')

        if "fc" in layer_name:
          w = tf.get_variable(name='weights',
                              trainable=True,
                              shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                              initializer=tf.contrib.layers.xavier_initializer(),
                              dtype='float')
          b = tf.get_variable(name='biases',
                              trainable=True,
                              shape=[out_channels],
                              initializer=tf.constant_initializer(0.0),
                              dtype='float')

        else:
          w = tf.get_variable(name='weights',
                              trainable=True,
                              shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                              initializer=tf.contrib.layers.xavier_initializer(),
                              dtype='float')
          b = tf.get_variable(name='biases',
                              trainable=True,
                              shape=[out_channels],
                              initializer=tf.constant_initializer(0.0),
                              dtype='float')
        # initializer=tf.constant_initializer(0.0))
        pw, pb = maintain_tools_pruning_continuous(ini_w, ini_b, w, b, prune_ratio)

        if "fc" in layer_name:
            new_w = tf.get_variable(name='weights',
                                    trainable=True,
                                    shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype='float').assign(pw)
            new_b = tf.get_variable(name='biases',
                                    trainable=True,
                                    shape=[out_channels],
                                    initializer=tf.constant_initializer(0.0),
                                    dtype='float').assign(pb)
        else:
            new_w = tf.get_variable(name='weights',
                                    trainable=True,
                                    shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype='float').assign(pw)
            new_b = tf.get_variable(name='biases',
                                    trainable=True,
                                    shape=[out_channels],
                                    initializer=tf.constant_initializer(0.0),
                                    dtype='float').assign(pb)

        x = tf.nn.conv2d(x, new_w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, new_b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


def prune_conv(layer_name, prune_ratio, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=True):
    """Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers.
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    """

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='ini_weights',
                            trainable=False,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype='float')
        b = tf.get_variable(name='ini_biases',
                            trainable=False,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0),
                            dtype='float')
        pw, pb = tools_pruning(w, b, prune_ratio)
        
        if "fc" in layer_name:
          new_w = tf.get_variable(name='weights',
                                  trainable=True,
                                  shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype='float').assign(pw)
          new_b = tf.get_variable(name='biases',
                                  trainable=True,
                                  shape=[out_channels],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype='float').assign(pb)
        else:
          new_w = tf.get_variable(name='weights',
                                  trainable=True,
                                  shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype='float').assign(pw)
          new_b = tf.get_variable(name='biases',
                                  trainable=True,
                                  shape=[out_channels],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype='float').assign(pb)

#        new_w = tf.assign(new_w, pw)
#        new_b = tf.assign(new_b, pb)
        x = tf.nn.conv2d(x, new_w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, new_b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x



def prune_conv_continuous(layer_name, prune_ratio, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=True):
    """Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers.
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    """

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='ini_weights',
                            trainable=False,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype='float')
        b = tf.get_variable(name='ini_biases',
                            trainable=False,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0),
                            dtype='float')

        pw, pb = tools_pruning_continuous(w, b, prune_ratio)

        if "fc" in layer_name:
          new_w = tf.get_variable(name='weights',
                                  trainable=True,
                                  shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype='float').assign(pw)
          new_b = tf.get_variable(name='biases',
                                  trainable=True,
                                  shape=[out_channels],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype='float').assign(pb)
        else:
          new_w = tf.get_variable(name='weights',
                                  trainable=True,
                                  shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype='float').assign(pw)
          new_b = tf.get_variable(name='biases',
                                  trainable=True,
                                  shape=[out_channels],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype='float').assign(pb)

        x = tf.nn.conv2d(x, new_w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, new_b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


# %%
def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    """Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    """
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x


# %%
def batch_norm(x):
    """Batch normlization(I didn't include the offset and scale)
    """
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


# %%
def FC_layer(layer_name, x, out_nodes):
    """Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    """
    print('#######', x)
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes])
                            # initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes])
                            # initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size])   # flatten into 1D
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x


# %%
def loss(logits, labels):
    """Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    """
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', loss)
        return loss


# %%
def accuracy(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)*100.0
        tf.summary.scalar(scope+'/accuracy', accuracy)
    return accuracy


# %%
def num_correct_prediction(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Return:
        the number of correct predictions
    """
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)  # 计算正确的数量
    return n_correct


# %%
def optimize(loss, learning_rate, global_step):
    """optimization, use Gradient Descent as default
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    

# %%
def load(data_path, session):
    data_dict = np.load(data_path, encoding='latin1').item()
    
    keys = sorted(data_dict.keys())
    for key in keys:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                session.run(tf.get_variable(subkey).assign(data))
                

# %%
def test_load():
    data_path = './vgg_pretrain/vgg19.npy'
    
    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('\n')
        print(key)
        print('weights shape: ', weights.shape)
        print('biases shape: ', biases.shape)

    
# %%
def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            net_key = 'vgg_16/'+key.split('_')[0]+'/'+key
            with tf.variable_scope(net_key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    print(subkey)
                    session.run(tf.get_variable(subkey).assign(data))

   
# %%
def print_all_variables(train_only=True):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)

    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try: # TF1.0
            t_vars = tf.global_variables()
        except: # TF0.12
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))   

# %%


# #***** the followings are just for test the tensor size at diferent layers *********# #

# %%
def weight(kernel_shape, is_uniform = True):
    """ weight initializer
    Args:
        shape: the shape of weight
        is_uniform: boolen type.
                if True: use uniform distribution initializer
                if False: use normal distribution initizalizer
    Returns:
        weight tensor
    """
    w = tf.get_variable(name='weights',
                        shape=kernel_shape,
                        initializer=tf.contrib.layers.xavier_initializer())    
    return w


# %%
def bias(bias_shape):
    """bias initializer
    """
    b = tf.get_variable(name='biases',
                        shape=bias_shape,
                        initializer=tf.constant_initializer(0.0))
    return b
