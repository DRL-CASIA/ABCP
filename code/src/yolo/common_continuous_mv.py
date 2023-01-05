#! /usr/bin/env python
# coding=utf-8


import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import utils

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
UPDATE_OPS = 'update_ops'
BN_EPSILON = 0.001

def tools_pruning_conv(ini_gamma, weights, layer_sparsity):
    l1_w = tf.abs(ini_gamma)
    l1_w_sort = tf.nn.top_k(l1_w, k=tf.shape(l1_w)[0])[0]
    sparsity = tf.subtract(1.0, layer_sparsity)
    l1_w_sort_f = tf.to_float(tf.shape(l1_w_sort)[0])
    thre_index = tf.cond(tf.equal(sparsity, 1.0), lambda: tf.to_int32(tf.subtract(tf.multiply(l1_w_sort_f, sparsity), 1.0)), lambda: tf.to_int32(tf.multiply(l1_w_sort_f, sparsity)))
    thre = l1_w_sort[thre_index]

    one_gamma= tf.ones_like(l1_w)
    zero_gamma = tf.zeros_like(l1_w)
    mask_gamma = tf.cond(tf.equal(sparsity, 0.0), lambda: tf.where(l1_w <= thre, x=zero_gamma, y=one_gamma),
                         lambda: tf.where(l1_w < thre, x=zero_gamma, y=one_gamma))
    mask_ini_w = tf.ones_like(weights)
    mask_w_fin = tf.multiply(mask_gamma, mask_ini_w)
    w = tf.multiply(weights, mask_w_fin)
    return w


def tools_pruning_bn(ini_gamma, gamma, beta, layer_sparsity):
    l1_w = tf.abs(ini_gamma)
    l1_w_sort = tf.nn.top_k(l1_w, k=tf.shape(l1_w)[0])[0]
    sparsity = tf.subtract(1.0, layer_sparsity)
    l1_w_sort_f = tf.to_float(tf.shape(l1_w_sort)[0])
    thre_index = tf.cond(tf.equal(sparsity, 1.0), lambda: tf.to_int32(tf.subtract(tf.multiply(l1_w_sort_f, sparsity), 1.0)), lambda: tf.to_int32(tf.multiply(l1_w_sort_f, sparsity)))
    thre = l1_w_sort[thre_index]

    one_gamma = tf.ones_like(l1_w)
    zero_gamma = tf.zeros_like(l1_w)
    mask_gamma = tf.cond(tf.equal(sparsity, 0.0), lambda: tf.where(l1_w <= thre, x=zero_gamma, y=one_gamma),
                         lambda: tf.where(l1_w < thre, x=zero_gamma, y=one_gamma))
    mask_ini_gamma = tf.ones_like(gamma)
    mask_gamma_fin = tf.multiply(mask_gamma, mask_ini_gamma)
    gamma = tf.multiply(gamma, mask_gamma_fin)

    mask_ini_beta = tf.ones_like(beta)
    mask_beta_fin = tf.multiply(mask_gamma, mask_ini_beta)
    beta = tf.multiply(beta, mask_beta_fin)
    return gamma, beta


def tools_pruning_mv(ini_gamma, mean, variance, layer_sparsity):
    l1_w = tf.abs(ini_gamma)
    l1_w_sort = tf.nn.top_k(l1_w, k=tf.shape(l1_w)[0])[0]
    sparsity = tf.subtract(1.0, layer_sparsity)
    l1_w_sort_f = tf.to_float(tf.shape(l1_w_sort)[0])
    thre_index = tf.cond(tf.equal(sparsity, 1.0), lambda: tf.to_int32(tf.subtract(tf.multiply(l1_w_sort_f, sparsity), 1.0)), lambda: tf.to_int32(tf.multiply(l1_w_sort_f, sparsity)))
    thre = l1_w_sort[thre_index]

    one_gamma = tf.ones_like(l1_w)
    zero_gamma = tf.zeros_like(l1_w)
    mask_gamma = tf.cond(tf.equal(sparsity, 0.0), lambda: tf.where(l1_w <= thre, x=zero_gamma, y=one_gamma),
                         lambda: tf.where(l1_w < thre, x=zero_gamma, y=one_gamma))
    mask_ini_gamma = tf.ones_like(mean)
    mask_gamma_fin = tf.multiply(mask_gamma, mask_ini_gamma)
    mean = tf.multiply(mean, mask_gamma_fin)

    mask_ini_beta = tf.ones_like(variance)
    mask_beta_fin = tf.multiply(mask_gamma, mask_ini_beta)
    variance = tf.multiply(variance, mask_beta_fin)
    return mean, variance


def last_conv_prune(input_data, filters_shape, H, trainable, name, flops,  downsample=False, activate=True, bn=True):
    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        ini_weight = tf.get_variable(name='ini_weight', dtype=tf.float32, trainable=False,
                                     shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                     shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01)).assign(ini_weight)


        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            ini_gamma = tf.get_variable(name='ini_gamma', dtype=tf.float32, trainable=False,
                                     shape=filters_shape[-1], initializer=tf.random_normal_initializer(stddev=0.01))
            ini_beta = tf.get_variable(name='ini_beta', dtype=tf.float32, trainable=False,
                                        shape=filters_shape[-1], initializer=tf.random_normal_initializer(stddev=0.01))
            beta = tf.get_variable(name='beta',
                                   shape=filters_shape[-1],
                                   trainable=True,
                                   initializer=tf.zeros_initializer,
                                   dtype='float').assign(ini_beta)
            gamma = tf.get_variable(name='gamma',
                                    shape=filters_shape[-1],
                                    trainable=True,
                                    initializer=tf.ones_initializer,
                                    dtype='float').assign(ini_gamma)

            ini_moving_mean = tf.get_variable(name='ini_moving_mean',
                                          shape=filters_shape[-1],
                                          trainable=False,
                                          initializer=tf.zeros_initializer,
                                          dtype='float')
            ini_moving_variance = tf.get_variable(name='ini_moving_variance',
                                              shape=filters_shape[-1],
                                              trainable=False,
                                              initializer=tf.ones_initializer,
                                              dtype='float')
                                              
            moving_mean = tf.get_variable(name='moving_mean',
                                          shape=filters_shape[-1],
                                          trainable=False,
                                          initializer=tf.zeros_initializer,
                                          dtype='float').assign(ini_moving_mean)
            moving_variance = tf.get_variable(name='moving_variance',
                                              shape=filters_shape[-1],
                                              trainable=False,
                                              initializer=tf.ones_initializer,
                                              dtype='float').assign(ini_moving_variance)
            x_shape = conv.get_shape()
            axis = list(range(len(x_shape) - 1))
            mean, variance = tf.nn.moments(conv, axis)
            #moving_mean, moving_variance = tools_pruning_mv(ini_gamma, moving_mean, moving_variance, prune_ratio)
            train_mean_op = tf.assign(moving_mean, moving_mean * BN_DECAY + mean * (1 - BN_DECAY))
            train_var_op = tf.assign(moving_variance, moving_variance * BN_DECAY + variance * (1 - BN_DECAY))

            def batch_statistics():
                with tf.control_dependencies([train_mean_op, train_var_op]):
                    return tf.nn.batch_normalization(conv, mean, variance, beta, gamma, BN_EPSILON)

            def population_statistics():
                return tf.nn.batch_normalization(conv, moving_mean, moving_variance, beta, gamma, BN_EPSILON)

            conv = tf.cond(trainable, batch_statistics, population_statistics)

            # conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
            #                                      gamma_initializer=tf.ones_initializer(),
            #                                      moving_mean_initializer=tf.zeros_initializer(),
            #                                      moving_variance_initializer=tf.ones_initializer(), training=trainable)

        else:
            ini_bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=False,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0)).assign(ini_bias)
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)
    C_in = filters_shape[2]
    C_out = filters_shape[3]
    layer_flops = tf.multiply(tf.multiply(tf.multiply(C_in, tf.multiply(H, H)), C_out),
                              tf.multiply(filters_shape[0], filters_shape[0]))
    flops += layer_flops / 1000

    return conv, flops


def last_conv_maintain(input_data, filters_shape, H, trainable, name, flops,  downsample=False, activate=True, bn=True):
    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                     shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))


        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            beta = tf.get_variable(name='beta',
                                   shape=filters_shape[-1],
                                   trainable=True,
                                   initializer=tf.zeros_initializer,
                                   dtype='float')
            gamma = tf.get_variable(name='gamma',
                                    shape=filters_shape[-1],
                                    trainable=True,
                                    initializer=tf.ones_initializer,
                                    dtype='float')

            moving_mean = tf.get_variable(name='moving_mean',
                                          shape=filters_shape[-1],
                                          trainable=False,
                                          initializer=tf.zeros_initializer,
                                          dtype='float')
            moving_variance = tf.get_variable(name='moving_variance',
                                              shape=filters_shape[-1],
                                              trainable=False,
                                              initializer=tf.ones_initializer,
                                              dtype='float')

            x_shape = conv.get_shape()
            axis = list(range(len(x_shape) - 1))
            mean, variance = tf.nn.moments(conv, axis)
            #moving_mean, moving_variance = tools_pruning_mv(ini_gamma, moving_mean, moving_variance, prune_ratio)
            train_mean_op = tf.assign(moving_mean, moving_mean * BN_DECAY + mean * (1 - BN_DECAY))
            train_var_op = tf.assign(moving_variance, moving_variance * BN_DECAY + variance * (1 - BN_DECAY))

            def batch_statistics():
                with tf.control_dependencies([train_mean_op, train_var_op]):
                    return tf.nn.batch_normalization(conv, mean, variance, beta, gamma, BN_EPSILON)

            def population_statistics():
                return tf.nn.batch_normalization(conv, moving_mean, moving_variance, beta, gamma, BN_EPSILON)

            conv = tf.cond(trainable, batch_statistics, population_statistics)

            # conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
            #                                      gamma_initializer=tf.ones_initializer(),
            #                                      moving_mean_initializer=tf.zeros_initializer(),
            #                                      moving_variance_initializer=tf.ones_initializer(), training=trainable)

        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)
    C_in = filters_shape[2]
    C_out = filters_shape[3]
    layer_flops = tf.multiply(tf.multiply(tf.multiply(C_in, tf.multiply(H, H)), C_out),
                              tf.multiply(filters_shape[0], filters_shape[0]))
    flops += layer_flops / 1000
    return conv, flops


def convolutional_prune(input_data, filters_shape, trainable, name, prune_ratio, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        ini_weights = tf.get_variable(name='ini_weight', dtype=tf.float32, trainable=False,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        ini_gamma = tf.get_variable(name='ini_gamma', dtype=tf.float32, trainable=False,
                                     shape=filters_shape[-1], initializer=tf.random_normal_initializer(stddev=0.01))
        pw = tools_pruning_conv(ini_gamma, ini_weights, prune_ratio)
        weights = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01)).assign(pw)
        conv = tf.nn.conv2d(input=input_data, filter=weights, strides=strides, padding=padding)

        if bn:
            ini_beta = tf.get_variable(name='ini_beta', dtype=tf.float32, trainable=False,
                                        shape=filters_shape[-1], initializer=tf.random_normal_initializer(stddev=0.01))
            pgamma, pbeta = tools_pruning_bn(ini_gamma, ini_gamma, ini_beta, prune_ratio)
            beta = tf.get_variable(name='beta',
                                   shape=filters_shape[-1],
                                   trainable=True,
                                   initializer=tf.zeros_initializer,
                                   dtype='float').assign(pbeta)
            gamma = tf.get_variable(name='gamma',
                                    shape=filters_shape[-1],
                                    trainable=True,
                                    initializer=tf.ones_initializer,
                                    dtype='float').assign(pgamma)

            ini_moving_mean = tf.get_variable(name='ini_moving_mean',
                                          shape=filters_shape[-1],
                                          trainable=False,
                                          initializer=tf.zeros_initializer,
                                          dtype='float')
            ini_moving_variance = tf.get_variable(name='ini_moving_variance',
                                              shape=filters_shape[-1],
                                              trainable=False,
                                              initializer=tf.ones_initializer,
                                              dtype='float')
                                              
            moving_mean = tf.get_variable(name='moving_mean',
                                          shape=filters_shape[-1],
                                          trainable=False,
                                          initializer=tf.zeros_initializer,
                                          dtype='float').assign(ini_moving_mean)
            moving_variance = tf.get_variable(name='moving_variance',
                                              shape=filters_shape[-1],
                                              trainable=False,
                                              initializer=tf.ones_initializer,
                                              dtype='float').assign(ini_moving_variance)

            x_shape = conv.get_shape()
            axis = list(range(len(x_shape) - 1))
            mean, variance = tf.nn.moments(conv, axis)
            #mean, variance = tools_pruning_mv(ini_gamma, mean, variance, prune_ratio)
            #moving_mean, moving_variance = tools_pruning_mv(ini_gamma, moving_mean, moving_variance, prune_ratio)
            train_mean_op = tf.assign(moving_mean, moving_mean * BN_DECAY + mean * (1 - BN_DECAY))
            train_var_op = tf.assign(moving_variance, moving_variance * BN_DECAY + variance * (1 - BN_DECAY))

            def batch_statistics():
                with tf.control_dependencies([train_mean_op, train_var_op]):
                    return tf.nn.batch_normalization(conv, mean, variance, beta, gamma, BN_EPSILON)

            def population_statistics():
                return tf.nn.batch_normalization(conv, moving_mean, moving_variance, beta, gamma, BN_EPSILON)

            conv = tf.cond(trainable, batch_statistics, population_statistics)

            # conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
            #                                      gamma_initializer=tf.ones_initializer(),
            #                                      moving_mean_initializer=tf.zeros_initializer(),
            #                                      moving_variance_initializer=tf.ones_initializer(), training=trainable)

        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def convolutional_maintain(input_data, filters_shape, trainable, name, prune_ratio, downsample=False, activate=True, bn=True):
    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        ini_gamma = tf.get_variable(name='ini_gamma', dtype=tf.float32, trainable=False,
                                    shape=filters_shape[-1], initializer=tf.random_normal_initializer(stddev=0.01))
        weights = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        pw = tools_pruning_conv(ini_gamma, weights, prune_ratio)
        new_weights = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                  shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01)).assign(pw)
        conv = tf.nn.conv2d(input=input_data, filter=new_weights, strides=strides, padding=padding)

        if bn:
            ini_gamma = tf.get_variable(name='ini_gamma', dtype=tf.float32, trainable=False,
                                        shape=filters_shape[-1], initializer=tf.random_normal_initializer(stddev=0.01))
            beta = tf.get_variable(name='beta',
                                   shape=filters_shape[-1],
                                   trainable=True,
                                   initializer=tf.zeros_initializer,
                                   dtype='float')
            gamma = tf.get_variable(name='gamma',
                                    shape=filters_shape[-1],
                                    trainable=True,
                                    initializer=tf.ones_initializer,
                                    dtype='float')
            pgamma, pbeta = tools_pruning_bn(ini_gamma, gamma, beta, prune_ratio)
            new_beta = tf.get_variable(name='beta',
                                   shape=filters_shape[-1],
                                   trainable=True,
                                   initializer=tf.zeros_initializer,
                                   dtype='float').assign(pbeta)
            new_gamma = tf.get_variable(name='gamma',
                                    shape=filters_shape[-1],
                                    trainable=True,
                                    initializer=tf.ones_initializer,
                                    dtype='float').assign(pgamma)

            moving_mean = tf.get_variable(name='moving_mean',
                                          shape=filters_shape[-1],
                                          trainable=False,
                                          initializer=tf.zeros_initializer,
                                          dtype='float')
            moving_variance = tf.get_variable(name='moving_variance',
                                              shape=filters_shape[-1],
                                              trainable=False,
                                              initializer=tf.ones_initializer,
                                              dtype='float')

            x_shape = conv.get_shape()
            axis = list(range(len(x_shape) - 1))
            mean, variance = tf.nn.moments(conv, axis)
            #mean, variance = tools_pruning_mv(ini_gamma, mean, variance, prune_ratio)
            #moving_mean, moving_variance = tools_pruning_mv(ini_gamma, moving_mean, moving_variance, prune_ratio)
            train_mean_op = tf.assign(moving_mean, moving_mean * BN_DECAY + mean * (1 - BN_DECAY))
            train_var_op = tf.assign(moving_variance, moving_variance * BN_DECAY + variance * (1 - BN_DECAY))

            def batch_statistics():
                with tf.control_dependencies([train_mean_op, train_var_op]):
                    return tf.nn.batch_normalization(conv, mean, variance, new_beta, new_gamma, BN_EPSILON)

            def population_statistics():
                return tf.nn.batch_normalization(conv, moving_mean, moving_variance, new_beta, new_gamma, BN_EPSILON)

            conv = tf.cond(trainable, batch_statistics, population_statistics)

            # conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
            #                                      gamma_initializer=tf.ones_initializer(),
            #                                      moving_mean_initializer=tf.zeros_initializer(),
            #                                      moving_variance_initializer=tf.ones_initializer(), training=trainable)

        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block_prune(input_data, input_channel, filter_num1, filter_num2, H, flops, trainable, name, arc, trans_ratio):

    short_cut = input_data
    with tf.variable_scope(name):

        input_data, flops, trans_ratio = enas_layer_pruning_conv(input_data, filters_shape=(1, 1, input_channel, filter_num1), H=H,
                                   trainable=trainable, name='conv1', prune_ratio=arc[0], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = enas_layer_pruning_conv(input_data, filters_shape=(3, 3, filter_num1, filter_num2), H=H,
                                   trainable=trainable, name='conv2', prune_ratio=arc[1], trans_ratio=trans_ratio, flops=flops)


        residual_output = input_data + short_cut

    return residual_output, flops, trans_ratio


def residual_block_maintain(input_data, input_channel, filter_num1, filter_num2, H, flops, trainable, name, arc, trans_ratio):

    short_cut = input_data

    with tf.variable_scope(name):

        input_data, flops, trans_ratio = maintain_pruning_conv(input_data, filters_shape=(1, 1, input_channel, filter_num1), H=H,
                                   trainable=trainable, name='conv1', prune_ratio=arc[0], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = maintain_pruning_conv(input_data, filters_shape=(3, 3, filter_num1, filter_num2), H=H,
                                   trainable=trainable, name='conv2', prune_ratio=arc[1], trans_ratio=trans_ratio, flops=flops)

        residual_output = input_data + short_cut

    return residual_output, flops, trans_ratio


def enas_layer_pruning_conv(input_data, filters_shape, H, trainable, name, prune_ratio, trans_ratio, flops, downsample=False, activate=True, bn=True):
  C_in = tf.cond(tf.equal(trans_ratio, 1.0), lambda: tf.to_int32(tf.multiply(tf.to_float(filters_shape[2]), trans_ratio)), lambda: tf.add(tf.to_int32(tf.multiply(tf.to_float(filters_shape[2]), trans_ratio)), 1))
  #C_in = filters_shape[2]
  ep_outs = convolutional_prune(input_data, filters_shape, trainable, name, prune_ratio, downsample, activate, bn)

  remain_ratio = tf.subtract(1.0, prune_ratio)
  C_out = tf.cond(tf.equal(remain_ratio, 1.0), lambda: tf.to_int32(tf.multiply(tf.to_float(filters_shape[3]), remain_ratio)), lambda: tf.add(tf.to_int32(tf.multiply(tf.to_float(filters_shape[3]), remain_ratio)), 1))
  layer_flops = tf.multiply(tf.multiply(tf.multiply(C_in, tf.multiply(H, H)), C_out),
                            tf.multiply(filters_shape[0], filters_shape[0]))
  flops += layer_flops / 1000
  #flops = layer_flops
  #flops = tf.cond(layer_flops < 0, lambda: flops, lambda: tf.add(flops, layer_flops))
  trans_ratio = tf.cond(tf.equal(remain_ratio, 0.0), lambda: trans_ratio, lambda: remain_ratio)
  return ep_outs, flops, trans_ratio


def maintain_pruning_conv(input_data, filters_shape, H, trainable, name, prune_ratio, trans_ratio, flops,  downsample=False, activate=True, bn=True):

  C_in = tf.cond(tf.equal(trans_ratio, 1.0), lambda: tf.to_int32(tf.multiply(tf.to_float(filters_shape[2]), trans_ratio)), lambda: tf.add(tf.to_int32(tf.multiply(tf.to_float(filters_shape[2]), trans_ratio)), 1))
  #C_in = filters_shape[2]
  ep_outs = convolutional_maintain(input_data, filters_shape, trainable, name, prune_ratio, downsample, activate, bn)

  remain_ratio = tf.subtract(1.0, prune_ratio)
  C_out = tf.cond(tf.equal(remain_ratio, 1.0), lambda: tf.to_int32(tf.multiply(tf.to_float(filters_shape[3]), remain_ratio)), lambda: tf.add(tf.to_int32(tf.multiply(tf.to_float(filters_shape[3]), remain_ratio)), 1))
  layer_flops = tf.multiply(tf.multiply(tf.multiply(C_in, tf.multiply(H, H)), C_out),
                            tf.multiply(filters_shape[0], filters_shape[0]))
  flops += layer_flops / 1000
  #flops = layer_flops
  #flops = tf.cond(layer_flops < 0, lambda: flops, lambda: tf.add(flops, layer_flops))
  trans_ratio = tf.cond(tf.equal(remain_ratio, 0.0), lambda: trans_ratio, lambda: remain_ratio)
  return ep_outs, flops, trans_ratio


def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

        return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]

        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output



