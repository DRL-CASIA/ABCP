#! /usr/bin/env python
# coding=utf-8


import common as common
import tensorflow as tf
import utils


def darknet53_prune(input_data, trainable, arc, flops, trans_ratio):

    with tf.variable_scope('darknet'):

        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, filters_shape=(3, 3,  3,  32), H=416, trainable=trainable, name='conv0', prune_id=arc[0], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, filters_shape=(3, 3, 32,  64), H=208,
                                          trainable=trainable, name='conv1', downsample=True, prune_id=arc[1], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = common.residual_block_prune(input_data,  64,  32, 64, H=208, trainable=trainable, name='residual0', arc=arc[2: 4], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, filters_shape=(3, 3,  64, 128), H=104,
                                          trainable=trainable, name='conv4', downsample=True, prune_id=arc[4], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 128, 64, 128,  H=104,trainable=trainable, name='residual2', arc=arc[7:9], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, filters_shape=(3, 3, 128, 256), H=52,
                                          trainable=trainable, name='conv9', downsample=True, prune_id=arc[9], trans_ratio=trans_ratio,  flops=flops)

        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual3', arc=arc[10:12], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual4', arc=arc[12:14], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual5', arc=arc[14:16], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual6', arc=arc[16:18], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual7', arc=arc[18:20], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual8', arc=arc[20:22], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual9', arc=arc[22:24], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual10', arc=arc[24:26], trans_ratio=trans_ratio,  flops=flops)

        route_1 = input_data
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, filters_shape=(3, 3, 256, 512), H=26,
                                          trainable=trainable, name='conv26', downsample=True, prune_id=arc[26], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual11', arc=arc[27:29], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual12', arc=arc[29:31], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual13', arc=arc[31:33], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual14', arc=arc[33:35], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual15', arc=arc[35:37], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual16', arc=arc[37:39], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual17', arc=arc[39:41], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual18', arc=arc[41:43], trans_ratio=trans_ratio, flops=flops)

        route_2 = input_data
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, filters_shape=(3, 3, 512, 1024), H=13,
                                          trainable=trainable, name='conv43', downsample=True, prune_id=arc[43], trans_ratio=trans_ratio,  flops=flops)

        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 1024, 512, 1024, H=13, trainable=trainable, name='residual19', arc=arc[44:46], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 1024, 512, 1024, H=13, trainable=trainable, name='residual20', arc=arc[46:48], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 1024, 512, 1024, H=13, trainable=trainable, name='residual21', arc=arc[48:50], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.residual_block_prune(input_data, 1024, 512, 1024, H=13, trainable=trainable, name='residual22', arc=arc[50:], trans_ratio=trans_ratio, flops=flops)

        return route_1, route_2, input_data, flops, trans_ratio

def darknet53_maintain(input_data, trainable, arc, flops, trans_ratio):

    with tf.variable_scope('darknet'):

        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, filters_shape=(3, 3,  3,  32), H=416, trainable=trainable, name='conv0', prune_id=arc[0], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, filters_shape=(3, 3, 32,  64), H=208,
                                          trainable=trainable, name='conv1', downsample=True, prune_id=arc[1], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = common.residual_block_maintain(input_data,  64,  32, 64, H=208, trainable=trainable, name='residual0', arc=arc[2: 4], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, filters_shape=(3, 3,  64, 128), H=104,
                                          trainable=trainable, name='conv4', downsample=True, prune_id=arc[4], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 128,  64, 128,  H=104, trainable=trainable, name='residual1', arc=arc[5:7], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 128, 64, 128,  H=104, trainable=trainable, name='residual2', arc=arc[7:9], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, filters_shape=(3, 3, 128, 256), H=52,
                                          trainable=trainable, name='conv9', downsample=True, prune_id=arc[9], trans_ratio=trans_ratio,  flops=flops)

        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual3', arc=arc[10:12], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual4', arc=arc[12:14], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual5', arc=arc[14:16], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual6', arc=arc[16:18], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual7', arc=arc[18:20], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual8', arc=arc[20:22], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual9', arc=arc[22:24], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 256, 128, 256, H=52, trainable=trainable, name='residual10', arc=arc[24:26], trans_ratio=trans_ratio,  flops=flops)

        route_1 = input_data
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, filters_shape=(3, 3, 256, 512), H=26,
                                          trainable=trainable, name='conv26', downsample=True, prune_id=arc[26], trans_ratio=trans_ratio,  flops=flops)

        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual11', arc=arc[27:29], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual12', arc=arc[29:31], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual13', arc=arc[31:33], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual14', arc=arc[33:35], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual15', arc=arc[35:37], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual16', arc=arc[37:39], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual17', arc=arc[39:41], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 512, 256, 512, H=26, trainable=trainable, name='residual18', arc=arc[41:43], trans_ratio=trans_ratio,  flops=flops)

        route_2 = input_data
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, filters_shape=(3, 3, 512, 1024), H=13,
                                          trainable=trainable, name='conv43', downsample=True, prune_id=arc[43], trans_ratio=trans_ratio, flops=flops)

        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 1024, 512, 1024, H=13, trainable=trainable, name='residual19', arc=arc[44:46], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 1024, 512, 1024, H=13, trainable=trainable, name='residual20', arc=arc[46:48], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 1024, 512, 1024, H=13, trainable=trainable, name='residual21', arc=arc[48:50], trans_ratio=trans_ratio,  flops=flops)
        input_data, flops, trans_ratio = common.residual_block_maintain(input_data, 1024, 512, 1024, H=13, trainable=trainable, name='residual22', arc=arc[50:], trans_ratio=trans_ratio,  flops=flops)

        return route_1, route_2, input_data, flops, trans_ratio





