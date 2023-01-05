#coding=UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from src.yolo.models import Model
from src.yolo.image_ops import conv
from src.yolo.image_ops import fully_connected
from src.yolo.image_ops import batch_norm
from src.yolo.image_ops import batch_norm_with_mask
from src.yolo.image_ops import relu
from src.yolo.image_ops import max_pool
from src.yolo.image_ops import global_avg_pool

from src.common_ops import create_weight
import src.yolo.common_continuous_mv as common
import src.yolo.backbone_continuous_mv as backbone
from src.yolo.config import cfg
from src.yolo.dataset import Dataset
import src.yolo.utils as utils


class GeneralChild(object):
  def __init__(self,
               name,
               epoch_lrdec,
               epoch_lrdec_ratio,
               cutout_size=None,
               whole_channels=False,
               fixed_arc=None,
               out_filters_scale=1,
               num_layers=2,
               num_branches=6,
               out_filters=24,
               keep_prob=1.0,
               batch_size=32,
               clip_mode=None,
               grad_bound=None,
               moving_average=0.9997,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=10000,
               lr_dec_rate=0.1,
               lr_cosine=False,
               lr_max=None,
               lr_min=None,
               lr_T_0=None,
               lr_T_mul=None,
               optim_algo=None,
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               data_format="NHWC",
               *args,
               **kwargs
              ):
    """
    """
    self.whole_channels = whole_channels
    self.epoch_lrdec = epoch_lrdec
    self.moving_average = moving_average
    self.lr_cosine = lr_cosine
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.lr_T_0 = lr_T_0
    self.lr_T_mul = lr_T_mul
    self.out_filters = out_filters * out_filters_scale
    self.num_layers = num_layers

    self.cutout_size = cutout_size
    self.batch_size = batch_size
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.epoch_lrdec = epoch_lrdec
    self.epoch_lrdec_ratio = epoch_lrdec_ratio
    self.moving_average = moving_average
    self.l2_reg = l2_reg
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_rate = lr_dec_rate
    self.keep_prob = keep_prob
    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.data_format = data_format
    self.name = name

    self.num_branches = num_branches
    self.fixed_arc = fixed_arc
    self.out_filters_scale = out_filters_scale

    pool_distance = self.num_layers // 3
    self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]


    self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
    self.num_class = len(self.classes)
    self.strides = np.array(cfg.YOLO.STRIDES)
    self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
    self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
    self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
    self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD

    self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
    self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
    self.num_classes = len(self.classes)
    self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT  # 1e-4
    self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END  # 1e-6
    self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS  # 20
    self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS  # 30
    self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS  # 2
    self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY  # 0.9995
    self.max_bbox_per_scale = 150
    self.trainset = Dataset('train')
    self.testset = Dataset('test')
    self.steps_per_period = len(self.trainset)


  def _get_C(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return x.get_shape()[3].value
    elif self.data_format == "NCHW":
      return x.get_shape()[1].value
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

  def _get_HW(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    return x.get_shape()[2].value

  def _get_strides(self, stride):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return [1, stride, stride, 1]
    elif self.data_format == "NCHW":
      return [1, 1, stride, stride]
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))


  def _model(self, input_data, trainable, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(self.name, reuse=reuse):
        flops = 0
        trans_ratio = 1.0
        route_1, route_2, input_data, flops, trans_ratio = backbone.darknet53_prune(input_data, trainable, self.sample_arc[0:52], trans_ratio=trans_ratio, flops=flops)
        # neck
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (1, 1, 1024, 512), 13, trainable, 'conv52', self.sample_arc[52], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (3, 3, 512, 1024), 13, trainable, 'conv53', self.sample_arc[53], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (1, 1, 1024, 512), 13, trainable, 'conv54', self.sample_arc[54], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (3, 3, 512, 1024), 13, trainable, 'conv55', self.sample_arc[55], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (1, 1, 1024, 512), 13, trainable, 'conv56', self.sample_arc[56], trans_ratio=trans_ratio, flops=flops)

        conv_lobj_branch, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (3, 3, 512, 1024), 13, trainable, 'conv_lobj_branch', self.sample_arc[57], trans_ratio=trans_ratio, flops=flops)
        # lbbox
        conv_lbbox, flops = common.last_conv_prune(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)), 13,
                                          trainable=trainable, name='conv_lbbox', activate=False, bn=False, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (1, 1, 512, 256), 13, trainable, 'conv57', self.sample_arc[58], trans_ratio=trans_ratio, flops=flops)
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (1, 1, 768, 256), 26, trainable, 'conv58', self.sample_arc[59], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (3, 3, 256, 512), 26, trainable, 'conv59', self.sample_arc[60], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (1, 1, 512, 256), 26, trainable, 'conv60', self.sample_arc[61], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (3, 3, 256, 512), 26, trainable, 'conv61', self.sample_arc[62], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (1, 1, 512, 256), 26, trainable, 'conv62', self.sample_arc[63], trans_ratio=trans_ratio, flops=flops)

        conv_mobj_branch, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (3, 3, 256, 512), 26, trainable, 'conv_mobj_branch', self.sample_arc[64], trans_ratio=trans_ratio, flops=flops)
        # mbbox
        conv_mbbox, flops = common.last_conv_prune(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)), 26,
                                          trainable=trainable, name='conv_mbbox', activate=False, bn=False, flops=flops)

        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (1, 1, 256, 128), 26, self.trainable, 'conv63', self.sample_arc[65], trans_ratio=trans_ratio, flops=flops)
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (1, 1, 384, 128), 52, trainable, 'conv64', self.sample_arc[66], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (3, 3, 128, 256), 52, trainable, 'conv65', self.sample_arc[67], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (1, 1, 256, 128), 52, trainable, 'conv66', self.sample_arc[68], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (3, 3, 128, 256), 52, trainable, 'conv67', self.sample_arc[69], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (1, 1, 256, 128), 52, trainable, 'conv68', self.sample_arc[70], trans_ratio=trans_ratio, flops=flops)

        conv_sobj_branch, flops, trans_ratio = common.enas_layer_pruning_conv(input_data, (3, 3, 128, 256), 52, trainable, 'conv_sobj_branch', self.sample_arc[71], trans_ratio=trans_ratio, flops=flops)
        # sbbox
        conv_sbbox, flops = common.last_conv_prune(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)), 52,
                                          trainable=trainable, name='conv_sbbox', activate=False, bn=False, flops=flops)

    return conv_lbbox, conv_mbbox, conv_sbbox, flops


  def _model_train(self, input_data, trainable, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(self.name, reuse=reuse):
        flops = 0
        trans_ratio = 1.0
        route_1, route_2, input_data, flops, trans_ratio = backbone.darknet53_maintain(input_data, trainable, self.sample_arc[0:52], trans_ratio=trans_ratio, flops=flops)
        # neck
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (1, 1, 1024, 512), 13, trainable, 'conv52', self.sample_arc[52], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (3, 3, 512, 1024), 13, trainable, 'conv53', self.sample_arc[53], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (1, 1, 1024, 512), 13, trainable, 'conv54', self.sample_arc[54], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (3, 3, 512, 1024), 13, trainable, 'conv55', self.sample_arc[55], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (1, 1, 1024, 512), 13, trainable, 'conv56', self.sample_arc[56], trans_ratio=trans_ratio, flops=flops)

        conv_lobj_branch, flops, trans_ratio = common.maintain_pruning_conv(input_data, (3, 3, 512, 1024), 13, trainable, 'conv_lobj_branch', self.sample_arc[57], trans_ratio=trans_ratio, flops=flops)
        # lbbox
        conv_lbbox, flops = common.last_conv_maintain(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)), 13,
                                          trainable=trainable, name='conv_lbbox', activate=False, bn=False, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (1, 1, 512, 256), 13, trainable, 'conv57', self.sample_arc[58], trans_ratio=trans_ratio, flops=flops)
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (1, 1, 768, 256), 26, trainable, 'conv58', self.sample_arc[59], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (3, 3, 256, 512), 26, trainable, 'conv59', self.sample_arc[60], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (1, 1, 512, 256), 26, trainable, 'conv60', self.sample_arc[61], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (3, 3, 256, 512), 26, trainable, 'conv61', self.sample_arc[62], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (1, 1, 512, 256), 26, trainable, 'conv62', self.sample_arc[63], trans_ratio=trans_ratio, flops=flops)

        conv_mobj_branch, flops, trans_ratio = common.maintain_pruning_conv(input_data, (3, 3, 256, 512), 26, trainable, 'conv_mobj_branch', self.sample_arc[64], trans_ratio=trans_ratio, flops=flops)
        # mbbox
        conv_mbbox, flops = common.last_conv_maintain(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)), 26,
                                          trainable=trainable, name='conv_mbbox', activate=False, bn=False, flops=flops)

        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (1, 1, 256, 128), 26, self.trainable, 'conv63', self.sample_arc[65], trans_ratio=trans_ratio, flops=flops)
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (1, 1, 384, 128), 52, trainable, 'conv64', self.sample_arc[66], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (3, 3, 128, 256), 52, trainable, 'conv65', self.sample_arc[67], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (1, 1, 256, 128), 52, trainable, 'conv66', self.sample_arc[68], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (3, 3, 128, 256), 52, trainable, 'conv67', self.sample_arc[69], trans_ratio=trans_ratio, flops=flops)
        input_data, flops, trans_ratio = common.maintain_pruning_conv(input_data, (1, 1, 256, 128), 52, trainable, 'conv68', self.sample_arc[70], trans_ratio=trans_ratio, flops=flops)

        conv_sobj_branch, flops, trans_ratio = common.maintain_pruning_conv(input_data, (3, 3, 128, 256), 52, trainable, 'conv_sobj_branch', self.sample_arc[71], trans_ratio=trans_ratio, flops=flops)
        # sbbox
        conv_sbbox, flops = common.last_conv_maintain(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)), 52,
                                          trainable=trainable, name='conv_sbbox', activate=False, bn=False, flops=flops)

    return conv_lbbox, conv_mbbox, conv_sbbox, flops


  def decode(self, conv_output, anchors, stride):
      """
      return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
             contains (x, y, w, h, score, probability)
      """

      conv_shape       = tf.shape(conv_output)
      batch_size       = conv_shape[0]
      output_size      = conv_shape[1]
      anchor_per_scale = len(anchors)

      conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

      conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
      conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
      conv_raw_conf = conv_output[:, :, :, :, 4:5]
      conv_raw_prob = conv_output[:, :, :, :, 5: ]

      y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
      x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

      xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
      xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
      xy_grid = tf.cast(xy_grid, tf.float32)

      pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
      pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
      pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

      pred_conf = tf.sigmoid(conv_raw_conf)
      pred_prob = tf.sigmoid(conv_raw_prob)

      return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)  # yolo output

  def focal(self, target, actual, alpha=1, gamma=2):
      focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
      return focal_loss


  def bbox_giou(self, boxes1, boxes2):

      boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                          boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
      boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                          boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

      boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                          tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
      boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                          tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

      boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
      boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

      left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
      right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

      inter_section = tf.maximum(right_down - left_up, 0.0)
      inter_area = inter_section[..., 0] * inter_section[..., 1]
      union_area = boxes1_area + boxes2_area - inter_area
      iou = inter_area / (union_area + 1e-6)
      # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

      enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
      enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
      enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
      enclose_area = enclose[..., 0] * enclose[..., 1]
      giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-6)
      return giou


  def bbox_iou(self, boxes1, boxes2):

      boxes1_area = boxes1[..., 2] * boxes1[..., 3]
      boxes2_area = boxes2[..., 2] * boxes2[..., 3]

      boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                          boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
      boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                          boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

      left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
      right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

      inter_section = tf.maximum(right_down - left_up, 0.0)
      inter_area = inter_section[..., 0] * inter_section[..., 1]
      union_area = boxes1_area + boxes2_area - inter_area
      iou = 1.0 * inter_area / union_area

      return iou


  def loss_layer(self, conv, pred, label, bboxes, anchors, stride):  # sigle yolo layer loss

      conv_shape  = tf.shape(conv)
      batch_size  = conv_shape[0]
      output_size = conv_shape[1]
      input_size  = stride * output_size
      conv = tf.reshape(conv, (batch_size, output_size, output_size,
                               self.anchor_per_scale, 5 + self.num_class))
      conv_raw_conf = conv[:, :, :, :, 4:5]
      conv_raw_prob = conv[:, :, :, :, 5:]

      pred_xywh     = pred[:, :, :, :, 0:4]
      pred_conf     = pred[:, :, :, :, 4:5]

      label_xywh    = label[:, :, :, :, 0:4]  # (self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0], self.anchor_per_scale, 5 + self.num_classes)
      respond_bbox  = label[:, :, :, :, 4:5]
      label_prob    = label[:, :, :, :, 5:]

      giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)  # iou
      input_size = tf.cast(input_size, tf.float32)

      bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
      giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

      iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
      max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

      respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

      conf_focal = self.focal(respond_bbox, pred_conf)

      conf_loss = conf_focal * (
              respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
              +
              respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
      )

      prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

      giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
      conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
      prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

      return giou_loss, conf_loss, prob_loss


  def compute_loss(self, conv_sbbox, pred_sbbox, conv_mbbox, pred_mbbox, conv_lbbox, pred_lbbox, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

      with tf.name_scope('smaller_box_loss'):
          loss_sbbox = self.loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                       anchors = self.anchors[0], stride = self.strides[0])

      with tf.name_scope('medium_box_loss'):
          loss_mbbox = self.loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                       anchors = self.anchors[1], stride = self.strides[1])

      with tf.name_scope('bigger_box_loss'):
          loss_lbbox = self.loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                       anchors = self.anchors[2], stride = self.strides[2])

      with tf.name_scope('giou_loss'):
          giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

      with tf.name_scope('conf_loss'):
          conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

      with tf.name_scope('prob_loss'):
          prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

      return giou_loss, conf_loss, prob_loss


  # override
  def _build_train(self):
    with tf.name_scope('define_input'):
        self.trainable = tf.placeholder(dtype=tf.bool, name='training')
        # self.input_data = tf.placeholder(shape=(1, 416, 416, 3),dtype=tf.float32, name='input_data')
        self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
        self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
        self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
        self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
        self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
        self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
        self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')

    print("-" * 80)
    print("Build train graph")

    self.if_enas = tf.placeholder(dtype=np.int32, shape=(), name="if_enas")
    self.A = tf.Variable(1, dtype=tf.int32, trainable=False, name="A")
    self.conv_lbbox, self.conv_mbbox, self.conv_sbbox, self.flops = tf.cond(tf.equal(self.if_enas, self.A), lambda: self._model(self.input_data, self.trainable), lambda: self._model_train(self.input_data, self.trainable))
    with tf.variable_scope('pred_sbbox'):
        self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])
    with tf.variable_scope('pred_mbbox'):
        self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])
    with tf.variable_scope('pred_lbbox'):
        self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    with tf.name_scope("define_loss"):
        self.giou_loss, self.conf_loss, self.prob_loss = self.compute_loss(
            self.conv_sbbox, self.pred_sbbox,
            self.conv_mbbox, self.pred_mbbox,
            self.conv_lbbox, self.pred_lbbox,
            self.label_sbbox, self.label_mbbox, self.label_lbbox,
            self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
        self.loss = self.giou_loss + self.conf_loss + self.prob_loss

    with tf.name_scope('learn_rate'):
        self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
        warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                   dtype=tf.float64, name='warmup_steps')  # warm up steps
        train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                  dtype=tf.float64, name='train_steps')  # total steps
        self.learn_rate = tf.cond(
            pred=self.global_step < warmup_steps,
            true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
            # false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
            #                  (1 + tf.cos(
            #                      (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            false_fn=lambda: tf.cast(self.learn_rate_init, tf.float64))
        global_step_update = tf.assign_add(self.global_step, 1.0)

    with tf.name_scope("define_weight_decay"):
        print(tf.trainable_variables())
        moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

    with tf.name_scope("define_first_stage_train"):  # firstly update yolo layer
        self.first_stage_trainable_var_list = []
        for var in tf.trainable_variables():
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            if var_name_mess[1] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                self.first_stage_trainable_var_list.append(var)

        first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                  var_list=self.first_stage_trainable_var_list)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                with tf.control_dependencies([moving_ave]):
                    self.train_op_with_frozen_variables = tf.no_op()

    with tf.name_scope("train"):  # secondly update all weights
        trainable_var_list = tf.trainable_variables()
        self.optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                  var_list=trainable_var_list)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([self.optimizer, global_step_update]):
                with tf.control_dependencies([moving_ave]):
                    self.train_op_with_all_variables = tf.no_op()  # the three steps above must be performed sequentially

    # tf_variables = [var
    #     for var in tf.all_variables() if var.name.startswith(self.name)]
    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    print(tf_variables)
    self.num_vars = utils.count_model_params(tf_variables)
    print("Model has {} params".format(self.num_vars))


  def connect_controller(self, controller_model):
    if self.fixed_arc is None:
      self.sample_arc = tf.placeholder(dtype=np.float32, shape=(self.num_layers), name="sample_arc")
    else:
      fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
      self.sample_arc = fixed_arc

    self._build_train()








