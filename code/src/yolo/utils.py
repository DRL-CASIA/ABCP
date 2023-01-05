#! /usr/bin/env python
# coding=utf-8


import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from config import cfg


def count_model_params(tf_variables):
    """
    Args:
      tf_variables: list of all model variables
    """

    num_vars = 0
    for var in tf_variables:
        num_vars += np.prod([dim.value for dim in var.get_shape()])
    return num_vars


def get_train_ops(
        loss,
        tf_variables,
        train_step,
        clip_mode=None,
        grad_bound=None,
        l2_reg=1e-4,
        lr_warmup_val=None,
        lr_warmup_steps=100,
        lr_init=0.1,
        lr_dec_start=0,
        lr_dec_every=10000,
        lr_dec_rate=0.1,
        lr_dec_min=None,
        lr_cosine=False,
        lr_max=None,
        lr_min=None,
        lr_T_0=None,
        lr_T_mul=None,
        num_train_batches=None,
        optim_algo=None,
        sync_replicas=False,
        num_aggregate=None,
        num_replicas=None,
        get_grad_norms=False,
        epoch_lrdec=False,
        epoch_lrdec_every=None,
        lr_values=None,
        moving_average=None,
        if_child=True):
    """
    Args:
      clip_mode: "global", "norm", or None.
      moving_average: store the moving average of parameters
    """

    if l2_reg > 0:
        l2_losses = []
        for var in tf_variables:
            l2_losses.append(tf.reduce_sum(var ** 2))
        l2_loss = tf.add_n(l2_losses)
        loss += l2_reg * l2_loss

    grads = tf.gradients(loss,
                         tf_variables)
    grad_norm = tf.global_norm(
        grads)

    grad_norms = {}
    for v, g in zip(tf_variables, grads):
        if v is None or g is None:
            continue
        if isinstance(g, tf.IndexedSlices):
            grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g.values ** 2))
        else:
            grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g ** 2))

    if clip_mode is not None:
        assert grad_bound is not None, "Need grad_bound to clip gradients."
        if clip_mode == "global":
            grads, _ = tf.clip_by_global_norm(grads, grad_bound)
        elif clip_mode == "norm":
            clipped = []
            for g in grads:
                if isinstance(g, tf.IndexedSlices):
                    c_g = tf.clip_by_norm(g.values, grad_bound)
                    c_g = tf.IndexedSlices(g.indices, c_g)
                else:
                    c_g = tf.clip_by_norm(g, grad_bound)
                clipped.append(g)
            grads = clipped
        else:
            raise NotImplementedError("Unknown clip_mode {}".format(clip_mode))

    if lr_cosine:
        assert lr_max is not None, "Need lr_max to use lr_cosine"
        assert lr_min is not None, "Need lr_min to use lr_cosine"
        assert lr_T_0 is not None, "Need lr_T_0 to use lr_cosine"
        assert lr_T_mul is not None, "Need lr_T_mul to use lr_cosine"
        assert num_train_batches is not None, ("Need num_train_batches to use"
                                               " lr_cosine")

        curr_epoch = train_step // num_train_batches  # curr_epoch = 0, 1, 2, 3, 4……

        last_reset = tf.Variable(0, dtype=tf.int32, trainable=False,  # last_reset = 0
                                 name="last_reset")
        T_i = tf.Variable(lr_T_0, dtype=tf.int32, trainable=False, name="T_i")  # Ti = 10, 20, 30, 40……
        T_curr = curr_epoch - last_reset  # T_curr = 0, 1, 2, 3, 4……

        def _update():
            update_last_reset = tf.assign(last_reset, curr_epoch,
                                          use_locking=True)
            update_T_i = tf.assign(T_i, T_i * lr_T_mul, use_locking=True)  # T_i = 20
            with tf.control_dependencies([update_last_reset, update_T_i]):
                rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926  # T_curr = 10; last_reset = 20
                lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
            return lr

        def _no_update():
            rate = tf.to_float(T_curr) / tf.to_float(
                T_i) * 3.1415926  # T_curr = 1,2,3,4……,11,12,13,14……； T_i = 10,20,30……
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
            return lr

        learning_rate = tf.cond(
            tf.greater_equal(T_curr, T_i), _update,
            _no_update)

    elif epoch_lrdec:
        curr_step = train_step % num_train_batches
        learning_rate = tf.train.piecewise_constant(curr_step, epoch_lrdec_every, lr_values)

    else:
        learning_rate = tf.train.exponential_decay(
            lr_init, tf.maximum(train_step - lr_dec_start, 0), lr_dec_every,
            lr_dec_rate, staircase=True)
        if lr_dec_min is not None:
            learning_rate = tf.maximum(learning_rate, lr_dec_min)

    if lr_warmup_val is not None:
        learning_rate = tf.cond(tf.less(train_step, lr_warmup_steps),
                                lambda: lr_warmup_val, lambda: learning_rate)

    # if get_grad_norms:
    #   g_1, g_2 = 0.0001, 0.0001
    #   for v, g in zip(tf_variables, grads):
    #     if g is not None:
    #       if isinstance(g, tf.IndexedSlices):
    #         g_n = tf.reduce_sum(g.values ** 2)
    #       else:
    #         g_n = tf.reduce_sum(g ** 2)
    #       if "enas_cell" in v.name:
    #         print("g_1: {}".format(v.name))
    #         g_1 += g_n
    #       else:
    #         print("g_2: {}".format(v.name))
    #         g_2 += g_n
    #   learning_rate = tf.Print(learning_rate, [g_1, g_2, tf.sqrt(g_1 / g_2)],
    #                            message="g_1, g_2, g_1/g_2: ", summarize=5)

    if optim_algo == "momentum":
        opt = tf.train.MomentumOptimizer(
            learning_rate, 0.9, use_locking=True, use_nesterov=True)
    elif optim_algo == "sgd":
        opt = tf.train.GradientDescentOptimizer(learning_rate, use_locking=True)
    elif optim_algo == "adam":
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.0, epsilon=1e-3,
                                     use_locking=True)
    else:
        raise ValueError("Unknown optim_algo {}".format(optim_algo))

    # if sync_replicas:
    #     assert num_aggregate is not None, "Need num_aggregate to sync."
    #     assert num_replicas is not None, "Need num_replicas to sync."
    #
    #     opt = tf.train.SyncReplicasOptimizer(
    #         opt,
    #         replicas_to_aggregate=num_aggregate,
    #         total_num_replicas=num_replicas,
    #         use_locking=True)

    if moving_average is not None:
        opt = tf.contrib.opt.MovingAverageOptimizer(
            opt, average_decay=moving_average)

    train_op = opt.apply_gradients(

        zip(grads, tf_variables), global_step=train_step)

    if if_child:
        batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(train_op, batchnorm_updates_op)
    # train_op = opt.minimize(loss, global_step=train_step, var_list=tf_variables)  # This method simply combines calls `compute_gradients()` and`apply_gradients()`
    if get_grad_norms:
        return train_op, learning_rate, grad_norm, opt, grad_norms
    else:
        return train_op, learning_rate, grad_norm, opt


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preporcess(image, target_size, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)


    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(float(iw)/float(w), float(ih)/float(h))

    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh), scale, scale)

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image



def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious



def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def remain_ratio(prune_id):
    prune_ratios = {}
    prune_ratios[tf.equal(prune_id, 0)] = lambda: tf.Variable([1.0], trainable=False)
    prune_ratios[tf.equal(prune_id, 1)] = lambda: tf.Variable([0.8], trainable=False)
    prune_ratios[tf.equal(prune_id, 2)] = lambda: tf.Variable([0.6], trainable=False)
    prune_ratios[tf.equal(prune_id, 3)] = lambda: tf.Variable([0.4], trainable=False)
    prune_ratios[tf.equal(prune_id, 4)] = lambda: tf.Variable([0.2], trainable=False)
    prune_ratios[tf.equal(prune_id, 5)] = lambda: tf.Variable([0.0], trainable=False)
    prune_ratio = tf.case(prune_ratios, default=lambda: tf.Variable([1.0], trainable=False), exclusive=True)
    return prune_ratio[0]

def get_C(x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    return x.get_shape()[3].value


def get_HW(x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    return x.get_shape()[2].value 




