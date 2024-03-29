#! /usr/bin/env python
# coding=utf-8


import os
import cv2
import random
import numpy as np
import tensorflow as tf
import utils as utils
from config import cfg



class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG  # True

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE  # [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
        self.strides = np.array(cfg.YOLO.STRIDES)  # 8 16 32  input image size//output image size(3 yolo layers)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE  # 3 anchors per scale
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0


    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations  # all annotations  xxyy

    def __iter__(self):  # create iter object
        return self

    def next(self):  # for python3, name iterator as '__next__'

        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))

            # yolo layer truth
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            # labels: grid, best_anchor, xywh, conf, class id
            # bboexs: xywh
            num = 0
            if self.batch_count < self.num_batchs:  # the batches that have processed
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:  # the index in all annotations
                        index -= self.num_samples
                    annotation = self.annotations[index]  # an image
                    image, bboxes = self.parse_annotation(annotation)  # extract images and bboxes(xxyy), reshape and pad them
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                    # yolo layer truth
                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    # boxes truth
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    # aug
    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):

        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = np.array(cv2.imread(image_path))
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
        # map: transfer each items in the latter to the type of the former
        # list: list(iterator), map(int, box.split(',') is a iterator
        # building a 2D array which shape is [-1, 5] to store the coordinates and class ids
        '''
        line = '0.jpg 263,211,324,339,8 165,264,253,372,8 241,194,295,299,8'
        line = line.split()
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])
        ->[[263 211 324 339   8]
           [165 264 253 372   8]
           [241 194 295 299   8]]
        '''
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        # reshaping and padding images and bboxes
        updated_bb = []
        for bb in bboxes:
            x1, y1, x2, y2, cls_label = bb
            
            if x2 <= x1 or y2 <= y1:
                # dont use such boxes as this may cause nan loss.
                continue

            x1 = int(np.clip(x1, 0, image.shape[1]))
            y1 = int(np.clip(y1, 0, image.shape[0]))
            x2 = int(np.clip(x2, 0, image.shape[1]))
            y2 = int(np.clip(y2, 0, image.shape[0]))
            # clipping coordinates between 0 to image dimensions as negative values 
            # or values greater than image dimensions may cause nan loss.
            updated_bb.append([x1, y1, x2, y2, cls_label])

        return image, np.array(updated_bb)


    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / (union_area + 1e-6)
        # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss


    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0  # one hot label
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)  # Equivalent probability distribution

            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # transfer xxyy to xywh
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            # np.newaxis： add an axis (1, ..)/ (.., 1)
            iou = []
            exist_positive = False
            for i in range(3):  # screen ground truth for each anchor of each grid where bboxes located in according to iou
                # change w and h in bbox_xywh_scaled as those of anchors
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # scaled x and y
                anchors_xywh[:, 2:4] = self.anchors[i]  # w and h

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)  # choose anchors according to iou  # 3 ious every bbox
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):  # if any True in iou_mask
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)  # anchors in every grid
                    xind = np.clip(xind, 0, self.train_output_sizes[i] - 1)     
                    yind = np.clip(yind, 0, self.train_output_sizes[i] - 1)     
                    # This will mitigate errors generated when the location computed by this is more the grid cell location. 
                    # e.g. For 52x52 grid cells possible values of xind and yind are in range [0-51] including both. 
                    # But sometimes the coomputation makes it 52 and then it will try to find that location in label array 
                    # which is not present and throws error during training.
                    # boundary problem

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh  # no scaled
                    label[i][yind, xind, iou_mask, 4:5] = 1.0  # confidence of ground truth
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)  # bbox_count[i]: the total number of boxes match with current anchor
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh  # rebuild bboxes_xywh: for every anchor, all the matched and unsacled bboxes
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:  # no ground truth whose iou > 0.3 : only choose the anchor whose iou is best
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)  # the best iou
                best_detect = int(best_anchor_ind / self.anchor_per_scale)  # because the total times of calculating iou is number of bbox * anchor_per_scale
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                xind = np.clip(xind, 0, self.train_output_sizes[i] - 1)     
                yind = np.clip(yind, 0, self.train_output_sizes[i] - 1)     
                # This will mitigate errors generated when the location computed by this is more the grid cell location. 
                # e.g. For 52x52 grid cells possible values of xind and yind are in range [0-51] including both. 
                # But sometimes the coomputation makes it 52 and then it will try to find that location in label array 
                # which is not present and throws error during training.

                label[best_detect][yind, xind, best_anchor, :] = 0  # ???
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes  # matched and unscaled labels and bboxes
        # labels: grid, best_anchor, xywh, conf, class id
        # bboexs: xywh

    def __len__(self):
        return self.num_batchs




