#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : create_dataset.py
#   Author      : ronen halevy 
#   Created date:  10/19/22
#   Description :
#
# ================================================================
import glob
from pathlib import Path

# from core.load_tfrecords import parse_tfrecords
# from core.create_dataset_from_files import create_dataset_from_files
# from core.load_tfrecords import parse_tfrecords

from utils.tf_general import segments2boxes
import os
from PIL import ExifTags, Image, ImageOps
import contextlib
import numpy as np
import tensorflow as tf

import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm
from utils.general import LOGGER, colorstr

#
# from load_train_data import LoadSegmentationData


# IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break



class CreateDataset:
    def __init__(self, imgsz):
        self.mosaic_border = [-imgsz // 2, -imgsz // 2]
        self.imgsz = imgsz

    # @tf.function
    def scatter_img_to_mosaic(self, dst_img, src_img, dst_xy):
        """
        :param dst_img: 2w*2h*3ch 4mosaic dst img
        :type dst_img:
        :param src_img:
        :type src_img:
        :param dst_xy:
        :type dst_xy:
        :return: 
        :rtype: 
        """
        y_range = tf.range(dst_xy[2], dst_xy[3])[..., None]
        y_ind = tf.tile(y_range, tf.constant([1, dst_xy[1] - dst_xy[0]]))
        x_range = tf.range(dst_xy[0], dst_xy[1])[None]
        x_ind = tf.tile(x_range, tf.constant([dst_xy[3] - dst_xy[2], 1]))
        indices = tf.squeeze(tf.concat([y_ind[..., None], x_ind[..., None]], axis=-1))
        dst = tf.tensor_scatter_nd_update(
            dst_img, indices, src_img
        )
        return dst

    # @tf.function
    def clip_boxes(self, boxes, shape):
        # Clip boxes (xyxy) to image shape (height, width)
        tf.clip_by_value( boxes[:, 0], 0, boxes.shape[1])
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

    def xyxy2xywhn(self, x, w=640, h=640, clip=False, eps=0.0):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
        if clip:
            self.clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip

        xc = ((x[..., 0:1] + x[..., 2:3]) / 2) / w  # x center
        yc = ((x[..., 1:2] + x[..., 3:4]) / 2) / h  # y center
        w = (x[..., 2:3] - x[..., 0:1]) / w  # width
        h= (x[..., 3:4] - x[..., 1:2]) / h  # height
        y = tf.concat(
            [xc, yc, w, h], axis=-1, name='concat'
        )
        return y

    def xyn2xy(self, x, w=640, h=640, padw=0, padh=0):
        # Convert normalized segments into pixel segments, shape (n,2)
        # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        # x=x.to_tensor()
        xcoord = w * x[:, 0:1] + padw  # top left x
        ycoord = h * x[:, 1:2] + padh  # top left y
        y = tf.concat(
            [xcoord, ycoord], axis=-1, name='stack'
        )
        # y= tf.RaggedTensor.from_tensor(y)
        return y

    def xywhn2xyxy(self, x, w=640, h=640, padw=0, padh=0):
        """
         transform scale and align bboxes: xywh to xyxy, scaled to image size, shift by padw,padh to location in mosaic
        :param x: xywh normalized bboxes
        :type x: float array, shape: [nboxes,4]
        :param w: dest image width
        :type w: int
        :param h: dest image height
        :type h: int
        :param padw: shift of src image left end from mosaic left end
        :type padw: float ]
        :param padh: shift of src image upper end from mosaic upper end
        :type padh: float
        :return: scaled bboxes in xyxy coords, aligned to shifts in mosaic
        :rtype: float array, shape: [nboxes, 4]
        """
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        xmin = w * (x[..., 0:1] - x[..., 2:3] / 2) + padw  # top left x
        ymin = h * (x[..., 1:2] - x[..., 3:] / 2) + padh  # top left y
        xmax = w * (x[..., 0:1] + x[..., 2:3] / 2) + padw  # bottom right x
        ymax = h * (x[..., 1:2] + x[..., 3:] / 2) + padh  # bottom right y
        y = tf.concat(
            [xmin, ymin, xmax, ymax], axis=-1, name='concat'
        )
        return y

    def decode_resize(self, filename, size):
        img_st = tf.io.read_file(filename)
        img_dec = tf.image.decode_image(img_st, channels=3, expand_animations=False)
        img11 = tf.cast(img_dec, tf.float32)
        img11 = tf.image.resize(img11 / 255, size)
        return img11

        # return img4, y_labels, filename, img.shape, y_masks

    # @tf.function
    def decode_and_resize_image(self, filenames, size, y_labels, y_segments):
        labels4, segments4 = [], []
        segments4 = None
        # randomly select mosaic center:
        yc, xc = (int(random.uniform(-x, 2 * self.imgsz + x)) for x in self.mosaic_border)  # mosaic center x, y

        img4 = tf.fill(
            (self.imgsz * 2, self.imgsz * 2, 3), 114/255
        ) # gray background

        w, h = size
        # arrange mosaic 4:
        for idx in range(4):
            if idx == 0:  # top left mosaic dest zone,  bottom-right aligned src image fraction:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc # xmin, ymin, xmax, ymax
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h # xmin, ymin, xmax, ymax: src image fraction
            elif idx == 1:  # top right mosaic dest zone, bottom-left aligned src image fraction:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h  # src image fraction
            elif idx == 2:  # bottom left mosaic dest zone, top-right aligned src image fraction:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(w * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h) # src image fraction: aligned right-up
            elif idx == 3:  # bottom right mosaic dest zone, top-left aligned src image fraction:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w * 2), min(w * 2, yc + h) #
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h) # src image fraction
            img = self.decode_resize(filenames[idx], size)

            img4 = self.scatter_img_to_mosaic(dst_img=img4, src_img=img[y1b:y2b, x1b:x2b], dst_xy=(x1a, x2a,y1a, y2a))
            padw = x1a - x1b # shift of src scattered image from mosaic left end. Used for bbox and segment alignment.
            padh = y1a - y1b # shift of src scattered image from mosaic top end. Used for bbox and segment alignment.
            y_l = y_labels[idx]
            xyxy = self.xywhn2xyxy(y_l[:, 1:], w, h, padw, padh)  # transform scale and align bboxes
            xywh= self.xyxy2xywhn(xyxy,  w=640, h=640, clip=False, eps=0.0)

            y_l = tf.concat([y_l[:, 0:1], xywh/2], axis=-1) # concat [cls,xywh] shape:[nt, 5]. div by 2 from 2w x 2h
            labels4.append(y_l)

            ys = y_segments[idx]
            segments = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=ys,
                                     fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
                                                                             ragged_rank=1));

            if segments4 is None:
                segments4 = segments
            else:
                segments4 = tf.concat([segments4, segments], axis=0)

        labels4 = tf.concat(labels4, axis=0) # concat 4 labels of 4 mosaic images
        segments4/=2. # rescale from mosaic expanded  2w x 2h to wxh
        img4 = tf.image.resize(img4, size) # rescale from 2w x 2h
        return img4, labels4, filenames, img4.shape, segments4


    def __call__(self, image_files, labels, segments):
        y_segments = tf.ragged.constant(list(segments))
        y_labels = tf.ragged.constant(list(labels))
        x_train = tf.convert_to_tensor(image_files)

        ds = tf.data.Dataset.from_tensor_slices((x_train, y_labels, y_segments))

        # debug loop:
        for x, lables, segments in ds:
            self.decode_and_resize_image(x, [self.imgsz, self.imgsz], lables, segments)
        dataset = ds.map(
            lambda x, lables, segments: self.decode_and_resize_image(x, [self.imgsz, self.imgsz], lables, segments))

        return dataset
