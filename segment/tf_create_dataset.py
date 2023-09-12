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

from utils.tf_general import segment2box, resample_segments
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


import cv2


from utils.segment.polygons2masks import polygons2masks_overlap, polygon2mask

def preprocess(bimages,bsegments):
    downsample_ratio=4
    bmasks, bsorted_idx = polygons2masks_overlap(bimages.shape[0:2],
                                                 bsegments,
                                                 downsample_ratio=downsample_ratio)

    return bmasks

def parse_func(img, img_segments_ragged):
    downsample_ratio=4
    bmask = tf.py_function(preprocess, [img,img_segments_ragged], tf.uint8)
    return bmask



class CreateDataset:
    def __init__(self, imgsz, mosaic4, degrees, translate, scale, shear, perspective):
        self.mosaic_border = [-imgsz // 2, -imgsz // 2] # mosaic center placed randimly at [-border, 2 * imgsz + border]
        self.imgsz = imgsz
        self.mosaic4=mosaic4

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective



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

    def xywhn2xyxy(self, x, w,h, padw, padh):
        xmin = x[..., 1:2] * w + padw  # top left x
        ymin = x[..., 2:3] * h + padh  # top left y
        y_w = x[:,3:4]*w
        y_h = x[:,4:5]*h
        y_l = tf.concat([x[:, 0:1], xmin / 2, ymin/2, y_w/2, y_h/2], axis=-1)  # [cls,xywh] shape:[nt, 5].div by 2: 2w x 2h
        return y_l


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

    # def xywhn2xyxy(self, x, w=640, h=640, padw=0, padh=0):
    #     """
    #      transform scale and align bboxes: xywh to xyxy, scaled to image size, shift by padw,padh to location in mosaic
    #     :param x: xywh normalized bboxes
    #     :type x: float array, shape: [nboxes,4]
    #     :param w: dest image width
    #     :type w: int
    #     :param h: dest image height
    #     :type h: int
    #     :param padw: shift of src image left end from mosaic left end
    #     :type padw: float ]
    #     :param padh: shift of src image upper end from mosaic upper end
    #     :type padh: float
    #     :return: scaled bboxes in xyxy coords, aligned to shifts in mosaic
    #     :rtype: float array, shape: [nboxes, 4]
    #     """
    #     # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #     # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    #     xmin = w * (x[..., 0:1] - x[..., 2:3] / 2) + padw  # top left x
    #     ymin = h * (x[..., 1:2] - x[..., 3:] / 2) + padh  # top left y
    #     xmax = w * (x[..., 0:1] + x[..., 2:3] / 2) + padw  # bottom right x
    #     ymax = h * (x[..., 1:2] + x[..., 3:] / 2) + padh  # bottom right y
    #     y = tf.concat(
    #         [xmin, ymin, xmax, ymax], axis=-1, name='concat'
    #     )
    #     return y

    def random_perspective(self, im,
                           targets=(),
                           segments=(),
                           degrees=10,
                           translate=.1,
                           scale=.1,
                           shear=10,
                           perspective=0.0,
                           border=(0, 0)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # targets = [cls, xyxy]

        height = im.shape[0] + border[0] * 2  # shape(h,w,c)
        width = im.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        print('angle', a)
        print('scale', s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)  # x translation (pixels)
        T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)  # y translation (pixels)
        print('translate', translate)

        print('T', T)
        print('C', C)

        # Combined rotation matrix
        M = T @ C  # order of operations (right to left) is IMPORTANT
        print('M', M)

        # M = T  @ C # order of operations (right to left) is IMPORTANT

        # M=C
        # M = np.eye(3)
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                from utils.segment.dataloaders import debugplt

                # imm = cv2.warpAffine(np.asarray(im), M[:2], dsize=(1280, 1280), borderValue=(114, 114, 114))
                # cv2.imshow('laaauuu', cv2.resize(imm, [640, 640]))
                # cv2.waitKey()
                im = cv2.warpAffine(np.asarray(im), M[:2], dsize=(width, height), borderValue=(114, 114, 114))

                # debugplt(im, segments, msg='aaauuu')
                cv2.imshow('aaauuu', im)
                cv2.waitKey()

        # Visualize
        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
        # ax[0].imshow(im[:, :, ::-1])  # base
        # ax[1].imshow(im2[:, :, ::-1])  # warped

        # Transform label coordinates
        n = len(targets)
        new_segments = []
        if n:
            new = np.zeros((n, 4))
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2])  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)
                new_segments.append(xy)

            # filter candidates
            i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01)
            targets = targets[i]
            targets[:, 1:5] = new[i]
            new_segments = np.array(new_segments)[i]

        return im, targets, new_segments

    def load_mosaic(self,  filenames, size, y_labels, y_segments):
        labels4, segments4 = [], []
        segments4 = None
        # randomly select mosaic center:
        yc, xc = (int(random.uniform(-x, 2 * self.imgsz + x)) for x in self.mosaic_border)  # mosaic center x, y
        # yc, xc = 496, 642  # ronen debug todo

        img4 = tf.fill(
            (self.imgsz * 2, self.imgsz * 2, 3), 114 / 255
        )  # gray background

        w, h = size
        # arrange mosaic 4:
        for idx in range(4):
            if idx == 0:  # top left mosaic dest zone,  bottom-right aligned src image fraction:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                            y2a - y1a), w, h  # xmin, ymin, xmax, ymax: src image fraction
            elif idx == 1:  # top right mosaic dest zone, bottom-left aligned src image fraction:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h  # src image fraction
            elif idx == 2:  # bottom left mosaic dest zone, top-right aligned src image fraction:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(w * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)  # src image fraction: aligned right-up
            elif idx == 3:  # bottom right mosaic dest zone, top-left aligned src image fraction:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w * 2), min(w * 2, yc + h)  #
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)  # src image fraction
            img = self.decode_resize(filenames[idx], size)

            img4 = self.scatter_img_to_mosaic(dst_img=img4, src_img=img[y1b:y2b, x1b:x2b], dst_xy=(x1a, x2a,y1a, y2a))
            padw = x1a - x1b # shift of src scattered image from mosaic left end. Used for bbox and segment alignment.
            padh = y1a - y1b # shift of src scattered image from mosaic top end. Used for bbox and segment alignment.

            # resize normalized and add pad values to bboxes and segments:
            y_l = self.xywhn2xyxy(y_labels[idx], w,h, padw, padh)
            y_s = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=y_segments[idx],
                                     fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
                                                                             ragged_rank=1));

            labels4.append(y_l)
            if segments4 is None:
                segments4 = y_s
            else:
                segments4 = tf.concat([segments4, y_s], axis=0)


            ########
            # y_s=y_segments[idx]
            # y_l = y_labels[idx]
            # if y_l.shape[0]:
            #     # rescale and add pad values - both bboxes and segments:
            #     y_l = self.xywhn2xyxy(y_l, w,h, padw, padh)
            #     y_s = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=y_s,
            #                          fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
            #                                                                  ragged_rank=1));
            # labels4.append(y_l)
            # if segments4 is None:
            #     segments4 = y_s
            # else:
            #     segments4 = tf.concat([segments4, y_s], axis=0)

            ########

            # y_s=y_segments[idx]
            # if y_l.shape[0]:
            #     y_l = self.xywhn2xyxy(y_labels[idx], w,h, padw, padh)
            #     y_s = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=y_s,
            #                          fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
            #                                                                  ragged_rank=1));
            # labels4.append(y_l)
            # if segments4 is None:
            #     segments4 = y_s
            # else:
            #     segments4 = tf.concat([segments4, y_s], axis=0)
            # modify rescale normalized segment coords and shift by pad values:

            # modify rescale normalized segment coords and shift by pad values:
            # segments = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=ys,
            #                      fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
            #                                                              ragged_rank=1));
            #
            # if segments4 is None:
            #     segments4 = segments
            # else:

        labels4 = tf.concat(labels4, axis=0)  # concat 4 labels of 4 mosaic images
        segments4 /= 2.  # rescale from mosaic expanded  2w x 2h to wxh
        img4 = tf.image.resize(img4, size)  # rescale from 2w x 2h

        img4, labels4, segments4 = self.random_perspective(img4,
                                                      labels4,
                                                      segments4,
                                                      degrees=self.degrees,
                                                      translate=self.translate,
                                                      scale=self.scale,
                                                      shear=self.shear,
                                                      perspective=self.perspective,
                                                      border=self.mosaic_border)  # border to remove

        return img4, labels4, segments4

    def decode_resize(self, filename, size):
        img_st = tf.io.read_file(filename)
        img_dec = tf.image.decode_image(img_st, channels=3, expand_animations=False)
        img11 = tf.cast(img_dec, tf.float32)
        img11 = tf.image.resize(img11 / 255, size)
        return img11

        # return img4, y_labels, filename, img.shape, y_masks

    # @tf.function
    def decode_and_resize_image(self, filename, size, y_labels, y_segments):

        if self.mosaic4:
            img, labels, segments=self.load_mosaic(filename, size, y_labels, y_segments)
        else:
            img = self.decode_resize(filename, size)
            labels = y_labels
            padw, padh=0,0
            w, h = size
            segments = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=y_segments,
                                 fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
                                                                         ragged_rank=1));





        bmask=parse_func(img, segments)
        return (img, labels, filename, img
                .shape,  bmask)




    def __call__(self, image_files, labels, segments):
        y_segments = tf.ragged.constant(list(segments))
        y_labels = tf.ragged.constant(list(labels))
        x_train = tf.convert_to_tensor(image_files)

        ds = tf.data.Dataset.from_tensor_slices((x_train, y_labels, y_segments))

        # debug loop:
        for x, lables, segments in ds:
            aa=self.decode_and_resize_image(x, [self.imgsz, self.imgsz], lables, segments)
        dataset = ds.map(
            lambda x, lables, segments: self.decode_and_resize_image(x, [self.imgsz, self.imgsz], lables, segments))

        for batch, (bimages,  btargets, bfilename, bshape, bmasks) in enumerate(dataset):
            pass

        # for idx, (img, img_labels_ragged, img_filenames, img_shape, img_segments_ragged) in enumerate(dataset):
        #     res=parse_func(img, img_labels_ragged, img_filenames, img_shape, img_segments_ragged)

        # dataset = dataset.map(parse_func)
        # dataset=dataset.batch(2)

        # dataset = dataset.map(lambda ds,aa,bb,cc,dd: parse_func(ds,aa,bb,cc,dd))
        dataset=dataset.batch(2)

        # dataset = ds.map(
        #     lambda img, img_labels_ragged, img_filenames, img_shape, img_segments_ragged: parse_func(img, img_labels_ragged, img_filenames, img_shape, img_segments_ragged))

        return dataset
