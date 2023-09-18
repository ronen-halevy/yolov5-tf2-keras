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

from utils.tf_general import segment2box,  xyxy2xywhn
import os
from PIL import ExifTags, Image, ImageOps
import contextlib
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp

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
from utils.tf_augmentations import box_candidates



def preprocess(imgsz,bsegments):
    downsample_ratio=4
    bmasks = polygons2masks_overlap(imgsz,
                                                 bsegments,
                                                 downsample_ratio=downsample_ratio)
    return bmasks

def parse_func( img_segments_ragged):
    downsample_ratio=4
    imgsz=[640, 640]# [640, 640] # todo - forced hereimg_segments_ragged
    bmask = tf.py_function(preprocess, [imgsz,img_segments_ragged], Tout=tf.float32)
    return bmask

# @tf.function
def affaine_preprocess(img):
    # img = cv2.warpAffine(img, M[:2], dsize=(1280, 1280), borderValue=(114, 114, 114))
    img=tf.keras.preprocessing.image.apply_affine_transform(
        img,
        theta=180,
        tx=4,
        ty=4,
        shear=20,
        zx=1,
        zy=1,
        row_axis=1,
        col_axis=2,
        channel_axis=0,
        fill_mode='nearest',
        cval=0.0,
        order=1
    )
    return img
# @tf.function
def affaine_transform(img):
    img = tf.py_function(affaine_preprocess, [img], tf.float32)
    return img



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
        y_ind = tf.tile(y_range, [1, dst_xy[1] - dst_xy[0]])
        x_range = tf.range(dst_xy[0], dst_xy[1])[None]
        x_ind = tf.tile(x_range,[dst_xy[3] - dst_xy[2], 1])
        indices = tf.squeeze(tf.concat([y_ind[..., None], x_ind[..., None]], axis=-1))
        dst = tf.tensor_scatter_nd_update(
            dst_img, indices, src_img
        )
        return dst


    def xyn2xy(self, x, w=640, h=640, padw=0, padh=0):
        # Convert normalized segments into pixel segments, shape (n,2)

        xcoord =  tf.math.multiply(tf.cast(w, tf.float32),  x[:, 0:1]) + tf.cast(padw, tf.float32)  # top left x
        ycoord = tf.math.multiply(tf.cast(h, tf.float32), x[:, 1:2]) + tf.cast(padh, tf.float32)  # top left y
        y = tf.concat(
            [xcoord, ycoord], axis=-1, name='stack'
        )
        return y

    def xywhn2xyxy(self, x, w=640, h=640, padw=0, padh=0):
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
        xmin = tf.math.multiply(tf.cast(w, tf.float32), (x[..., 1:2] - x[..., 3:4] / 2)) + tf.cast(padw, tf.float32)  # top left x
        ymin = tf.math.multiply(tf.cast(h, tf.float32), (x[..., 2:3] - x[..., 4:5] / 2)) + tf.cast(padh, tf.float32) # top left y
        xmax = tf.math.multiply(tf.cast(w, tf.float32), (x[...,1:2] + x[..., 3:4] / 2)) + tf.cast(padw, tf.float32)   # bottom right x
        ymax = tf.math.multiply(tf.cast(h, tf.float32),(x[..., 2:3] + x[..., 4:5] / 2)) + tf.cast(padh, tf.float32)  # bottom right y
        y = tf.concat(
            [x[..., 0:1], xmin, ymin, xmax, ymax], axis=-1, name='concat'
        )
        return y

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
        random.seed(0) # ronen todo!!
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
        # print('angle', a)
        # print('scale', s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)  # x translation (pixels)
        T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)  # y translation (pixels)
        # print('translate', translate)

        # print('T', T)

        # print('C', C)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        # M = T  @ C # order of operations (right to left) is IMPORTANT
        # return im, targets, segments

        # M=C
        # M = np.eye(3)
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                from utils.segment.dataloaders import debugplt
                # im = tf.keras.preprocessing.image.apply_affine_transform(
                #     im,
                #     theta=tf.constant(4.),
                #     tx=10,
                #     ty=0,
                #     shear=0,
                #     zx=1,
                #     zy=1,
                #     row_axis=1,
                #     col_axis=2,
                #     channel_axis=0,
                #     fill_mode='nearest',
                #     cval=0.0,
                #     order=1
                # )
                # im = cv2.warpAffine(np.asarray(im), M[:2], dsize=(1280, 1280), borderValue=(114, 114, 114))
                # cv2.imshow('laaauuu', cv2.resize(imm, [640, 640]))
                # cv2.waitKey()
                # im = cv2.warpAffine(np.asarray(im), M[:2], dsize=(width, height), borderValue=(114, 114, 114))

                # debugplt(im, segments, msg='aaauuu')
                # cv2.imshow('aaauuu', im)
                # cv2.waitKey()

        # Visualize
        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
        # ax[0].imshow(im[:, :, ::-1])  # base
        # ax[1].imshow(im2[:, :, ::-1])  # warped

        # Transform label coordinates
        # n = len(targets)
        ## debug!!
        # return im, targets, segments
        new_segments = []

##


        ##

        if True: #if n:
            # new = np.zeros((n, 4))
            new = []
            # segments = resample_segments(segments)  # upsample
            # # set homogenius coords before transform:
            # segments = tf.map_fn(fn=lambda segment: self.create_hcoords(segment, M,s), elems=[segments, targets],
            #                       fn_output_signature=tf.TensorSpec(shape=[1, 4], dtype=tf.float32,
            #                                                               ));
            ########################################################
            segments = tf.clip_by_value(
                segments, 0, 2 * 640, name=None  # todo 640
            )

            x = tf.cast(tf.linspace(0., 5 - 1, 1000), dtype=tf.float32)  # n interpolation points. n points array

            segments = tf.map_fn(fn=lambda segment: self.seg_interp(x, segment), elems=segments,
                                  fn_output_signature=tf.TensorSpec(shape=[1000,3], dtype=tf.float32,
                                                                          ));
            segments = tf.matmul(segments, tf.cast(tf.transpose(M), tf.float32))  # transform
            segments = tf.gather(segments, [0, 1], axis=-1)
            # bboxes = segment2box(segments)  # replace with this:

            #
            bboxes = tf.map_fn(fn=lambda segment: self.create_hcoords(segment), elems=segments,
                                  fn_output_signature=tf.TensorSpec(shape=[4,], dtype=tf.float32
                                                                          ));
        indices = box_candidates(box1=tf.transpose(targets.to_tensor()[...,1:]) * s, box2=tf.transpose(bboxes), area_thr=0.01)
        bboxes=bboxes[indices]

        targets = targets.to_tensor()[indices]
        bboxes=tf.concat([targets[:,0:1], bboxes], axis=-1)
        segments=segments[indices]
        return im, bboxes, segments

    def load_mosaic(self,  filenames, size, y_labels, y_segments):
        labels4, segments4 = [], []
        segments4 = None
        # randomly select mosaic center:
        yc, xc = (tf.random.uniform((), -x, 2 * self.imgsz + x, dtype=tf.int32) for x in self.mosaic_border)  # mosaic center x, y

        # yc, xc = 496, 642  # ronen debug todo

        img4 = tf.fill(
            (self.imgsz * 2, self.imgsz * 2, 3), 114 / 255
        )  # gray background

        w, h = size
        # arrange mosaic 4:
        for idx in range(4):
            if idx == 0:  # top left mosaic dest zone,  bottom-right aligned src image fraction:
                x1a, y1a, x2a, y2a = tf.math.maximum(xc - w, 0), tf.math.maximum(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                            y2a - y1a), w, h  # xmin, ymin, xmax, ymax: src image fraction
            elif idx == 1:  # top right mosaic dest zone, bottom-left aligned src image fraction:
                x1a, y1a, x2a, y2a = xc, tf.math.maximum(yc - h, 0), tf.math.minimum(xc + w, w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), tf.math.minimum(w, x2a - x1a), h  # src image fraction
            elif idx == 2:  # bottom left mosaic dest zone, top-right aligned src image fraction:
                x1a, y1a, x2a, y2a = tf.math.maximum(xc - w, 0), yc, xc, tf.math.minimum(w * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, tf.math.minimum(y2a - y1a, h)  # src image fraction: aligned right-up
            elif idx == 3:  # bottom right mosaic dest zone, top-left aligned src image fraction:
                x1a, y1a, x2a, y2a = xc, yc, tf.math.minimum(xc + w, w * 2), tf.math.minimum(w * 2, yc + h)  #
                x1b, y1b, x2b, y2b = 0, 0, tf.math.minimum(w, x2a - x1a), tf.math.minimum(y2a - y1a, h)  # src image fraction
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




        labels4 = tf.concat(labels4, axis=0)  # concat 4 labels of 4 mosaic images

        clipped_bboxes =tf.clip_by_value(
            labels4[:,1:], 0, 2*tf.cast(w, tf.float32), name=None
        )
        labels4=tf.concat([labels4[...,0:1], clipped_bboxes],axis=-1)

        # for x in (labels4[:, 1:], *segments4):
        #     np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # temp resize::
        # segments4 /= 2.  # rescale from mosaic expanded  2w x 2h to wxh
        # img4 = tf.image.resize(img4, size)  # rescale from 2w x 2h
        #
        img4, labels4, segments4 = self.random_perspective(img4,
                                                      labels4,
                                                      segments4,
                                                      degrees=self.degrees,
                                                      translate=self.translate,
                                                      scale=self.scale,
                                                      shear=self.shear,
                                                      perspective=self.perspective,
                                                      border=self.mosaic_border)  # border to remove
        # segments4 /= 2.  # rescale from mosaic expanded  2w x 2h to wxh
        # img4 = tf.image.resize(img4, size)  # rescale from 2w x 2h

        return img4, labels4, segments4





        # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
        x, y = segment.T  # segment xy
        inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
        x, y, = x[inside], y[inside]
        return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((4))  # xyxy

    def get_shape0(self, tens):
        return tens.shape[0]

    def arrange_bbox(self, seg_coords):
        width=height=640

        bbox = segment2box(seg_coords) # replace with this:

        # x, y = tf.transpose(seg_coords)  # segment xy
        # ge = tf.math.logical_and(tf.math.greater_equal(x, 0), tf.math.greater_equal(y, 0))
        # le = tf.math.logical_and(tf.math.less_equal(x, width), tf.math.less_equal(y, height))
        # inside = tf.math.logical_and(ge, le)
        # # inside = (x >= 0) & (y >= 0) & (x <= width) & (y<= height)
        # x, y, = x[inside], y[inside]
        # # tf.print('xy,',x,y)
        # bbox = tf.stack([tf.math.reduce_min(x), tf.math.reduce_min(y), tf.math.reduce_max(x), tf.math.reduce_max(y)],
        #                 axis=0) if any(x) else tf.zeros((4))
        # any_positive = tf.math.greater(tf.reduce_max(tf.math.abs(x)), 0)
        # bbox = tf.where(any_positive,
        #                 tf.stack([tf.math.reduce_min(x), tf.math.reduce_min(y), tf.math.reduce_max(x),
        #                           tf.math.reduce_max(y)], axis=0),
        #                 tf.zeros((4)))

        return bbox



    def seg_interp(self, x, seg_coords):
        seg_coords=seg_coords.to_tensor()

        seg_coords = tf.concat([seg_coords, seg_coords[0:1,:]], axis=0) #  last polygon's section for interpolation

        # y_ref=tf.reshape(y_ref.to_tensor(), [-1])
        segment = [tfp.math.interp_regular_1d_grid(
            x=x,
            x_ref_min=0,  # tf.constant(0.),
            x_ref_max=4,  # shape0,  # tf.constant(len(s)),
            y_ref=seg_coords[...,idx],
            axis=-1,
            fill_value='constant_extension',
            fill_value_below=None,
            fill_value_above=None,
            grid_regularizing_transform=None,
            name=None
        ) for idx in range(2)]
        ####
        segment = tf.concat([segment], axis=0)
        segment = tf.reshape(segment, [2,-1])
        segment = tf.transpose(segment)

        segment=tf.concat([segment, tf.ones([1000,1], dtype=tf.float32)], axis=-1)
        return segment
        ###

    # create h coords - add ones row
    def create_hcoords(self, seg_coords):
        n=1000
        bbox = tf.py_function(self.arrange_bbox, [seg_coords],  tf.float32)
        # bbox = segment2box(seg_coords) # replace with this:

        return bbox

        # tf.print(xx.shape[1])
        # xx = tf.concat([xx[...,0:1], xx[...,1:2], tf.ones([xx.shape[0],1], dtype=tf.float32)], axis=-1)
        # xx = tf.concat([xx, tf.ones([xx.shape[0],1], dtype=tf.float32)], axis=-1)
        # xx=tf.expand_dims(xx, axis=-1)
        # pass
        # return xx
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
        img=tf.image.resize(img, size)

        # tf.print('img.shape[1], h=img.shape[0]',img.shape[1],img.shape[0])
        labels = xyxy2xywhn(labels, w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3) # return xywh normalized
        # img=self.image_affine(img) # todo this added - is the problem


        # im = cv2.warpAffine(np.asarray(im), M[:2], dsize=(1280, 1280), borderValue=(114, 114, 114))
        # hsegments = tf.map_fn(fn=lambda segment: self.create_hcoords(segment), elems=segments,
        #                 fn_output_signature=tf.RaggedTensorSpec(shape=[None, 3], dtype=tf.float32,
        #                                                         ragged_rank=1));
        # bmask=parse_func(img, segments)
        # image=affaine_transform(img)
        # img=image
        labels=tf.RaggedTensor.from_tensor(labels, padding=-1)

        return(img, labels,   filename, img.shape, segments, )
        # return (img, labels, filename, img
        #         .shape,  bmask)

    def image_affine(self, bimages):
        image=affaine_transform(bimages)
        return image
    # def segment_affine(self, segment):
    #     segment=segment_affaine_transform(segment)
    #     return segment

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

        dataset = dataset.map(
            lambda bimages,  btargets, bfilename, bshape, segments: (self.image_affine(bimages),  btargets, bfilename, bshape, segments))

        # dataset = dataset.map(
        #     lambda bimages, btargets, bfilename, bshape, segments: (
        #     bimages, btargets, bfilename, bshape, self.segment_affine(segments)))
        # debug loop:
        for batch, (bimages,  btargets, bfilename, bshape, segments) in enumerate(dataset):
            mask= parse_func(segments)

        dataset = dataset.map(
            lambda bimages,  btargets, bfilename, bshape, segments: (bimages,  btargets, parse_func(segments), bfilename, bshape))


        # dataset = dataset.map(
        #     lambda bimages,  btargets, bfilename, bshape, bmasks: self.segments_affine(bimages,  btargets, bfilename, bshape, bmasks))


        for batch, (bimages,  btargets, bmasks, bshape, bfilename) in enumerate(dataset):
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


