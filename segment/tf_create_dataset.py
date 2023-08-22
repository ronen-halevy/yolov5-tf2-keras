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


# from tensorflow.python.ops.numpy_ops import np_config
#
# np_config.enable_numpy_behavior()


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


class LoadImagesAndLabels:
    def exif_size(self, img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        with contextlib.suppress(Exception):
            rotation = dict(img._getexif().items())[orientation]
            if rotation in [6, 8]:  # rotation 270 or 90
                s = (s[1], s[0])
        return s

    def verify_image_label(self, idx, im_file, lb_file, prefix=''):
        # Verify one image-label pair
        # im_file, lb_file, prefix = args
        nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
        try:
            # verify images
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = self.exif_size(im)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
            assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
            if im.format.lower() in ('jpg', 'jpeg'):
                with open(im_file, 'rb') as f:
                    f.seek(-2, 2)
                    if f.read() != b'\xff\xd9':  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                        msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

            # verify labels
            if os.path.isfile(lb_file):
                nf = 1  # label found
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb):  # is segment
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        img_index = tf.tile(tf.expand_dims(tf.constant([idx]), axis=-1), [classes.shape[0], 1])
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                        xxx = classes.reshape(-1, 1)
                        rrr = segments2boxes(segments)
                        vvv = img_index
                        lb = np.concatenate((img_index, classes.reshape(-1, 1), segments2boxes(segments)),
                                            1)  # (cls, xywh)
                    lb = np.array(lb, dtype=np.float32)
                nl = len(lb)
                if nl:
                    assert lb.shape[1] == 6, f'labels require 6 columns, {lb.shape[1]} columns detected'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                    assert (lb[:,
                            2:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 2:][lb[:, 2:] > 1]}'
                    _, i = np.unique(lb, axis=1, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        lb = lb[i]  # remove duplicates
                        if segments:
                            segments = [segments[x] for x in i]
                        msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
                else:
                    ne = 1  # label empty
                    lb = np.zeros((0, 5), dtype=np.float32)
            else:
                nm = 1  # label missing
                lb = np.zeros((0, 5), dtype=np.float32)
            # print('ret', im_file, lb, segments)
            return im_file, lb, segments
        except Exception as e:

            nc = 1
            msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
            return [None, None, None, None, nm, nf, ne, nc, msg]

    def img2label_paths(self, img_paths):
        # Define label paths as a function of image paths
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    def load(self,
             path):
        # self.img_size = img_size
        # self.augment = augment
        # self.hyp = hyp
        # self.image_weights = image_weights
        # self.rect = False if image_weights else rect
        # self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        # self.mosaic_border = [-img_size // 2, -img_size // 2]
        # self.stride = stride
        self.path = path
        # self.albumentations = Albumentations(size=img_size) if augment else None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f'{"prefix"}No images found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}') from e

        self.label_files = self.img2label_paths(self.im_files)  # labels
        # image_files, lables, segments = zip(*[self.verify_image_label(idx, self.im_file, self.label_file) for idx, (self.im_file, self.label_file) in enumerate(zip(self.im_files, self.label_files))])
        image_files = []
        labels = []
        segments = []
        for idx, (self.im_file, self.label_file) in enumerate(zip(self.im_files, self.label_files)):
            image_file, label, segment = self.verify_image_label(idx, self.im_file, self.label_file)
            image_files.append(image_file)
            labels.append(label)
            segments.append(segment)
        mosaic=True
        if mosaic:
            image_files_mosaic = []
            labels_mosaic = []
            segments_mosaic = []
            for im_file, label, segment in zip(image_files, labels, segments):
                indices = random.choices(range(len(image_files)), k=3)  # 3 additional image indices
                files4 = [im_file]
                labels4 = [label]
                segments4 = [segment]

                for idx in indices:
                    files4.append(image_files[idx])
                    labels4.append(labels[idx])
                    segments4.append(segments[idx])
                image_files_mosaic.append(files4)
                labels_mosaic.append(labels4)
                segments_mosaic.append(segments4)
            image_files=image_files_mosaic
            labels=labels_mosaic
            segments=segments_mosaic



        # image_files: list, str, size ds_size, labels: list of arrays(nt, 6), size: ds_size segments: list of arrays(nt, 6), size: ds_size


        return image_files, labels, segments

# def collate_fn(img, label, path, shapes, masks):
#     segments = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=ys,
#                          fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
#                                                                  ragged_rank=1));
#     return tf.stack(img, 0), tf.concat(label, 0), path, shapes, batched_masks
#
# def collate_fn(img, label, path, shapes, masks):
#     # img, label, path, shapes, masks = zip(*batch)  # transposed
#
#     for i, l in enumerate(label):
#         id = tf.cast(tf.fill([l.shape[0], 1], i), tf.float32)
#         l = tf.concat([l, id], axis=-1)
#
#         # l[:, 0] = i  # add target image index for build_targets()
#     return tf.stack(img, 0), tf.concat(label, 0), path, shapes, batched_masks

class CreateDataset:
    def __init__(self, imgsz):
        self.mosaic_border = [-imgsz//2 , -imgsz//2]
        self.imgsz = imgsz

    # @tf.function
    def scatter_image_to_mosaic( self, dst_image, src_image, dst_x, dst_y):
        # tf.print("innnnnnnnnnnnnnnnnnnnn scatter_image_to_mosaic")
        y_range = tf.range(dst_y[0], dst_y[1])[...,None]
        y_ind=tf.tile(y_range,  tf.constant([1,dst_x[1]-dst_x[0]]))
        x_range = tf.range(dst_x[0], dst_x[1])[None]
        x_ind=tf.tile(x_range,  tf.constant([dst_y[1]-dst_y[0],1]))
        indices = tf.squeeze(tf.concat([y_ind[..., None], x_ind[..., None]], axis=-1))

        dst =tf.tensor_scatter_nd_update(
            dst_image, indices, src_image
        )
        # tf.print("out scatter_image_to_mosaic")

        return dst


    # @tf.function

    def testt(self, img, filename, y_lables, y_masks, rimages):
        # w,h = img.shape[1:2]
        tf.print('!!!!!!!!!!!!!!!!!!!!!!!!!filename', filename)

        w,h = img.shape[0:2]
        padw=[]
        padh=[]
        yc, xc = (int(random.uniform(-x, 2 * self.imgsz  + x)) for x in self.mosaic_border)  # mosaic center x, y

        indices = [[]]+random.choices(range(len(rimages)), k=3)  # 3 additional image indices
        random.shuffle(indices)
        # img4 = np.full((self.imgsz  * 2, self.imgsz  * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
        img4=  tf.fill(
            (self.imgsz  * 2, self.imgsz  * 2, img.shape[2]), 0.0
        )

        img5 = tf.fill(
            (self.imgsz  * 2, self.imgsz  * 2, img.shape[2]), 0.0
        )

        x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # new xmin, ymin, xmax, ymax (large image)
        x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # Delta(w), Delta(h), w, h (small image)
        padw.append(x1a - x1b)
        padh.append(y1a - y1b)

        y_range = tf.range(y1a, y2a)[...,None]
        y_ind=tf.tile(y_range,  tf.constant([1,x2a-x1a]))
        x_range = tf.range(x1a, x2a)[None]
        x_ind=tf.tile(x_range,  tf.constant([y2a-y1a,1]))
        indices = tf.squeeze(tf.concat([y_ind[...,None],x_ind[...,None]], axis=-1))
        img4= tf.tensor_scatter_nd_add(
            img4, indices, img[y1b:y2b, x1b:x2b], name=None
        )
        # #


        x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.imgsz * 2), yc
        x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        padw.append(x1a - x1b)
        padh.append(y1a - y1b)
        # img4[y1a:y2a, x1a:x2a] = rimages[0][0][y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

        indices1=self.scatter_image_to_mosaic(  [x1a, x2a], [y1a, y2a])
        img4 = tf.tensor_scatter_nd_add(
            img4, indices1, rimages[0][0][y1b:y2b, x1b:x2b], name=None
        )
        # #
        x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.imgsz * 2, yc + h)
        x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        padw.append(x1a - x1b)
        padh.append(y1a - y1b)
        # img4[y1a:y2a, x1a:x2a] = rimages[1][0][y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

        # img4=self.scatter_image_to_mosaic( img4, rimages[1][0][y1b:y2b, x1b:x2b] , [x1a, x2a], [y1a, y2a])
        indices1 = self.scatter_image_to_mosaic( [x1a, x2a], [y1a, y2a])
        img4 = tf.tensor_scatter_nd_add(
            img4, indices1, rimages[1][0][y1b:y2b, x1b:x2b], name=None
        )
        # tf.print('rimages',rimages[1][0][y1b:y2b, x1b:x2b].shape)

        x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.imgsz * 2), min(self.imgsz * 2, yc + h)
        x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        padw.append(x1a - x1b)
        padh.append(y1a - y1b)
        # img4[y1a:y2a, x1a:x2a] = rimages[2][0][y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # img4=self.scatter_image_to_mosaic( img4, rimages[2][0][y1b:y2b, x1b:x2b] , [x1a, x2a], [y1a, y2a])
        indices1 = self.scatter_image_to_mosaic( [x1a, x2a], [y1a, y2a])
        img4 = tf.tensor_scatter_nd_add(
            img4, indices1, rimages[2][0][y1b:y2b, x1b:x2b], name=None
        )
        # labels, segments = self.labels[index].copy(), self.segments[index].copy()
        # if labels.size:
        #     labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
        #     segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        # labels4.append(labels)
        # segments4.extend(segments)
        img4=tf.cast(img4, tf.float32)
        img4 = tf.image.resize(img4, [640,640])
        img5 = tf.image.resize(img5, [640,640])

    def xyn2xy(self,x, w=640, h=640, padw=0, padh=0):
        # Convert normalized segments into pixel segments, shape (n,2)
        # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        # x=x.to_tensor()
        xcoord = w * x[:, 0:1] + padw  # top left x
        ycoord = h * x[:, 1:2] + padh  # top left y
        y=tf.concat(
            [xcoord, ycoord], axis=-1, name='stack'
        )
        # y= tf.RaggedTensor.from_tensor(y)
        return y


    def xywhn2xyxy(self, x, w=640, h=640, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        xmin = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
        ymin = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
        xmax = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
        ymax = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
        y=tf.stack(
            [xmin, ymin, xmax,ymax], axis=-1, name='stack'
        )
        return y

    def dec_res(self, filename, size):
        img_st = tf.io.read_file(filename)
        img_dec = tf.image.decode_image(img_st, channels=3, expand_animations=False)
        img11 = tf.cast(img_dec, tf.float32)
        img11 = tf.image.resize(img11 / 255, size)
        return img11

        # return img4, y_lables, filename, img.shape, y_masks
    # @tf.function
    def decode_and_resize_image(self, filenames, size,  y_lables, y_segments):
        # tf.print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', rimages[0][0])
        # tf.print(y_lables)
        # y_lables=lables[indices[2]]
        # y_masks=segments[indices[2]]
        labels4, segments4 = [], []
        segments4 = None
        # padw = []
        # padh = []
        yc, xc = (int(random.uniform(-x, 2 * self.imgsz + x)) for x in self.mosaic_border)  # mosaic center x, y
        nch = 3

        img4=  tf.fill(
            (self.imgsz  * 2, self.imgsz  * 2, nch), 0.0
        )

        w,h = size
        s=w

        for idx in range(4):
            if idx == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif idx == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif idx == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif idx == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img = self.dec_res(filenames[idx], size)
            img4 = self.scatter_image_to_mosaic(img4, img[y1b:y2b, x1b:x2b], (x1a, x2a), (y1a, y2a))
            padw = x1a - x1b
            padh = y1a - y1b
            tf.print('1111111', idx, y_lables[idx].to_tensor().shape)

            if True: # y_lables[idx].to_tensor().shape[0]:
                y_l = y_lables[idx].to_tensor()
                xyxy = self.xywhn2xyxy(y_l[:, 2:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                y_l = tf.concat( [y_l[:, 0:2], xyxy], axis=-1)
                labels4.append(y_l)

                ys = y_segments[idx]#.to_tensor()
                # segments = [self.xyn2xy(x, w, h, padw, padh) for x in ys]
                segments = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=ys,
                                fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
                                                                        ragged_rank=1));


                # segments4.extend(segments)
                if segments4 is None:
                    segments4=segments
                else:
                    segments4 = tf.concat([segments4, segments], axis=0)

        #     img = self.dec_res(filenames[0], size)
        #     img4 = self.scatter_image_to_mosaic(img4, img[y1b:y2b, x1b:x2b], (x1a, x2a), (y1a, y2a))
        #     padw = x1a - x1b
        #     padh = y1a - y1b


         #1
        # x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # new xmin, ymin, xmax, ymax (large image)
        # x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # Delta(w), Delta(h), w, h (small image)
        # padw = x1a - x1b
        # padh = y1a - y1b
        # img = self.dec_res(filenames[0], size)
        # img4 = self.scatter_image_to_mosaic(img4, img[y1b:y2b, x1b:x2b], (x1a, x2a), (y1a, y2a))
        # if y_lables[0].shape[0]:
        #     y_l = y_lables[0].to_tensor()
        #     xyxy = self.xywhn2xyxy(y_l[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
        #     yl = tf.concat( [y_l[:, 0][...,None], xyxy], axis=-1)
        #     ys = y_segments[0].to_tensor()
        #     segments = [self.xyn2xy(x, w, h, padw, padh) for x in ys]
        #     labels4.append(ys)
        #     segments4.extend(segments)


        #2
        # x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.imgsz * 2), yc
        # x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        # img = self.dec_res(filenames[1], size)
        # img4 = self.scatter_image_to_mosaic(img4, img[y1b:y2b, x1b:x2b], (x1a, x2a), (y1a, y2a))
        # padw= x1a - x1b
        # padh=y1a - y1b
        # # img4[y1a:y2a, x1a:x2a] = rimages[0][0][y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        #
        # # 3
        # x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.imgsz * 2, yc + h)
        # x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        # img = self.dec_res(filenames[2], size)
        # img4 = self.scatter_image_to_mosaic(img4, img[y1b:y2b, x1b:x2b], (x1a, x2a), (y1a, y2a))
        # padw=x1a - x1b
        # padh=y1a - y1b
        # # img4[y1a:y2a, x1a:x2a] = rimages[1][0][y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        #
        # # tf.print('rimages',rimages[1][0][y1b:y2b, x1b:x2b].shape)
        # #4
        # x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.imgsz * 2), min(self.imgsz * 2, yc + h)
        # x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        # padw=x1a - x1b
        # padh=y1a - y1b
        #
        # img = self.dec_res(filenames[3], size)
        # img4 = self.scatter_image_to_mosaic(img4, img[y1b:y2b, x1b:x2b], (x1a, x2a), (y1a, y2a))
        # labels, segments = self.labels[index].copy(), self.segments[index].copy()


        # indices1 = self.scatter_image_to_mosaic(img4, [x1a, x2a], [y1a, y2a])
        # img4 = tf.tensor_scatter_nd_add(
        #     img4, indices1, rimages[0][0][y1b:y2b, x1b:x2b], name=None
        # )
        # tf.print('plabels4',labels4)
        #
        # if labels4:
        #     # tf.print('plabels4',labels4)
        #     labels4 = tf.concat(labels4, 0)
        #     # tf.print('pplabels4',labels4)
        # # if segments4:
        #     lent=[t.shape[0] for t in segments4]
        #     segments4 = tf.concat(segments4, 0)
        #     segments4=tf.RaggedTensor.from_row_lengths(segments4, row_lengths=lent)
        #
        # segments4 = tf.concat(segments4, 0)

        labels4 = tf.concat(labels4, axis=0)
        img4 = tf.image.resize(img4, [640, 640])

        return img4,labels4, filenames, img4.shape, segments4


    def decode_and_resize_image_mult(self, filename, size,  y_lables, y_masks):


        index=3

        ######################3
        labels4, segments4 = [], []
        s = size[0]
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

        # 3 additional image indices
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            fname=self.image_files[index]
            img_st = tf.io.read_file(fname)
            img_dec = tf.image.decode_image(img_st, channels=3, expand_animations=False)
            img = tf.cast(img_dec, tf.float32)
            # resize w/o keeping aspect ratio - no prob for normal sized images
            img = tf.image.resize(img / 255, size)

            h, w=size

        #     # place img in img4
        #     if i == 0:  # top left
        #         img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
        #         x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
        #         x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        #     elif i == 1:  # top right
        #         x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
        #         x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        #     elif i == 2:  # bottom left
        #         x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
        #         x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        #     elif i == 3:  # bottom right
        #         x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
        #         x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        #
        #     img4[0:2, 4:9] = img[0:2, 4:9]
        #     img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        #     padw = x1a - x1b
        #     padh = y1a - y1b
        #
        #     labels, segments = self.labels[index].copy(), self.segments[index].copy()
        #
        #     if labels.size:
        #         labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
        #         segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        #     labels4.append(labels)
        #     segments4
        # #####################
        #
        # # mosaic = self.mosaic and random.random() < hyp['mosaic']
        #
        # img_st = tf.io.read_file(aa)
        # img_dec = tf.image.decode_image(img_st, channels=3, expand_animations=False)
        # img = tf.cast(img_dec, tf.float32)
        # # resize w/o keeping aspect ratio - no prob for normal sized images
        # img = tf.image.resize(img/255, size)

        return img, y_lables[0], filename[0], img.shape, y_masks[0]

    def __call__(self, train_path):
        lial = LoadImagesAndLabels()
        image_files, lables, segments = lial.load(train_path)
        # self.image_files=image_files
        # self.lables=lables
        #
        # self.segments=segments
        y_segments = tf.ragged.constant(list(segments))
        y_lables = tf.ragged.constant(list(lables))
        x_train = tf.convert_to_tensor(image_files)
        # self.image_files=image_files
        # img_indices = tf.reshape(tf.convert_to_tensor(range(len(image_files))), (-1,1,1))
        # print('y_lables',y_lables.shape)
        # print('img_indices',img_indices.shape)

        # y_lables =tf.concat([y_lables, img_indices], axis=-1)
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_lables, y_segments))
        # indices = random.choices(range(len(image_files)), k=3)  # 3 additional image indices
        # indices = [0,0,0]
        # ds = ds.map(lambda x, lables, segments: self.decode_and_resize_image_mult(x, [self.imgsz, self.imgsz],  lables, segments,
        #  [tf.data.experimental.at(
        # ds, indices[0]),tf.data.experimental.at(ds, 1),tf.data.experimental.at(ds, 2) ]  ))

        for x,  lables, segments in ds:
            self.decode_and_resize_image(x, [self.imgsz, self.imgsz],  lables, segments)
        dataset = ds.map(lambda x, lables, segments: self.decode_and_resize_image(x, [self.imgsz, self.imgsz],  lables, segments))
        # dataset = dataset.map(lambda img,  y_lables, filename, shape, y_masks : collate_fn(img,  y_lables, filename, shape, y_masks))

        for img, y_lables, filename, shape, y_masks in dataset:
            pass

        # for idx, (x,  lables, filename, shapes, masks) in enumerate(ds):
        #     res = self.testt(x, filename, lables, masks, [tf.data.experimental.at(
        #         ds, indices[0]), tf.data.experimental.at(ds, indices[1]), tf.data.experimental.at(ds, indices[2])])
        #     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2', res)
        #     tf.keras.utils.save_img(f'testtttttt{idx}.jpg', res[0])

        # ds=ds.take(4)


        # dataset = ds.map(lambda x,  lables, filename, shapes, masks: self.testt(x, filename,  lables, masks ,[tf.data.experimental.at(
        # ds, indices[0]),tf.data.experimental.at(ds, indices[1]),tf.data.experimental.at(ds, indices[2]) ]         ))
# ) )

        return dataset
        # for img,  y_lables, filename, shape, y_masks in dataset:
        #     collate_fn(img,  y_lables, filename, img.shape, y_masks )
        dataset = dataset.map(lambda img,  y_lables, filename, shape, y_masks : collate_fn(img,  y_lables, filename, shape, y_masks))
        # for img,  y_lables, filename, img.shape, y_masks in dataset:
        #     pass
