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

# from tensorflow.python.ops.numpy_ops import np_config
#
# np_config.enable_numpy_behavior()


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


class LoadTrainData:
    def exif_size(self, img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        with contextlib.suppress(Exception):
            rotation = dict(img._getexif().items())[orientation]
            if rotation in [6, 8]:  # rotation 270 or 90
                s = (s[1], s[0])
        return s

    def _create_entries(self, idx, im_file, lb_file, prefix=''):
        nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
        # try:
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
                    # img_index = tf.tile(tf.expand_dims(tf.constant([idx]), axis=-1), [classes.shape[0], 1])
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)),
                                        1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:,
                        2:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 2:][lb[:, 2:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
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
        return im_file, lb, segments
    # except Exception as e:

    # nc = 1
    # msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
    # raise Exception(e)
    # return [None, None, None, None, nm, nf, ne, nc, msg]


    def img2label_paths(self, img_paths):
        # Define label paths as a function of image paths
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    def make_file_list(self, path, ext_list):
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
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ext_list)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'{"prefix"}No images found'
            return im_files
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}') from e

    def prepare_entries(self, path):
        self.im_files = self.make_file_list(path, IMG_FORMATS)

        self.label_files = self.img2label_paths(self.im_files)  # labels
        # image_files, lables, segments = zip(*[self._create_entries(idx, self.im_file, self.label_file) for idx, (self.im_file, self.label_file) in enumerate(zip(self.im_files, self.label_files))])
        image_files = []
        labels = []
        segments = []
        for idx, (self.im_file, self.label_file) in enumerate(zip(self.im_files, self.label_files)):
            image_file, label, segment = self._create_entries(idx, self.im_file, self.label_file)
            image_files.append(image_file)
            labels.append(label)
            segments.append(segment)
        return image_files, labels, segments

    def prepare_mosaic4_entries(self, image_files, labels, segments):
        image_files_mosaic = []
        labels_mosaic = []
        segments_mosaic = []
        for entry_idx, (im_file, label, segment) in enumerate(zip(image_files, labels, segments)):
            mos_sel_indices = random.choices(range(len(image_files)), k=3)  # select 3 reandom images todo!!!
            files4 = [image_files[0]]
            labels4 = [labels[0]]
            segments4 = [segments[0]]

            for mos_ind in mos_sel_indices:
                files4.append(image_files[mos_ind])
                labels4.append(labels[mos_ind])
                segments4.append(segments[mos_ind])
            image_files_mosaic.append(files4)
            # img_index =

            labels_mosaic.append(labels4)
            segments_mosaic.append(segments4)
        return image_files_mosaic, labels_mosaic, segments_mosaic

    def load_data(self,
             path, mosaic):
        # self.img_size = img_size
        # self.augment = augment
        # self.hyp = hyp
        # self.image_weights = image_weights
        # self.rect = False if image_weights else rect
        # self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        # self.mosaic_border = [-img_size // 2, -img_size // 2]
        # self.stride = stride
        # self.path = path
        # self.albumentations = Albumentations(size=img_size) if augment else None
        image_files,labels,segments= self.prepare_entries(path)

        if mosaic:
            image_files_mosaic, labels_mosaic, segments_mosaic=self.prepare_mosaic4_entries(image_files, labels, segments)
            image_files = image_files_mosaic
            labels = labels_mosaic
            segments = segments_mosaic


        # image_files: list, str, size ds_size, labels: list of arrays(nt, 6), size: ds_size segments: list of arrays(nt, 6), size: ds_size

        return image_files, labels, segments
