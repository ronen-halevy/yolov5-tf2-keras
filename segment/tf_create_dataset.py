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
                        img_index = tf.tile(tf.expand_dims(tf.constant([idx]), axis=-1), [classes.shape[0],1])
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                        xxx = classes.reshape(-1, 1)
                        rrr=segments2boxes(segments)
                        vvv = img_index
                        lb = np.concatenate( (img_index, classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
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
            print('ret', im_file, lb, segments)
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
        image_files=[]
        lables=[]
        segments=[]
        for idx, (self.im_file, self.label_file) in enumerate(zip(self.im_files, self.label_files)):
            image_file, lable, segment = self.verify_image_label(idx, self.im_file, self.label_file)
            image_files.append(image_file)
            lables.append(lable)
            segments.append(segment)


        # image_files: list, str, size ds_size, labels: list of arrays(nt, 6), size: ds_size segments: list of arrays(nt, 6), size: ds_size
        return image_files, lables, segments


def collate_fn(img, label, path, shapes, masks):
    # img, label, path, shapes, masks = zip(*batch)  # transposed
    batched_masks = tf.concat(masks, 0)
    for i, l in enumerate(label):
        id = tf.cast(tf.fill([l.shape[0],1], i), tf.float32)
        l=tf.concat([l, id], axis=-1)

        # l[:, 0] = i  # add target image index for build_targets()
    return tf.stack(img, 0), tf.concat(label, 0), path, shapes, batched_masks

def decode_and_resize_image(filename, size,  y_lables, y_masks):
    img_st = tf.io.read_file(filename)
    img_dec = tf.image.decode_image(img_st, channels=3, expand_animations=False)
    img = tf.cast(img_dec, tf.float32)
    # resize w/o keeping aspect ratio - no prob for normal sized images
    img = tf.image.resize(img/255, size)
    return img,  y_lables, filename, img.shape, y_masks

def create_dataset(train_path, imgsz):
    lial = LoadImagesAndLabels()
    image_files, lables, segments = lial.load(train_path)
    y_segments = tf.ragged.constant(list(segments))
    y_lables = tf.ragged.constant(list(lables))
    x_train = tf.convert_to_tensor(image_files)
    # img_indices = tf.reshape(tf.convert_to_tensor(range(len(image_files))), (-1,1,1))
    # print('y_lables',y_lables.shape)
    # print('img_indices',img_indices.shape)

    # y_lables =tf.concat([y_lables, img_indices], axis=-1)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_lables, y_segments))
    dataset = dataset.map(lambda x, y_lables, y_segments: decode_and_resize_image(x, imgsz,  y_lables, y_segments))
    return dataset
    # for img,  y_lables, filename, shape, y_masks in dataset:
    #     collate_fn(img,  y_lables, filename, img.shape, y_masks )
    # dataset = dataset.map(lambda img,  y_lables, filename, shape, y_masks : collate_fn(img,  y_lables, filename, shape, y_masks))
    # for img,  y_lables, filename, img.shape, y_masks in dataset:
    #     pass
