



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
import cv2
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm
# from utils.general import LOGGER, colorstr

# from tensorflow.python.ops.numpy_ops import np_config
#
# np_config.enable_numpy_behavior()
from utils.tf_general import segment2box, xyxy2xywhn, segments2boxes_exclude_outbound_points
from utils.tf_augmentations import box_candidates

from utils.segment.tf_augmentations import Augmentation


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


# class LoadTrainData:
class LoadImagesAndLabelsAndMasks:
    def __init__(self, path, imgsz, mosaic,augment, degrees, translate, scale, shear, perspective):
        self.im_files = self._make_file(path, IMG_FORMATS)

        self.label_files = self._img2label_paths(self.im_files)  # labels
        # image_files, lables, segments = zip(*[self._create_entries(idx, self.im_file, self.label_file) for idx, (self.im_file, self.label_file) in enumerate(zip(self.im_files, self.label_files))])
        self.image_files = []
        labels = []
        segments = []
        for idx, (self.im_file, self.label_file) in enumerate(zip(self.im_files, self.label_files)):
            image_file, label, segment = self._create_entry(idx, self.im_file, self.label_file)
            self.image_files.append(image_file)
            labels.append(label)
            segments.append(segment)
        # return image_files, labels, segments
        self.indices =  range(len(self.image_files))
        self.mosaic=mosaic


        self.y_segments = tf.ragged.constant(list(segments)) # [nimg, nsegments, npoints, 2] ,nsegments, npoints vary # TODO update for mosaic
        self.y_labels = tf.ragged.constant(list(labels))  #  [nimg, nlabels, 5], nlabels vary

        #######
        self.mosaic_border = [-imgsz[0] // 2,
                              -imgsz[1] // 2]  # mosaic center placed randimly at [-border, 2 * imgsz + border]
        self.imgsz = imgsz
        self.augment, self.degrees, self.translate, self.scale, self.shear, self.perspective=augment, degrees, translate, scale, shear, perspective
        # self.augment = augment
        # self.degrees = degrees
        # self.translate = translate
        # # affaine params:
        # self.scale = scale
        # self.shear = shear
        # self.perspective = perspective
        # # # augmentation params:
        # # self.hsv_h=hgain
        # # self.hsv_s=sgain
        # # self.hsv_v=vgain
        # # self.flipud = flipud
        # # self.fliplr = fliplr
        # self.augmentation = Augmentation(hgain, sgain, vgain, flipud, fliplr)

    def exif_size(self, img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        with contextlib.suppress(Exception):
            rotation = dict(img._getexif().items())[orientation]
            if rotation in [6, 8]:  # rotation 270 or 90
                s = (s[1], s[0])
        return s

    def _create_entry(self, idx, im_file, lb_file, prefix=''):
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

    def _img2label_paths(self, img_paths):
        # Define label paths as a function of image paths
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    def _make_file(self, path, ext_list):
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

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights


        masks = []
        if self.mosaic:
            # Load mosaic
            # img, labels, segments =
            img4, labels4, segments4  =self.load_mosaic(index)
            return img4, labels4, segments4

    def iter(self):
        for i in self.indices :
            yield self[i]

    def decode_resize(self, filename, size):
        img_st = tf.io.read_file(filename)
        img_dec = tf.image.decode_image(img_st, channels=3, expand_animations=False)
        img11 = tf.cast(img_dec, tf.float32)
        img11 = tf.image.resize(img11 / 255, size)
        return img11

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
        x_ind = tf.tile(x_range, [dst_xy[3] - dst_xy[2], 1])
        indices = tf.squeeze(tf.concat([y_ind[..., None], x_ind[..., None]], axis=-1))
        dst = tf.tensor_scatter_nd_update(
            dst_img, indices, src_img
        )
        return dst
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
        xmin = tf.math.multiply(tf.cast(w, tf.float32), (x[..., 1:2] - x[..., 3:4] / 2)) + tf.cast(padw,
                                                                                                   tf.float32)  # top left x
        ymin = tf.math.multiply(tf.cast(h, tf.float32), (x[..., 2:3] - x[..., 4:5] / 2)) + tf.cast(padh,
                                                                                                   tf.float32)  # top left y
        xmax = tf.math.multiply(tf.cast(w, tf.float32), (x[..., 1:2] + x[..., 3:4] / 2)) + tf.cast(padw,
                                                                                                   tf.float32)  # bottom right x
        ymax = tf.math.multiply(tf.cast(h, tf.float32), (x[..., 2:3] + x[..., 4:5] / 2)) + tf.cast(padh,
                                                                                                   tf.float32)  # bottom right y
        y = tf.concat(
            [x[..., 0:1], xmin, ymin, xmax, ymax], axis=-1, name='concat'
        )
        return y
    def xyn2xy(self, x, w=640, h=640, padw=0, padh=0):
        # Convert normalized segments into pixel segments, shape (n,2)

        xcoord = tf.math.multiply(tf.cast(w, tf.float32), x[:, 0:1]) + tf.cast(padw, tf.float32)  # top left x
        ycoord = tf.math.multiply(tf.cast(h, tf.float32), x[:, 1:2]) + tf.cast(padh, tf.float32)  # top left y
        y = tf.concat(
            [xcoord, ycoord], axis=-1, name='stack'
        )
        return y

    def affaine_transform(self, img, M):
        # img = cv2.warpAffine(img.numpy(), M[:2].numpy(), dsize=(1280, 1280), borderValue=(114, 114, 114))
        img = cv2.warpAffine(np.asarray(img), M[:2].numpy(), dsize=(640, 640),
                             borderValue=(114. / 255, 114. / 255, 114. / 255))

        # img = tf.keras.preprocessing.image.apply_affine_transform(img,theta=0,tx=0,ty=0,shear=0,zx=1,zy=1,row_axis=0,col_axis=1,channel_axis=2,fill_mode='nearest',cval=0.0,order=1 )
        return img

    def resample_segments(self, x, seg_coords):
        seg_coords = seg_coords.to_tensor()

        seg_coords = tf.concat([seg_coords, seg_coords[0:1, :]], axis=0)  # last polygon's section for interpolation

        # y_ref=tf.reshape(y_ref.to_tensor(), [-1])
        segment = [tfp.math.interp_regular_1d_grid(
            x=x,
            x_ref_min=0,  # tf.constant(0.),
            x_ref_max=4,  # shape0,  # tf.constant(len(s)),
            y_ref=seg_coords[..., idx],
            axis=-1,
            fill_value='constant_extension',
            fill_value_below=None,
            fill_value_above=None,
            grid_regularizing_transform=None,
            name='interp_segment'
        ) for idx in range(2)]
        ####
        segment = tf.concat([segment], axis=0)
        segment = tf.reshape(segment, [2, -1])
        segment = tf.transpose(segment)

        segment = tf.concat([segment, tf.ones([1000, 1], dtype=tf.float32)], axis=-1)
        return segment
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
        # random.seed(0)
        tf.random.set_seed(
            0
        )  # ronen todo!!
        height = im.shape[0] + border[0] * 2  # shape(h,w,c)
        width = im.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

        # Perspective
        presp = tf.random.uniform([2], -perspective, perspective, dtype=tf.float32)
        P = tf.tensor_scatter_nd_update(tf.eye(3), [[2, 0], [2, 1]], presp)  # x perspective (about y)

        # Rotation and Scale
        a = tf.random.uniform((), -degrees, degrees, dtype=tf.float32)
        s = tf.random.uniform((), 1 - scale, 1 + scale, dtype=tf.float32)
        R = [[s * tf.math.cos(a), s * tf.math.sin(a), 0], [- s * tf.math.sin(a), s * tf.math.cos(a), 0], [0, 0, 1]]
        # Shear
        shearval = tf.math.tan(tf.random.uniform([2], -shear, shear, dtype=tf.float32) * math.pi / 180)  # x shear (deg)
        S = tf.tensor_scatter_nd_update(tf.eye(3), [[0, 1], [1, 0]], shearval)  # x perspective (about y)

        # Translation
        transn = tf.random.uniform([2], 0.5 - translate, 0.5 + translate) * [width, height]
        T = tf.tensor_scatter_nd_update(tf.eye(3), [[0, 2], [1, 2]], transn)  # x perspective (about y)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                im = tf.py_function(self.affaine_transform, [im, M], Tout=tf.float32)

        if True:  # if n:
            x = tf.cast(tf.linspace(0., 5 - 1, 1000), dtype=tf.float32)  # n interpolation points. n points array
            # before resample, segments are ragged (variable npoints per segment), accordingly tf.map_fn required:
            segments = tf.map_fn(fn=lambda segment: self.resample_segments(x, segment), elems=segments,
                                 fn_output_signature=tf.TensorSpec(shape=[1000, 3], dtype=tf.float32,
                                                                   ));
            segments = tf.matmul(segments, tf.cast(tf.transpose(M), tf.float32))  # transform
            segments = tf.gather(segments, [0, 1], axis=-1)

            bboxes = segments2boxes_exclude_outbound_points(segments)
        indices = box_candidates(box1=tf.transpose(targets.to_tensor()[..., 1:]) * s, box2=tf.transpose(bboxes),
                                 area_thr=0.01)
        bboxes = bboxes[indices]

        targets = targets.to_tensor()[indices]
        bboxes = tf.concat([targets[:, 0:1], bboxes], axis=-1)
        segments = segments[indices]
        return im, bboxes, segments

    def load_mosaic(self, index, ): # filenames, size, y_labels, y_segments):
        # labels4, segments4 = [], []
        segments4 = None
        labels4=None
        # randomly select mosaic center:

        xc = tf.random.uniform((), -self.mosaic_border[0], 2 * self.imgsz[0] + self.mosaic_border[0], dtype=tf.int32)
        yc = tf.random.uniform((), -self.mosaic_border[1], 2 * self.imgsz[1] + self.mosaic_border[1], dtype=tf.int32)
        #
        #          for x in
        #           self.mosaic_border)  # mosaic center x, y
        # tf.split(xc_yc, 2, axis=0)

        # indices =tf.random.uniform([3], 0, len(self.indices)-1, dtype=tf.int32)  # 3 additional image indices #debug:  [0] + [1,2,3] #r
        # indices = tf.squeeze(tf.concat([index[None][None], indices[None]], axis=1),axis=0)
        indices = random.choices(self.indices, k=3)  # 3 additional image indices
        indices.insert(0,index)
        # yc, xc = 496, 642  # ronen debug todo

        img4 = tf.fill(
            (self.imgsz[0] * 2, self.imgsz[1] * 2, 3), 114 / 255
        )  # gray background

        w, h = self.imgsz[0], self.imgsz[1]
        # arrange mosaic 4:
        for idx in range(4):
            index = indices[idx]
            if idx == 0:  # top left mosaic dest zone,  bottom-right aligned src image fraction:
                x1a, y1a, x2a, y2a = tf.math.maximum(xc - w, 0), tf.math.maximum(yc - h,
                                                                                 0), xc, yc  # xmin, ymin, xmax, ymax
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                        y2a - y1a), w, h  # xmin, ymin, xmax, ymax: src image fraction
            elif idx == 1:  # top right mosaic dest zone, bottom-left aligned src image fraction:
                x1a, y1a, x2a, y2a = xc, tf.math.maximum(yc - h, 0), tf.math.minimum(xc + w, w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), tf.math.minimum(w, x2a - x1a), h  # src image fraction
            elif idx == 2:  # bottom left mosaic dest zone, top-right aligned src image fraction:
                x1a, y1a, x2a, y2a = tf.math.maximum(xc - w, 0), yc, xc, tf.math.minimum(w * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, tf.math.minimum(y2a - y1a,
                                                                            h)  # src image fraction: aligned right-up
            elif idx == 3:  # bottom right mosaic dest zone, top-left aligned src image fraction:
                x1a, y1a, x2a, y2a = xc, yc, tf.math.minimum(xc + w, w * 2), tf.math.minimum(w * 2, yc + h)  #
                x1b, y1b, x2b, y2b = 0, 0, tf.math.minimum(w, x2a - x1a), tf.math.minimum(y2a - y1a,
                                                                                          h)  # src image fraction
            img = self.decode_resize(self.image_files[index], self.imgsz)

            img4 = self.scatter_img_to_mosaic(dst_img=img4, src_img=img[y1b:y2b, x1b:x2b], dst_xy=(x1a, x2a, y1a, y2a))
            padw = x1a - x1b  # shift of src scattered image from mosaic left end. Used for bbox and segment alignment.
            padh = y1a - y1b  # shift of src scattered image from mosaic top end. Used for bbox and segment alignment.

            # resize normalized and add pad values to bboxes and segments:
            y_l = self.xywhn2xyxy(self.y_labels[index], w, h, padw, padh)
            # map_fn since segments is a ragged tensor:
            y_s = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=self.y_segments[index],
                            fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
                                                                    ragged_rank=1));
            # concat 4 mosaic elements together. idx=0 is the first concat element:
            labels4=tf.cond(tf.equal(idx, 0), true_fn= lambda:  y_l, false_fn=lambda : tf.concat([labels4, y_l], axis=0))
            segments4=tf.cond( segments4==None, true_fn= lambda:  y_s, false_fn=lambda : tf.concat([segments4, y_s], axis=0))


        clipped_bboxes = tf.clip_by_value(
            labels4[:, 1:], 0, 2 * tf.cast(w, tf.float32), name='labels4'
        )
        labels4 = tf.concat([labels4[..., 0:1], clipped_bboxes], axis=-1)

        segments4 = tf.clip_by_value(
            segments4, 0, 2 * 640, name='segments4'  # todo 640
        )

        img4, labels4, segments4 = self.random_perspective(img4,
                                                           labels4,
                                                           segments4,
                                                           degrees=self.degrees,
                                                           translate=self.translate,
                                                           scale=self.scale,
                                                           shear=self.shear,
                                                           perspective=self.perspective,
                                                           border=self.mosaic_border)  # border to remove
        segments4= tf.RaggedTensor.from_tensor(segments4)
        labels4= tf.RaggedTensor.from_tensor(labels4)

        return img4, labels4, segments4


def polygons2mask(is_ragged, img_size, polygon, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons: [1, npoints, 2]
    """
    polygon = tf.cond(is_ragged, true_fn=lambda:polygon, false_fn=lambda: polygon)

    mask = np.zeros(img_size, dtype=np.uint8)
    cv2.fillPoly(mask, np.asarray(polygon), color=1)
    return mask # shape: [img_size]
class CreateData:
    def __init__(self, imgsz, augment, hgain, sgain, vgain, flipud, fliplr):
        self.imgsz = imgsz
        self.augment=augment
        self.augmentation = Augmentation(hgain, sgain, vgain, flipud, fliplr)


    def polygons2masks(self, segments, size, is_ragged):
        downsample_ratio = 4  # yolo training requires downsampled by 4 mask
        color = 1  # default value 1 is later modifed to a color per mask
        segments = tf.cast(segments, tf.int32)
        # polygons2mask done as loop for a seperatee mask per segment. Merge masks after size-sorting and coloring:
        masks = tf.map_fn(fn=lambda segment:
        tf.py_function(polygons2mask, [is_ragged, size, segment[None], color, downsample_ratio],
                       Tout=tf.float32), elems=segments,
                          fn_output_signature=tf.TensorSpec(shape=[640, 640], dtype=tf.float32, ));
        # NOTE: fillPoly firstly then resize is trying the keep the same way
        # of loss calculation when mask-ratio=1.

        nh, nw = (size[0] // downsample_ratio, size[1] // downsample_ratio)
        masks = tf.squeeze(tf.image.resize(masks[..., None], [nh, nw]), axis=3) # expand-mult-sqeuuze

        # shape: [nmasks, 160, 160]
        # masks = tf.concat([ms])
        # sort masks in arreas descending order - larger first
        areas = tf.math.reduce_sum(masks, axis=[1, 2])  # shape: [nmasks]
        index = tf.argsort(areas, axis=-1, direction='DESCENDING', stable=False, name=None)  # shape: [nmasks]
        masks = tf.gather(masks, index, axis=0)  # shape: [nmasks]
        # set value to masks pixels - increasing cpint from 1 to index, (=num of masks):
        index = tf.sort(index, axis=-1, direction='ASCENDING', name=None)
        index = tf.cast(index, tf.float32) + 1.
        masks = tf.math.multiply(masks, tf.reshape(index, [-1, 1, 1])) # mult by index to set value
        masks = tf.reduce_max(masks, axis=0) # reduce to merge and keep smallest mask pixes if overlap
        return masks

    def __call__(self, img, labels, segments):
        is_ragged=True
        masks = self.polygons2masks(segments, self.imgsz, is_ragged)
        # img = tf.cast(img, tf.uint8)



        labels = xyxy2xywhn(labels, w=640, h=640, clip=True, eps=1e-3)  # return xywh normalized
        if self.augment:
            img, labels, masks = self.augmentation(img, labels, masks)
            img = tf.cast(img, tf.float32)/255
        # labels = tf.RaggedTensor.from_tensor(labels, padding=-1)
        return (img, labels,  masks,)



    # for training/testing
if __name__ == '__main__':

    data_path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train'
    imgsz = [640, 640]
    mosaic = True
    hyp = '../data/hyps/hyp.scratch-low.yaml'
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    degrees, translate, scale, shear, perspective = hyp['degrees'],hyp['translate'],hyp['scale'],hyp['shear'],hyp['perspective']
    hgain, sgain, vgain, flipud, fliplr =hyp['hsv_h'],hyp['hsv_s'],hyp['hsv_v'],hyp['flipud'],hyp['fliplr']
    augment=True

    loader =  LoadImagesAndLabelsAndMasks(data_path, imgsz, mosaic, augment, degrees, translate, scale, shear, perspective)
    loader[0]
    dataset = tf.data.Dataset.from_generator(loader.iter,
                                             output_signature=(
                                                 tf.TensorSpec(shape=[imgsz[0], imgsz[1], 3], dtype=tf.float32, ),
                                                 tf.RaggedTensorSpec(shape=[None, 5], dtype=tf.float32,
                                                                     ragged_rank=1),
                                                 tf.RaggedTensorSpec(shape=[None, 1000, 2], dtype=tf.float32,
                                                                     ragged_rank=1)
                                             )
                                             )

    augment=True
    dp = CreateData(imgsz,augment,hgain, sgain, vgain, flipud, fliplr)
    dataset=dataset.map(lambda img, lables, segments: dp(img, lables, segments))


    for img, labels, mask in dataset:
        pass
        # print(dd)



    # dataset = tf.data.Dataset.from_generator(loader.iter,
    #                                          output_types=tf.int32,
    #                                          output_shapes=())
    # img4, labels4, segments4 = loader[0]
    # res = loader[1]
    # res = loader[2]
    # res = loader[3]

    pass
