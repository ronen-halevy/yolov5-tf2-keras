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

from utils.tf_general import segment2box, xyxy2xywhn, segments2boxes_exclude_outbound_points
import tensorflow as tf
import tensorflow_probability as tfp
import math
import random
import numpy as np
from PIL import ExifTags, Image, ImageOps



for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

import cv2

from utils.segment.polygons2masks import polygons2masks_overlap, polygon2mask
from utils.tf_augmentations import box_candidates


def preprocess(imgsz, bsegments):
    downsample_ratio = 4
    bmasks = polygons2masks_overlap(imgsz,
                                    bsegments,
                                    downsample_ratio=downsample_ratio)
    return bmasks


def parse_func(img_segments_ragged):
    downsample_ratio = 4
    imgsz = [640, 640]  # [640, 640] # todo - forced hereimg_segments_ragged
    bmask = tf.py_function(preprocess, [imgsz, img_segments_ragged], Tout=tf.float32)
    return bmask


# @tf.function


class CreateDataset:
    def __init__(self, imgsz, mosaic4, degrees, translate, scale, shear, perspective):
        self.mosaic_border = [-imgsz // 2,
                              -imgsz // 2]  # mosaic center placed randimly at [-border, 2 * imgsz + border]
        self.imgsz = imgsz
        self.mosaic4 = mosaic4

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

    def affaine_transform(self, img, M):
        # img = cv2.warpAffine(img.numpy(), M[:2].numpy(), dsize=(1280, 1280), borderValue=(114, 114, 114))
        img = cv2.warpAffine(np.asarray(img), M[:2].numpy(), dsize=(640, 640), borderValue=(114, 114, 114))

        # img = tf.keras.preprocessing.image.apply_affine_transform(img,theta=0,tx=0,ty=0,shear=0,zx=1,zy=1,row_axis=0,col_axis=1,channel_axis=2,fill_mode='nearest',cval=0.0,order=1 )
        return img

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
        x_ind = tf.tile(x_range, [dst_xy[3] - dst_xy[2], 1])
        indices = tf.squeeze(tf.concat([y_ind[..., None], x_ind[..., None]], axis=-1))
        dst = tf.tensor_scatter_nd_update(
            dst_img, indices, src_img
        )
        return dst

    def xyn2xy(self, x, w=640, h=640, padw=0, padh=0):
        # Convert normalized segments into pixel segments, shape (n,2)

        xcoord = tf.math.multiply(tf.cast(w, tf.float32), x[:, 0:1]) + tf.cast(padw, tf.float32)  # top left x
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
        random.seed(0)  # ronen todo!!
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


        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)  # x translation (pixels)
        T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                im = tf.py_function(self.affaine_transform, [im, M], Tout=tf.float32)



        if True:  # if n:
            # new = np.zeros((n, 4))
            new = []
            # segments = resample_segments(segments)  # upsample
            # # set homogenius coords before transform:
            # segments = tf.map_fn(fn=lambda segment: self.create_hcoords(segment, M,s), elems=[segments, targets],
            #                       fn_output_signature=tf.TensorSpec(shape=[1, 4], dtype=tf.float32,
            #                                                               ));
            ########################################################


            x = tf.cast(tf.linspace(0., 5 - 1, 1000), dtype=tf.float32)  # n interpolation points. n points array

            segments = tf.map_fn(fn=lambda segment: self.seg_interp(x, segment), elems=segments,
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

    def load_mosaic(self, filenames, size, y_labels, y_segments):
        labels4, segments4 = [], []
        segments4 = None
        # randomly select mosaic center:
        yc, xc = (tf.random.uniform((), -x, 2 * self.imgsz + x, dtype=tf.int32) for x in
                  self.mosaic_border)  # mosaic center x, y

        yc, xc = 496, 642  # ronen debug todo

        img4 = tf.fill(
            (self.imgsz * 2, self.imgsz * 2, 3), 114 / 255
        )  # gray background

        w, h = size
        # arrange mosaic 4:
        for idx in range(4):
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
            img = self.decode_resize(filenames[idx], size)

            img4 = self.scatter_img_to_mosaic(dst_img=img4, src_img=img[y1b:y2b, x1b:x2b], dst_xy=(x1a, x2a, y1a, y2a))
            padw = x1a - x1b  # shift of src scattered image from mosaic left end. Used for bbox and segment alignment.
            padh = y1a - y1b  # shift of src scattered image from mosaic top end. Used for bbox and segment alignment.

            # resize normalized and add pad values to bboxes and segments:
            y_l = self.xywhn2xyxy(y_labels[idx], w, h, padw, padh)
            y_s = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=y_segments[idx],
                            fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
                                                                    ragged_rank=1));

            labels4.append(y_l)
            if segments4 is None:
                segments4 = y_s
            else:
                segments4 = tf.concat([segments4, y_s], axis=0)

        labels4 = tf.concat(labels4, axis=0)  # concat 4 labels of 4 mosaic images

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
        return img4, labels4, segments4

        # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
        # x, y = segment.T  # segment xy
        # inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
        # x, y, = x[inside], y[inside]
        # return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((4))  # xyxy


    def seg_interp(self, x, seg_coords):
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
        ###

    def decode_resize(self, filename, size):
        img_st = tf.io.read_file(filename)
        img_dec = tf.image.decode_image(img_st, channels=3, expand_animations=False)
        img11 = tf.cast(img_dec, tf.float32)
        img11 = tf.image.resize(img11 / 255, size)
        return img11


    # @tf.function
    def decode_and_resize_image(self, filename, size, y_labels, y_segments):

        if self.mosaic4:
            img, labels, segments = self.load_mosaic(filename, size, y_labels, y_segments)
        else:
            img = self.decode_resize(filename, size)
            labels = y_labels
            padw, padh = 0, 0
            w, h = size
            segments = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=y_segments,
                                 fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
                                                                         ragged_rank=1));

        downsample_ratio = 4
        masks = tf.py_function(polygons2masks_overlap, [[self.imgsz, self.imgsz], segments, downsample_ratio], Tout=tf.float32)


        labels = xyxy2xywhn(labels, w=640, h=640, clip=True, eps=1e-3)  # return xywh normalized
        labels = tf.RaggedTensor.from_tensor(labels, padding=-1)

        return (img, labels, filename, masks,)


    def __call__(self, image_files, labels, segments):
        y_segments = tf.ragged.constant(list(segments))
        y_labels = tf.ragged.constant(list(labels))
        x_train = tf.convert_to_tensor(image_files)

        ds = tf.data.Dataset.from_tensor_slices((x_train, y_labels, y_segments))

        # debug loop:
        for x, lables, segments in ds:
            aa = self.decode_and_resize_image(x, [self.imgsz, self.imgsz], lables, segments)
        dataset = ds.map(
            lambda x, lables, segments: self.decode_and_resize_image(x, [self.imgsz, self.imgsz], lables, segments))


        dataset = dataset.batch(2)


        return dataset
