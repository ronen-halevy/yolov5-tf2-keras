# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
TensorFlow, Keras and TFLite versions of YOLOv5
Authored by https://github.com/zldrobit in PR https://github.com/ultralytics/yolov5/pull/1127

Usage:
    $ python models/tf.py --weights yolov5s.pt

Export:
    $ python export.py --weights yolov5s.pt --include saved_model pb tflite tfjs
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

import numpy as np
import tensorflow as tf

from tensorflow import keras

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior() # allows running NumPy code, accelerated by TensorFlow
from keras import mixed_precision


# from models.common import (C3, SPP, SPPF, Bottleneck, BottleneckCSP, C3x, Concat, Conv, CrossConv, DWConv,
#                            DWConvTranspose2d, Focus, autopad)
# from models.experimental import  attempt_load
# from models.yolo import Detect, Segment
from utils.tf_general import LOGGER, make_divisible, print_args

from utils.tf_plots import feature_visualization

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class TFBN(keras.layers.Layer):
    # TensorFlow BatchNormalization wrapper
    def __init__(self, w=None):
        super().__init__()
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w.bias.numpy()),
            gamma_initializer=keras.initializers.Constant(w.weight.numpy()),
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.numpy()),
            moving_variance_initializer=keras.initializers.Constant(w.running_var.numpy()),
            epsilon=w.eps)

    def call(self, inputs):
        return self.bn(inputs)

    # # Solution for model saving error
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'bn': self.bn,
    #     })
    #     return config
    #

class TFPad(keras.layers.Layer):
    # Pad inputs in spatial dimensions 1 and 2
    def __init__(self, pad):
        super().__init__()
        if isinstance(pad, int):
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        else:  # tuple/list
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])

    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode='constant', constant_values=0)


class TFConv(keras.layers.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        # ch_in, ch_out, weights, kernel, stride, padding, groups
        super().__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        # TensorFlow convolution padding is inconsistent with PyTorch (e.g. k=3 s=2 'SAME' padding)
        # see https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch
        conv = keras.layers.Conv2D(
            filters=c2,
            kernel_size=k,
            strides=s,
            padding='SAME' if s == 1 else 'VALID',
            use_bias=not hasattr(w, 'bn'),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01) if w is None else  keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer='zeros' if w is None else 'zeros' if hasattr(w, 'bn') else keras.initializers.Constant(w.conv.bias.numpy()))
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        # If weights converted from pytorch, BN weights might be fused to adjacent Conv layer, excluding bn attribute
        self.bn = TFBN(w.bn ) if hasattr(w, 'bn') else keras.layers.BatchNormalization(
            momentum=0.03) #tf.identity
        self.act = keras.activations.swish if  w is None else  activations(w.act) if act else tf.identity

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs)))

    # # Solution for model saving error:
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'conv':self.conv,
    #         'bn': self.bn,
    #         'act':self.act,
    #     })
    #     return config


class TFDWConv(keras.layers.Layer):
    # Depthwise convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True, w=None):
        # ch_in, ch_out, weights, kernel, stride, padding, groups
        super().__init__()
        assert c2 % c1 == 0, f'TFDWConv() output={c2} must be a multiple of input={c1} channels'
        conv = keras.layers.DepthwiseConv2D(
            kernel_size=k,
            depth_multiplier=c2 // c1,
            strides=s,
            padding='SAME' if s == 1 else 'VALID',
            use_bias=not hasattr(w, 'bn'),
            depthwise_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer='zeros' if hasattr(w, 'bn') else keras.initializers.Constant(w.conv.bias.numpy()))
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        self.bn = TFBN(w.bn) if hasattr(w, 'bn') else tf.identity
        self.act = activations(w.act) if act else tf.identity

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs)))


class TFDWConvTranspose2d(keras.layers.Layer):
    # Depthwise ConvTranspose2d
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0, w=None):
        # ch_in, ch_out, weights, kernel, stride, padding, groups
        super().__init__()
        assert c1 == c2, f'TFDWConv() output={c2} must be equal to input={c1} channels'
        assert k == 4 and p1 == 1, 'TFDWConv() only valid for k=4 and p1=1'
        weight, bias = w.weight.permute(2, 3, 1, 0).numpy(), w.bias.numpy()
        self.c1 = c1
        self.conv = [
            keras.layers.Conv2DTranspose(filters=1,
                                         kernel_size=k,
                                         strides=s,
                                         padding='VALID',
                                         output_padding=p2,
                                         use_bias=True,
                                         kernel_initializer=keras.initializers.Constant(weight[..., i:i + 1]),
                                         bias_initializer=keras.initializers.Constant(bias[i])) for i in range(c1)]

    def call(self, inputs):
        return tf.concat([m(x) for m, x in zip(self.conv, tf.split(inputs, self.c1, 3))], 3)[:, 1:-1, 1:-1]


class TFFocus(keras.layers.Layer):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = TFConv(c1 * 4, c2, k, s, p, g, act, w.conv)

    def call(self, inputs):  # x(b,w,h,c) -> y(b,w/2,h/2,4c)
        # inputs = inputs / 255  # normalize 0-255 to 0-1
        inputs = [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :], inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]]
        return self.conv(tf.concat(inputs, 3))


class TFBottleneck(keras.layers.Layer):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1 if w is not None else None)
        self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2 if w is not None else None)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))

    # # Solution for model saving error:
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'cv1': self.cv1,
    #         'cv2': self.cv2,
    #         'add': self.add,
    #     })
    #     return config




class TFCrossConv(keras.layers.Layer):
    # Cross Convolution
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, w=None):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, (1, k), (1, s), w=w.cv1)
        self.cv2 = TFConv(c_, c2, (k, 1), (s, 1), g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFConv2d(keras.layers.Layer):
    # Substitution for PyTorch nn.Conv2D
    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None):
        super().__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        self.conv = keras.layers.Conv2D(filters=c2,
                                        kernel_size=k,
                                        strides=s,
                                        padding='VALID',
                                        use_bias=bias,
                                        kernel_initializer='zeros' if w is None else  keras.initializers.Constant(
                                            w.weight.permute(2, 3, 1, 0).numpy()),
                                        bias_initializer='zeros' if w is None else keras.initializers.Constant(w.bias.numpy()) if bias else None)

    def call(self, inputs):
        return self.conv(inputs)



# class TFBottleneckCSP(keras.layers.Layer):
#     # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
#         # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
#         self.cv2 = TFConv2d(c1, c_, 1, 1, bias=False, w=w.cv2)
#         self.cv3 = TFConv2d(c_, c_, 1, 1, bias=False, w=w.cv3)
#         self.cv4 = TFConv(2 * c_, c2, 1, 1, w=w.cv4)
#         self.bn = TFBN(w.bn)
#         self.act = lambda x: keras.activations.swish(x)
#         self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])
#
#     def call(self, inputs):
#         y1 = self.cv3(self.m(self.cv1(inputs)))
#         y2 = self.cv2(inputs)
#         return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))


class TFC3(keras.layers.Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1 if w is not None else None)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2 if w is not None else None)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3 if w is not None else None)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j] if w is not None else None) for j in range(n)])

    def call(self, inputs):
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))

    # # Solution for model saving error:
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'cv1': self.cv1,
    #         'cv2': self.cv2,
    #         'cv3': self.cv3,
    #         'm': self.m,
    #
    #     })
    #     return config


class TFC3x(keras.layers.Layer):
    # 3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([
            TFCrossConv(c_, c_, k=3, s=1, g=g, e=1.0, shortcut=shortcut, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))

    # # Solution for model saving error:
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'cv1': self.cv1,
    #         'cv2': self.cv2,
    #         'cv3': self.cv3,
    #         'm': self.m,
    #     })
    #     return config


class TFSPP(keras.layers.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), w=None):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding='SAME') for x in k]

    def call(self, inputs):
        x = self.cv1(inputs)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))

    # # Solution for model saving error:
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'cv1': self.cv1,
    #         'cv2': self.cv2,
    #         'm': self.m,
    #     })
    #     return config



class TFSPPF(keras.layers.Layer):
    # Spatial pyramid pooling-Fast layer
    def __init__(self, c1, c2, k=5, w=None):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1 if w is not None else None)
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2 if w is not None else None)
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding='SAME')

    def call(self, inputs):
        x = self.cv1(inputs)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3))

    # # Solution for model saving error:
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'cv1': self.cv1,
    #         'cv2': self.cv2,
    #         'm': self.m,
    #     })
    #     return config

class TFDetect(keras.layers.Layer):
    # TF YOLOv5 Detect layer
    # todo detail params w - weights
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), training=True, w=None):  # detection layer
        super().__init__()
        self.stride = tf.convert_to_tensor(w.stride.numpy() if w is not None else [8,16,32], dtype=tf.float32)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        # reshape anchors and normalize by stride values:
        self.anchors = tf.convert_to_tensor(w.anchors.numpy(), dtype=tf.float16) if w is not None else \
            tf.convert_to_tensor(anchors, dtype=tf.float16).reshape([self.nl, self.na, 2]) / self.stride.reshape([self.nl, 1, 1])
        # rescale anchors by stride values and reshape to
        self.anchor_grid = self.anchors.reshape([self.nl, 1, self.na, 1, 2])
        self.anchor_grid = self.anchor_grid.transpose( [0, 1, 3, 2, 4])  # shape: [nl, 1,1,na,2]

        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i] if w is not None else None) for i, x in enumerate(ch)]
        self.training = training  # set to False after building model
        self.imgsz = imgsz
        for i in range(self.nl):
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            xv, yv = tf.meshgrid(tf.range(nx, dtype=tf.float32), tf.range(ny, dtype=tf.float32)) # shapes: [ny,nx]
            # stack to grid, reshape & transpose to predictions matching shape:
            self.grid[i]= tf.stack([xv, yv], 2).reshape( [1, 1, ny, nx, 2])
            self.grid[i] = self.grid[i].transpose( [0, 2, 3, 1, 4]) # shape: [1, ny, nx, 1, 2]

    # @tf.function
    def call(self, inputs):
        """
        Model's detection layaer. exeutes conv2d operator on 3 input layers, and then if Training, reshapes result and return.
        process bbox and
        :param inputs: list[3], shapes:[[bsize,nyi,nxi,128] for i=0:2] where ny0,nx0:80,80, ny1,nx1:40,40, ny2,nx2:20,20
        :return:
        if Training:
        list[3] grid layers, with shapes: [[b,na,nyi,nxi,no], for i=0:2], no=4+1+nc+nm
        else:
        tuple(2), z: packed output for nms, shape: [b,25200,no], x: list[3] grid layers, with shape detailed above
        """
        z = []  # inference output
        x = []
        for i in range(self.nl):
            x.append(self.m[i](inputs[i])) # shape: [bs,nyi,nxi,na*no] where ny,nx=[[80,80],[40,40],[20,20]], no=117
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            x[i] = x[i].reshape( [-1,ny, nx, self.na,  self.no]) # from [bs,nyi,nxi,na*no] to [bs,nyi,nxi,na,no]
            if not self.training:  # for inference & validation - process preds according to yolo spec,ready for nms:
                y = x[i] # shape: [bs, ny,nx,na,no] where no=xywh+conf+nc+nm
                # operate yolo adaptations on box:
                xy = (tf.sigmoid(y[..., 0:2]) * 2 - 0.5 + self.grid[i]) / [nx, ny] # xy bbox formula, normalized  to 0-1
                wh = (2*tf.sigmoid(y[..., 2:4])) ** 2 * self.anchor_grid[i]/[ny,ny] # fwh bbox formula. noremalized.
                # concat modified values back together, operate yolo specified sigmoid on confs:
                y = tf.concat([xy, wh, tf.sigmoid(y[..., 4:5 + self.nc]).astype(tf.float32), y[..., 5 + self.nc:].astype(tf.float32)], -1)
                z.append(y.reshape([-1, self.na * ny * nx, self.no])) # reshape [bs,ny,nx,na,no]->[bs,nxi*nyi*na,no]
                x[i] = x[i].transpose([0, 3, 1, 2, 4])
            else: # train output a list of x[i] arrays , i=0:nl-1,  array shape:  [bs,na,ny,nx,no]
                x[i] = x[i].transpose([0,3,1,2,4]) # from shape [bs,nyi,nxi,na, no] to [bs,na,nyi,nxi,no]

        return  x if self.training else (tf.concat(z, axis=1), x) # x:[bs,nyi,nxi,na,no] for i=0:2], z: [b,25200,no]


class TFSegment(TFDetect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), imgsz=(640, 640), training=False, w=None):
        super().__init__(nc, anchors, ch, imgsz, training, w)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i] if w is not None else None) for i, x in enumerate(ch)]  # output conv
        self.proto = TFProto(ch[0], self.npr, self.nm, w=w.proto if w is not None else None)  # protos
        self.detect = TFDetect.call

    def call(self, x):
        """
        Module's segment layer: envokes proto to generate mask protos and  detection layer. Layer's output combines both
        :param x: list[3], shapes:[[bsize,nyi,nxi,128] for i=0:2] where ny0,nx0:80,80, ny1,nx1:40,40, ny2,nx2:20,20
        :return:
        Detect output, with proto addition:
        if Training:
            list[3] grid layers, with shapes: [[b,na,nyi,nxi,no], for i=0:2], no=4+1+nc+nm
            proto, shape: [b,32,160,160], tf.float32
        else: (validation & inference)
            tuple(3):
            x[0]: packed output for nms, shape: [b,25200,no],
            proto, shape: [b,32,160,160], tf.float32
            x[1]: list[3] grid layers, with shape detailed above



        x: detect output.  list[3] grid layers output. shape:[[bsize,80,80,128],[bsize,40,40,2],[bsize,20,20,128]],tf.float32
        p: mask protos proto. shape: [b,32,160,160]
        detect output x if Training and x[0],x[1] otherwise - see detect layer, and mask proto,
        shape:[b,32,160,160], tf.float32
        """
        p = self.proto(x[0])
        # p = TFUpsample(None, scale_factor=4, mode='nearest')(self.proto(x[0]))  # (optional) full-size protos
        p = p.transpose( [0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160)
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p, x[1])

        # return [x[0], x[1], x[2], p] if self.training else (x[0], p, x[1])


class TFProto(keras.layers.Layer):

    def __init__(self, c1, c_=256, c2=32, w=None):
        super().__init__()
        self.cv1 = TFConv(c1, c_, k=3, w=w.cv1 if w is not None else None)
        self.upsample = TFUpsample(None, scale_factor=2, mode='nearest')
        self.cv2 = TFConv(c_, c_, k=3, w=w.cv2 if w is not None else None)
        self.cv3 = TFConv(c_, c2, w=w.cv3 if w is not None else None)

    def call(self, inputs):
        """
        Produce mask prod tensor: Apply to conv stages on upsampled input, to produce prod with shape: [b,160,160,32]
        :param inputs: module pred[0], i.e smallest stride (stride=8) layer. shape:[b,h/8,w/8,128], h=w=640, tf.float32
        :return: prod shape[b160,160,32], tf.loaf32
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(inputs))))

class TFUpsample(keras.layers.Layer):
    # TF version of torch.nn.Upsample()
    def __init__(self, size, scale_factor, mode, w=None):  # warning: all arguments needed including 'w'
        super().__init__()
        assert scale_factor % 2 == 0, 'scale_factor must be multiple of 2'
        self.upsample=  tf.keras.layers.UpSampling2D(
            size=(scale_factor, scale_factor), data_format=None, interpolation=mode
        )
        # self.upsample = keras.layers.UpSampling2D(size=scale_factor, interpolation=mode)
        # with default arguments: align_corners=False, half_pixel_centers=False
        # self.upsample = lambda x: tf.raw_ops.ResizeNearestNeighbor(images=x,
        #                                                            size=(x.shape[1] * 2, x.shape[2] * 2))

    def call(self, inputs):
        return self.upsample(inputs)

class TFConcat(keras.layers.Layer):
    # TF version of torch.concat()
    def __init__(self, dimension=1, w=None):
        super().__init__()
        assert dimension == 1, 'convert only NCHW to NHWC concat'
        self.d = 3

    def call(self, inputs):
        return tf.concat(inputs, self.d)

def parse_model(anchors, nc, gd, gw, mlist, ch, ref_model_seq, imgsz, training):  # model_dict, input_channels(3)
    """
    Constructs the model by parsing model layers' configuration
    :param anchors: list[nl[na*2] of anchor sets per layer. int
    :param nc: nof classes. Needed to determine no -nof outputs, which is used to check for last stage. int
    :param gd: depth gain. A scaling factor. float
    :param gw: width gain. A scaling factor. float
    :param mlist: model layers list. A layer is a list[4] structured: [from,number dup, module name,args]
    :param ch: list of nof in channels to layers. Initiated as [3], then an entry is appended each layers loop iteration
    :param ref_model_seq: A trained ptorch ref model seq, used for offline torch to tensorflow format weights conversion
    :param imgsz: Input img size, Typ [640,640], required by detection module to find grid size as (imgsz/strides). int
    :param training: For detect layer. Bool. if False (inference & validation), output a processed tensor ready for nms.
    :return:
     model - tf.keras Sequential linear stack of layers.
     savelist: indices list of  layers indices which output is a source to a next but not adjacent (i.e. not -1) layer.
    """
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(mlist):  # mlist-list of layers configs. from, number, module, args
        m_str = m
        # m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m_str in [
                'nn.Conv2d', 'Conv', 'DWConv', 'DWConvTranspose2d', 'Bottleneck', 'SPP', 'SPPF',  'Focus', 'CrossConv',
                'BottleneckCSP', 'C3', 'C3x']:
            c1, c2 = ch[f], args[0] # c1: nof layer's in channels, c2: nof layer's out channels
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]] # [nch_in, nch_o, args]
            if m_str in ['BottleneckCSP', 'C3', 'C3x']:
                args.insert(2, n)
                n = 1
        # elif m_str is 'nn.BatchNorm2d':
        #     args = [ch[f]]
        elif m_str == 'Concat':
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        elif m_str in ['Detect', 'Segment']:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m_str == 'Segment':
                args[3] = make_divisible(args[3] * gw, 8)
            args.append(imgsz)
            args.append(training)
        else:
            c2 = ch[f]

        tf_m = eval('TF' + m_str.replace('nn.', ''))
        if ref_model_seq: # feed weights directly
            m_ = keras.Sequential([tf_m(*args, w=ref_model_seq[i][j]) for j in range(n)]) if n > 1 \
                else tf_m(*args, w=ref_model_seq[i])  # module
        else:
            m_ = keras.Sequential([tf_m(*args) for j in range(n)]) if n > 1 \
                else tf_m(*args)  # module


        # torch_m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # np = sum(x.numel() for x in torch_m_.parameters())  # number params
        np=0
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{str(n):>3}{np:>10}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return keras.Sequential(layers), sorted(save)


class TFModel:
    # TF YOLOv5 model
    def __init__(self, cfg='', ch=3, nc=None, ref_model_seq=None, imgsz=(640, 640), training=True):  # model, channels, classes
        # todo - check mixed precision. looks slow on amd cpu
        # mixed_precision.set_global_policy('mixed_float16') # mixed precision train runs slow in cpu, upsample2d issue?

        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding {cfg} nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value


        d = deepcopy(self.yaml)
        anchors, nc, gd, gw, mlist = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d['backbone'] + d['head']
        self.anchors, self.nc = anchors, nc
        self.model, self.savelist = parse_model(anchors, nc, gd, gw, mlist, ch=[ch], ref_model_seq=ref_model_seq, imgsz=imgsz, training=training)

    def predict(self,
                inputs,
                tf_nms=False,
                agnostic_nms=False,
                topk_per_class=100,
                topk_all=100,
                iou_thres=0.45,
                conf_thres=0.25,
                visualize=False):
        y = []  # outputs
        x = inputs

        for m in self.model.layers:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.savelist else None)  # save output
            if visualize:
                feature_visualization(x, m.name, m.i, save_dir=visualize)

        # Add TensorFlow NMS
        # if tf_nms:
        #     boxes = self._xywh2xyxy(x[0][..., :4])
        #     probs = x[0][:, :, 4:5]
        #     classes = x[0][:, :, 5:]
        #     scores = probs * classes
        #     if agnostic_nms:
        #         nms = AgnosticNMS()((boxes, classes, scores), topk_all, iou_thres, conf_thres)
        #     else:
        #         boxes = tf.expand_dims(boxes, 2)
        #         nms = tf.image.combined_non_max_suppression(boxes,
        #                                                     scores,
        #                                                     topk_per_class,
        #                                                     topk_all,
        #                                                     iou_thres,
        #                                                     conf_thres,
        #                                                     clip_boxes=False)
        #     return (nms,)
        return x  # output [1,6300,85] = [xywh, conf, class0, class1, ...]
        # x = x[0]  # [x(1,6300,85), ...] to x(6300,85)
        # xywh = x[..., :4]  # x(6300,4) boxes
        # conf = x[..., 4:5]  # x(6300,1) confidences
        # cls = tf.reshape(tf.cast(tf.argmax(x[..., 5:], axis=1), tf.float32), (-1, 1))  # x(6300,1)  classes
        # return tf.concat([conf, cls, xywh], 1)

    @staticmethod
    def _xywh2xyxy(xywh):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


class AgnosticNMS(keras.layers.Layer):
    # TF Agnostic NMS
    def call(self, input, topk_all, iou_thres, conf_thres):
        # wrap map_fn to avoid TypeSpec related error https://stackoverflow.com/a/65809989/3036450
        return tf.map_fn(lambda x: self._nms(x, topk_all, iou_thres, conf_thres),
                         input,
                         fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.int32),
                         name='agnostic_nms')

    @staticmethod
    def _nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25):  # agnostic NMS
        boxes, classes, scores = x
        class_inds = tf.cast(tf.argmax(classes, axis=-1), tf.float32)
        scores_inp = tf.reduce_max(scores, -1)
        selected_inds = tf.image.non_max_suppression(boxes,
                                                     scores_inp,
                                                     max_output_size=topk_all,
                                                     iou_threshold=iou_thres,
                                                     score_threshold=conf_thres)
        selected_boxes = tf.gather(boxes, selected_inds)
        padded_boxes = tf.pad(selected_boxes,
                              paddings=[[0, topk_all - tf.shape(selected_boxes)[0]], [0, 0]],
                              mode='CONSTANT',
                              constant_values=0.0)
        selected_scores = tf.gather(scores_inp, selected_inds)
        padded_scores = tf.pad(selected_scores,
                               paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
                               mode='CONSTANT',
                               constant_values=-1.0)
        selected_classes = tf.gather(class_inds, selected_inds)
        padded_classes = tf.pad(selected_classes,
                                paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
                                mode='CONSTANT',
                                constant_values=-1.0)
        valid_detections = tf.shape(selected_inds)[0]
        return padded_boxes, padded_scores, padded_classes, valid_detections


def activations(act):
    if 'LeakyReLU' in str(act): #  in ['nn.LeakyReLU']:
        return lambda x: keras.activations.relu(x, alpha=0.1)
    elif 'Hardswish' in str(act): #  in ['Hardswish']:
        return lambda x: x * tf.nn.relu6(x + 3) * 0.166666667
    elif 'SiLU' in  str(act): #  in ['nn.SiLU', 'SiLU']:
        return lambda x: keras.activations.swish(x)
    else:
        raise Exception(f'no matching TensorFlow activation found for PyTorch activation {act}')


def representative_dataset_gen(dataset, ncalib=100):
    # Representative dataset generator for use with converter.representative_dataset, returns a generator of np arrays
    for n, (path, img, im0s, vid_cap, string) in enumerate(dataset):
        im = np.transpose(img, [1, 2, 0])
        im = np.expand_dims(im, axis=0).astype(np.float32)
        im /= 255
        yield [im]
        if n >= ncalib:
            break
