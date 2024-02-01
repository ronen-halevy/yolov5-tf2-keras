from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input, Model
from tensorflow.python.ops.numpy_ops import np_config

import sys
from copy import deepcopy
from pathlib import Path
import yaml  # for torch hub
import tensorflow as tf
import math

np_config.enable_numpy_behavior()  # allows running NumPy code, accelerated by TensorFlow

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


def make_divisible(x, divisor):
    """
    Returns nearest x divisible by divisor
    """

    return math.ceil(x / divisor) * divisor


def parse_reshape(x, dim0, dim1, dim2, dim3):
    x = tf.keras.layers.Reshape((dim0, dim1, dim2, dim3))(x)
    return x


def _parse_maxpool(x, pool_size, stride_xy, pad='same'):
    padding = pad
    x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=stride_xy,
                                     padding=padding)(x)

    return x


def _parse_concat(x):
    x = tf.keras.layers.Concatenate(axis=3)(x)

    return x


def _parse_upsample(x, size, interpolation):
    x = tf.keras.layers.UpSampling2D(size=size, interpolation=interpolation)(x)

    return x


def _parse_shortcut(x):
    x = tf.keras.layers.Add()(x)

    return x


def mask_proto(x, decay_factor, npr, nm):
    x = _parse_convolutional(x, decay_factor, npr, kernel_size=3)
    size = (2, 2)
    x = _parse_upsample(x, size, interpolation='nearest')
    x = _parse_convolutional(x, decay_factor, npr, kernel_size=3)
    x = _parse_convolutional(x, decay_factor, nm)
    return x


def decoder(y, nx, ny, nc, na, nm, grid, anchor_grid):
    xy = tf.keras.activations.sigmoid(y[..., 0:2])
    xy=tf.keras.layers.Multiply()([xy, tf.constant([2])])
    xy = tf.keras.layers.subtract([xy,tf.constant([0.5])])
    xy = tf.keras.layers.add([xy,grid])
    gg=1. / nx
    gg=tf.reshape(gg, [1])
    xy=tf.keras.layers.Multiply()([xy,gg])

    wh = tf.sigmoid(y[..., 2:4])# ** 2 * anchor_grid / [ny, ny]  # fwh bbox formula. noremalized.
    wh=tf.keras.layers.Multiply()([wh, tf.constant([2])])
    wh=tf.keras.layers.Multiply()([wh, wh])
    wh=tf.keras.layers.Multiply()([wh, anchor_grid])
    wh=tf.keras.layers.Multiply()([wh, gg])
    cls = tf.keras.activations.sigmoid(y[..., 4:5 + nc])
    mask = y[..., 5 + nc:]
    y = tf.keras.layers.Concatenate(axis=-1)([xy, wh, cls, mask])
    y = tf.keras.layers.Reshape([ na * int(nx.numpy() )* int(ny.numpy()), 5+nc+nm])(y)
    return y


def detect(inputs, decay_factor, nc=80, anchors=(), nm=32, imgsz=(640, 640), training=False, w=None):
    stride = tf.convert_to_tensor(w.stride.numpy() if w is not None else [8, 16, 32], dtype=tf.float32)
    no = 5 + nc + nm  # number of outputs per anchor
    nl = len(anchors)  # number of detection layers
    na = len(anchors[0]) // 2  # number of anchors
    grid = []
    for i in range(nl):
        ny, nx = imgsz[0] // stride[i], imgsz[1] // stride[i]
        xv, yv = tf.meshgrid(tf.range(nx, dtype=tf.float32), tf.range(ny, dtype=tf.float32))  # shapes: [ny,nx]
        # stack to grid, reshape & transpose to predictions matching shape:
        layer_grid = tf.stack([xv, yv], 2).reshape([1, 1, ny, nx, 2])
        layer_grid = layer_grid.transpose([0, 2, 3, 1, 4])  # shape: [1, ny, nx, 1, 2]
        grid.append(layer_grid)

    # reshape anchors and normalize by stride values:
    anchors = tf.convert_to_tensor(w.anchors.numpy(), dtype=tf.float16) if w is not None else \
        tf.convert_to_tensor(anchors, dtype=tf.float16).reshape([nl, na, 2]) / stride.reshape([nl, 1, 1])
    # rescale anchors by stride values and reshape to
    anchor_grid = anchors.reshape([nl, 1, na, 1, 2])
    anchor_grid = anchor_grid.transpose([0, 1, 3, 2, 4])  # shape: [nl, 1,1,na,2]
    #####
    decoder_out = []  # inference output
    x = []

    for i in range(nl):
        y = _parse_convolutional(inputs[i], decay_factor, no * na, kernel_size=1, stride=1, bn=0,
                                 activation=0)
        # x.append(y)
        ny, nx = imgsz[0] // stride[i], imgsz[1] // stride[i] # ronen todo check reuse calc
        y = tf.reshape(y, [-1, ny, nx, na, no])  # from [bs,nyi,nxi,na*no] to [bs,nyi,nxi,na,no]
        if not training:  # for inference & validation - process preds according to yolo spec,ready for nms:
            z = decoder(y, nx, ny, nc, na, nm, grid[i], anchor_grid[i])
            decoder_out.append(z)
        y = tf.transpose(y, [0, 3, 1, 2, 4])  # from shape [bs,nyi,nxi,na, no] to [bs,na,nyi,nxi,no]
        x.append(y)
    return x if training else (tf.concat(decoder_out, axis=1), x)  # x:[bs,nyi,nxi,na,no] for i=0:2], z: [b,25200,no]


def _parse_segmment(x, decay_factor, nc=80, anchors=(), nm=32, npr=256, imgsz=(640, 640), training=False):
    p = mask_proto(x[0], decay_factor, npr, nm)
    p = tf.transpose(p, perm=[0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160)
    x = detect(x, decay_factor, nc, anchors, nm, imgsz, training)
    y = (x, p) if training else (x[0], p, x[1])
    return y


def _parse_sppf(x, decay_factor, c2, pool_size):
    # e=0.5 # todo ronen
    c_ = x.shape[3] // 2  # hidden channels
    x = _parse_convolutional(x, decay_factor, c_, kernel_size=1, stride=1, bn=1, activation=1)
    y1 = _parse_maxpool(x, pool_size, stride_xy=1, pad='same')
    y2 = _parse_maxpool(y1, pool_size, stride_xy=1, pad='same')
    y3 = _parse_maxpool(y2, pool_size, stride_xy=1, pad='same')
    x = tf.keras.layers.Concatenate(axis=3)([x, y1, y2, y3])
    x = _parse_convolutional(x, decay_factor, c2, kernel_size=1, stride=1, bn=1, activation=1)
    #
    return x


def _parse_c3(x, decay, n, kernel_size, stride, filters, shortcut=True):
    e = 0.5
    c_ = int(filters * e)  # hidden channels
    x1 = _parse_convolutional(x, decay, c_, kernel_size, stride, bn=1, activation=1)
    for idx in range(n):
        x1 = _parse_bottleneck(x1, c_, shortcut, e=1.0)
    x2 = _parse_convolutional(x, decay, c_, kernel_size, stride, bn=1, activation=1)
    x = tf.keras.layers.Concatenate(axis=3)([x1, x2])
    x = _parse_convolutional(x, decay, filters, kernel_size, stride, bn=1, activation=1)
    return x


def _parse_bottleneck(x, filters2, decay=0.01, shortcut=True, e=0.5):
    c_ = int(filters2 * e)  # hidden channels
    kernel_size = 1
    stride = 1
    x1 = _parse_convolutional(x, decay, c_, kernel_size, stride)
    kernel_size = 3
    x1 = _parse_convolutional(x1, decay, filters2, kernel_size, stride)

    add = shortcut and x.shape[3] == filters2  # n ch in = n ch out
    if add:
        x1 = tf.keras.layers.Add()([x, x1])
    return x1


def _parse_convolutional(x, decay, filters, kernel_size=1, stride=1, bn=1, activation=1):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=(stride, stride),
                               padding='SAME',
                               use_bias=not bn,
                               activation='linear',
                               kernel_regularizer=l2(decay))(x)

    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    if activation:
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    #

    return x


def parse_model(x, anchors, nc, gd, gw, mlist, ch, imgsz, decay_factor, training):  # model_dict, input_channels(3)
    """

    @param x: model inputs, KerasTensor, shape:[b,w,h,ch], float
    @param anchors:  anchors, shape: [nl, na,2]
    @param nc: nof classes, int
    @param gd: depth gain. Factors nof layers' repeats.  float
    @param gw: width gain, Factors nof layers' filters. float
    @param mlist: layers config list read from yaml
    @param ch: nof channels, list, iteratively populated, init value: ch=[3]
    @param imgsz: image size, list[2] (can be set to [None,None] )
    @param decay_factor: value for conv kernel_regularizer, float
    @param training:  if false (inference  & validation), model returns also decoded output for nms process, bool
    @return:
    layers: list of parsed layers
    """

    x_in = x
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    y = []  # outputs
    for i, (f, n, m, args) in enumerate(mlist):  # mlist-list of layers configs. from, number, module, args
        m_str = m
        if f != -1:  # if not from previous layer
            x_in = x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  # from earlier layers
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m_str in ['C3', 'Conv', 'SPPF']:
            c2 = args[0]  # c2: nof out channels
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c2, *args[1:]]  # [nch_in, nch_o, args]
        if m_str in ['C3']:
            args = [*args]  # [nch_in, nch_o, args]

        if m_str == 'Concat':
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        elif m_str in ['Detect', 'Segment']:
            if m_str == 'Segment':
                args[3] = make_divisible(args[3] * gw, 8)
            # args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            args.append(imgsz)
            args.append(training)
        if m_str == 'Conv':
            x = _parse_convolutional(x, decay_factor, *args)
        elif m_str == 'Shortcut':
            x = _parse_shortcut(x)

        elif m_str == 'nn.Upsample':
            size = (args[1], args[1])
            interpolation = args[2]
            x = _parse_upsample(x, size, interpolation)
        elif m_str == 'Concat':
            x = _parse_concat(x)  # todo no args needed, *args)
        elif m_str == 'C3':
            kernel_size, stride = 1, 1
            x = _parse_c3(x, decay_factor, n, kernel_size, stride, *args)
        elif m_str == 'SPPF':
            x = _parse_sppf(x, decay_factor, *args)
        elif m_str == 'Maxpool':
            x = _parse_maxpool(x, *args)
        elif m_str == 'Reshape':
            x = parse_reshape(x, *args)
        elif m_str == 'Segment':

            x = _parse_segmment(x, decay_factor, *args)
        else:
            print('\n! Warning!! Unknown module name:', m_str)
        ch.append(c2)
        # pack laers as models for a more compact network .summary() display:
        x = Model(x_in, x, name=f'{m_str}_{i}')(x_in)
        x_in = x  # for next iter
        y.append(x)  # save output
        layers.append(x)

    return layers


def build_model(cfg, imgsz, training):
    """
    layers: list of parsed layers
    @param inputs:model inputs, KerasTensor, shape:[b,w,h,ch], float
    @param anchors:  anchors, shape: [nl, na,2]
    @param nc:nof classes, int
    @param gd: depth gain. Factors nof layers' repeats.  float
    @param gw: width gain, Factors nof layers' filters. float
    @param mlist: layers config list read from yaml
    @param ch: nof channels, list, iteratively populated, init value: ch=[3]
    @param imgsz: image size, list[2] (can be set to [None,None] )
    @param decay_factor: value for conv kernel_regularizer, float
    @param training: if not training, model returns also decoded output for nms process, bool

    @return:
        model : functional model
    """
    with open(cfg) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    d = deepcopy(model_cfg)
    anchors, nc, gd, gw, mlist = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d['backbone'] + d[
        'head']
    inputs = Input(shape=(640, 640, 3))
    ch = [3]
    decay_factor = 0.01
    layers = parse_model(inputs, anchors, nc, gd, gw, mlist, ch, imgsz, decay_factor, training)
    model = Model(inputs, layers[-1])
    return model


if __name__ == '__main__':
    def demo():
        cfg = '/home/ronen/devel/PycharmProjects/tf_yolov5/models/segment/yolov5s-seg.yaml'
        # cfg = '/home/ronen/devel/PycharmProjects/yolo-v3-tf2/config/models/yolov3/yolov3.yaml'


        # mlist = d['backbone'] + d['head']
        # nc = 80
        imgsz = [640, 640]
        ch = 3
        # bs = 2

        decay_factor = 0.01
        # na = 3
        # gd = 0.33  # depth multiply
        # gw = 0.5  # width multiply
        # anchors, nc, gd, gw, mlist = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d['backbone'] + d[
        #     'head']
        model = build_model(cfg,  imgsz=imgsz,
                            training=False)
        model = build_model(cfg,  imgsz=imgsz,
                            training=True)

        import numpy as np

        np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])

        xx = tf.zeros([1, 640, 640, 3], dtype=tf.float32)
        outp = model(xx)
        print(outp)
        print(model.summary())



    demo()
