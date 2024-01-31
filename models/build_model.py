from tensorflow.keras.regularizers import l2

from tensorflow.keras import Input, Model
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior() # allows running NumPy code, accelerated by TensorFlow

import sys
from copy import deepcopy
from pathlib import Path
import yaml  # for torch hub


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import tensorflow as tf
import math

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    # if isinstance(divisor, torch.Tensor):
    #     divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def _parse_output(x,  nc):
    #
    return x


def parse_reshape(x,  dim0, dim1, dim2, dim3):
    x = tf.keras.layers.Reshape((dim0, dim1, dim2, dim3))(x)

    return x


def _parse_maxpool(x,  pool_size, stride_xy, pad='same'):
    padding = pad
    x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=stride_xy,
                                     padding=padding)(x)


    return x


def _parse_concat(x):
    x = tf.keras.layers.Concatenate(axis=3)(x)

    return x

def _parse_upsample(x,  size, interpolation):
    x = tf.keras.layers.UpSampling2D(size=size, interpolation=interpolation)(x)

    return x

def _parse_shortcut(x):
    x = tf.keras.layers.Add()(x)

    return x


def mask_proto(x,  decay_factor, npr, nm):
    x=_parse_convolutional(x,  decay_factor, npr, kernel_size=3)
    size = (2,2)
    x = _parse_upsample(x,  size, interpolation= 'nearest')
    x=_parse_convolutional(x,  decay_factor,  npr, kernel_size=3)
    x=_parse_convolutional(x,  decay_factor,  nm)
    return x



def decoder(y, nx, ny, grid, anchor_grid):

    wh = (2 * tf.sigmoid(y[..., 2:4])) ** 2 * anchor_grid / [ny, ny]  # fwh bbox formula. noremalized.
    # concat modified values back together, operate yolo specified sigmoid on confs:

    y = tf.concat([xy, wh, tf.sigmoid(y[..., 4:5 + nc]).astype(tf.float32), y[..., 5 + nc:].astype(tf.float32)], -1)
    no = nc + 5  # number of outputs per anchor
    y = y.reshape([-1, na * ny * nx, no])  # reshape [bs,ny,nx,na,no]->[bs,nxi*nyi*na,no]
    return y

def detect(inputs, nc=80, anchors=(), nm=32, npr=256, ch=(), imgsz=(640, 640), training=False, w=None):
        stride = tf.convert_to_tensor(w.stride.numpy() if w is not None else [8,16,32], dtype=tf.float32)
        no = 5 + nc + nm  # number of outputs per anchor
        nl = len(anchors)  # number of detection layers
        na = len(anchors[0]) // 2  # number of anchors
        grid = [tf.zeros(1)] * nl  # init grid
        # reshape anchors and normalize by stride values:
        anchors = tf.convert_to_tensor(w.anchors.numpy(), dtype=tf.float16) if w is not None else \
            tf.convert_to_tensor(anchors, dtype=tf.float16).reshape([nl, na, 2]) / stride.reshape([nl, 1, 1])
        # rescale anchors by stride values and reshape to
        anchor_grid = anchors.reshape([nl, 1, na, 1, 2])
        anchor_grid = anchor_grid.transpose( [0, 1, 3, 2, 4])  # shape: [nl, 1,1,na,2]
        #####
        decoder_out = []  # inference output
        x = []
        layers=[]

        for i in range(nl):
            y =_parse_convolutional(inputs[i],  decay_factor, no * na, kernel_size=1, stride=1, pad=1, bn=0,
                                 activation=0)
            x.append(y)
            ny, nx = imgsz[0] // stride[i], imgsz[1] // stride[i]
            x[i] = tf.reshape(x[i], [-1, ny, nx, na, no])  # from [bs,nyi,nxi,na*no] to [bs,nyi,nxi,na,no]
            if not training:  # for inference & validation - process preds according to yolo spec,ready for nms:
                y = decoder(x[i], nx, ny, grid[i], anchor_grid[i])
                decoder_out.append(y)
            x[i] = tf.transpose(x[i], [0, 3, 1, 2, 4])  # from shape [bs,nyi,nxi,na, no] to [bs,na,nyi,nxi,no]
        return x if training else (tf.concat(decoder_out, axis=1), x)  # x:[bs,nyi,nxi,na,no] for i=0:2], z: [b,25200,no]


def _parse_segmment(x,  decay_factor, nc=80, anchors=(), nm=32, npr=256, ch=(),  imgsz=(640, 640), training=False):
    p = mask_proto(x[0],  decay_factor, npr, nm)
    p = tf.transpose(p, perm=[0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160)
    x=detect(x, nc, anchors, nm, npr, ch, imgsz, training)
    y = (x, p) if training else (x[0], p, x[1])
    # layers.append(y)
    return y






def _parse_sppf(x,  decay_factor, kernel_size, stride, c2, pad=1):

    # e=0.5 # todo ronen
    c_ = x.shape[3] // 2  # hidden channels
    x=_parse_convolutional(x,  decay_factor, c_, kernel_size=1, stride=1, pad=1, bn=1, activation=1)
    y1=_parse_maxpool(x,  pool_size=5, stride_xy=1, pad='same')
    y2=_parse_maxpool(y1,  pool_size=5, stride_xy=1, pad='same')
    y3=_parse_maxpool(y2,  pool_size=5, stride_xy=1, pad='same')
    x = tf.keras.layers.Concatenate(axis=3)([x,y1,y2,y3])
    x = _parse_convolutional(x,  decay_factor, c2, kernel_size=1, stride=1, pad=1, bn=1, activation=1)
    #
    return x



def _parse_c3(x,  decay, n,kernel_size, stride, c1, filters, pad=1, bn=1, activation=1):
    e = 0.5
    c_ = int(filters * e)  # hidden channels
    x1=_parse_convolutional(x,  decay, c_, kernel_size, stride, pad, bn=1, activation=1)
    for idx in range(n):
        x1=_parse_bottleneck(x1, c_, c_, shortcut=True, g=1, e=1.0)
    x2 = _parse_convolutional(x,  decay, c_, kernel_size, stride, pad, bn=1, activation=1)
    x = tf.keras.layers.Concatenate(axis=3)([x1, x2])
    x = _parse_convolutional(x,  decay,filters, kernel_size, stride, pad, bn=1, activation=1)
    #
    return x

def _parse_bottleneck(x, c1, c2, decay=0.01, shortcut=True, g=1, e=0.5):
    c_ = int(c2 * e)  # hidden channels
    layers=[]
    kernel_size=1
    stride=1
    x1 = _parse_convolutional(x,  decay, c_, kernel_size, stride)
    kernel_size=3
    x1=_parse_convolutional(x1,  decay, c2, kernel_size, stride)

    add = shortcut and c1 == c2
    if add:
        x1 = tf.keras.layers.Add()([x, x1])
    return x1


def _parse_convolutional(x,  decay, filters, kernel_size=1, stride=1, pad=1, bn=1, activation=1):
    # padding = 'same' if pad == 1 and stride == 1 else 'valid'
    # if stride > 1:
    #     x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)

    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=(stride, stride),
                               padding='SAME',# if stride == 1 else 'VALID',
                               use_bias=not bn,
                               activation='linear',
                               kernel_regularizer=l2(decay))(x)

    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    if activation:
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    #

    return x


def parse_model(inputs, na, nc, gd, gw,mlist, ch, imgsz, decay_factor):  # model_dict, input_channels(3)
    """

    @param inputs: model inputs, KerasTensor, shape:[b,w,h,ch], float
    @param na: nof anchors per grid layer,  int
    @param nc: nof classes, int
    @param mlist: layers config list read from yaml
    @param ch: nof channels, list, iteratively populated, init value: ch=[3]
    @param imgsz: image size, list[2] (can be set to [None,None] )
    @param decay_factor: value for conv kernel_regularizer, float
    @return:
    layers: list of parsed layers
    """

    x = inputs
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    y = []  # outputs
    for i, (f, n, m, args) in enumerate(mlist):  # mlist-list of layers configs. from, number, module, args
        m_str = m
        if f != -1:  # if not from previous layer
            x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  # from earlier layers
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m_str in [
                 'C3', 'Conv', 'SPPF']:
            c2 = args[0] # c1: nof layer's in channels, c2: nof layer's out channels
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c2, *args[1:]] # [nch_in, nch_o, args]
        if m_str in [
                 'C3']:
            c1 = ch[f] # c1: nof layer's in channels, c2: nof layer's out channels
            args = [c1,  *args] # [nch_in, nch_o, args]

        if m_str == 'Concat':
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        elif m_str in ['Detect', 'Segment']:
            if m_str == 'Segment':
                args[3] = make_divisible(args[3] * gw, 8)
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            args.append(imgsz)
            args.append(training)
        if m_str == 'Conv':
            inputs = x
            x = _parse_convolutional(x,  decay_factor, *args)
            mmmodel = Model(inputs, x)
            x = Model(inputs, x, name=f'Conv{i}')(inputs)
            # layers.append(x)

        elif m_str == 'Shortcut':
            x = _parse_shortcut(x)

        elif m_str == 'nn.Upsample':
            size = (args[1], args[1])
            interpolation = args[2]
            x = _parse_upsample(x,  size, interpolation)
        elif m_str == 'Concat':
            x = _parse_concat(x) # todo no args needed, *args)
        elif m_str == 'C3':
            kernel_size, stride=1,1
            # for idx in range(n):
            inputs = x
            x = _parse_c3(x,  decay_factor,  n, kernel_size, stride, *args)

            mmmodel = Model(inputs, x)
            x = Model(inputs, x, name=f'C3_{i}')(inputs)
            # layers.append(x)
            # print(mmmodel.summary())

        elif m_str == 'SPPF':
            inputs = x
            x = _parse_sppf(x,  decay_factor, kernel_size, stride, *args)


            mmmodel = Model(inputs, x)
            x = Model(inputs, x, name=f'SPPF_{i}')(inputs)
            # layers.append(x)
            # print(mmmodel.summary())

        elif m_str == 'Maxpool':
            x = _parse_maxpool(x,  *args)
        elif m_str == 'Reshape':
            x = parse_reshape(x,  *args)
        elif m_str == 'Output':
            x = _parse_output(x,  *args)
        elif m_str == 'Segment':
            inputs=x
            x = _parse_segmment(x,  decay_factor,*args)
            x = Model(inputs, x, name=f'Segment')(inputs)

            # mmmodel = Model(inputs, x)
            # print(mmmodel.summary())
        else:
            print('\n! Warning!! Unknown module name:', m_str)
        ch.append(c2)
        # print('\n x.shape', x.shape)
        y.append(x)  # save output
        layers.append(x)

    return layers


def build_model(inputs, anchors, nc, gd, gw, mlist, ch, imgsz, decay_factor):
    """
    layers: list of parsed layers
    @param inputs:model inputs, KerasTensor, shape:[b,w,h,ch], float
    @param na:nof anchors per grid layer,  int
    @param nc:nof classes, int
    @param mlist: layers config list read from yaml
    @param ch: nof channels, list, iteratively populated, init value: ch=[3]
    @param imgsz: image size, list[2] (can be set to [None,None] )
    @param decay_factor: value for conv kernel_regularizer, float
    @return:
        model : functional model
    """
    layers = parse_model(inputs, anchors, nc, gd, gw, mlist, ch, imgsz=imgsz, decay_factor=decay_factor)
    model = Model(inputs, layers[-1])
    return model


if __name__ == '__main__':
    cfg = '/home/ronen/devel/PycharmProjects/tf_yolov5/models/segment/yolov5s-seg.yaml'
    # cfg = '/home/ronen/devel/PycharmProjects/yolo-v3-tf2/config/models/yolov3/yolov3.yaml'
    with open(cfg) as f:
        yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    d = deepcopy(yaml)

    mlist = d['backbone'] + d['head']
    nc = 80
    training = True
    imgsz = [640, 640]
    ch = 3
    bs=2
    inputs = Input(shape=(640, 640, 3))

    decay_factor = 0.01
    # na = 3
    gd = 0.33# depth multiply
    gw =0.5# width multiply
    anchors, nc, gd, gw, mlist = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d['backbone'] + d[
        'head']

    model = build_model(inputs, anchors, nc, gd, gw, mlist, ch=[ch], imgsz=imgsz, decay_factor=decay_factor)
    import numpy as np

    hh=model.trainable_variables
    np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])

    xx = tf.zeros([1, 640, 640, 3], dtype=tf.float32)
    outp = model(xx)
    print(model.summary())
