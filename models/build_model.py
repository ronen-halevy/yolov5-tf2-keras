from tensorflow.keras.regularizers import l2

from tensorflow.keras import Input, Model

import sys
from copy import deepcopy
from pathlib import Path
import yaml  # for torch hub

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import tensorflow as tf


def _parse_output(x, layers, nc):
    layers.append(x)
    return x, layers


def parse_reshape(x, layers, dim0, dim1, dim2, dim3):
    x = tf.keras.layers.Reshape((dim0, dim1, dim2, dim3))(x)
    layers.append(x)
    return x, layers


def _parse_maxpool(x, layers, pool_size, stride_xy, pad='same'):
    padding = pad
    x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=stride_xy,
                                     padding=padding)(x)
    layers.append(x)

    return x, layers


def _parse_concat(x, layers):
    x = tf.keras.layers.Concatenate(axis=3)(x)
    layers.append(x)
    return x, layers


def _parse_upsample(x, layers, stride):
    x = tf.keras.layers.UpSampling2D(size=stride)(x)
    layers.append(x)
    return x, layers

    pass


def _parse_shortcut(x, layers):
    x = tf.keras.layers.Add()(x)
    layers.append(x)
    return x, layers



def _parse_convolutional(x, layers, decay, filters, kernel_size, stride, pad, bn=1, activation=1):
    padding = 'same' if pad == 1 and stride == 1 else 'valid'
    if stride > 1:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)

    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=(stride, stride),
                               padding=padding,
                               use_bias=not bn,
                               activation='linear',
                               kernel_regularizer=l2(decay))(x)

    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    if activation:
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    layers.append(x)

    return x, layers


def parse_model(inputs, na, nc, mlist, ch, imgsz, decay_factor):  # model_dict, input_channels(3)
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
        if m_str == 'Concat':
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        elif m_str in ['Detect', 'Segment']:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            args.append(imgsz)
            args.append(training)
        if m_str == 'Conv':
            x, layers = _parse_convolutional(x, layers, decay_factor, *args)
        elif m_str == 'Shortcut':
            x, layers = _parse_shortcut(x, layers)

        elif m_str == 'Upsample':
            x, layers = _parse_upsample(x, layers, *args)
        elif m_str == 'Concat':
            x, layers = _parse_concat(x, layers, *args)
        elif m_str == 'Maxpool':
            x, layers = _parse_maxpool(x, layers, *args)
        elif m_str == 'Reshape':
            x, layers = parse_reshape(x, layers, *args)
        elif m_str == 'Output':
            x, layers = _parse_output(x, layers, *args)
        ch.append(c2)
        y.append(x)  # save output
    return layers


def build_model(inputs, na, nc, mlist, ch, imgsz, decay_factor):
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
    layers = parse_model(inputs, na, nc, mlist, ch, imgsz=imgsz, decay_factor=decay_factor)
    model = Model(inputs, layers[-1], name='model')
    return model


if __name__ == '__main__':
    cfg = '/home/ronen/devel/PycharmProjects/yolo-v3-tf2/config/models/yolov3_tiny/yolov3_tiny.yaml'
    # cfg = '/home/ronen/devel/PycharmProjects/yolo-v3-tf2/config/models/yolov3/yolov3.yaml'
    with open(cfg) as f:
        yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    d = deepcopy(yaml)

    mlist = d['backbone'] + d['head']
    nc = 7
    training = True
    imgsz = [None, None]
    ch = 3
    inputs = Input(shape=(416, 416, 3))
    decay_factor = 0.01
    na = 3
    model = build_model(inputs, na, nc, mlist, ch=[ch], imgsz=imgsz, decay_factor=decay_factor)
    xx = tf.zeros([1, 416, 416, 3], dtype=tf.float32)
    outp = model(xx)
    print(model.summary())
