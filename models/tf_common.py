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
