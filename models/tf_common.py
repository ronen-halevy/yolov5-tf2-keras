import sys
from pathlib import Path
import tensorflow as tf

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


def mask_proto(x, decay_factor, npr, nm, w=None):
    x = _parse_conv(x, decay_factor, npr, kernel_size=3, w=w.cv1 if w is not None else None)
    size = (2, 2)
    x = _parse_upsample(x, size, interpolation='nearest')
    x = _parse_conv(x, decay_factor, npr, kernel_size=3, w=w.cv2 if w is not None else None)
    x = _parse_conv(x, decay_factor, nm, w=w.cv3 if w is not None else None)
    return x


def _parse_sppf(x, decay_factor, c2, pool_size, w=None):
    c_ = x.shape[3] // 2  # hidden channels
    x = _parse_conv(x, decay_factor, c_, kernel_size=1, stride=1, bn=1, activation=1,
                    w=w.cv1 if w is not None else None)
    y1 = _parse_maxpool(x, pool_size, stride_xy=1, pad='same')
    y2 = _parse_maxpool(y1, pool_size, stride_xy=1, pad='same')
    y3 = _parse_maxpool(y2, pool_size, stride_xy=1, pad='same')
    x = tf.keras.layers.Concatenate(axis=3)([x, y1, y2, y3])
    x = _parse_conv(x, decay_factor, c2, kernel_size=1, stride=1, bn=1, activation=1,
                    w=w.cv2 if w is not None else None)
    return x


def _parse_c3(x, decay, n, kernel_size, stride, filters, shortcut=True, w=None):
    e = 0.5
    c_ = int(filters * e)  # hidden channels
    cv1 = _parse_conv(x, decay, c_, kernel_size, stride, bn=1, activation=1, w=w.cv1 if w is not None else None)
    for idx in range(n):
        cv1 = _parse_bottleneck(cv1, c_, decay, shortcut, e=1.0, w=w.m[idx] if w is not None else None)
    cv2 = _parse_conv(x, decay, c_, kernel_size, stride, bn=1, activation=1, w=w.cv2 if w is not None else None)
    x = tf.keras.layers.Concatenate(axis=3)([cv1, cv2])
    cv3 = _parse_conv(x, decay, filters, kernel_size, stride, bn=1, activation=1, w=w.cv3 if w is not None else None)
    return cv3


def _parse_bottleneck(x, filters2, decay=0.01, shortcut=True, e=0.5, w=None):
    c_ = int(filters2 * e)  # hidden channels
    kernel_size = 1
    stride = 1
    x1 = _parse_conv(x, decay, c_, kernel_size, stride, w=w.cv1 if w is not None else None)
    kernel_size = 3
    x1 = _parse_conv(x1, decay, filters2, kernel_size, stride, w=w.cv2 if w is not None else None)
    add = shortcut and x.shape[3] == filters2  # n ch in = n ch out
    if add:
        x1 = tf.keras.layers.Add()([x, x1])
    return x1


def _parse_conv(x, decay, filters, kernel_size=1, stride=1, p=None, bn=1, activation=1, w=None):
    if stride != 1:
        pad = p if p is not None else kernel_size // 2
        x = tf.keras.layers.ZeroPadding2D(padding=pad)(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=(stride, stride),
                               padding='SAME' if stride == 1 else 'VALID',
                               use_bias=not hasattr(w, 'bn'),
                               kernel_initializer=tf.keras.initializers.RandomNormal(
                                   stddev=0.01) if w is None else tf.keras.initializers.Constant(
                                   w.conv.weight.permute(2, 3, 1, 0).numpy()),
                               bias_initializer='zeros' if w is None else 'zeros' if hasattr(w, 'bn') else
                               tf.keras.initializers.Constant(w.conv.bias.numpy()))(x)
    if bn:
        x = tf.keras.layers.BatchNormalization(momentum=0.03)(x)
    if activation:
        x = tf.keras.activations.swish(x)
    return x
