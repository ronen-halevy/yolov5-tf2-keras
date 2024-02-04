from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input, Model
from tensorflow.python.ops.numpy_ops import np_config

import sys
from copy import deepcopy
from pathlib import Path
import yaml  # for torch hub
import tensorflow as tf
import math

from models.tf_common import parse_reshape,_parse_maxpool,_parse_concat,_parse_upsample,_parse_shortcut,mask_proto,_parse_sppf,_parse_c3, _parse_conv

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
class Decoder:
    def __init__(self, nc, nm, anchors, imgsz):
        stride = tf.convert_to_tensor( [8, 16, 32], dtype=tf.float32)
        self.nc=nc
        self.no = 5 + nc + nm  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid, self.ny, self.nx = [], [], []
        for i in range(self.nl):
            ny, nx = imgsz[0] // stride[i], imgsz[1] // stride[i]
            self.nx.append(nx)
            self.ny.append(ny)

            xv, yv = tf.meshgrid(tf.range( self.nx[i], dtype=tf.float32), tf.range( self.ny[i], dtype=tf.float32))  # shapes: [ny,nx]
            # stack to grid, reshape & transpose to predictions matching shape:
            layer_grid = tf.stack([xv, yv], 2).reshape([1, 1,  self.ny[i],  self.nx[i], 2])
            # layer_grid = layer_grid.transpose([0, 2, 3, 1, 4])  # shape: [1, ny, nx, 1, 2]
            self.grid.append(layer_grid)

        # reshape anchors and normalize by stride values:
        anchors = tf.convert_to_tensor(anchors, dtype=tf.float16).reshape([self.nl, self.na, 2]) / stride.reshape([self.nl, 1, 1])
        # rescale anchors by stride values and reshape to
        self.anchor_grid = anchors.reshape([self.nl, 1, self.na, 1, 2])
        self.anchor_grid = self.anchor_grid.transpose([0, 2, 1, 3,  4])  # shape: [nl, 1,1,na,2]

    def decoder(self, y, layer_idx):
        # operate yolo adaptations on box:
        dd = tf.sigmoid(y[..., 0:2]) * 2 - 0.5 + self.grid[layer_idx]
        xy = (tf.sigmoid(y[..., 0:2]) * 2 - 0.5 + self.grid[layer_idx]) / [self.nx[layer_idx],
                                                                   self.ny[layer_idx]]  # xy bbox formula, normalized  to 0-1
        wh = (2 * tf.sigmoid(y[..., 2:4])) ** 2 * self.anchor_grid[layer_idx] / [self.ny[layer_idx], self.ny[layer_idx]]  # fwh bbox formula. noremalized.
        # concat modified values back together, operate yolo specified sigmoid on confs:
        y = tf.concat([xy, wh, tf.sigmoid(y[..., 4:5 + self.nc]).astype(tf.float32),
                       y[..., 5 + self.nc:].astype(tf.float32)], -1)
        y = y.reshape([-1, self.na * self.ny[layer_idx] * self.nx[layer_idx], self.no])  # reshape [bs,ny,nx,na,no]->[bs,nxi*nyi*na,no]
        return y

    # def decoder1(self, y):
    #     xy = tf.keras.activations.sigmoid(y[..., 0:2])
    #     xy = tf.keras.layers.Multiply()([xy, tf.constant([2])])
    #     xy = tf.keras.layers.subtract([xy, tf.constant([0.5])])
    #     xy = tf.keras.layers.add([xy, self.grid])
    #     gg = 1. / nx
    #     gg = tf.reshape(gg, [1])
    #     xy = tf.keras.layers.Multiply()([xy, gg])
    #
    #     wh = tf.sigmoid(y[..., 2:4])  # ** 2 * anchor_grid / [ny, ny]  # fwh bbox formula. noremalized.
    #     wh = tf.keras.layers.Multiply()([wh, tf.constant([2])])
    #     wh = tf.keras.layers.Multiply()([wh, wh])
    #     wh = tf.keras.layers.Multiply()([wh, anchor_grid])
    #     wh = tf.keras.layers.Multiply()([wh, gg])
    #     cls = tf.keras.activations.sigmoid(y[..., 4:5 + nc])
    #     mask = y[..., 5 + nc:]
    #     y = tf.keras.layers.Concatenate(axis=-1)([xy, wh, cls, mask])
    #     y = tf.keras.layers.Reshape([na * int(nx) * int(ny), 5 + nc + nm])(y)
    #     return y


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
    y = tf.keras.layers.Reshape([ na * int(nx )* int(ny), 5+nc+nm])(y)
    return y


def detect(inputs, decay_factor, nc=80, anchors=(), nm=32, imgsz=(640, 640), w=None):
    stride = tf.convert_to_tensor(w.stride.numpy() if w is not None else [8, 16, 32], dtype=tf.float32)
    no = 5 + nc + nm  # number of outputs per anchor
    nl = len(anchors)  # number of detection layers
    na = len(anchors[0]) // 2  # number of anchors
    x = []

    for i in range(nl):
        y = _parse_conv(inputs[i], decay_factor, no * na, kernel_size=1, stride=1, bn=0,
                                 activation=0)
        # x.append(y)
        ny, nx = imgsz[0] // stride[i], imgsz[1] // stride[i] # ronen todo check reuse calc
        y = tf.keras.layers.Reshape([int(ny), int(nx), na, no])(y) # from [bs,nyi,nxi,na*no] to [bs,nyi,nxi,na,no]
        y = tf.keras.layers.Permute([3, 1, 2, 4])(y)# from shape [bs,nyi,nxi,na, no] to [bs,na,nyi,nxi,no]
        x.append(y)
    return x #if training else (tf.keras.layers.Concatenate(axis=1)(decoder_out), x)  # x:[bs,nyi,nxi,na,no] for i=0:2], z: [b,25200,no]


def _parse_segmment(x, decay_factor, nc=80, anchors=(), nm=32, npr=256, imgsz=(640, 640)):
    p = mask_proto(x[0], decay_factor, npr, nm)
    p = tf.transpose(p, perm=[0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160)
    x = detect(x, decay_factor, nc, anchors, nm, imgsz)
    y = (x, p) #if training else x[0], (p, x[1])
    return y


def parse_model(x, anchors, nc, gd, gw, mlist, ch, imgsz, decay_factor, ref_model_seq=None):  # model_dict, input_channels(3)
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
            # args.append(training)
        if m_str == 'Conv':
            if ref_model_seq:  # feed weights directly - used for pytorch to keras weights conversion
                x = _parse_conv(x, decay_factor, *args, w=ref_model_seq[i])
            else:
                x = _parse_conv(x, decay_factor, *args)
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

            if ref_model_seq:  # feed weights directly - used for pytorch to keras weights conversion
                x = _parse_c3(x, decay_factor, n, kernel_size, stride, *args, w=ref_model_seq[i])
            else:
                x = _parse_c3(x, decay_factor, n, kernel_size, stride, *args)
        elif m_str == 'SPPF':
            x = _parse_sppf(x, decay_factor, *args)
        elif m_str == 'Maxpool':
            x = _parse_maxpool(x, *args)
        elif m_str == 'Reshape':
            x = parse_reshape(x, *args)
        elif m_str == 'Segment':
            x = _parse_segmment(x, decay_factor, *args)
            # decoder_out = x[0]
            # x = x[1]
        else:
            print('\n! Warning!! Unknown module name:', m_str)
        ch.append(c2)
        # pack laers as models for a more compact network .summary() display:
        x = Model(x_in, x, name=f'{m_str}_{i}')(x_in)
        x_in = x  # for next iter
        y.append(x)  # save output
        layers.append(x)

    return layers


def build_model(cfg, imgsz,ref_model_seq=None):
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
    if isinstance(cfg, dict):
        model_cfg = cfg  # model dict
    else:  # is *.yaml
        import yaml  # for torch hub
        with open(cfg) as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    d = deepcopy(model_cfg)
    anchors, nc, gd, gw, mlist = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d['backbone'] + d[
        'head']
    inputs = Input(shape=(640, 640, 3))
    ch = [3]
    decay_factor = 0.01
    layers = parse_model(inputs, anchors, nc, gd, gw, mlist, ch, imgsz, decay_factor, ref_model_seq=ref_model_seq)
    model = Model(inputs, layers[-1])

    return model


if __name__ == '__main__':
    def demo():
        cfg = '/home/ronen/devel/PycharmProjects/tf_yolov5/models/segment/yolov5s-seg.yaml'
        # cfg = '/home/ronen/devel/PycharmProjects/yolo-v3-tf2/config/models/yolov3/yolov3.yaml'


        # mlist = d['backbone'] + d['head']
        # nc = 80
        imgsz = [640, 640]
        # ch = 3 # input image nof channels
        # decay_factor = 0.01
        im = tf.zeros([1, 640, 640, 3], dtype=tf.float32)

        train_model = build_model(cfg,  imgsz=imgsz)
        pred = train_model(im)
        print(train_model.summary())
        for idx,p in enumerate(pred[0]):
            print(f'training pred layer {idx} shape: {p.shape}')
        print(f'train proto shape: {pred[1].shape}')
        import numpy as np
        np.sum([np.prod(v.get_shape().as_list()) for v in train_model.trainable_variables])

        nm=32
        with open(cfg) as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        d = deepcopy(model_cfg)
        anchors, nc, gd, gw, mlist = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d['backbone'] + d['head']
        dd = Decoder(nc, nm, anchors, imgsz)
        layer_idx=0
        dd.decoder(pred[0][0], layer_idx)


    # inference_model = build_model(cfg,  imgsz=imgsz,
        #                     training=True)
        # pred = inference_model(im)
        # print(f'inference decoded out shape: {pred[1].shape}')
        # print(f'inference proto shape: {pred[0].shape}')
        # for idx,p in enumerate(pred[2]):
        #     print(f'inference pred layer {idx} shape: {p.shape}')
        # inference_model.set_weights(train_model.get_weights())

    demo()
