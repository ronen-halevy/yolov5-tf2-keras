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
        """

        :param nc:
        :type nc:
        :param nm:
        :type nm:
        :param anchors:
        :type anchors:
        :param imgsz:
        :type imgsz:
        """
        stride = tf.convert_to_tensor( [8, 16, 32], dtype=tf.float32)
        self.nc=nc
        self.no = 5 + nc + nm  # number of outputs per anchor
        # self.nl = anchors.shape[0]  # number of detection layers
        # self.na = anchors.shape[1]  # number of anchors
        self.nl = 3 #len(anchors)  # number of detection layers
        self.na = 3 # len(anchors[0]) // 2  # number of anchors
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


def detect(inputs, decay_factor, nc=80, nl=3, nm=32, imgsz=(640, 640), na=3,w=None):
    """

    :param inputs:
    :type inputs:
    :param decay_factor:
    :type decay_factor:
    :param nc:
    :type nc:
    :param nm:
    :type nm:
    :param imgsz:
    :type imgsz:
    :param nl: nof grid layers, int
    :type nl:
    :param na: nof anchor sets per layer, int
    :type na:
    :param w:
    :type w:
    :return:
    :rtype:
    """
    stride = tf.convert_to_tensor([8, 16, 32], dtype=tf.float32) # todo used config for stride
    no = 5 + nc + nm  # number of outputs per anchor

    x = []
    for i in range(nl):
        # reason for avoiding common conv: pytorch weights structure arrangement is slightly different (no w.conv attr):
        bias = True
        y = tf.keras.layers.Conv2D(filters=no * na,
                                        kernel_size=1,
                                        strides=1,
                                        padding='VALID',
                                        use_bias=bias,
                                        kernel_initializer='zeros' if w is None else  tf.keras.initializers.Constant(
                                            w.m[i].weight.permute(2, 3, 1, 0).numpy()),
                                        bias_initializer='zeros' if w is None else tf.keras.initializers.Constant(w.m[i].bias.numpy()) if bias else None
                                   )(inputs[i])

        ny, nx = imgsz[0] // stride[i], imgsz[1] // stride[i] # ronen todo check reuse calc
        y = tf.keras.layers.Reshape([int(ny), int(nx), na, no])(y) # from [bs,nyi,nxi,na*no] to [bs,nyi,nxi,na,no]
        y = tf.keras.layers.Permute([3, 1, 2, 4])(y)# from shape [bs,nyi,nxi,na, no] to [bs,na,nyi,nxi,no]
        x.append(y)
    return x #if training else (tf.keras.layers.Concatenate(axis=1)(decoder_out), x)  # x:[bs,nyi,nxi,na,no] for i=0:2], z: [b,25200,no]


def _parse_segment(x, decay_factor, nc=8, nl=3, nm=32, npr=256, imgsz=(640, 640), na=3, w=None):
    """

    :param x:
    :type x:
    :param decay_factor:
    :type decay_factor:
    :param nc:
    :type nc:
    :param nm:
    :type nm:
    :param npr:
    :type npr:
    :param imgsz:
    :type imgsz:
    :param nl: nof grid layers, int
    :type nl:
    :param na: nof anchor sets per layer, int
    :type na:
    :param w:
    :type w:
    :return:
    :rtype:
    """
    p = mask_proto(x[0], decay_factor, npr, nm, w.proto if w is not None else None)
    p = tf.transpose(p, perm=[0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160)
    x = detect(x, decay_factor, nc, nl, nm, imgsz, na, w)
    y = (x, p) #if training else x[0], (p, x[1])
    return y


def parse_model(x, nl,na, nc, gd, gw, mlist, ch, imgsz, decay_factor, ref_model_seq=None):  # model_dict, input_channels(3)
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
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    y = []  # outputs
    for i, (f, n, m, args) in enumerate(mlist):  # mlist-list of layers configs. from, number, module, args
        m_str = m
        if f != -1:  # if not from previous layer
            x_in = x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  # from earlier layers
        for j, a in enumerate(args):
            try:
                # patch effective for pytorch weights porting: Remove anchors from args. Pytorch trained weights
                # model hold it but uneeded here.
                if a == 'anchors':
                    args[j]=nl
                    continue
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
            args.append(imgsz)
        if m_str == 'Conv':
            x = _parse_conv(x, decay_factor, *args, w=ref_model_seq[i] if ref_model_seq else None)
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

            x = _parse_c3(x, decay_factor, n, kernel_size, stride, *args, w=ref_model_seq[i] if ref_model_seq else None)
        elif m_str == 'SPPF':
            x = _parse_sppf(x, decay_factor, *args, w=ref_model_seq[i] if ref_model_seq else None)
        elif m_str == 'Maxpool':
            x = _parse_maxpool(x, *args)
        elif m_str == 'Reshape':
            x = parse_reshape(x, *args)
        elif m_str == 'Segment':
            x = _parse_segment(x, decay_factor, *args, na=3, w=ref_model_seq[i] if ref_model_seq else None)
        else:
            print('\n! Warning!! Unknown module name:', m_str)
        ch.append(c2)
        # pack laers as models for a more compact network .summary() display:
        x = Model(x_in, x, name=f'{m_str}_{i}')(x_in)
        x_in = x  # for next iter
        y.append(x)  # save output
        layers.append(x)

    return layers


def build_model(cfg, nl,na, imgsz,ref_model_seq=None):
    """
    layers: list of parsed layers
    @param inputs:model inputs, KerasTensor, shape:[b,w,h,ch], float
    @param anchors:  anchors, shape: [nl, na,2]
    @param nc:nof classes, int
    @param gd: depth gain.  a factor used in n layers repeats settings.  float
    @param gw: width gain, a factor used in n layers repeats settings. float
    @param mlist: layers config list read from yaml
    @param ch: nof channels, list, iteratively populated for all layers, init value: ch=[3]
    @param imgsz: image size, list[2] (can be set to [None,None] )
    @param decay_factor: value for conv kernel_regularizer, float
    @param training: if not training, model returns also decoded output for nms process, bool

    @return:
        model : functional model
    """
    # read cfg from either dict (in pytorch weights porting mode) or yaml file (in normal operation):
    if isinstance(cfg, dict):
        model_cfg = cfg  # model dict
    else:  # is *.yaml
        import yaml  # for torch hub
        with open(cfg) as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    d = deepcopy(model_cfg)
    nc, gd, gw, mlist = d['nc'], d['depth_multiple'], d['width_multiple'], d['backbone'] + d[
        'head']
    inputs = Input(shape=(None, None, 3))
    ch = [3]
    decay_factor = 0.01 # todo use or eliminate
    layers = parse_model(inputs, nl,na, nc, gd, gw, mlist, ch, imgsz, decay_factor, ref_model_seq=ref_model_seq)
    model = Model(inputs, layers[-1])

    return model


if __name__ == '__main__':
    def demo():
        cfg = '/home/ronen/devel/PycharmProjects/tf_yolov5/models/segment/yolov5s-seg.yaml'
        # cfg = '/home/ronen/devel/PycharmProjects/yolo-v3-tf2/config/models/yolov3/yolov3.yaml'
        # mlist = d['backbone'] + d['head']
        nc = 80 # todo from cfg
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
        dataset_cfg_file = 'data/coco128-seg-short.yaml'
        with open(cfg) as dataset_cfg_file:
            data_cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        anchors = data_cfg['anchors']
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
