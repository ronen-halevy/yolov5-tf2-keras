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
import torch
import torch.nn as nn
import tensorflow as tf

from tensorflow import keras

from models.common import (C3, SPP, SPPF, Bottleneck, BottleneckCSP, C3x, Concat, Conv, CrossConv, DWConv,
                           DWConvTranspose2d, Focus, autopad)
from models.experimental import MixConv2d, attempt_load
from models.yolo import Detect, Segment
from utils.activations import SiLU
from utils.general import LOGGER, make_divisible, print_args

from models.tf_model import TFModel


def run(
        weights=ROOT / 'yolov5l-seg.pt',  # weights path
        imgsz=(640, 640),  # inference size h,w
        batch_size=1,  # batch size
        tf_weights_dir='.',
        tf_model_dir='.',
        **kwargs
):
    ref_model = attempt_load(weights, device=torch.device('cpu'), inplace=True, fuse=True)
    # PyTorch model
    im = torch.zeros((batch_size, 3, *imgsz))  # BCHW image
    # fuse is essential for porting weights to keras. TBD
    ref_model = attempt_load(weights, device=torch.device('cpu'), inplace=True, fuse=True)
    _ = ref_model(im)  # inference
    ref_model.info()
    ref_model_seq = ref_model.model
    # ref_model = ref_model.model
    # TensorFlow model
    im = tf.zeros((batch_size, *imgsz, 3))  # BHWC image
    tf_model = TFModel(cfg=ref_model.yaml, ref_model_seq=ref_model_seq, nc=ref_model.nc, imgsz=imgsz)
    # _ = tf_model.predict(im)  # inference

    # Keras model
    im = keras.Input(shape=(*imgsz, 3), batch_size=None) # assume input is dataset - no batch size to be specified
    keras_model = keras.Model(inputs=im, outputs=tf_model.predict(im))
    keras_model.summary()

    LOGGER.info(f'Source Weights: {weights}')
    LOGGER.info('PyTorch, TensorFlow and Keras models successfully verified.\nUse export.py for TF model export.')
    keras_model.save_weights(tf_weights_dir)
    LOGGER.info(f'Keras Weights saved to {tf_weights_dir}')
    # keras_model.trainable=True
    # tf.keras.models.save_model(keras_model, tf_model_dir)
    keras_model.save(tf_model_dir)
    LOGGER.info(f'Keras Model saved to {tf_model_dir}')
    return keras_model, tf_model


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / '/home/ronen/Downloads/best_2_colors_squares.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--tf_weights_dir', type=str, default='./keras_weights/rrcoco.tf', help='produced weights target location')
    parser.add_argument('--tf_model_dir', type=str, default='./keras_model', help='produced weights target location')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
