from multiprocessing import Pool
import tqdm
import time
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import tensorflow as tf





####

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, ROOT, Profile,  check_suffix, check_version, colorstr,
                           increment_path, is_jupyter, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, smart_inference_mode
###

from models.yolo import parse_model

np.random.seed(88883)

weights = '/home/ronen/devel/PycharmProjects/tf_yolov5/yolov5s-seg.pt'
ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
ref_model = ckpt['model']

from models.build_model import build_model
# Keras model
imgsz=[640,640]
na=3
nl=3
keras_model = build_model(cfg=ref_model.yaml, nl=nl, na=na, imgsz=imgsz, ref_model_seq=ref_model.model)
keras_model.summary()

# train:
import yaml  # for torch hub
cfg = '/home/ronen/devel/PycharmProjects/tf_yolov5/models/segment/yolov5s-seg.yaml'
yaml_file = Path(cfg).name
with open(cfg, encoding='ascii', errors='ignore') as f:
    model_yaml = yaml.safe_load(f)  # model dict
ch=3
model, save = parse_model(model_yaml, ch=[ch])  # model, savelist

y, dt = [], []  # outputs
batch_size=1
x = torch.rand(batch_size, 3, 640, 640)#.to(device)

for m in model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        x = m(x)  # run
        y.append(x if m.i in save else None)  # save output

pass
im = torch.rand(batch_size, 3, 640, 640)#.to(device)
x, xx = model(im)

filters=32
kernel_size=7
stride=2
nfilters=64
#initialize the layers respectively
torch_layer = torch.nn.Conv2d(
    in_channels=3,
    out_channels=nfilters,
    kernel_size=(kernel_size, kernel_size),
    stride=(stride, stride),
    bias=False
)
torch_model = torch.nn.Sequential(
              torch.nn.ZeroPad2d((2,3,2,3)),
              torch_layer
              )

tf_layer = tf.keras.layers.Conv2D(
    filters=nfilters,
    kernel_size=(kernel_size, kernel_size),
    strides=(stride, stride),
    padding='same',
    use_bias=True
)

#setting weights in torch layer and tf layer respectively
torch_weights = np.random.rand(nfilters, 3, kernel_size, kernel_size)
torch_bias = np.random.rand(nfilters)

tf_weights = np.transpose(torch_weights, (2, 3, 1, 0))

with torch.no_grad():
  torch_layer.weight = torch.nn.Parameter(torch.Tensor(torch_weights))
  torch_layer.bias = torch.nn.Parameter(torch.Tensor(torch_bias))


tf_layer(np.zeros((1,256,256,3)))
tf_layer.kernel.assign(tf_weights)

#prepare inputs and do inference
torch_inputs = torch.Tensor(np.random.rand(1, 3, 256, 256))
tf_inputs = np.transpose(torch_inputs.numpy(), (0, 2, 3, 1))

with torch.no_grad():
  tttorch_output = torch_model(torch_inputs)
tttf_output = tf_layer(tf_inputs)
tttf_outputt = tf.transpose(tttf_output, [0,3,1,2])
rrr = np.allclose(tttf_output.numpy() ,np.transpose(tttorch_output.numpy(),(0, 2, 3, 1))) #True

# Edit: from pytorch to tensorflow

torch_layer = torch.nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False
)

tf_layer=tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(7, 7),
    strides=(2, 2),
    padding='valid',
    use_bias=False
    )

tf_model = tf.keras.Sequential([
           tf.keras.layers.ZeroPadding2D((3, 3)),
           tf_layer
           ])

###






















filters=32
kernel_size=6
stride=2
x=tf.ones([1,640,640,3])
w=None
convtf = tf.keras.layers.Conv2D(filters=filters,
                           kernel_size=kernel_size,
                           strides=(stride, stride),
                           padding='SAME' if stride == 1 else 'VALID',
                           use_bias=True,
                           kernel_initializer='zeros',
                           bias_initializer='zeros' if w is None else 'zeros' if hasattr(w, 'bn') else
                           tf.keras.initializers.Constant(w.conv.bias.numpy()))(x)

xpt=torch.ones(1,3, 640,640)

conv = nn.Conv2d(3, filters, kernel_size, stride, padding=0, groups=1, dilation=1, bias=False)(xpt)
pass
# bn = nn.BatchNorm2d(conv)
# act=True
# # acti = default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()