# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license


import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
from models.experimental import MixConv2d, attempt_load
from utils.general import LOGGER, make_divisible, print_args

# from models.tf_model import TFModel
from models.build_model import build_model


def run(
        weights=ROOT / 'yolov5s-seg.pt',  # weights path
        imgsz=(640, 640),  # inference size h,w
        nl=3,
        na=3,
        batch_size=1,  # batch size
        tf_weights_dir='.',
        tf_model_dir='.',
        **kwargs
):
    # PyTorch model
    im = torch.zeros((batch_size, 3, *imgsz))  # BCHW image
    # fuse is essential for porting weights to keras. TBD
    ref_model = attempt_load(weights, device=torch.device('cpu'), inplace=True, fuse=True)
    _ = ref_model(im)  # inference
    ref_model.info()
    ref_model_seq = ref_model.model

    # Keras model
    keras_model=build_model(cfg=ref_model.yaml, nl=nl,na=na, imgsz=imgsz, ref_model_seq=ref_model_seq)
    keras_model.summary()

    LOGGER.info(f'Source Weights: {weights}')
    LOGGER.info('PyTorch, TensorFlow and Keras models successfully verified.\nUse export.py for TF model export.')
    keras_model.save_weights(tf_weights_dir, overwrite=True)
    LOGGER.info(f'Keras Weights saved to {tf_weights_dir}')
    # keras_model.trainable=True
    # tf.keras.models.save_model(keras_model, tf_model_dir)
    keras_model.save(tf_model_dir)
    LOGGER.info(f'Keras Model saved to {tf_model_dir}')
    return keras_model


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / '/home/ronen/devel/PycharmProjects/tf_yolov5/yolov5s-seg.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--tf_weights_dir', type=str, default='./keras_weights/rrcoco.h5', help='produced weights target location')
    parser.add_argument('--tf_model_dir', type=str, default='./keras_model', help='produced weights target location')
    parser.add_argument('--nl', type=str, default=3, help='number of grid layers') # todo move to a cfg file
    parser.add_argument('--na', type=str, default=3, help='number of anchors per layer') # todo move to a cfg file

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
