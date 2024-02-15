import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
from utils.general import LOGGER, make_divisible, print_args
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
    # 1. load ref weights from pytorch weights file:
    ckpt = torch.load(weights, map_location='cpu') # dict
    ref_model = ckpt['model'] # object inherits from nn.BaseModel
    ref_model_seq = ref_model.model # Sequential type model object

    # 2. Constrict a Keras model, fed by ref model's weights
    keras_model=build_model(cfg=ref_model.yaml, nl=nl,na=na, imgsz=imgsz, ref_model_seq=ref_model_seq)
    keras_model.summary()

    LOGGER.info(f'Source Weights: {weights}')
    LOGGER.info('PyTorch, TensorFlow and Keras models successfully verified.\nUse export.py for TF model export.')
    # save resultant weights:
    keras_model.save_weights(tf_weights_dir, overwrite=True)
    LOGGER.info(f'Keras Weights saved to {tf_weights_dir}')
    # save weights loaded model:
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

if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
