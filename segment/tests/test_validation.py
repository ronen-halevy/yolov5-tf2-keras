
from pathlib import Path
import sys

import tensorflow as tf
from tensorflow import keras

import segment.tf_val as validate  # for end-of-epoch mAP
from utils.tf_general import increment_path

from segment.tf_dataloaders import create_dataloader
from models.tf_model import TFModel


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

def create_model(cfg):
    dynamic = False
    tf_model = TFModel(cfg=cfg,
                       ref_model_seq=None, nc=80, imgsz=imgsz, training=True)
    im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
    # tf_model.predict(im)
    # s=640
    ch = 3

    keras_model = tf.keras.Model(inputs=im, outputs=tf_model.predict(im))
    return model












if __name__ == '__main__':
    imgsz = 640
    name='exp'
    # save_dir = increment_path(Path(f'{Path.cwd()}/results/dataset')  / name, exist_ok=False)  # increment run
    save_dir = increment_path(Path(f'{ROOT}/runs/tests/') / name, exist_ok=False)  # increment run

    data_path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train'

    save_dir.mkdir(parents=True, exist_ok=True)
    batch_size=2
    imgsz=640
    mosaic=True

    hyp = '../../data/hyps/hyp.scratch-low.yaml'
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    mosaic=True# False
    degrees, translate, scale, shear, perspective = hyp['degrees'],hyp['translate'],hyp['scale'],hyp['shear'],hyp['perspective']
    augment=True
    hgain, sgain, vgain, flipud, fliplr =hyp['hsv_h'],hyp['hsv_s'],hyp['hsv_v'],hyp['flipud'],hyp['fliplr']
    # create_dataset = CreateDataset(imgsz, mosaic, augment, degrees, translate, scale, shear, perspective, hgain, sgain, vgain, flipud, fliplr)
    # ds = create_dataset(image_files, labels, segments)
    batch_size=2
    debug=True
    ds = create_dataloader(data_path, batch_size,[imgsz, imgsz], mosaic, augment, degrees, translate, scale, shear, perspective ,hgain, sgain, vgain, flipud, fliplr, debug)



    ds = create_dataloader(data_path, batch_size, [imgsz, imgsz], mosaic, augment, degrees, translate, scale, shear,
                           perspective, hgain, sgain, vgain, flipud, fliplr, debug)

    print(f"Results saved to {save_dir}")


    ####


    classes_name_file = './data/class-names/coco.names'
    class_names = [c.strip() for c in open(classes_name_file).readlines()]

    data_dict={'names': class_names}
    amp=False
    model='TBD'
    single_cls=False
    val_loader='val_loader'
    save_dir='runs/exp/train-seg/ex42'
    callbacks='tbd'
    compute_loss='compute_loss'
    mask_ratio=4
    overlap=True

    cfg = '../models/segment/yolov5s-seg.yaml'
    model = create_model(cfg)



    results, maps, _ = validate.run(data_dict,
                                    batch_size=batch_size,
                                    imgsz=imgsz,
                                    half=amp,
                                    model=ema.ema,
                                    single_cls=single_cls,
                                    dataloader=val_loader,
                                    save_dir=save_dir,
                                    plots=False,
                                    callbacks=callbacks,
                                    compute_loss=compute_loss,
                                    mask_downsample_ratio=mask_ratio,
                                    overlap=overlap)