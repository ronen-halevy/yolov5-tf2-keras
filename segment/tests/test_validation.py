
from pathlib import Path
import sys

import yaml

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

from segment.tf_loss import ComputeLoss
from utils.segment.tf_metrics import fitness

def create_model(cfg):
    dynamic = False
    tf_model = TFModel(cfg=cfg,
                       ref_model_seq=None, nc=80, imgsz=imgsz, training=False) # produce both inference&training outputs
    im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
    # tf_model.predict(im)
    # s=640
    ch = 3

    keras_model = tf.keras.Model(inputs=im, outputs=tf_model.predict(im))
    return keras_model, tf_model



if __name__ == '__main__':
    name='exp'
    # save_dir = increment_path(Path(f'{Path.cwd()}/results/dataset')  / name, exist_ok=False)  # increment run
    save_dir = increment_path(Path(f'{ROOT}/runs/tests/') / name, exist_ok=False)  # increment run

    data_path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train'

    save_dir.mkdir(parents=True, exist_ok=True)
    batch_size=2
    imgsz=[640, 640]
    mosaic=False

    hyp = '../../data/hyps/hyp.scratch-low.yaml'
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    degrees, translate, scale, shear, perspective = hyp['degrees'],hyp['translate'],hyp['scale'],hyp['shear'],hyp['perspective']
    augment= False
    hgain, sgain, vgain, flipud, fliplr =hyp['hsv_h'],hyp['hsv_s'],hyp['hsv_v'],hyp['flipud'],hyp['fliplr']
    # create_dataset = CreateDataset(imgsz, mosaic, augment, degrees, translate, scale, shear, perspective, hgain, sgain, vgain, flipud, fliplr)
    # ds = create_dataset(image_files, labels, segments)
    batch_size=2
    debug_dataloader=True
    mask_ratio=4
    ds, ds_len = create_dataloader(data_path, batch_size,imgsz,mask_ratio, mosaic, augment,  hyp, debug_dataloader)


    print(f"Results saved to {save_dir}")

    classes_name_file = '/home/ronen/devel/PycharmProjects/tf_yolov5/data/class-names/coco.names'
    class_names = [c.strip() for c in open(classes_name_file).readlines()]

    data_dict={'nc':80, 'names': class_names}

    amp=False
    model='TBD'
    single_cls=False
    val_loader='val_loader'
    save_dir='segment/runs/train-seg/exp' # /ex42'
    callbacks='tbd'
    compute_loss='compute_loss'
    mask_ratio=4
    overlap=True

    cfg = '/home/ronen/devel/PycharmProjects/tf_yolov5/models/segment/yolov5s-seg.yaml'
    keras_model, tf_model = create_model(cfg)

    weights_load_path = ROOT / '../models/keras_weights/rr.tf' # 'models/keras_weights/rr.tf',  # used if load_model=False

    load_weights=True
    if load_weights:  # normally True when load_model is false
        keras_model.load_weights(weights_load_path)

    anchors=tf_model.anchors
    stride =[32,16,8]


    anchors = tf.reshape(tf_model.anchors, [len(tf_model.anchors), -1, 2]) # shape: [nl, na, 2]
    anchors = tf.cast(anchors / tf.reshape(stride, (-1, 1, 1)), tf.float32) # scale by stride to nl grid layers


    nl = anchors.shape[0] # number of layers (output grids)
    na = anchors.shape[1]  # number of anchors
    nc = len(class_names)-1
    nm=32
    compute_loss = ComputeLoss( na,nl,nc,nm, anchors, hyp['fl_gamma'], hyp['box'], hyp['obj'], hyp['cls'], hyp['anchor_t'], autobalance=False)  # init loss class

    best_fitness, start_epoch = 0.0, 0
    epochs=2

    # save_dir = str(increment_path(Path(save_dir) , exist_ok=opt.exist_ok))
    save_dir = increment_path(Path(f'{ROOT}/runs/tests/') / name, exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)


    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        mloss = tf.zeros([4], dtype=tf.float32)  # mean losses

        results, maps, _ = validate.run(ds,
                                        data_dict,
                                        batch_size=batch_size,
                                        imgsz=imgsz,
                                        half=amp,
                                        model=keras_model,
                                        single_cls=single_cls,
                                        # dataloader=val_loader,
                                        save_dir=save_dir,
                                        plots=False,
                                        callbacks=callbacks,
                                        compute_loss=compute_loss,
                                        mask_downsample_ratio=mask_ratio,
                                        overlap=overlap)

        import numpy as np
        ####
        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        # stop = stopper(epoch=epoch, fitness=fi)  # early stop check
        if fi > best_fitness:
            best_fitness = fi
        # log_vals = list(mloss) + list(results) + lr
        # # callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
        # # Log val metrics and media
        # metrics_dict = dict(zip(KEYS, log_vals))
        # logger.log_metrics(metrics_dict, epoch)
        #
        # # Save model
        # if (not nosave) or (final_epoch and not evolve):  # if save
        #     ckpt = {
        #         'epoch': epoch,
        #         'best_fitness': best_fitness,
        #         'model': deepcopy(de_parallel(model)).half(),
        #         'ema': deepcopy(ema.ema).half(),
        #         'updates': ema.updates,
        #         'optimizer': optimizer.state_dict(),
        #         'opt': vars(opt),
        #         'git': GIT_INFO,  # {remote, branch, commit} if a git repo
        #         'date': datetime.now().isoformat()}
        #
        #     # Save last, best and delete
        #     torch.save(ckpt, last)
        #     if best_fitness == fi:
        #         torch.save(ckpt, best)
        #     if opt.save_period > 0 and epoch % opt.save_period == 0:
        #         torch.save(ckpt, w / f'epoch{epoch}.pt')
        #         logger.log_model(w / f'epoch{epoch}.pt')
        #     del ckpt
        #     # callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)
