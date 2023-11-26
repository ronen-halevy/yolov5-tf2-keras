# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 segment model on a segment dataset
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640  # from pretrained (recommended)
    $ python segment/train.py --data coco128-seg.yaml --weights '' --cfg yolov5s-seg.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.callbacks import Callbacks
from utils.downloads import is_url

from utils.tf_general import (LOGGER, TQDM_BAR_FORMAT,  check_file,
                             check_yaml, colorstr,
                            increment_path,print_args, check_dataset,print_mutation, yaml_save)

from segment.tb import GenericLogger
from utils.tf_plots import plot_evolve, plot_labels
from tf_dataloaders import create_dataloader,LoadImagesAndLabelsAndMasks
from tf_loss import ComputeLoss
from utils.segment.tf_metrics import KEYS, fitness
from utils.segment.tf_plots import plot_images_and_masks, plot_results_with_masks
from utils.tf_utils import (EarlyStopping)


import tensorflow as tf
from tensorflow import keras
import numpy as np
from models.tf_model import TFModel
import segment.tf_val as validate  # for end-of-epoch mAP
from optimizer import LRSchedule


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))



# def collate_fn(img,  y_lables, y_segments):
#     img, label, path, shapes, masks = zip(*batch)  # transposed
#     batched_masks = torch.cat(masks, 0)
#     for i, l in enumerate(label):
#         l[:, 0] = i  # add target image index for build_targets()
#     return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks



def train(hyp, opt, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, pretrained, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, mask_ratio, augment, mosaic = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.pretrained, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.mask_ratio, opt.augment, opt.mosaic
    # callbacks.run('on_pretrain_routine_start')
    # todo to config:
    imgsz = [640, 640] # Todo ronen
    nm = 32 # todo TBD
    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.h5', w / 'best.h5'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # affine params:
    # degrees,translate,scale,shear,perspective = hyp['degrees'],hyp['translate'], hyp['scale'],hyp['shear'],hyp['perspective']
    # augmentation params:
    # hgain, sgain, vgain, flipud, fliplr =hyp['hsv_h'],hyp['hsv_s'],hyp['hsv_v'],hyp['flipud'],hyp['fliplr']


    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    # if RANK in {-1, 0}:
    logger = GenericLogger(opt=opt, console_logger=LOGGER)


    # Config
    plots = not evolve and not opt.noplots  # create plots
    overlap = not opt.no_overlap
    # cuda = device.type != 'cpu'
    # init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # with torch_distributed_zero_first(LOCAL_RANK):
    # with open(data, errors='ignore') as f:
    #     data_dict= yaml.safe_load(f)
    data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']

    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    # is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
    # Model
    dynamic = False
    tf_model = TFModel(cfg=cfg,
                       ref_model_seq=None, nc=nc, imgsz=imgsz, training=True)
    # im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
    im = keras.Input(shape=(None,None, 3), batch_size=None if dynamic else batch_size)

    ch=3

    keras_model = tf.keras.Model(inputs=im, outputs=tf_model.predict(im), name='train')

    val_tf_model = TFModel(cfg=cfg,
                       ref_model_seq=None, nc=nc, imgsz=imgsz, training=False)

    val_keras_model = tf.keras.Model(inputs=im, outputs=val_tf_model.predict(im), name='validation')

    # extract stride to adjust anchors:
    stride =[imgsz[0] / x.shape[2] for x in keras_model.predict(tf.zeros([1,*imgsz, ch]))[0]]
    # keras_model.compile()
    print(keras_model.summary())
    # tf_model.run_eagerly = True
    # pred = keras_model(im)  # forward
    best_fitness, start_epoch = 0.0, 0

    # check_suffix(weights, '.pt')  # check weights
    if pretrained:
        keras_model.load_weights(weights)

    # keras_model.trainable = False # freeze
    # Freeze
    # freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # for k, v in model.named_parameters():
    #     v.requires_grad = True  # train all layers
    #     # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
    #     if any(x in k for x in freeze):
    #         LOGGER.info(f'freezing {k}')
    #         v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay # see notes: https://github.com/ultralytics/yolov5/issues/6757 https://github.com/ultralytics/yolov5/discussions/2452
    # optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])


    # Scheduler   # todo check scheduler issue:
    # if opt.cos_lr:
    #     lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    # else:
    #     lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    # ema = tf.train.ExponentialMovingAverage(decay=0.9999) # todo check ema
    # Resume - TBD todo
    # nc = tf_model.nc  # number of classes
    debug = False # use for dataloaders step-by-set debug
    if debug:
        dataset = LoadImagesAndLabelsAndMasks(train_path, imgsz, mask_ratio, mosaic, augment, hyp)
        dbg_entries=len(dataset)
        for idx in range(dbg_entries):
            ds=dataset[idx]

    train_loader, labels, nb = create_dataloader(train_path, batch_size, imgsz, mask_ratio, mosaic, augment, hyp)
    val_path=train_path # todo debug need a chang2
    val_loader, _ ,_ = create_dataloader(val_path, batch_size, imgsz, mask_ratio, mosaic=False, augment=False, hyp=hyp)

    if not resume:
        if plots:
            plot_labels(labels, names, save_dir)




    anchors = tf.reshape(tf_model.anchors, [len(tf_model.anchors), -1, 2]) # shape: [nl, np, 2]
    anchors = tf.cast(anchors, tf.float32) / tf.reshape(stride, (-1, 1, 1)) # scale by stride to nl grid layers

    nl = anchors.shape[0] # number of layers (output grids)
    na = anchors.shape[1]  # number of anchors

    compute_loss = ComputeLoss(na,nl,nc,nm, stride, anchors, overlap, hyp['fl_gamma'], hyp['box'], hyp['obj'], hyp['cls'], hyp['anchor_t'], autobalance=False)  # init loss class
    stopper, stop = EarlyStopping(patience=opt.patience), False

    # train_dataset = train_dataset.batch(batch_size)
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of lr warmup iterations, max(3 epochs, 100 iterations)
    warmup_bias_lr=hyp['warmup_bias_lr']
    optimizer = tf.keras.optimizers.Adam(learning_rate= LRSchedule( hyp['lr0'], hyp['lrf'], nb, nw, warmup_bias_lr, epochs,False),  weight_decay=hyp['weight_decay'])



    for epoch in range(epochs):
        # pbar = enumerate(train_dataset)
        pbar = tqdm(train_loader, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar

        mloss = tf.zeros([4], dtype=tf.float32)  # mean losses
        # train:
        for batch_idx, (bimages,  btargets, bmasks, paths, shapes) in enumerate(pbar):
            ni = batch_idx + nb * epoch  # number batches (since train start), used to scheduke debug plots and logs

            # Convert targets ragged tensor shape: [b, None,5] to rectangle tensor shape:[nt,imidx+cls+xywh] i.e. [nt,6]
            new_btargets = []
            for idx, targets in enumerate(btargets):
                if targets.shape[0]: # if any target:
                    im_idx=tf.cast([idx], tf.float32)[None]
                    im_idx = tf.tile(im_idx, [targets.shape[0],  1])
                    new_btargets.extend(tf.concat([im_idx, targets.to_tensor()[...,0:]], axis=-1)) # [im_idx,cls, xywh]
                else: # if no targets, zeros([0,6]):
                    targets=tf.zeros([0,6], tf.float32)
                    new_btargets.extend( targets)
            new_btargets=tf.stack(new_btargets, axis=0) # list[nt] of shape[6] to tensor shaoe[nt,6]

            with (tf.GradientTape() as tape):
                # model forward, with training=True, outputs a tuple:2 - preds list:3 & proto. Details:
                # preds shapes: [b,na,gyi,gxi,xywh+conf+cls+masks] where na=3,gy,gx[i=1:3]=size/8,/16,/32,masks:32 words
                # proto shape: [b,32,size/4,size/4]
                pred = keras_model(bimages)
                loss, loss_items = compute_loss(pred, new_btargets, bmasks) # returns: sum(loss),  [lbox, lseg, lobj, lcls]
                #  lbox, lseg, lobj, lcls= tf.split(loss_items, num_or_size_splits=4, axis=-1)

            grads = tape.gradient(loss, keras_model.trainable_variables)
            optimizer.apply_gradients(
                    zip(grads, keras_model.trainable_variables))

            mloss = (mloss * batch_idx + loss_items) / (batch_idx + 1)  # update mean losses

            LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'box_loss', 'obj_loss', 'cls_loss', 'mask_loss','Instances', 'Size'))

            pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                 (f'{epoch}/{epochs - 1}',  *mloss.numpy(), new_btargets.shape[0], bimages.shape[1]))
            #
            # Mosaic plots
            if plots:
                if ni < 3:
                    plot_images_and_masks(bimages,  new_btargets, bmasks, paths, save_dir / f'train_batch{ni}.jpg')
                if ni == 10:
                    files = sorted(save_dir.glob('train*.jpg'))
                    logger.log_images(files, 'Mosaics', epoch)
        # end batch ------------------------------------------------------------------------------------------------
        # classes_name_file = '/home/ronen/devel/PycharmProjects/tf_yolov5/data/class-names/coco.names'
        # class_names = [c.strip() for c in open(classes_name_file).readlines()]

        # data_dict = {'nc': 80, 'names': class_names}
        val_keras_model.set_weights(keras_model.get_weights())
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop

        # results, list[12] - mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask,  box_loss, obj_loss, cls_loss, mask_loss
        # maps: array[nc]:  ap-bbox+ap-masks per cla
        results, maps, _ = validate.run(val_loader,
                                        data_dict,
                                        batch_size=batch_size,
                                        imgsz=imgsz,
                                        half=False, # half precision model
                                        model=val_keras_model, # todo use ema
                                        single_cls=single_cls,
                                        save_dir=save_dir,
                                        plots=False,
                                        callbacks=callbacks,
                                        compute_loss=compute_loss,
                                        mask_downsample_ratio=mask_ratio,
                                        overlap=overlap)


        keras_model.save_weights(
                    last)

        # Update best mAP
        # fi=0.1*map50_bbox +0.9*map_bbox+0.1*map50_mask+0.9 map_maskzs
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        stop = stopper(epoch=epoch, fitness=fi)  # early stop check
        if fi > best_fitness:
            best_fitness = fi
        log_vals = list(mloss) + list(results) + [optimizer.learning_rate]
        # callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
        # Log val metrics and media
        metrics_dict = dict(zip(KEYS, log_vals))
        logger.log_metrics(metrics_dict, epoch)

        # Save model
        if (not nosave) or (final_epoch and not evolve):  # if save
            # Save last, best
            keras_model.save_weights(
                last)
            if best_fitness == fi:
                keras_model.save_weights(
                    best)
                # torch.save(ckpt, best)
            if opt.save_period > 0 and epoch % opt.save_period == 0:
                keras_model.save_weights(
                    w / f'epoch{epoch}.tf')
                logger.log_model(w / f'epoch{epoch}.pt')
            # callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping

        if stop:
            break  # must break all DDP ranks

    # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    # if RANK in {-1, 0}:
    LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    val_keras_model.load_weights(best)
    results, _, _ = validate.run(val_loader,
                                    data_dict,
                                    batch_size=batch_size,
                                    imgsz=imgsz,
                                    half=False,  # half precision model
                                    model=val_keras_model,  # todo use ema
                                    single_cls=single_cls,
                                    save_dir=save_dir,
                                    verbose=True,
                                    plots=True,
                                    callbacks=callbacks,
                                    compute_loss=compute_loss,
                                    mask_downsample_ratio=mask_ratio,
                                    overlap=overlap)

    logger.log_metrics(metrics_dict, epochs)

    if plots:
        plot_results_with_masks(file=save_dir / 'results.csv')  # save results.png
        files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
        files = [(save_dir / f) for f in files if (save_dir / f).exists()]  # filter
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
        logger.log_images(files, 'Results', epoch + 1)
        logger.log_images(sorted(save_dir.glob('val*.jpg')), 'Validation', epoch + 1)
    # # torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'utilities/keras_weights/yolov5s-seg.tf', help='initial weights path')
    parser.add_argument('--pretrained', action='store_true', help='load model from weights file')
    parser.add_argument('--cfg', type=str, default='../models/segment/yolov5s-seg.yaml', help='model.yaml path')
    shapes=True
    if shapes:
        parser.add_argument('--data', type=str, default=ROOT / 'data/shapes-seg.yaml', help='dataset.yaml path')
    else:
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128-seg.yaml', help='dataset.yaml path')

    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=40, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-segt', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--augment', action='store_true', help='enable training dataset augmentation')
    parser.add_argument('--mosaic', action='store_true', help='enable training mosaic dataset. mosaic requires augment enabled (tbd-change that?)')

    # Instance Segmentation Args
    parser.add_argument('--mask-ratio', type=int, default=4, help='Downsample the truth masks to saving memory')
    parser.add_argument('--no-overlap', action='store_true', help='Overlap masks train faster at slightly less mAP')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    # Checks
    # if RANK in {-1, 0}:
    print_args(vars(opt))
        # check_git_status()
        # check_requirements(ROOT / 'requirements.txt')

    # Resume
    if opt.resume and not opt.evolve:  # resume from specified or most recent last.pt
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train-seg'):  # if default project name, rename to runs/evolve-seg
                opt.project = str(ROOT / 'runs/evolve-seg')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))



    # Train
    if not opt.evolve:
        train(opt.hyp, opt, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run([
                'gsutil',
                'cp',
                f'gs://{opt.bucket}/evolve.csv',
                str(evolve_csv),])

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 12] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            print_mutation(KEYS[4:16], results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')




def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
