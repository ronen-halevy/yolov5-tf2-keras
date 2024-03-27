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

import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm
import pathlib
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.callbacks import Callbacks
from utils.downloads import is_url

from utils.tf_general import (LOGGER, TQDM_BAR_FORMAT,  check_file,
                             check_yaml, colorstr,
                            increment_path,print_args, check_dataset,print_mutation, yaml_save, get_latest_run)

from segment.tf_tb import GenericLogger
from utils.tf_plots import plot_evolve, plot_labels
from tf_dataloaders import DataLoader
# from simple_dataset import SimpleDataset

from tf_data_reader import LoadImagesAndLabelsAndMasks

from tf_loss import ComputeLoss
from utils.segment.tf_metrics import KEYS, fitness
from utils.segment.tf_plots import plot_images_and_masks, plot_images_and_masks2, plot_results_with_masks
from utils.tf_utils import (EarlyStopping)


import tensorflow as tf
from tensorflow import keras

import numpy as np
from models.tf_model import TFModel#  todo remove old TFmodel
from models.build_model import build_model, Decoder

import segment.tf_val as validate  # for end-of-epoch mAP
from optimizer import LRSchedule

from tf_config import parse_opt
from tf_train_utils import flatten_btargets
from utils.tf_autoanchor import check_anchors

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp, opt, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, pretrained, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, mask_ratio, augment, mosaic, anchors_data, debug = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.pretrained, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.mask_ratio, opt.augment, opt.mosaic, opt.anchors_data, opt.debug
    if debug:
        tf.config.run_functions_eagerly(True)
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
    results = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)


    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    # if RANK in {-1, 0}:
    logger = GenericLogger(opt=opt, res_table_cols=KEYS, console_logger=LOGGER)


    # Config
    plots = not evolve and not opt.noplots  # create plots
    overlap =  opt.overlap
    # cuda = device.type != 'cpu'
    # init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # with torch_distributed_zero_first(LOCAL_RANK):
    # with open(data, errors='ignore') as f:
    #     data_dict= yaml.safe_load(f)
    data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']

    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    class_names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    # is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
    # Model
    dynamic = False

    # read anchors:
    with open(anchors_data, 'r') as stream:
        anchors = yaml.safe_load(stream)['anchors']
    na=3 # nof anchors per layer
    anchors = tf.reshape(anchors, [-1, na,2]) # shape: [nl,na,2]
    nl =anchors.shape[0]

    keras_model=build_model(cfg,  nl, na, imgsz=imgsz)
    print(keras_model.summary())

    # decoder = Decoder(nc, nm, anchors, imgsz)
    # extract 3 layers grid shapes strides:
    grids =[[80,80],[40,40],[20,20]] # todo grids and strides from config
    strides = [8.,16.,32.]
    grids = tf.constant(grids)#  todo arranmge configl


    best_fitness, start_epoch = 0.0, 0

    if pretrained or resume:
        LOGGER.info(f'Loading pretrained weights from {colorstr("bold", weights)} ')
        keras_model.load_weights(weights)
    # if resume: # Todo - in addition to picking weights, pick data from resumed log
    #     best_fitness, start_epoch, epochs = opt.best_fitness, opt.start_epoch, opt.epochs


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
    if debug:
        dataset = LoadImagesAndLabelsAndMasks(train_path, imgsz, mask_ratio, mosaic, augment, hyp, overlap, debug)
        dbg_entries=len(dataset)
        for idx in range(dbg_entries):
            ds=dataset[idx][1]
    create_dataloader=DataLoader()
    # val_path=train_path
    train_loader,  nb, dataset = create_dataloader(train_path, batch_size, imgsz, mask_ratio, mosaic, augment, hyp, overlap)
    create_dataloader_val=DataLoader()
    val_loader ,val_nb,_ = create_dataloader_val(val_path, batch_size, imgsz, mask_ratio, mosaic=False, augment=False, hyp=hyp, overlap=overlap)

    # train_ds=SimpleDataset(imgsz, hyp, overlap, train_path, mask_ratio, mosaic, augment,batch_size)
    # val_ds=SimpleDataset(imgsz, hyp, overlap, train_path, mask_ratio, mosaic, augment,batch_size)
    # train_loader=train_ds.train_loader
    # val_loader=val_ds.train_loader
    # val_nb=nb = train_ds.nb
    anchors = tf.cast(anchors, tf.float32) / tf.reshape(strides, (-1, 1, 1)) # scale 3 layers anchors by layer's strides

    if not resume:
        if not opt.noautoanchor:
            anchors=check_anchors(dataset, strides, anchors, thr=hyp['anchor_t'], imgsz=imgsz[0])  # run AutoAnchor
            LOGGER.info(f'{anchors}')
        if plots:
            labels = tf.concat(dataset.labels, 0)
            plot_labels(labels, class_names, save_dir)  # todo implement this!!

    decoder = Decoder(nc, nm, anchors, imgsz)

    nl = anchors.shape[0] # number of layers (output grids)
    na = anchors.shape[1]  # number of anchors

    # grids = tf.concat([imgsz[0]/tf.constant(strides)[...,None], imgsz[1]/tf.constant(strides)[...,None]], axis=-1)
    compute_loss = ComputeLoss(na,nl,nc,nm, strides, grids, anchors, overlap, hyp['fl_gamma'], hyp['box'], hyp['obj'], hyp['cls'], hyp['anchor_t'], autobalance=False)  # init loss class
    stopper, stop = EarlyStopping(patience=opt.patience), False

    # train_dataset = train_dataset.batch(batch_size)
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of lr warmup iterations, max(3 epochs, 100 iterations)
    warmup_bias_lr=hyp['warmup_bias_lr']
    optimizer = tf.keras.optimizers.Adam(learning_rate= LRSchedule( hyp['lr0'], hyp['lrf'], nb, nw, warmup_bias_lr, epochs,False), ema_momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
    # optimizer = tf.keras.optimizers.SGD(learning_rate= 0.01, ema_momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    # keras_model.compile(optimizer='sgd', loss=compute_loss)
    # keras_model.fit(train_loader, epochs=500, verbose=1)
    # train loop:

    for epoch in range(epochs):
        LOGGER.info(('\n' + '%11s' * 9) % ('Epoch', 'box_loss', 'mask_loss', 'obj_loss','cls_loss','Instances', 'Size', 'lr','gpu_mem'))
        pbar = tqdm(train_loader, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar

        mloss = tf.zeros([4], dtype=tf.float32)  # mean losses
        for batch_idx, (b_images, y_train ) in enumerate(pbar):
            # y_train is unpacked to 4 elements:
            # batch_targets, shape:[Nt,6],
            # b_masks, shape:[b,h/4,w/4],
            # paths of img src, shape:[b]
            # shapes: (shape0, shape old/shape new, paddings), shape :[b,3,2]
            b_targets, b_masks, paths, shapes=y_train
            ni = batch_idx + nb * epoch  # number batches (since train start), used to scheduke debug plots and logs
            # if non-overlap=mask per target, tensor is ragged, shape:[b,None,160,160], otherwise shape is [b, 160,160]
            if not overlap:  # convert ragged shape [b,nti,160,160] to tensor [b*nti,160,160]
                b_masks = tf.reshape(b_masks.flat_values, [-1, 160,160]) # flatten ragged tensor shape: [b, 160,160]

            # Flatten batched targets ragged tensor from shape [b, nti,5] to shape:[nt,imidx+cls+xywh] i.e. [nt,6],
            # ready gor loss calc:
            targets = flatten_btargets(b_targets, tf.shape(b_images)[0]) # shape: [bnt, 6] entry: [bi, cls,bbox]

            with tf.GradientTape() as tape:
                # model forward, with training=True, outputs a tuple:2 - preds list:3 & proto. Details:
                # preds shapes: [b,na,gyi,gxi,xywh+conf+cls+masks] where na=3,gy,gx[i=1:3]=size/8,/16,/32,masks:32 words
                # proto shape: [b,32,size/4,size/4]
                pred = keras_model(b_images, training=True) # Reaining=True essential for bn adaptation.
                loss, loss_items = compute_loss((targets, b_masks),pred)  # returns: sum(loss),  [lbox, lseg, lobj, lcls]
            grads = tape.gradient(loss, keras_model.trainable_variables)
            optimizer.apply_gradients(
                    zip(grads, keras_model.trainable_variables))
            mloss = (mloss * batch_idx + loss_items) / (batch_idx + 1)  # update mean losses

            gpu_devices = tf.config.list_physical_devices('GPU')
            gpu_mem = f'{tf.config.experimental.get_memory_usage("GPU:0") / 1E9 if gpu_devices else 0:.3g}G'

            pbar.set_description(('%11s' * 1 + '%11.4g' * 7 + '%11s' * 1) %
                                 (f'{epoch}/{epochs - 1}', *mloss.numpy(), targets.shape[0], b_images.shape[1], optimizer.lr, gpu_mem))
                                     #
            # Mosaic plots
            if plots:
                if ni < 3:
                    # plot_images_and_masks2(b_images,  targets, b_masks, paths, class_names, f'train_batch', ni)
                    plot_images_and_masks(b_images,  targets, b_masks, paths, save_dir / f'train_batch{ni}.jpg')

                # if ni == 10: todo check that
                #     files = sorted(save_dir.glob('train*.jpg'))
                #     logger.log_images(files, 'Mosaics', epoch)
        # end batch ------------------------------------------------------------------------------------------------
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
        if not noval or final_epoch:  # Calculate mAP
            # results, list[12] - mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask,  box_loss, obj_loss, cls_loss, mask_loss
            # maps: array[nc]:  ap-bbox+ap-masks per cla
            results, maps, _ = validate.run(val_loader,
                                            data_dict,
                                            batch_size=batch_size,
                                            imgsz=imgsz,
                                            nb=val_nb,
                                            half=False, # half precision model
                                            model=keras_model, # todo use ema
                                            decoder=decoder.decoder,
                                            single_cls=single_cls,
                                            save_dir=save_dir,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss,
                                            mask_downsample_ratio=mask_ratio,
                                            overlap=overlap)

        # Update best mAP
        # fi=0.1*map50_bbox +0.9*map_bbox+0.1*map50_mask+0.9 map_maskzs
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        stop = stopper(epoch=epoch, fitness=fi)  # early stop check
        if fi > best_fitness:
            best_fitness = fi
        log_vals = list(mloss.numpy()) + list(results) + [float(optimizer.learning_rate)]
        # callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
        # Log metrics to table:
        metrics_dict = dict(zip(KEYS, log_vals))
        logger.log_metrics(metrics_dict, epoch) # log results to csv
        # Save model
        if (not nosave) or (final_epoch and not evolve):  # if save
            # Save last, best weights:
            keras_model.save_weights(last)
            if best_fitness == fi:
                keras_model.save_weights(best)
            if opt.save_period > 0 and epoch % opt.save_period == 0:
                keras_model.save_weights(w / f'epoch{epoch}.tf')
                logger.log_model(w / f'epoch{epoch}.pt')
            # callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping

        if stop:
            break  # must break all DDP ranks

    # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    # if RANK in {-1, 0}:
    LOGGER.info(f'\n{epochs - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    # todo note: last validation runs with ploits True
    results, _, _ = validate.run(val_loader,
                                    data_dict,
                                    batch_size=batch_size,
                                    imgsz=imgsz,
                                    nb=val_nb,
                                    half=False,  # half precision model
                                    model=keras_model,  # todo use ema
                                    decoder=decoder.decoder,
                                    single_cls=single_cls,
                                    save_dir=save_dir,
                                    verbose=True,
                                    plots=True,
                                    callbacks=callbacks,
                                    compute_loss=compute_loss,
                                    mask_downsample_ratio=mask_ratio,
                                    overlap=overlap)

    # todo originaly last log skipped loss prints:
    # log results to talbe:
    log_vals = list(mloss.numpy()) + list(results) + [float(optimizer.learning_rate)]
    metrics_dict = dict(zip( KEYS, log_vals))
    logger.log_metrics(metrics_dict, epoch) # log metrics to table

    if plots:
        plot_results_with_masks(file=save_dir / 'results.csv')  # save results.png # todo old remove
        files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
        files = [(save_dir / f) for f in files if (save_dir / f).exists()]  # filter
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
        # todo- cancel copy images from log to wandb
        logger.log_images(files, 'Results', epoch + 1)
        logger.log_images(sorted(save_dir.glob('val*.jpg')), 'Validation', epoch + 1)

    return results

# todo - pick data from resumed run:
# # import csv
# # import pandas as pd
# #
# #
# def smart_resume(last, epochs):
#
#     # extract start epoch from resume csv file:
#     try:
#         with open(last.parent.parent / 'results.csv', errors='ignore') as f:
#             header = [h.strip() for h in f.readline().split(',')]
#             reader_obj = csv.DictReader(f, fieldnames=header)
#             for row in reader_obj:  # iterate till las row, to fetch last epoch index
#                 pass
#             epoch= int(row['epoch'].strip())
#             start_epoch =epoch + 1
#             if opt.epochs < start_epoch:
#                 LOGGER.info(
#                     f"{opt.weights} has been already trained for {epoch} epochs. Fine-tuning for {opt.epochs} more epochs.")
#                 opt.epochs += epoch  # finetune additional epochs
#         # find best fitness of all generations:
#         import pandas as pd
#
#         data = pd.read_csv(last.parent.parent / 'results.csv', skipinitialspace=True)
#         data = data.rename(columns=lambda x: x.strip())  # strip keys
#         best_fitness = np.amax(fitness(data.values))
#     except Exception as e:
#         best_fitness = 0
#         start_epoch = 0
#
#     return best_fitness, start_epoch, epochs

def main(opt, callbacks=Callbacks()):
    print_args(vars(opt))

    # Resume
    if opt.resume and not opt.evolve:  # resume from specified or most recent last.pt
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        LOGGER.info(f'Resuming with {colorstr("bold", last)} ')
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        # if opt_yaml.is_file():
        with open(opt_yaml, errors='ignore') as f:
            d = yaml.safe_load(f)
        # else:
        #     d = torch.load(last, map_location='cpu')['opt']
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
            opt.exist_ok, opt.resume, opt.pretrained = opt.resume, False, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)) # anyway save in new dir

    # ##
    # # Resume
    # if opt.resume and not opt.evolve:  # resume from specified or most recent configuration, hypermarams and weights
    #     # 1. take weights specified by opt.resume otherwise take last:
    #     last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run('../runs'))
    #     LOGGER.info(f'Resuming with {colorstr("bold", last)} ')
    #     # # 2. Replace current opt config with resume config file contents, if exists:
    #     # if pathlib.Path.exists(last.parent.parent/ 'opt.yaml'):
    #     #     with open(last.parent.parent / 'opt.yaml', errors='ignore') as f:
    #     #         opt_dict = yaml.safe_load(f)
    #
    #     # opt = argparse.Namespace(**opt_dict)  # convert dict to a Namespace object
    #     # store resume weights:
    #     opt.weights = last
    #     opt.data = check_file(opt.data )  # if url then download url to file
    #
    #     LOGGER.info(f'Resuming with {colorstr("bold", opt.weight)} ')
    #
    #     # # todo fetch from csv resume best_fitness, start_epoch & epochs:
    #     # best_fitness, start_epoch, epochs = smart_resume(last,  opt.epochs)
    #     # opt.epochs = epochs
    #     # opt.start_epoch = start_epoch
    #     # opt.best_fitness = best_fitness
    #     # opt.resume_source = last.parent.parent
    #
    # # else:
    # #     if opt.evolve:
    # #     #     # todo consider separate evolve projects:
    # #         if opt.project == str(ROOT / "runs/train-segt"):  # if default project name, rename to runs/evolve-seg
    # #             opt.project = str(ROOT / "runs/evolve-seg")
    # #         opt.exist_ok, opt.resume, opt.pretrained = opt.resume, False, False  # pass resume to exist_ok and disable resume
    # if opt.name == 'cfg':
    #     opt.name = Path(opt.cfg).stem  # use model.yaml as name
    # opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)) # anyway save in new dir
    #
    # # # log hyp and data yamls, just for contents debug logging:
    # # with open(opt.hyp, errors='ignore') as f:
    # #         hyp = yaml.safe_load(f)
    # # with open(opt.data, errors='ignore') as f: # dataset input paths, anchors, namestbd)
    # #         data = yaml.safe_load(f)
    # # wandb.config.update({ 'hyp': hyp, 'dataset':data})
    # #
    # # assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    # # # if opt.evolve:
    # #     #     if opt.project == str(ROOT / 'runs/train-seg'):  # if default project name, rename to runs/evolve-seg
    # #     #         opt.project = str(ROOT / 'runs/evolve-seg')
    # #     #     opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
    # #
    # # # todo - check modify run dir bane - tbd!!
    # # wandb.run.name = wandb.run.id
    # # wandb.run.save()

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, callbacks)

    # Evolve hyperparameters (optional)
    else:
        #     # todo consider separate evolve projects:
        if opt.project == str(ROOT / "runs/train-segt"):  # if default project name, rename to runs/evolve-seg
            opt.project = str(ROOT / "runs/evolve-seg")
        opt.exist_ok, opt.resume, opt.pretrained = opt.resume, False, False  # pass resume to exist_ok and disable resume

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
            results = train(hyp.copy(), opt,  callbacks)
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
    opt = parse_opt()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)

if __name__ == '__main__':

    opt = parse_opt()
    main(opt)
