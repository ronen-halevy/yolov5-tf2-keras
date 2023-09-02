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
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np


import yaml
# from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.callbacks import Callbacks
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_dataset, check_file, check_git_info,
                             check_yaml, colorstr,
                           get_latest_run, increment_path,
                           print_args, print_mutation, yaml_save)
from utils.loggers import GenericLogger
from utils.plots import plot_evolve, plot_labels
from utils.segment.dataloaders import create_dataloader
from utils.segment.tf_loss import ComputeLoss
from utils.segment.metrics import KEYS, fitness
from utils.segment.plots import plot_images_and_masks, plot_results_with_masks
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)


import tensorflow as tf
from tensorflow import keras
import numpy as np
from models.tf_model import TFModel

from tf_create_dataset import CreateDataset

from utils.segment.polygons2masks import polygons2masks_overlap, polygon2mask

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()



# def collate_fn(img,  y_lables, y_segments):
#     img, label, path, shapes, masks = zip(*batch)  # transposed
#     batched_masks = torch.cat(masks, 0)
#     for i, l in enumerate(label):
#         l[:, 0] = i  # add target image index for build_targets()
#     return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks



def train(hyp, opt, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, mask_ratio = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.mask_ratio
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

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        logger = GenericLogger(opt=opt, console_logger=LOGGER)


    # Config
    plots = not evolve and not opt.noplots  # create plots
    overlap = not opt.no_overlap
    # cuda = device.type != 'cpu'
    # init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # with torch_distributed_zero_first(LOCAL_RANK):
    data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']

    create_dataset=CreateDataset(imgsz[0])
    ds_train=create_dataset(train_path)
    ds_val=create_dataset(val_path)

    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model
    dynamic = False
    tf_model = TFModel(cfg=cfg,
                       ref_model_seq=None, nc=80, imgsz=imgsz, training=True)
    im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
    # tf_model.predict(im)
    # s=640
    ch=3

    keras_model = tf.keras.Model(inputs=im, outputs=tf_model.predict(im))
    # extract stride to adjust anchors:
    stride =[imgsz[0] / x.shape[2] for x in keras_model.predict(tf.zeros([1,*imgsz, ch]))[0]]
    # keras_model.compile()
    print(keras_model.summary())
    # tf_model.run_eagerly = True
    # pred = keras_model(im)  # forward
    best_fitness, start_epoch = 0.0, 0

    # check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.h5')
    if pretrained:
        keras_model.load_weights(weights)


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
    optimizer = tf.keras.optimizers.Adam(learning_rate= hyp['lr0'],  weight_decay=hyp['weight_decay'])


    # Scheduler
    # if opt.cos_lr:
    #     lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    # else:
    #     lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = tf.train.ExponentialMovingAverage(decay=0.9999) # todo

    # nc = tf_model.nc  # number of classes
    anchors = tf.reshape(tf_model.anchors, [len(tf_model.anchors), -1, 2])
    anchors = tf.cast(anchors, tf.float32) / tf.reshape(stride, (-1, 1, 1))

    nl = anchors.shape[0] # number of layers (output grids)
    na = anchors.shape[1]  # number of anchors

    compute_loss = ComputeLoss( na,nl,nc,nm, anchors, hyp['fl_gamma'], hyp['box'], hyp['obj'], hyp['cls'], hyp['anchor_t'], autobalance=False)  # init loss class
    ds_train = ds_train.batch(batch_size)

    for epoch in range(epochs):
        # train:
        for batch, (bimages,  btargets, bfilename, bshape, bsegments) in enumerate(ds_train):
            bmasks, bsorted_idx = polygons2masks_overlap(bimages.shape[1:3],
                                                               bsegments,
                                                               downsample_ratio=mask_ratio)
            bmasks = tf.stack(bmasks, axis=0)  # (b, 640, 640)


            # concat image index word to targets. result targets shape: [nt, 6] where nt total of target objectd
            new_btargets = []
            for idx, (targets, sorted_idx) in enumerate(zip(btargets, bsorted_idx)):
                bindex=tf.cast([idx], tf.float32)[None]
                bindex = tf.tile(bindex, [targets.shape[0],  1])
                new_btargets.extend(tf.concat([bindex, targets.to_tensor()], axis=-1)[sorted_idx]) # [bindex,cls, xywh]

            new_btargets=tf.stack(new_btargets, axis=0)

            with (tf.GradientTape() as tape):
                # im = tf.expand_dims(image,axis=0)
                pred = keras_model(bimages)  # forward
                loss, loss_items = compute_loss(pred, new_btargets, bmasks)
                lbox, lobj, lcls, lseg= tf.split(loss_items, num_or_size_splits=4, axis=-1)

            grads = tape.gradient(loss, keras_model.trainable_variables)
            optimizer.apply_gradients(
                    zip(grads, keras_model.trainable_variables))

            print(
                f'{epoch}_train_{batch}_lr:{optimizer.lr.numpy():.4f}, '
                f'totLoss:{loss.numpy()[0]:.4f}, '
                f'lbox: {lbox.numpy()[0]:.4f}, '
                f'lobj: {lobj.numpy()[0]:.4f}, '
                f'lcls: {lcls.numpy()[0]:.4f}, '
                f'lseg: {lseg.numpy()[0]:.4f}, ')

        keras_model.save_weights(
                    last)




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / '/segment/saved_weights/yolov5s-seg_weights.tf.', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='../models/segment/yolov5s-seg.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/shapes-seg.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
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
    parser.add_argument('--project', default=ROOT / 'runs/train-seg', help='save to project/name')
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

    # Instance Segmentation Args
    parser.add_argument('--mask-ratio', type=int, default=4, help='Downsample the truth masks to saving memory')
    parser.add_argument('--no-overlap', action='store_true', help='Overlap masks train faster at slightly less mAP')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
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
from tensorflow.keras.models import Sequential
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
