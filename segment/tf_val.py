# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 segment model on a segment dataset

Usage:
    $ bash data/scripts/get_coco.sh --val --segments  # download COCO-segments val split (1G, 5000 images)
    $ python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate COCO-segments

Usage - formats:
    $ python segment/val.py --weights yolov5s-seg.pt                 # PyTorch
                                      yolov5s-seg.torchscript        # TorchScript
                                      yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s-seg_openvino_label     # OpenVINO
                                      yolov5s-seg.engine             # TensorRT
                                      yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                      yolov5s-seg_saved_model        # TensorFlow SavedModel
                                      yolov5s-seg.pb                 # TensorFlow GraphDef
                                      yolov5s-seg.tflite             # TensorFlow Lite
                                      yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                      yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# import torch
import tensorflow as tf
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# import torch.nn.functional as F

# from models.common import DetectMultiBackend
# from models.yolo import SegmentationModel
# from utils.callbacks import Callbacks
from utils.tf_general import (LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, Profile,
                             coco80_to_coco91_class, colorstr, increment_path,
                            print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.tf_metrics import ConfusionMatrix, box_iou
from utils.tf_plots import output_to_target, plot_val_study
# from utils.segment.dataloaders import create_dataloader
from utils.segment.tf_general import mask_iou, process_mask, process_mask_native, scale_image
from utils.segment.tf_metrics import Metrics, ap_per_class_box_and_mask
from utils.segment.tf_plots import plot_images_and_masks
# from utils.torch_utils import de_parallel, select_device, smart_inference_mode
from .nms import non_max_suppression


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = tf.convert_to_tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(tf.convert_to_tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map, pred_masks):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    # from pycocotools.mask import encode

    def single_encode(x):
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5),
            'segmentation': rles[i]})


def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    """
    Returns a tp (true positive) vector  of tp_i entries, i=0:9. tp_i is true if iou(gt_val,pred_val)>iouv[i], i=0:9 and
    gt_class==pred_class.
    shape of the correct vector with N entries : [N, 10]
    Detection can be either a mask or a bbox, and iou is computed accordingly.

    Arguments:
        detections (array[Np, 6]), x1, y1, x2, y2, conf, class
        labels (array[nl, 5]), class, x1, y1, x2, y2
        iouv (array [10]), iou vector for mAP@0.5:0.95  linspace(start: 0.5, end: 0.95,steps: 10)
        pred_masks: shape: A mask map per each of the N predictions  1-object pixel 0-background, shape:[Np,h/4, w/4]
        gt_masks: shape: [1,h/4, w/4], mask target pixels marked 1:Nt or 0 if pixel is not of a target object
        masks: bool, if True calculate tp_m else tp_box

    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    # part 1: calculate iou between all target and preds, masks or bboxes according to masks flag.
    if masks:
        # part 1 (if masks): calculate iou between target masks and pred masks.
        # 1.a Extract object masks from the single gt_mask, where nl masks are marked by nl colors
        nl = len(labels) # num of target labels
        colors = tf.reshape(tf.range(nl), [nl, 1, 1]) + 1  # create nl slices valued 1:nl. shape: [nl,1,1]
        gt_masks = tf.tile(gt_masks, (nl, 1, 1)) # duplicate mask nl times. shape:[nl,h/4,w,4]
        gt_masks = tf.where(gt_masks == colors, 1.0, 0.0)  # nl masks by color. set mask pixels to 1.shape:[nl,h/4,w,4]
        # 1.b perform iou between nl gt_masks and Npi pred masks. resultant shape: [nl,Npi]
        if gt_masks.shape[1:] != pred_masks.shape[1:]: # if diff sizes - rescale
            gt_masks = tf.image.resize(gt_masks[...,None], pred_masks.shape[1:])
            gt_masks = gt_masks.gt_(0.5) # thresh after resize interpolation
        # iou gt prepare: flatten gt masks shape to shape: [nl, w/4*h/4):
        gt_mask_reshaped = tf.reshape(gt_masks, (gt_masks.shape[0], -1))
        # iou pred prepare: flatten pred masks shape to shape: [np, w/4*h/4):
        pred_mask_reshaped = tf.reshape(pred_masks, (pred_masks.shape[0], -1))
        # iou between nl and Npi pred masks. iou resultant shape: [nl,Npi]
        iou = mask_iou(gt_mask_reshaped, pred_mask_reshaped) # shape: [nl, Np]
    else:  # boxes
        # part 1 (if boxes): calculate iou between target bboxes and pred bboxes.
        iou = box_iou(labels[:, 1:], detections[:, :4]) # iou of Nti tboxes and Npi pboxes. resultant shape: [nl,Np]
    # part 2: init `correct` shape[Np,10] to False, 10 tp entries per each prediction. tp_i=True if iou>iouv[i] thresh.
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool) # shape: [Np, nmAp], nmAp=10. Init vals: False
    correct_class = labels[:, 0:1] == detections[:, 5] # shape: [Nti, Npi] True if tclass=pclass

    # part 3: Loop on 10 thresholds 0.5<=iouv[i]<=0.95, determine tp per each
    #  on thresholds.`matches`shape: [N_match, 3], keeps[label_ind, pred_ind, iou_v] for thresholded iou vals.
    # set True value in `correct` according to `matches` unique pred_ind.

    for i in range(len(iouv)):
        # x holds survived ious indices (i,j). ie label and pred indices. shape: [n,2], where nof thresh passed entries
        x = tf.where(tf.math.logical_and((iou >= iouv[i]) , correct_class))
        if x.shape[0]: # if any iou thresh survivors:
            # `matches`holds surviers ious, concats (i,j,iou) of all n thresh surviers. shape[n,3]:
            matches = (tf.concat((x.astype(tf.float32),tf.gather_nd(iou, x)[..., None]), 1).numpy())
            # if x[0].shape[0] > 1: #
            # Remove entries of label duplicates, if label ind contained in multi entries, select that with biggest iou
            matches = matches[matches[:, 2].argsort()[::-1]]  #assending sort by iou, b4 unique takes latest occurance.
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # unique takes latest occurances
            # Remove entries of preds duplicates, if pred ind contained in multi entries, select that with biggest iou
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]] # unique takes latest occurances
            correct[matches[:, 1].astype(int), i] = True # set correct[Np,10] True in entries pointed by matched preds
    return tf.convert_to_tensor(correct, dtype=tf.bool) # correct[Np,10] tf.bool is True for matching pred entries


# @smart_inference_mode()
def run(
        dataloader,
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val-seg',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        save_dir=Path(''),
        plots=True,
        overlap=False,
        mask_downsample_ratio=1,
        compute_loss=None,
        callbacks=None,
):
    if save_json:
        # check_requirements('pycocotools>=2.0.6')
        process = process_mask_native  # more accurate
    else:
        process = process_mask  # faster

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        # pt, jit, engine = True, False, False  # get model device, PyTorch model
        # half &= device.type != 'cpu'  # half precision only supported on CUDA
        # model.half() if half else model.float()
        nm = 32 # Todo ronen de_parallel(model).model[-1].nm  # number of masks

        # Data
        # data = check_dataset(data)  # check

    # Configure
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = tf.linspace(0.5, 0.95, 10)  # vector for mAP calculated for ious 0.5:0.95
    niou = tf.size(iouv) # len of iouv vector   # Dataloader

    seen = 0 # loop on predictions counter
    confusion_matrix = ConfusionMatrix(nc=nc) #
    names =  data['names'] # model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95)", "Mask(P", "R",
                                  "mAP50", "mAP50-95)")
    dt = Profile(), Profile(), Profile() # timing profilers
    metrics = Metrics()
    loss = tf.zeros([4] ) # 4 loss sources: [lbox, lseg, lobj, lcls]
    jdict, stats = [], []
    # callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    # batch loop on gt dataloader entries. batch size: b
    # shape: batch_targets, shape:[Nt,6], batch_masks, shape:[b,h/4,w/4], paths of img src, shape:[b]
    # shapes: shape0, shape old/shape new, pad:[b,3,2]
    for batch_i, (batch_im, batch_targets,  batch_masks, paths, shapes) in enumerate(pbar):# dataset batch by batch loop
        # next loop concats bidx to targets: ragged [b,nt,5] -> list size Nt of: [bidx, class, bbox4]
        new_targets = []
        for idx, targets in enumerate(batch_targets): # loop on targets' batch
            if targets.shape[0]: # escape empty targets
                bindex = tf.cast([idx], tf.float32)[None]
                bindex = tf.tile(bindex, [targets.shape[0], 1]) # shape: [nt,1] , nt num of example's targets
                new_targets.extend(tf.concat([bindex, targets.to_tensor()], axis=-1))  #[bindex,cls, xywh] shape: [nt,6]
        # list size Nt: [bidx, class, bbox4]-> tensor[Nt, 6]
        batch_targets = tf.stack(new_targets, axis=0) # stack all targets. shape:[Nt,6], Nt sum of all batches targets

        nb, height, width, _ = batch_im.shape  # batch size, channels, height, width

        # inference + profiler:
        with dt[0]:
            # inference outputs 3 objects:
            # 1.train_out is a list of 3 tensors per 3 out layers. shape:[bsize,gyi,gxi,na,xywh+conf+nc+nm] where
            # gyi,gxi=size/8,/16,/32 for i=0:2, na:nof anchors (=3), nc: nof classes
            # 2. preds is a train_out but bbox and conf are post-processed and packed. shape:[b,Np,xywh+conf+cls+masks]
            # where Np: (na*sum(gyi*gxi)) i=0:2]
            # 3. proto holds 32 proto masks. shape: [b,32,size/4,size/4]
            preds, protos, train_out = model(batch_im)


        with dt[1]:
            # Loss:
            loss += compute_loss((train_out, protos), batch_targets, batch_masks)[1]  # [lbox, lseg, lobj, lcls]

        # NMS
        tbboxes = batch_targets[:, 2:] * (width, height, width, height) # scale tbbox

        batch_targets = tf.concat([batch_targets[:, 0:2], tbboxes], axis=-1) # re-concat targets [si, cl, bbox]
        # lb =  []  # for autolabelling
        plot_masks = []  # batch masks for plotting
        # Calc stats - a list of size contains tp-bbox and tp-masks.  ize list of bboxes and masks mAp
        # svae text and json and plots per prediction according to config flags.
        list_preds=[]
        for si, (pred, proto) in enumerate(zip(preds, protos)): # loop on preds batch - image by image
            with dt[2]: # nof outputs limitted by max_det:
                b, h, w, ch = tf.cast(batch_im.shape, tf.float32)  # dataset images shape: batch,height,width,channel
                pbboxes = pred[..., :4] * [w, h, w, h]  # xywh scale pred bboxes
                pred = tf.concat([pbboxes, pred[..., 4:]], axis=-1) # pack scaled preds [back. shape: [Nt, 4+1+1+32]
                pred=non_max_suppression(pred, conf_thres, iou_thres, multi_label=False, agnostic=single_cls, max_det=max_det) #shape:[Nt, 38] (xyxy+conf+cls+32masks)

            # scale nms selected pred boxes:
                # b, h, w, ch = tf.cast(batch_im.shape, tf.float32)  # dataset images shape: batch,height,width,channel
                # pbboxes = pred[..., :4] * [w, h, w, h]  # xywh scale pred bboxes
                # pred = tf.concat([pbboxes, pred[..., 4:]], axis=-1) # pack scaled preds [back. shape: [Nt, 4+1+1+32]
            # Metrics

            labels = batch_targets[batch_targets[:, 0] == si, 1:] # pick gt labels with sample_idx=si to match preds[si]
            nl, npr = labels.shape[0], pred.shape[0]  # ntlabels and npreds of batche's si'th element. shapes:[Nti, Npi]
            path, old_shape = Path(str(paths[si])), shapes[si][0] # target paths for out info, orig shapes for rescale
            correct_masks = tf.zeros([npr, niou], dtype=tf.bool)  # init masks-tp (true positive)
            correct_bboxes = tf.zeros([npr, niou], dtype=tf.bool)  # init boxes-tp
            seen += 1 # loop counter
            # if no nms selected preds and/or no related tlabels, store zero'ed stats:
            if npr == 0:# if no nms selected preds remained
                if nl: # ntlabels not 0, i.e. expected targets for this bn. store zerod stats.
                    stats.append((correct_masks, correct_bboxes, *tf.zeros((2, 0)), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Masks
            midx = [si] # mask idx
            # gt masks:
            gt_masks = batch_masks[midx] # ground truth masks for the si-th pred. shape: [1,h/4. w/4] i.e. [1,160,160]

            # pred masks: calc mask=mask@proto and crop to dounsampled by 4 predicted bbox bounderies:
            pred_masks = process(proto, pred[:, 6:], pred[:, :4], shape=batch_im[si].shape[:2]) # shape: [Npi, h/4,w/4]

            # Predictions
            if single_cls:
                pred[:, 5] = 0 # set class to 0
            predn =pred
            # scale pred bboxes - remove padding, scale by orig size factor & clip to original size:
            bboxes = scale_boxes(batch_im[si].shape[:2], predn[:, :4], old_shape, shapes[si][1:3])  # native-space pred
            predn = tf.concat([bboxes, predn[:, 4:]], axis=-1) # re-concat to shape: [Nt, 38]

            # Evaluate
            if nl: # if any tbboxes:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes shape: [Nt,4]
                # scale tbboxes - remove padding, scale by orig shape, clip:
                tbox=scale_boxes(batch_im[si].shape[1:], tbox, old_shape, shapes[si][1:3])  # native-space labels
                labelsn = tf.concat((labels[:, 0:1], tbox), 1)  # native-space labels, entry:[tclass, tbox] shape:[Nt,5]
                # Find bboxes tp (true positive) preds. result type: bool shape: [Npi, nMap] , where nmAp=10:
                correct_bboxes = process_batch(predn, labelsn, iouv)
                # Find masks tp (true positive) preds. result type: bool shape: [Npi, nMap] , where nmAp=10:
                correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap=overlap, masks=True)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)

            #  append per current pred: [tp-bbox, tp-masks, pclass, pconf, tclass]:
            stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))

            pred_masks = tf.cast(pred_masks, dtype=tf.uint8) # shaoe: [Npi, h/4,w/4]
            if plots and batch_i < 3: # plot masks of first 3 examples
                plot_masks.append(pred_masks[:15])  # filter top 15 pred objects to plot

            # Save/log Todo ronen support this logs:
            if save_txt:
                save_one_txt(predn, save_conf, old_shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                pred_masks = scale_image(batch_im[si].shape[1:],
                                         pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), old_shape, shapes[si][1])
                save_one_json(predn, jdict, path, class_map, pred_masks)  # append to COCO-JSON dictionary
            # arrange preds in a target-like flattened, preds marked by si_tag, for calling plot_images_and_masks().
            box, conf, cls = tf.split(pred[:, :6], (4, 1, 1), axis=1)
            xywh=xyxy2xywh(box)
            si_tag = tf.fill(cls.shape, si).astype(tf.float32) # tags pred index. shape: [np], val: pred index
            pred=tf.concat((si_tag, cls,xywh , conf), axis=1) # pred: [Npi,(si,cls,xywh,conf)] shape:[Npi, 7]
            if pred.shape[0]:
                list_preds.append(pred[:15])  #  keep top 15 images to plot
            # end pred in preds-batch loop
        # Plot images
        if plots and batch_i < 3:
            arrange_pred = tf.concat(list_preds, axis=0) if list_preds else tf.constant([]) # flattened preds arry (preds indexed by si_tag).shape: [Np,7]
            if len(plot_masks):
                plot_masks = tf.concat(plot_masks, axis=0) # concat batch preds' top 15 masks. shape:[Np*15, h/4,w/4]
            plot_images_and_masks(batch_im, batch_targets, batch_masks, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names) # targets
            plot_images_and_masks(batch_im, arrange_pred, plot_masks, paths,
                                  save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred
    # end dataset batches loop
    # callbacks.run('on_val_batch_end')
    # Compute metrics.
    # Rearrange stats: list [Np] to list [5]. Details: stat entry (list of npreds entries: [tp_m,tp_b,conf,pcls,tclss]

    # stat entry shape: [[[Npi,10],[Npi,10],[Npi],[Npi],[Nti]] for i=0:npreds]->[[sum(Npi),10],[sum(Npi),10],[sum(Npi)],[sum(Npi)],[sum(Nti)]]
    stats = [tf.concat(x, 0).numpy() for x in zip(*stats)]
    if len(stats) and stats[0].any(): # todo run this
        # Returns [tp,fp,precision,recall,f1,ap,unique classes]  per bbox&masks"
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names)
        metrics.update(results)
    nt = np.bincount(stats[4].astype(int), minlength=nc)  # list[nc], histogram of number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 8  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), *metrics.mean_results()))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i)))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms inference, %.1fms loss, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    # callbacks.run('on_val_end')

    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = metrics.mean_results()

    # Save JSON
    if save_json and len(jdict):

        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path('../datasets/coco/annotations/instances_val2017.json'))  # annotations
        pred_json = f'{save_dir}/{w}_predictions.json'  # predictions
        pred_json=str(ROOT / f'{pred_json}')
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            results = []
            for eval in COCOeval(anno, pred, 'bbox'), COCOeval(anno, pred, 'segm'):
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # img ID to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                results.extend(eval.stats[:2])  # update results (mAP@0.5:0.95, mAP@0.5)
            map_bbox, map50_bbox, map_mask, map50_mask = results
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    # model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    final_metric = mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask
    # return: final_metric, [boxloss, objloss, clsloss, maskloss]/len(dataloader), ap[nc], t[3]
    return (*final_metric, *(loss / len(list(dataloader))).tolist()), metrics.get_maps(nc), t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128-seg.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    # opt.data = check_yaml(opt.data)  # check YAML
    # opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.warning(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.warning('WARNING ⚠️ --save-hybrid returns high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        # opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)



