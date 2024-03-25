# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from utils import TryExcept, threaded


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=''):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True) # nt:nof times unqiue value comes
    nc = unique_classes.shape[0]  # number of classes

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes): # loop on classes
        i = pred_cls == c
        n_l = nt[ci]  # number of labels for current unique class
        n_p = i.sum()  # number of predictions  for current unique class
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0) # sum all tp=False for preds of current class preds
        tpc = tp[i].cumsum(0)  # sum all tp=True for preds of current class preds

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve. shape: [Np]
        precision: The precision curve. shape: [Np]
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def find_iou_matched_preds(labels, preds, iou_thresh=0.45):
    """
    Finds matching between preds and gt entries based on best bbox iou values.
    Preds entries are first thresholded by conf threshold then best iou matches are selected as follows:
     - concat [thresholded [label_idx, pred_idx], iou(
     and by bbox iou thresholds, and them
    Best iou unique couples are

    :param preds: Model predictions. shape: [np, 7], entry structure:  [si,cls,xywh,conf] where si: prediction index in batch
    :param labels: dataset's gt labels. shape: [nt, 5], entry structure: [class, x, y, w, h]

    :param preds: Model predictions. shape: [np, 7], entry structure:  [si,cls,xywh,conf] where si: prediction index in batch
    :param labels: dataset's gt labels. shape: [nt, 5], entry structure: [cls, x, y, w, h]
    :param iou_thresh: bbox iou threshold, float
    :return:
        m0:matched labels indices shape:[nt_threshold], int
        m1:matched dets indices, shape[nt_threshold], int
    :rtype:
    """

    # 2. Compute iou between all nl labels boxes and np pred boxes:
    iou = box_iou(labels[:, 1:], preds[:, :4])  # iou(tbbox, pbbox) , Shape [nl,np]
    # 3 Theshold results:
    x = tf.where(iou > iou_thresh)  # thresh. survived [label_idx, pred_idx] . shape: [N, 2],
    if x.shape[0]:  # if any matches survived thresh:
        # 4. arrange matches:  [N_survivors, (iou,index_label,index_pred)]
        matches = tf.concat([x.astype(tf.float32), iou[x[:, 0], x[:, 1]][:, None]],
                            axis=1)  # concat (label_idx, pred_idx,iou), shape:[N,3]
        if x[0].shape[0] > 1: # if multiple survivors, cancel duplicates if any:
            # 5. remove duplicates: each label or detection may belong to a single match entry. Filter unique:
            matches = matches[tf.argsort(matches[:, 2])[
                              ::-1]]  # sort by iou and reverse, since unique takes first occurance. shape[N,3]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # keep unique preds matches
            matches = matches[
                tf.argsort(matches[:, 2])[::-1]]  # sort by iou and reverse, since unique takes first occurance
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # take unique targets. shape[Nf,3]
    else:
        matches = tf.zeros((0, 3), dtype=tf.float32)

    # n = matches.shape[0] > 0
    m0, m1, _ = matches.transpose().astype(tf.int32)  # m0:matched labels indices, m1:matched dets indices, shape[N]
    return m0,m1

def find_matched_classes(preds, labels, nc):
    """
    Finds matching between ground truth class_id labels entries and predicted class_id enitres. Matches are unique,
    selection done according the best bbox iou matcing.
    :param preds: Model predictions. shape: [np, 7], entry structure:  [si,cls,xywh,conf] where si: prediction index in batch
    :param labels: dataset's gt labels. shape: [nt, 5], entry structure: [class, x, y, w, h]
    :param nc: Number of classes. Used as class_id for unmatched prediction (i.e. background) value used as a replacement in the absence of a matched prediction.
    :return:
    gt_classes: list size: nt+n_fd (fd: false detection). Related gt_+labels of fd are set to nc
    matched_pred_class_ids: list size: nt+n_fd, where a miss prediction is filled with nc.


    0. prepare gt_class and matched_pred_class_ids arrays, initiated latter with mc values (i.e 'miss detected')
    1 Threshold preds using conf and iou thresholds    :
    2. find_iou_matched_preds: return matched entries indices to labels and predictions.
    3. 4 scatter matched predicted class ids to related matched labels location:
    4. Add false detection to matched prediction list and fill matched_labels_list peer entries with nc

    """

    # 0.1 extract labels from preds:
    gt_classes = labels[:, 0].astype(tf.int32)
    # 0.2 init matced predicted class id array, filled with nc representing 'miss detection' values:
    matched_pred_class_ids =  np.full(gt_classes.shape, nc)
    # exit if no preds:
    if preds is None:
        return gt_classes, matched_pred_class_ids

    # 1. Threshold preds: filter preds according to predicted conf bbox iou thresholds:
    conf_thresh = 0.25
    thresholded_preds = preds[preds[..., 4] > conf_thresh]  # threshold preds
    # 2.  find_iou_matched_preds
    lmatched_idx, dmatched_idx = find_iou_matched_preds(labels, thresholded_preds, iou_thresh=0.45)

    # 3 scatter matched predicted class ids to related matched labels location:
    detection_classes = thresholded_preds[:, 5].astype(tf.int32) # exrtact predicted class ids:
    matched_pred_class_ids[lmatched_idx]=detection_classes[dmatched_idx] #

    # Add false detection to matched prediction list and fill matched_labels_list peer entries with nc:
    n = lmatched_idx.shape[0]
    if n: # false detections are added to matrix only if any matches (following ultralytics
        # extract false detection by deleting matched detections from list:
        detection_classes_fd=np.delete(detection_classes, dmatched_idx) # remove matched from detection list
        # add the false class id to the preds ids list:
        matched_pred_class_ids = np.concatenate([matched_pred_class_ids, detection_classes_fd])
        # add false detections as 'backrounds' to labels class ids:
        label_clssses_fd = np.full(len(detection_classes_fd), nc)
        gt_classes = np.concatenate([gt_classes, label_clssses_fd])

    return gt_classes, matched_pred_class_ids

import wandb # todo move wandb parts to file
import seaborn as sn


def plot_confusion_matrix(labels, preds, class_names, normalize=False):
    """ Plots a confusuion plot for any 2 entry lists: labels and preds. Plot confusion matrix for labels vs preds.
     Confusion matrix is an ncxnc matrix, where an entry at (x=m,y=n) counts occurrences of label=m.prediction=n

    :param labels: list of labels, len: n_match, int
    :param preds: list of preds len: n_match, int
    :param class_names: list of classnames len:nc, string.
    :return: No returns
    """

    labels = np.array(labels)
    preds = np.array(preds)

    # 1. Construct an empty confusion ncxnc matrix:
    matrix = tf.zeros([len(class_names), len(class_names)])
    # 2. Arrange 2d indices of matching label-preds couples:
    indices = tf.concat([ preds[..., None], labels[..., None]], axis=1)
    # 3 Scatter 1s to each lael-pred index. Use scatter_add to accumulate hits of identical indices.
    matrix=tf.tensor_scatter_nd_add(matrix, indices, tf.ones(indices.shape[0])).numpy()

    # normalize if required - todo tbd:
    array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (appears as a neutralize background color) (relevant for normalize=True)

    # arrange plot:
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    nc =  len(class_names)  # number of classes, names
    sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
    labels = (0 < nc < 99)  # apply names to ticklabels
    ticklabels = class_names  if labels else 'auto'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
        sn.heatmap(array,
                   ax=ax,
                   annot=nc < 30, # annotate cells with data values
                   annot_kws={
                       'size': 8},
                   cmap='Blues',
                   fmt='.2f',
                   square=True,
                   vmin=0.0, # min color
                   xticklabels=ticklabels,
                   yticklabels=ticklabels).set_facecolor((1, 1, 1))
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix')
    wandb.log({"plot": wandb.Image(fig)})
    plt.close


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels): # todo todel
        """
        Updates confusion matrix, with matching detection/label entries:
        1. Threshold detections using conf.
        2. Compute iou (Jaccard index) of all label-detection boxes.
        3. Threshold iou by iou_thres
        4. arrange matched_. [N_survivors, (iou,index_l,index_p)] shape: [N_survivors,3]
        5. remove duplicates from matches: each label or detection may belong to a single match entry.
        6. results: incr confusion `matrix` histogram by 1, at location [det_class, label_class]

        Arguments:
            detections (Array[np, 6]), x1, y1, x2, y2, conf, class
            labels (Array[nl, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly. matrix shape: [nc+1, nc+1]
        """
        if detections is None:
            gt_classes = labels.astype(tf.int32)
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return
        # 1. Threshold detections
        detections = detections[detections[:, 4] > self.conf] # threshold detections
        # 2. Compute iou between all nl labels boxes and np pred boxes:
        iou = box_iou(labels[:, 1:], detections[:, :4])# iou (tbbox, pbbox) , Shape [nl,np]
        # 3 Theshold results:
        x = tf.where(iou > self.iou_thres) # thresh.  N numof thresh survivors. shape: [N, 2],
        if x.shape[0]: # if any matches survived thresh:
            # 4. arrange matches:  [N_survivors, (iou,index_label,index_pred)]
            matches = tf.concat([x.astype(tf.float32), iou[x[:,0], x[:,1]][:, None] ], axis=1) # concat (cordx,cordy,iou) shape[N,3]
            if x[0].shape[0] > 1:
                # 5. remove duplicates: each label or detection may belong to a single match entry. Filter unique:
                matches = matches[tf.argsort(matches[:, 2])[::-1]] #sort by iou since unique takes last occurance.
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # keep unique preds matches
                matches = matches[tf.argsort(matches[:, 2])[::-1]] #sort by iou, since unique takes last occurance
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # take unique targets. shape[Nf,3]
        else:
            matches = tf.zeros((0, 3),dtype=tf.float32)

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(tf.int32) # m0:matched labels indices, m1:matched dets indices, shape[N]
        detection_classes = detections[:, 5].astype(tf.int32)#.int()
        gt_classes = labels[:, 0].astype(tf.int32) # shape: [Nt,1]
        for i, gc in enumerate(gt_classes):
            j = m0 == i # j bool,  shape: [N], true if gc found in matches.
            # if gt_class in matches, incr related dets/gt histogram. Otherwise, incr nonmatch/gt hostogram:
            if n and sum(j) == 1: #  Note that m0 holds unique vals, so sum(j) takes 0 or 1 value
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background
        if n:
            # incr dets/unmatched detection histogram:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    def plot(self, normalize=True, save_dir='', names=()): #todo todel
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close(fig)

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = tf.split(box1, 4, axis=-1), tf.split(box2, 4, axis=-1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        from tensorflow.python.ops.numpy_ops import np_config

        np_config.enable_numpy_behavior()
        b1_x1, b1_y1, b1_x2, b1_y2 = tf.split(box1, 4, axis=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 =  tf.split(box2, 4, axis=-1)
        w1, h1 = b1_x2 - b1_x1, tf.minimum(b1_y2 - b1_y1, eps)
        w2, h2 = b2_x2 - b2_x1, tf.minimum(b2_y2 - b2_y1,eps)

    # Intersection area
    inter = tf.maximum((tf.minimum(b1_x2, b2_x2) - tf.maximum(b1_x1, b2_x1)), 0) * \
            tf.maximum((tf.minimum(b1_y2,b2_y2) - tf.maximum(b1_y1,b2_y1)),0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = tf.maximum(b1_x2, b2_x2) - tf.minimum(b1_x1,b2_x1)  # convex (smallest enclosing box) width
        ch = tf.maximum(b1_y2, b2_y2) - tf.maximum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * tf.math.pow(tf.math.atan(w2 / h2) - tf.math.atan(w1 / h1), 2)
                # with tf.no_gradient():
                alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    (a1, a2) = tf.split(tf.expand_dims(box1, 1), 2,axis=2) # xy_min, xy_max of boxa, shapes each: [N,1,2]
    (b1, b2) = tf.split(tf.expand_dims(box2, 0), 2,axis=2)# xy_min, xy_max of boxb,  shapes each: [1,M,2]
    # find intersection area between N boxes a & M boxes b:
    min_xy_max =  tf.math.minimum(a2, b2) # shape: [N,M,2]
    max_xy_min =  tf.math.maximum(a1, b1) # shape: [N,M,2]
    y = min_xy_max-max_xy_min # Intersections widths and heights shape: [N,M,2]
    y=tf.math.maximum(y,0) # clamp 0  shape: [N,M,2]
    inter = tf.math.reduce_prod(y, 2) # intersection areas w*h shape: [N,M]
    area_a,area_b =  tf.math.reduce_prod(a2 - a1, axis=2), tf.math.reduce_prod(b2 - b1, axis=2) #shape:[N,1],shape:[1,M]
    return inter / (area_a+area_b-inter + eps) # Shape [N,M]



def bbox_ioa(box1, box2, eps=1e-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2, eps=1e-7):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = tf.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------


@threaded
def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
