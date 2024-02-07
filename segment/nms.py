import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior() # allows running NumPy code, accelerated by TensorFlow


from utils.tf_general import (xywh2xyxy)

# @tf.function
def non_max_suppression(pred, conf_thres,iou_thres, classes=None, agnostic=False, multi_label=False,labels=(), max_det=300, nm = 32,):
    """
    Calc non_max_suppression
    :param pred: all preds, shape: [25200, no], where no=4+1+nc+nm, where nm=32 and nc (coco)=80=>117
    :param conf_thres: nms conf threshold, detections bbelow threshold are excluded. float
    :param iou_thres: nms iou threshold. Detectons with iou above thresh wrt a selected detection, are excluded. float
    :param classes:
    :param agnostic: class agnostic. If False, iou thresholding is taken only for same class detections. Bool
    :param multi_label: Multi class for same detection, e.g. animal, cat. Bool
    :param labels:
    :param max_det: Limit for max detections
    :param nm:
    :return: nms_pred, shape: [Np_nms,38] (bbox+conf+class+nmn)
    :rtype:
    """
    # 1. filter out entries below conf thresh:
    xc = pred[..., 4] > conf_thres  # filter above thersh
    pred = pred[xc]  # confidence
    # 1. filter out entries below conf thresh:
    pred_class_conf = pred[:, 5:] * pred[:, 4:5]  # conf = obj_conf * cls_conf
    pred = tf.concat([pred[:, :5], pred_class_conf], axis=1)

    # 2. subst class nc confidence fields by class id: In case of ulti label, duplicate entry for classes above conf
    # thresh. Otherwise, select class by argmax:
    boxes = xywh2xyxy(pred[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)

    mi = pred.shape[1] -nm  # mask start index
    mask = pred[:, mi:]  # shape: [N,32]
    if multi_label: # take all labels above threshold:
        inds=tf.where(pred[:,5:mi]> conf_thres).T #  [entry_id, class_id] of Nthresholded_class entries. shape: [2, Nclass_thresh]
        i, j = inds[0], inds[1] # entry_id shape:[Nclass_Thresh], class_id shape:[Nclass_Thresh]
        # construct: bbox, classes, mask
        pred = tf.concat((boxes[i],  pred[i, 5 + j][..., None], j[:, None].astype(tf.float32), mask[i]), 1) #shape: [Nclass_thresh, 38] (box+conf+class+nm)
    else:  # best class only
        class_sel_idx = tf.math.argmax(pred[:, 5:mi], axis=-1, output_type=tf.int32)[
            ..., tf.newaxis].astype(tf.float32)  # take idx of preds' max prob classes
        pred = tf.concat((boxes, pred[:,4:5],class_sel_idx, mask), axis=1)# Concat b4 index gather:box,conf,class,mask,shape: [Np,38]

    max_wh = 7680  # (pixels) maximum box width and height - prevents iou overlap

    # 3. IF non class agnostic, separate bboxes with different classes by max_wh, to avoid iou overlap:
    c = pred[:, 5:6] * (0 if agnostic else max_wh)  # if not class agnostic, set an offset per class
    boxes = pred[:, :4] + c  # add offset per class to avoid iou overlap between interlcass boxes

    scores =  pred[:, 4]  # add offset per class to avoid iou overlap between interlcass boxes

    ind = tf.image.non_max_suppression(boxes, scores, max_output_size=max_det, iou_threshold=iou_thres,
                                       score_threshold=conf_thres)
    # Npw gather selected indices from pred:
    nms_pred = tf.gather(pred, indices=ind)

    return nms_pred


