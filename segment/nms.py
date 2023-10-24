import tensorflow as tf

from utils.tf_general import (xywh2xyxy)

def non_max_suppression(pred, conf_thres,iou_thres, max_det):
    nm = 32  # number of masks
    xywh_nbytes = 4
    obj_nbytes = 1
    nc = pred.shape[1] - (nm + xywh_nbytes + obj_nbytes)  # number of classes
    mi = 5 + nc  # mask start index
    class_sel_prob = tf.reduce_max(pred[:, 5:mi], axis=-1, keepdims=False)
    scores = pred[:, 4] * class_sel_prob
    class_sel_idx = tf.math.argmax(pred[:, 5:mi], axis=-1, output_type=tf.int32)[..., tf.newaxis] # take idx of preds' max prob classes
    class_sel_idx = tf.cast(class_sel_idx, dtype=tf.float32)
    boxes = xywh2xyxy(pred[:, :4])
    # ind = [idx  for idx, score in enumerate(scores) if score>0.5]

    ind = tf.image.non_max_suppression(boxes, scores, max_output_size=max_det, iou_threshold=iou_thres,
                                       score_threshold=conf_thres)
    # Npw gather selected indices from pred:
    pred = tf.concat((boxes, scores[..., tf.newaxis], class_sel_idx, pred[:, mi:]), axis=1)# Concat all before gather
    nms_pred = tf.gather(pred, indices=ind)
    return nms_pred