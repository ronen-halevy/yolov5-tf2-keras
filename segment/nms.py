import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior() # allows running NumPy code, accelerated by TensorFlow


from utils.tf_general import (xywh2xyxy)


def non_max_suppression(pred, conf_thres,iou_thres,classes=None,agnostic=False, multi_label=False,labels=(), max_det=300, nm = 32):
    nm = 32  # number of masks
    len_xywh = 4 # bbox entry length
    len_conf = 1 #
    nc = pred.shape[1] - (nm + len_xywh + len_conf)  # number of classes
    mi = len_xywh +len_conf + nc  # mask start index
    xc = pred[..., 4] > conf_thres  # filter above thersh
    pred = pred[xc]  # confidence
    max_wh = 7680  # (pixels) maximum box width and height
    pred_cls_mask = pred[:, 5:] * pred[:, 4:5]  # conf = obj_conf * cls_conf
    pred = tf.concat([pred[:, :5], pred_cls_mask], axis=1)

    boxes = xywh2xyxy(pred[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
    mask = pred[:, mi:]  # zero columns if no masks
    if multi_label:
        inds=tf.where(pred[:,5:mi]> conf_thres).T # returns 2d indices of above thresh entries: entry_id, class_id
        i, j = inds[0], inds[1] # entry_id, class_id
        pred = tf.concat((boxes[i],  pred[i, 5 + j][..., None], j[:, None].astype(tf.float32), mask[i]), 1)
    else:  # best class only
        class_sel_idx = tf.math.argmax(pred[:, 5:mi], axis=-1, output_type=tf.int32)[
            ..., tf.newaxis].astype(tf.float32)  # take idx of preds' max prob classes
        pred = tf.concat((boxes, pred[:,4:5],class_sel_idx, pred[:, mi:]), axis=1)# Concat all before gather

    # Batched NMS
    c = pred[:, 5:6] * (0 if agnostic else max_wh)  # if not class agnostic, set an offset per class
    boxes, scores = pred[:, :4] + c, pred[:, 4]  # add offset per class to avoid iou overlap between interlcass boxes

    ind = tf.image.non_max_suppression(boxes, scores, max_output_size=max_det, iou_threshold=iou_thres,
                                       score_threshold=conf_thres)
    # Npw gather selected indices from pred:
    nms_pred = tf.gather(pred, indices=ind)

    return nms_pred


