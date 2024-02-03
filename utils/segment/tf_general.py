import cv2
import numpy as np
import tensorflow as tf


def crop_mask(masks, boxes):
    """
    Crop masks pixels to within bboxes regions.
    :param masks: masks , value range: [0,1], shape:[nt,160,160], tf.float32. Note: actually mask pixels loss values
    :param boxes: target bboxes, value range: [0,160], shape:[nt4], tf.float32
    :return:
    :rtype: cropped masks, value range: [0,1], shape:[nt,160,160], tf.float32
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = tf.split(boxes[:, :, None], num_or_size_splits=4, axis=1)  # coords of n bbox , shapes (n,1,1)
    r = tf.range(w, dtype=x1.dtype)[None, None, :]  # mask rows shape(b,1,w)
    c = tf.range(h, dtype=x1.dtype)[None, :, None]  # mask cols shape(b,h,1)
    crop =(r >= x1).astype(tf.float32) * (r < x2).astype(tf.float32) * (c >= y1).astype(tf.float32) * (c < y2).astype(tf.float32) # shape [nt,160,160]
    cropped_masks = masks * crop  # zero pixels outside bounderies.  ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    return cropped_masks # shape: [nt,160,160], tf.float32

def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    Crop after upsample.
    protos: [mask_dim, mask_h, mask_w]
    masks_in: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape: input_image_size, (h, w)

    return: h, w, n
    """

    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.5)


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Description: alc mask=mask@proto and crop to dounsampled by 4 predicted bbox bounderies. Then threshold mask pixels
    by 0.5
    :param protos: model's output of nm masks' protos, shape:[nm, h/4, w/4], whrere nm=32, w=h=640
    :type protos: tf.float32
    :param masks_in: predicted mask fields, shape: [Np, 32]
    :type masks_in: tf.float32
    :param bboxes: Predicted xyxy boxes, 0<=w,h<=640, shape: [Np,4]
    :type bboxes: tf.float32
    :param shape: target shape tuple (640,640). USed by upsample, currently N/A
    :type shape: (int, int)
    :param upsample: Indicates if to upsample mask to image's size. Currently: False
    :type upsample: Bool
    :return: predicted downsampled by 4 mask, shape: [Np, 160, 160]
    :rtype: tf.float32
    """
    ch, mh, mw = protos.shape  # CHW
    ih, iw = shape # image shape

    masks =  tf.sigmoid(masks_in @ tf.reshape(protos, [ch, -1])  )# CHW
    masks = tf.reshape(masks, (-1, mh, mw))

    downsampled_bboxes = tf.concat([bboxes[:, 0:1]* mw / iw, bboxes[:, 1:2]* mh / ih,bboxes[:, 2:3]* mw / iw,bboxes[:, 3:4]* mh / ih], axis=-1)
    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = tf.image.resize(masks[...,tf.newaxis], size=shape )
    ret_val = tf.math.greater( # 0.5 min limit.
        masks, 0.5, name=None
    )
    return tf.cast(ret_val, tf.float32)

    # return masks.gt_(0.5)


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    Crop after upsample.
    protos: [mask_dim, mask_h, mask_w]
    masks_in: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape: input_image_size, (h, w)

    return: h, w, n
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2  # wh padding
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[:, top:bottom, left:right]

    masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.5)

# scale mask to orig image dims: crop padding margins and resize
def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    # resize w pad? TBD todo
    masks = tf.image.resize(masks[tf.newaxis,...], (im0_shape[0], im0_shape[1]))[0]

    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks


def mask_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [M, n] m2 means number of gt objects
    Note: n means image_w x image_h

    return: masks iou, [N, M]
    """
    intersection = tf.maximum(tf.matmul(mask1, tf.transpose(mask2)), 0) # clamp(0). shape: [Nt,Np]
    union = (tf.math.reduce_sum(mask1, axis=1)[:, None] + tf.math.reduce_sum(mask2, axis=1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def masks_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [N, n] m2 means number of gt objects
    Note: n means image_w x image_h

    return: masks iou, (N, )
    """
    intersection = (mask1 * mask2).sum(1).clamp(0)  # (N, )
    union = (mask1.sum(1) + mask2.sum(1))[None] - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def masks2segments(masks, strategy='largest'):
    # Convert masks(n,160,160) into segments(n,xy)
    segments = []
    for x in masks.numpy().astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype('float32'))
    return segments
