import cv2
import numpy as np
import tensorflow as tf


def crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [n, h, w] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = tf.split(boxes[:, :, None], num_or_size_splits=4, axis=1)  # x1 shape(n,1,1)
    r = tf.range(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = tf.range(h, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)
    crop =(tf.cast((r >= x1), dtype=tf.float32) * tf.cast((r < x2), dtype=tf.float32) * tf.cast((c >= y1), dtype=tf.float32) * tf.cast((c < y2), dtype=tf.float32))
    return masks * crop # ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

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
    Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """

    ch, mh, mw = protos.shape  # CHW
    ih, iw = shape # image shape

    masks =  tf.sigmoid(masks_in @ tf.reshape(protos, [ch, -1])  )# CHW
    masks = tf.reshape(masks, (-1, mh, mw))

    downsampled_bboxes = tf.concat([bboxes[:, 0:1]* mw / iw, bboxes[:, 1:2]* mh / ih,bboxes[:, 2:3]* mw / iw,bboxes[:, 3:4]* mh / ih], axis=-1)
    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        # masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
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
    # intersection = tf.matmul(mask1, tf.transpose(mask2.t()).clamp(0)
    intersection = tf.maximum(tf.matmul(mask1, tf.transpose(mask2)), 0) # clamp(0)

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
