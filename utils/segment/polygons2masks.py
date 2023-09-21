import cv2
import numpy as np
import tensorflow as tf

def polygon2mask(img_size, polygon, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygon = np.asarray(polygon)
    polygon = polygon.astype(np.int32)
    shape = polygon.shape
    polygon = polygon.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygon, color=[255,255,255])
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    mask=tf.image.resize(mask[None][...,None], [ nh, nw])

    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    # mask = cv2.resize(mask, (nw, nh))
    return tf.squeeze(mask)


def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M],
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


# @tf.function
# def polygons2masks_overlap(image_size, segments, downsample_ratio=1):
#     """Return a (640, 640) overlap mask."""
#     # bindex = []
#     # bmasks = []
#     # for segments in bsegments:
#     # masks = np.zeros((tf.math.floordiv(img_size[0], downsample_ratio), tf.math.floordiv(img_size[1], downsample_ratio)),
#     #                  dtype=np.int32 if segments.shape[0] > 255 else np.uint8)
#     # imgsz=640
#     # masks = tf.zeros((tf.math.floordiv(image_size[0], downsample_ratio), tf.math.floordiv(image_size[1], downsample_ratio)),
#     #                  dtype=np.int32 )
#     # areas = []
#     ms = []
#     # downsample_ratio=4
#     for si in range(segments.shape[0]):
#         mask = polygon2mask(
#                 image_size,
#
#                 [ tf.reshape(segments[si], [-1])],
#                 downsample_ratio=downsample_ratio,
#                 color=1,
#             )
#         ms.append(mask)
#         # areas.append(tf.math.reduce_sum(mask))
#     ms = np.asarray(ms) # shape: [nmasks, 160,160]
#     return ms

    # mask_id = np.arange(1, ms.shape[0]+1, dtype=np.float32) # s !ms.shape is only avail in eager mode (py_function)
    # ms = ms * np.reshape(mask_id, [-1,1,1])
    #
    # return ms

    # return ms
    areas = np.asarray(areas)

    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(segments.shape[0]):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)

    # bmasks.append(masks)
    # bindex.append(index)

    return masks#, index #, bindex