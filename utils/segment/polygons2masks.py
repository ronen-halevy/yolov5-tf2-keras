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
    cv2.fillPoly(mask, polygon, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask

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
def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    # bindex = []
    # bmasks = []
    # for segments in bsegments:
    masks = np.zeros((img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
                     dtype=np.int32 if segments.shape[0] > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(segments.shape[0]):
        mask = polygon2mask(
                img_size,

                [ tf.reshape(segments[si], [-1])],
                downsample_ratio=downsample_ratio,
                color=1,
            )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(segments.shape[0]):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)

    # bmasks.append(masks)
    # bindex.append(index)
    # print('len(bmasks)', len(bmasks))

    return masks, index #, bindex