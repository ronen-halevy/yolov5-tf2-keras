import contextlib
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from .. import threaded
from ..tf_general import xywh2xyxy
from ..tf_plots import Annotator, colors

@threaded
def plot_images_and_masks(images, targets, masks, paths=None, fname='images.jpg', names=None):
    """
     Plots input images as a grid of subplots, overlayed with masks, bboxes and class annotations.
     Max number of images in plots' grid  is limmited to [4x4 = 16].

    :param images:  batch of images. shape:[b,640,640,3], tf.float32 [0,1]. Note: pixels can be either normalized or not
    :param targets: batch's targets. shape: [nt,6] entry: [bi,cls,bbox (normalized)], tf.float32
    :type masks: shape: [b,160,160], tf.flaot32
    :param paths: path to images, used for labels. shape: [b] tf str
    :param fname: output file name
    :param names: class names list, for label annotations. string (optional)
    :return:
    :rtype:
    """

    # tf to numpy:
    if isinstance(images, tf.Tensor):
        images = images.numpy().astype(np.float32)
    if isinstance(targets, tf.Tensor):
        targets = targets.numpy()
    if isinstance(masks, tf.Tensor):
        masks = masks.numpy().astype(int)
    paths = paths.numpy().astype(np.string_)
    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs,  h, w, _ = images.shape  # unpack shape to batch size, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots in each square's axis (bs=ns*ns)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Fill images into ns*ns mosaic grid:
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init ns*ns mosaic
    for i, im in enumerate(images):
        if i == max_subplots:  # limit plotted images
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # mosaic sub-image origin: placement order is row after row
        mosaic[y:y + h, x:x + w, :] = im

    # Resize (optional): rescale if ns*images may exeed mosaic size in either w or h dim.
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  #  mosaic sub-image origin offsets
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths[i]:
            annotator.text((x + 5, y + 5), text=Path(str(paths[i])).name[:40], txt_color=(220, 220, 220))  # filenames
        if targets.shape[0]:
            # targets[:,0] indicates bs id:
            idx = targets[:, 0] == i# mask targets which belong to image i. shape: [Nt]
            ti = targets[idx]  # current image's targets

            boxes = tf.transpose(xywh2xyxy(ti[:, 2:6]))# shape: [4, nboxes]
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 6  # labels if no conf column
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]: # if nboxes>0, Note that boxes is transposed here
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    # scale to pixels
                    boxes = tf.concat( [boxes[[0]] * w, boxes[[1]] * h, boxes[[2]] * w, boxes[[3]] * h], axis=0)
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            # concat bboxes while adding subimage's origin offset in mosaic to bbox coords:
            boxes = tf.concat([boxes[[0]] + x, boxes[[1]] +y, boxes[[2]] +x, boxes[[3]] +y], axis=0)

            # loop on boxes, annotate with class name, conf and color:
            for j, box in enumerate(tf.transpose(boxes).tolist()):
                cls = classes[j]
                color = colors(cls) # select color from list
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)

            # Plot masks
            if len(masks):
                # if overlap & multi targets, separate masks[i] to nti masks. else, pick image's  idx masks
                if tf.reduce_max(masks) > 1.0:  #  in overlap mode mask contains all image's targets masks colored 1:n
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nti = len(ti) # nof targets in image i
                    index = np.arange(nti).reshape(nti, 1, 1) + 1 # shape: [nl,1,1]
                    image_masks = np.repeat(image_masks, nti, axis=0) # dup masks per nof labels, shape: [nl,160,160]
                    image_masks = np.where(image_masks == index, 1.0, 0.0) # image_masks holds nti masks colored 1 or 0
                else: # non-overlap mode or a single target image: mask[i] holds a single target colored by is.
                    image_masks = masks[idx] # take masks which belong to current image
                #  pick color by class value and draw overlay on the image:
                im = np.asarray(annotator.im).copy()
                for j, box in enumerate(boxes.T.tolist()):
                    if labels or conf[j] > 0.25:  # if labels (i.e. no conf column ) or 0.25 conf thresh
                        color = colors(classes[j])  # pick color per class
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w: # is mask[j] downsampled? normally it is
                            # resize mask[j], cast to bool:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else: # no resize, just cast:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            # draw masks: mix image pixel and mask:
                            im[y:y + h, x:x + w, :][mask] = im[y:y + h, x:x + w, :][mask] * 0.4 + np.array(color) * 0.6
                # convert numpy array to image:
                annotator.fromarray(im)
    annotator.im.save(fname)  # save


def plot_results_with_masks(file="path/to/results.csv", dir="", best=True):
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            index = np.argmax(0.9 * data.values[:, 8] + 0.1 * data.values[:, 7] + 0.9 * data.values[:, 12] +
                              0.1 * data.values[:, 11])
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for idx, j in enumerate([1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]):
                y = data.values[:, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[idx].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=2)
                if best:
                    # best
                    ax[idx].scatter(index, y[index], color="r", label=f"best:{index}", marker="*", linewidth=3)
                    ax[idx].set_title(s[j] + f"\n{round(y[index], 5)}")
                else:
                    # last
                    ax[idx].scatter(x[-1], y[-1], color="r", label="last", marker="*", linewidth=3)
                    ax[idx].set_title(s[j] + f"\n{round(y[-1], 5)}")
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[idx].get_shared_y_axes().join(ax[idx], ax[idx - 5])
        except Exception as e:
            print(f"Warning: Plotting error for {f}: {e}")
    ax[1].legend()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close()
