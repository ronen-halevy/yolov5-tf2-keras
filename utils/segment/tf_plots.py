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


# @threaded
def plot_images_and_masks(images, targets, masks, paths=None, fname='images.jpg', names=None):
    # Plot image grid with labels
    if isinstance(images, tf.Tensor):
        images = images.numpy().astype(np.float32)
    if isinstance(targets, tf.Tensor):
        targets = targets.numpy()
    if isinstance(masks, tf.Tensor):
        masks = masks.numpy().astype(int)
    paths = paths.numpy().astype(np.string_)
    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs,  h, w, _ = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots in each square's axis (bs=ns*ns)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init ns*ns mosaic
    for i, im in enumerate(images):
        if i == max_subplots:  # limit plotted images
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # mosaic sub-image origin: placement order is row after row
        # im = im.transpose(1, 2, 0)
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
            # .name[:40]
            annotator.text((x + 5, y + 5), text=Path(str(paths[i])).name[:40], txt_color=(220, 220, 220))  # filenames
        if targets.shape[0]:
            # targets[:,0] indicates bs id:
            idx = targets[:, 0] == i# mask targets which belong to current image i. shape: [Nt]
            ti = targets[idx]  # image targets

            boxes = tf.transpose(xywh2xyxy(ti[:, 2:6]))# shape: [4, nboxes]
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 6  # labels if no conf column
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]: # if nboxes>0, Note that boxes is transposed here
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    # boxes[[0, 2]] *= w  # scale to pixels
                    # boxes[[1, 3]] *= h
                    boxes = tf.concat( [boxes[[0]] * w, boxes[[1]] * h, boxes[[2]] * w, boxes[[3]] * h], axis=0)
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            boxes = tf.concat([boxes[[0]] + x, boxes[[1]] +y, boxes[[2]] +x, boxes[[3]] +y], axis=0)

            # boxes[[0, 2]] += x # add origin offset in mosaic to current image's bbox xmin, xmax
            # boxes[[1, 3]] += y  # add origin offset in mosaic to current image's to ymin,ymax
            for j, box in enumerate(tf.transpose(boxes).tolist()): # loop on boxes, annotate by class name, conf and color
                cls = classes[j]
                color = colors(cls) # select color from list
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)

            # Plot masks
            if len(masks):
                if masks.max() > 1.0:  # mean that masks are overlap, where masks are marked by 1:nt
                    image_masks = masks[[i]]  # (1, 640, 640) # TODO - check this and rest
                    nl = len(ti)
                    index = np.arange(nl).reshape(nl, 1, 1) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)
                else: # either a single object or non-overlapping masks.
                    try:
                        image_masks = masks[idx] # take current sample's single mask
                    except Exception as e:
                        msg= f'\nDebug~~~masks!!!!, {e}, {masks}, {masks.shape} idx, {idx}, exiting!!!'
                        print(msg)
                        raise(e)

                im = np.asarray(annotator.im).copy()
                for j, box in enumerate(boxes.T.tolist()):
                    if labels or conf[j] > 0.25:  # 0.25 conf thresh
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            im[y:y + h, x:x + w, :][mask] = im[y:y + h, x:x + w, :][mask] * 0.4 + np.array(color) * 0.6
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
            for i, j in enumerate([1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]):
                y = data.values[:, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=2)
                if best:
                    # best
                    ax[i].scatter(index, y[index], color="r", label=f"best:{index}", marker="*", linewidth=3)
                    ax[i].set_title(s[j] + f"\n{round(y[index], 5)}")
                else:
                    # last
                    ax[i].scatter(x[-1], y[-1], color="r", label="last", marker="*", linewidth=3)
                    ax[i].set_title(s[j] + f"\n{round(y[-1], 5)}")
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print(f"Warning: Plotting error for {f}: {e}")
    ax[1].legend()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close()
