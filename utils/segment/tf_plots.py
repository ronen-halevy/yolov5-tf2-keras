import contextlib
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb

from .. import threaded
from ..tf_general import xywh2xyxy
from ..tf_plots import Annotator, colors

# @threaded
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
                # elif scale < 1:  # absolute coords need scale if image scales
                #     boxes *= scale
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
                            opacity = 0.4
                            im[y:y + h, x:x + w, :][mask] = im[y:y + h, x:x + w, :][mask] * opacity + np.array(color) * (1-opacity)
                # convert numpy array to image:
                annotator.fromarray(im)
    annotator.im.save(fname)  # save


# @threaded
def plot_images_and_masks2(images, targets, masks, paths=None, class_names=None, fname='images.jpg', index=0):
    """
     Plots input images as a grid of subplots, overlayed with masks, bboxes and class annotations.
     Max number of images in plots' grid  is limmited to [4x4 = 16].

    :param images:  batch of images. shape:[b,640,640,3], tf.float32 [0,1]. Note: pixels can be either normalized or not
    :param targets: batch's targets. shape: [nt,6] entry: [nt, bi+cls+normalized_bbox]], tf.float32
    :type masks: shape: [b,160,160], tf.flaot32
    :param paths: path to images, used for labels. shape: [b] tf str
    :param fname: output file name
    :param class_names: class names dictionary, used for label annotations
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
    mosaic_mask = np.full((int(ns * h), int(ns * w)), 255, dtype=np.uint8)  # init ns*ns mosaic

    for i, im in enumerate(images):
        if i == max_subplots:  # limit plotted images
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # mosaic sub-image origin: placement order is row after row
        mosaic[y:y + h, x:x + w, :] = im

    examples = []
    batch_classes = []
    batch_confs = []
    batch_boxes=[]
    for i, im in enumerate(images):

        # extract currnet image's targets:
        if targets.shape[0]:
            x, y = int(w * (i // ns)), int(h * (i % ns))  # origin inside mosaic

            # extract currnet image's targets:
            idx = targets[:, 0] == i# mask targets which belong to image i. bool, shape: [nt] (targets[:,0] is bs id:)
            ti = targets[idx]  # current image's targets
            # unpack targets:
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 6  # labels if no conf column (epected either [bi,cls,bbox] or [bi,cls,bbox,conf])
            confs = np.full(ti.shape[0], None) if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            boxes = xywh2xyxy(ti[:, 2:6])# shape: [4, nboxes]

            if boxes.shape[1]: # if nboxes>0, Note that boxes is transposed here
                sw,sh = (w,h) if boxes.max() <= 1.01 else (1,1)  # rescale if normalized with tolerance 0.01
                # scale to pixels, add x,y placement offsets in mosaic location:
                boxes = tf.concat( [boxes[:,0][...,None] * sw+x, boxes[:,1][...,None]  * sh+y, boxes[:,2] [...,None] * sw+x, boxes[:,3][...,None]  * sh+y], axis=1)
            batch_boxes = tf.concat([batch_boxes, boxes], axis=0) if i else boxes # boxes on first loop

            # arrange masks: Pick current images mask, according to overlapping mode:
            if len(masks): #overlap: all targets in a single mask template, pix values: 0: b.ground 1:nt for masks:
                if tf.reduce_max(masks) > 1.0:  # if non-overlap n overlap mode mask contains all image's targets masks colored 1:n
                    image_masks = masks[i] # overlap mode: pixels' val in ranges 1:numof masks. shape:(1, 640, 640),
                else: # non overlap mode (or a single target mask): mask template per target, pix 0: (b.ground) 1: mask)
                    image_masks = masks[idx] # non-overlap: pick current image masks.  shape:[nti,160,160]
                    mask_pixel_mult = np.arange(image_masks.shape[0]).reshape(image_masks.shape[0],1,1)+1
                    image_masks = image_masks*mask_pixel_mult # set mask(i) # set mask[i]=i+1, i=0:(nt-1)
                    image_masks = np.sum(image_masks, axis=0) # shape: [160,160]


                mh, mw = image_masks.shape[0:2]
                if mh != h or mw != w:  # is mask[j] downsampled? normally it is
                    mask = cv2.resize(image_masks.astype(np.uint8), (w, h))  # .astype(np.uint8)
                mosaic_mask[y:y + h, x:x + w]=mask
                # concat mosaoic images to a single list:
                batch_classes = batch_classes + list(classes)
                batch_confs = batch_confs + list(confs)
                # offset class ids to 1 based:
    mask_class_names = {key+1: value for key, value in class_names.items()}

    masked_image = wandb.Image(
                    mosaic,
                    caption= f'{index}',
                    masks={
                    "predictions": {
                        "mask_data":mosaic_mask,
                        "class_labels": mask_class_names}
                    },
                    boxes={
                        "predictions": {
                            "box_data": [
                                {
                                    # one box expressed in the default relative/fractional domain
                                    "position": {"minX": bbox[0].numpy() / (640*ns), "maxX": bbox[2].numpy() / (640*ns),
                                                 "minY": bbox[1].numpy() / (640*ns), "maxY": bbox[3].numpy() / (640*ns)},
                                    "class_id": int(class_id),
                                    "box_caption": f'{class_id}:{class_names[class_id]} {conf}' if conf else f'{class_id}:{class_names[class_id]}',
                                    "scores": {"acc": 0.2, "loss": 1.2},
                                }
                                for bbox, class_id, conf  in zip(batch_boxes, batch_classes, batch_confs) ],
                            "class_labels": class_names
                        }
                    }
                )
    wandb.log({fname: masked_image})

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
