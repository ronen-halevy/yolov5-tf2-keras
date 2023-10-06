#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : create_dataset.py
#   Author      : ronen halevy
#   Created date:  10/19/22
#   Description :
#
# ================================================================
import yaml

import sys
from pathlib import Path
from PIL import ImageFont
import numpy as np
import cv2
from PIL import ImageDraw
from PIL import Image as im

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

from segment.load_train_data import LoadTrainData
from segment.tf_dataloaders import create_dataloader
from utils.tf_general import increment_path
from utils.segment.tf_general import masks2segments, process_mask, process_mask_native

import tensorflow as tf


def draw_text_on_bounding_box(image, ymin, xmin, color, display_str_list=(), font_size=30):
    """
    Description: Draws a text which starts at xmin,ymin bbox corner

    """

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  font_size)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    text_margin_factor = 0.05
    if display_str_list:
        left, top, right, bottom = zip(*[font.getbbox(display_str) for display_str in display_str_list])
        text_heights = tuple(map(lambda i, j: i - j, bottom, top))
        text_widths = tuple(map(lambda i, j: i - j, right, left))

        text_margins = np.ceil(text_margin_factor * np.array(text_heights))
        text_bottoms = ymin * (ymin > text_heights) + (ymin + text_heights) * (ymin <= text_heights)

        for idx, (display_str, xmint, text_bottom, text_width, text_height, text_margin) in enumerate(
                zip(display_str_list, xmin, text_bottoms, text_widths, text_heights, text_margins)):

            left, top, right, bottom = font.getbbox(display_str)
            text_height = bottom - top
            text_width = right - left

            text_margin = np.ceil(text_margin_factor * text_height)

            draw.rectangle(((xmint, text_bottom - text_height - 2 * text_margin),
                            (xmint + text_width + text_margin, text_bottom)),
                           fill=tuple(color))
            draw.text((xmint + text_margin, text_bottom - text_height - 3 * text_margin),
                      display_str,
                      fill="black",
                      font=font)
    return image

def draw_dataset_entry(img, bboxes, cls, img_segments, line_thickness):
    img = np.array(img * 255)

    # bboxes = np.array(img_labels)[:, 1:]
    # use category id for category name:
    category_names = [str(int(name)) for name in (np.array(cls))]

    for segment in img_segments:
        segment = np.array(segment)  # from ragged to tensor - todo check if needed - ndarray already?
        polygon = segment.reshape(1, segment.shape[0], -1, 2).astype(np.int32)

        color = np.random.randint(low=0, high=255, size=3).tolist()
        cv2.fillPoly(img, polygon, color=color)

    image = im.fromarray((img).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        xc, yc, w, h = tf.math.multiply(bbox, (img.shape[0], img.shape[1], img.shape[0], img.shape[1]))
        color = tuple(np.random.randint(low=0, high=255, size=3).tolist())
        draw.line([(xc - w / 2, yc - h / 2), (xc - w / 2, yc + h / 2), (xc + w / 2, yc + h / 2),
                   (xc + w / 2, yc - h / 2),
                   (xc - w / 2, yc - h / 2)],
                  width=line_thickness,
                  fill=color)
    text_box_color = [255, 255, 255]
    box_xmin = (np.array(bboxes)[..., 0] - np.array(bboxes)[..., 2] / 2) * img.shape[0]
    box_ymin = (np.array(bboxes)[..., 1] - np.array(bboxes)[..., 3] / 2) * img.shape[1]
    draw_text_on_bounding_box(image, box_ymin, box_xmin, text_box_color, category_names, font_size=15)
    ImageDraw.Draw(image)
    return image


if __name__ == '__main__':
    FILE = Path(__file__).resolve()

    ROOT = FILE.parents[2]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH


def test_dataset_creation(data_path, imgsz=640, line_thickness=3, nexamples=3, save_dir='./dataset'):
    hyp = '../../data/hyps/hyp.scratch-low.yaml'
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    mosaic = False
    degrees, translate, scale, shear, perspective = hyp['degrees'], hyp['translate'], hyp['scale'], hyp['shear'], hyp[
        'perspective']
    augment = True
    hgain, sgain, vgain, flipud, fliplr = hyp['hsv_h'], hyp['hsv_s'], hyp['hsv_v'], hyp['flipud'], hyp['fliplr']
    batch_size = 2
    data_loader_debug = True  # False
    mask_ratio = 4

    ds = create_dataloader(data_path, batch_size, [imgsz, imgsz], mask_ratio, mosaic, augment, degrees, translate,
                           scale, shear, perspective, hgain, sgain, vgain, flipud, fliplr, data_loader_debug)

    # ds = ds.shuffle(10)
    sel_ds = ds.take(nexamples)
    # for bidx, (bimg, bimg_labels_ragged, bimg_filenames, bimg_shape, bmask) in enumerate(sel_ds):
    for bidx, (bimg, bimg_labels_ragged, bmask, bpaths, bshapes) in enumerate(sel_ds):
        # for bidx, (img, img_labels_ragged, img_filenames, img_shape, img_segments_ragged) in enumerate(sel_ds):
        #     for idx, (img, img_labels_ragged, img_filenames, img_shape, mask) in enumerate(zip(bimg, bimg_labels_ragged, bimg_filenames, bimg_shape, bmask)):
        for idx, (img, img_labels_ragged, mask, paths, shapes) in enumerate(zip(bimg, bimg_labels_ragged, bmask, bpaths, bshapes)):
            img_labels = img_labels_ragged  # .to_tensor() # convert from ragged
            d_s_factor = 4
            mask = tf.squeeze(
                tf.image.resize(mask[..., None], [mask.shape[0] * d_s_factor, mask.shape[1] * d_s_factor]))
            segments = cv2.findContours(mask.numpy().astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            bboxes = tf.gather(img_labels, [1, 2, 3, 4], axis=-1)
            cls = tf.gather(img_labels, [0], axis=-1)
            image = draw_dataset_entry(img, bboxes, cls, segments, line_thickness)
            image.save(save_dir / f'annotatedm_{bidx}_{idx}.jpeg')
            im.fromarray((img.numpy() * 255).astype(np.uint8)).save(save_dir / f'annotationless_{bidx}_{idx}.jpeg')


if __name__ == '__main__':
    imgsz = 640
    name = 'exp'
    # save_dir = increment_path(Path(f'{Path.cwd()}/results/dataset')  / name, exist_ok=False)  # increment run
    save_dir = increment_path(Path(f'{ROOT}/runs/tests/') / name, exist_ok=False)  # increment run

    data_path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train'

    save_dir.mkdir(parents=True, exist_ok=True)
    test_dataset_creation(data_path, imgsz, line_thickness=3, nexamples=3, save_dir=save_dir)
    print(f"Results saved to {save_dir}")
