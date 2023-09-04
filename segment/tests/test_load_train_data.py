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
from pathlib import Path
from utils.tf_general import increment_path




from PIL import ImageFont

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


if __name__ == '__main__':
    import os
    import platform
    import sys
    from pathlib import Path
    FILE = Path(__file__).resolve()

    ROOT = FILE.parents[2]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH

    # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from segment.load_train_data import LoadTrainData

import numpy as np
import cv2
from PIL import ImageDraw
from PIL import Image as im

from segment.tf_create_dataset import CreateDataset


def test_dataset_creation(imgsz=640, line_thickness = 3, save_dir='./dataset'):


    ltd = LoadTrainData()
    mosaic=True
    train_data_path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train'
    image_files, labels, segments = ltd.load_data(train_data_path, mosaic)





    create_dataset = CreateDataset(imgsz)
    ds = create_dataset(image_files, labels, segments)

    # ds = ds.shuffle(10)
    sel_ds = ds.take(3)

    for idx, (img, img_labels, img_filenames, img_shape, img_segments) in enumerate(sel_ds):
        img = np.array(img* 255)

        bboxes = np.array(img_labels.to_tensor())[:,1:]
        category_names = [str(name) for name in(np.array(img_labels.to_tensor())[:,0])]


        for label, segment in zip(img_labels, img_segments):
            label =np.array(label)
            category = label[0]
            segment =np.array(segment.to_tensor()) # from ragged to tensor
            polygon = segment.reshape(1, segment.shape[0], -1, 2).astype(np.int32)

            color = np.random.randint(low=0, high=255, size=3).tolist()
            cv2.fillPoly(img, polygon, color=color)

        image = im.fromarray((img).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            xmin, ymin, w, h = bbox*imgsz
            color = tuple(np.random.randint(low=0, high=255, size=3).tolist())
            draw.line([(xmin-w/2, ymin-h/2), (xmin-w/2, ymin + h/2), (xmin + w/2, ymin + h/2), (xmin+w/2, ymin-h/2),
                               (xmin-w/2, ymin-h/2)],
                              width=line_thickness,
                              fill=color)
        text_box_color = [255, 255, 255]
        draw_text_on_bounding_box(image, np.array(bboxes)[..., 1],
                                  np.array(bboxes)[..., 0], text_box_color,
                                  category_names, font_size=15)
        ImageDraw.Draw(image)


        image.save(save_dir/f'annotated_{idx}.jpeg')

if __name__ == '__main__':
    name='exp'
    save_dir = increment_path(Path(f'{Path.cwd()}/results/dataset')  / name, exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)
    test_dataset_creation(imgsz=640, line_thickness=3, save_dir=save_dir)
    print(f"Results saved to {save_dir}")

