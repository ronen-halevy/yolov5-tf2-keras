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

from pathlib import Path
import random

from utils.tf_general import increment_path

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

from PIL import ImageFont

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


import numpy as np
import cv2
from PIL import ImageDraw
from PIL import Image as im


from utils.segment.dataloaders import LoadImagesAndLabelsAndMasks


def masks2segments(masks, strategy='largest'):
    # Convert masks(n,160,160) into segments(n,xy)
    segments = []
    for x in masks.numpy().astype('uint8'):
        # x=y.copy()
        # x=np.resize(x, ([640,640])).astype(np.uint8)
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

def draw_dataset_entry(img, img_labels, img_segments, line_thickness):

    bboxes = np.array(img_labels)[:, 2:]
    # use category id for category name:
    category_names = [str(int(name)) for name in (np.array(img_labels)[:, 0])]

    for idx, segment in enumerate(img_segments):

        # label = np.array(label)
        # category = label[0]
        segment = np.array(segment)  # from ragged to tensor
        # polygon = segment.reshape(-1, segment.shape[0], 2).astype(np.int32)
        polygon = segment.reshape(1, segment.shape[0], -1, 2).astype(np.int32)

        print('polygon', idx, polygon, polygon.shape)

        color = np.random.randint(low=0, high=255, size=3).tolist()
        mm = np.zeros([640,640,3]).astype(np.uint8).fill(255)
        # cv2.imshow('ffpre', img)
        # cv2.waitKey()
        cv2.fillPoly(img, polygon, color=color)
        # cv2.imshow('ffpo', img)
        # cv2.waitKey()
    image = im.fromarray((img).astype(np.uint8))
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        xmin, ymin, w, h = bbox# * imgsz
        xmin*=imgsz
        ymin*=imgsz
        w*=imgsz
        h*=imgsz

        print('bbox', (xmin - w / 2, ymin - h / 2), (xmin - w / 2, ymin + h / 2), (xmin + w / 2, ymin + h / 2),
              (xmin + w / 2, ymin - h / 2))
        color = tuple(np.random.randint(low=0, high=255, size=3).tolist())
        draw.line([(xmin - w / 2, ymin - h / 2), (xmin - w / 2, ymin + h / 2), (xmin + w / 2, ymin + h / 2),
                   (xmin + w / 2, ymin - h / 2),
                   (xmin - w / 2, ymin - h / 2)],
                  width=line_thickness,
                  fill=color)
    text_box_color = [255, 255, 255]
    # draw_text_on_bounding_box(image, (np.array(bboxes)[..., 1] - np.array(bboxes)[..., 3] / 2) * imgsz,
    #                           (np.array(bboxes)[..., 0] - np.array(bboxes)[..., 2] / 2) * imgsz, text_box_color,
    #                           category_names, font_size=15)
    ImageDraw.Draw(image)
    return image



if __name__ == '__main__':
    random.seed(42)
    path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train'
    imgsz = 640
    batch_size=16
    augment=True
    hyp='../../data/hyps/hyp.scratch-low.yaml'
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    rect=False
    cache=False
    single_cls=False
    stride=32
    pad=0
    image_weights=False
    prefix=''
    mask_downsample_ratio = 4
    overlap_mask=True
    dataset = LoadImagesAndLabelsAndMasks(
        path,
        imgsz,
        batch_size,
        augment=augment,  # augmentation
        hyp=hyp,  # hyperparameters
        rect=rect,  # rectangular batches
        cache_images=cache,
        single_cls=single_cls,
        stride=int(stride),
        pad=pad,
        image_weights=image_weights,
        prefix=prefix,
        downsample_ratio=mask_downsample_ratio,
        overlap=overlap_mask)
    idx=0
    nexamples=3
    imgsz = 640
    name = 'exp'

    save_dir = increment_path(Path(f'{ROOT}/runs/tests/') / name, exist_ok=False)  # increment run

    # save_dir = increment_path(Path(f'{Path.cwd()}/results/dataset') / name, exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)

    data_path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train'
    line_thickness = 3
    nexamples=1
    # nexamples = min(nexamples, len(dataset) )
    for idx in range(nexamples):
        if True:
            idx = 0# debug ronen random.randint(0, len(dataset)-1)
            img, labels_out, im_files, shapes, masks = dataset[idx]

            # print(labels_out)
            img = np.transpose(img, [1, 2, 0])
            img = np.ascontiguousarray(img, dtype=np.uint8)
            # cv2.imshow('gg44', img)
            #
            # cv2.waitKey()

            segments=masks2segments(masks)
            img = np.array(img)
            labels_out = np.array(labels_out)
            segments = np.array(segments)*4
            print('labels_out',labels_out)
            print('segments',segments)



        else:
            img, labels_out, segments = dataset[idx]

        # cv2.imshow('ghg', img)
        # cv2.waitKey()





        image = draw_dataset_entry(img, labels_out, segments, line_thickness)
        mm = np.array(image)

        # cv2.imshow('gg55', mm)
        #
        # cv2.waitKey()
        image.save(save_dir / f'annotated_{idx}.jpeg')
        print(f"Results saved to {save_dir}")


