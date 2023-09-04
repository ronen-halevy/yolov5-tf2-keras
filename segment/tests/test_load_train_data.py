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
import glob
import random
from pathlib import Path

# from core.load_tfrecords import parse_tfrecords
# from core.create_dataset_from_files import create_dataset_from_files
# from core.load_tfrecords import parse_tfrecords
from utils.tf_plots import Annotator, colors, save_one_box

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


if __name__ == '__main__':
    ltd = LoadTrainData()
    mosaic=True
    train_data_path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train'
    image_files, labels, segments = ltd.load_data(train_data_path, mosaic)


    imgsz=640
    line_thickness = 3

    create_dataset = CreateDataset(imgsz)
    ds = create_dataset(image_files, labels, segments)

    # ds = ds.shuffle(10)
    sel_ds = ds.take(1)
    print('!!!!!!!??????????????????????!!!!!!!!!!!!!!!!!!!!')

    for img, img_labels, img_filenames, img_shape, img_segments in sel_ds:
        img = np.array(img* 255)

        # ##
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # img = np.array(img)
        # cv2.imshow('ff', img)
        # cv2.waitKey()
        # image = im.fromarray((img*255).astype(np.uint8))
        # image.save('tt.jpeg')


        # ##


        bboxes = np.array(img_labels.to_tensor())[:,1:]
        for label, segment in zip(img_labels, img_segments):
            label =np.array(label)
            category = label[0]
            # bboxes = label[1:]
            segment =np.array(segment.to_tensor())
            polygon = segment# np.asarray(polygon)
            polygon = polygon.astype(np.int32)
            shape = polygon.shape
            polygon = polygon.reshape(shape[0], -1, 2)
            pp = np.expand_dims(polygon, 0)

            color = np.random.randint(low=0, high=255, size=3).tolist()
            print('color ', color)
            cv2.fillPoly(img, pp, color=color)

        image = im.fromarray((img).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            xmin, ymin, w, h = bbox*320
            color = tuple(np.random.randint(low=0, high=255, size=3).tolist())
            print(color)
            draw.line([(xmin-w/2, ymin-h/2), (xmin-w/2, ymin + h/2), (xmin + w/2, ymin + h/2), (xmin+w/2, ymin-h/2),
                               (xmin-w/2, ymin-h/2)],
                              width=line_thickness,
                              fill=color)
            #
        ImageDraw.Draw(image)

        cv2.waitKey()
        image.save('tt.jpeg')

