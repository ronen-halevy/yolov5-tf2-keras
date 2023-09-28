
import albumentations as A
import cv2
import numpy as np
import tensorflow as tf


from utils.tf_general import (LOGGER, colorstr)


class Albumentations:
    def __init__(self):
        self.transform = None
        prefix = colorstr('albumentations: ')
        try:
            T= [# there are some augmentation that won't change size, boxes and masks, so just be it for now.
                # A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0)
            ]  # transforms
            self.transform =A.Compose(T)


        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')


    def run(self, im, p=1.0):
        if self.transform and tf.random.uniform((),0,1) < p:
            new = self.transform(image=im.numpy())
            im = new['image']
        return im


class Augmentation:
    def __init__(self, hsv_h, hsv_s, hsv_v, flipud, fliplr):
        self.albumentations = Albumentations()
        self.hsv_h, self.hsv_s, self.hsv_v, self.flipud, self.fliplr=hsv_h, hsv_s, hsv_v, flipud, fliplr

    @staticmethod
    def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains

            hue, sat, val = cv2.split(cv2.cvtColor(im.numpy().astype(np.uint8), cv2.COLOR_BGR2HSV))
            dtype = im.numpy().dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype) # 255 entry target colors table
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)  # no return needed
        return im

    def __call__(self, img, labels, masks, p=1.0):
        img = tf.py_function(self.albumentations.run, [tf.cast(img*255, tf.uint8)], Tout=np.uint8)
        img = tf.py_function(self.augment_hsv, [img, self.hsv_h, self.hsv_s, self.hsv_v], Tout=tf.uint8)

        if tf.random.uniform((),0,1) < self.flipud:
            img = tf.image.flip_left_right(img)
            # if nl:
            y_min = 1 - labels[:, 2:3]  # y_min moves
            labels = tf.concat([labels[:, 0:2], y_min, labels[:, 3:5]], axis=-1)
            masks = tf.expand_dims(masks, axis=[-1])[None]
            masks = tf.image.flip_left_right(masks)
            masks = tf.squeeze(masks)

            # Flip left-right
        if tf.random.uniform((), 0, 1) < self.fliplr:
            img = tf.image.flip_left_right(img)
            # if nl:
            x_min = (1 - labels[:, 1:2])  # x_min moves
            labels = tf.concat([labels[:, 0:1], x_min, labels[:, 2:5]], axis=-1)
            masks = tf.expand_dims(masks, axis=[-1])[None]
            masks = tf.image.flip_left_right(masks)
            masks = tf.squeeze(masks)
        return img, labels, masks
