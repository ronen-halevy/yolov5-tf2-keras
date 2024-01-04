import tensorflow as tf
from utils.tf_general import segments2boxes
import tensorflow_probability as tfp
import glob
import os
import random
from pathlib import Path
import numpy as np
from PIL import ExifTags, Image, ImageOps
import cv2
import contextlib
import math


from utils.tf_general import xyxy2xywhn, segments2boxes_exclude_outbound_points
from utils.tf_augmentations import box_candidates
from utils.segment.tf_augmentations import Augmentation

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


# Note: rect, not implemented
# class LoadTrainData:
class LoadImagesAndLabelsAndMasks:
    """
    Creates dataset entries, consist of images, labels and masks.
    Main method is __getitem__ which is normally envoked by genetator itterations to produce a dataset entry__

    """

    def __init__(self, path, imgsz, mask_ratio, mosaic, augment, hyp, overlap, debug=False):
        """
        Produces 3 main lists:
        image_files - a bi size list of inout file paths, where bi nof input images
        labels: a bi size image labels arrays, shape [nti, 5], nti nof targets, entry struct: [cls, nxywh]
        segments: a bi list of nti lists, each holds array segments with shapes: [nsi,2], nsi: nof polygon's vertices
        :param path: path in files, Images expected at path/images and labels at path/labels, same names but .txt ext
        :param imgsz: size of model's input. list:2 ints. in yolo5: [640,640]
        :param mask_ratio: downsample_ratio of mask size wrt input image. default: 4, giving mask size [160,160], int
        :param mosaic: set mosaic-4 on data. Requires True augment, bool
        :param augment: set augmentation on data, bool
        :param hyp: config params for augmentation attributes
        :param debug: used to select static  mosaic selection for debug only, bool
        """
        self.im_files = self._make_file(path, IMG_FORMATS)

        self.label_files = self._img2label_paths(self.im_files)  # labels

        self.image_files = []
        self.labels = []
        self.segments = []
        for idx, (im_file, label_file) in enumerate(zip(self.im_files, self.label_files)):
            # extract class, bbox and segment from label file entry:
            image_file, label, segment = self._create_entry(idx, im_file, label_file)
            self.image_files.append(image_file)
            self.labels.append(label)
            self.segments.append(
                segment)  # list[nlabels], entries (not rectangular):list[nti] of [n_v_ij,2], where nti: nobjects in imagei, n_v_ij,2:nvertices in objecrtj of imagei

        self.indices = range(len(self.image_files))
        self.mosaic = mosaic
        self.debug = debug
        self.hyp = hyp
        self.overlap=overlap

        self.mosaic_border = [-imgsz[0] // 2,
                              -imgsz[1] // 2]  # mosaic center placed randimly at [-border, 2 * imgsz + border]
        self.imgsz = imgsz
        # self.augment, self.degrees, self.translate, self.scale, self.shear, self.perspective=augment, degrees, translate, scale, shear, perspective
        self.augment = augment
        self.downsample_ratio = mask_ratio  # yolo training requires downsampled by 4 mask

        self.augmentation = Augmentation(hsv_h=hyp["hsv_h"], hsv_s=hyp["hsv_s"], hsv_v=hyp["hsv_v"],
                                         flipud=hyp["flipud"], fliplr=hyp["fliplr"])
        # self.augmentation = Augmentation(hgain, sgain, vgain, flipud, fliplr)

    @property
    def __len__(self):
        return len(self.image_files)

    def exif_size(self, img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        with contextlib.suppress(Exception):
            rotation = dict(img._getexif().items())[orientation]
            if rotation in [6, 8]:  # rotation 270 or 90
                s = (s[1], s[0])
        return s

    def exif_transpose(image):
        """
        Transpose a PIL image accordingly if it has an EXIF Orientation tag.
        Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

        :param image: The image to transpose.
        :return: An image.
        """
        exif = image.getexif()
        orientation = exif.get(0x0112, 1)  # default 1
        if orientation > 1:
            method = {
                2: Image.FLIP_LEFT_RIGHT,
                3: Image.ROTATE_180,
                4: Image.FLIP_TOP_BOTTOM,
                5: Image.TRANSPOSE,
                6: Image.ROTATE_270,
                7: Image.TRANSVERSE,
                8: Image.ROTATE_90}.get(orientation)
            if method is not None:
                image = image.transpose(method)
                del exif[0x0112]
                image.info['exif'] = exif.tobytes()
        return image

    def read_label_from_file(self, fname):
        """
        Reads segments label file, retrun class and bbox.
        Input File format-a row per object structured: class, sx1,sy1....sxn,syn
        :param fname: labels file name, str. Fi
        :return:
        lb:  tensor of class,bbox. shape: [1,5], tf.float32
        """
        with open(fname) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            if any(len(x) > 6 for x in lb):  # is segment
                classes = np.array([x[0] for x in lb], dtype=np.float32)
                # img_index = tf.tile(tf.expand_dims(tf.constant([idx]), axis=-1), [classes.shape[0], 1])
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)),
                                    1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)
        return lb, segments

    def fix__corrupted_jpeg(self, im_file, warning_msg_prefix):
        with open(im_file, 'rb') as f:
            f.seek(-2, 2)
            if f.read() != b'\xff\xd9':  # corrupt JPEG
                ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                msg = f'{warning_msg_prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

    def _create_entry(self, idx, im_file, lb_file, warning_msg_prefix=''):
        nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = self.exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            self.fix__corrupted_jpeg(im_file, warning_msg_prefix)
        # verify labels
        if os.path.isfile(lb_file):
            lb, segments = self.read_label_from_file(lb_file)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:,
                        2:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 2:][lb[:, 2:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{warning_msg_prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, segments

    def _img2label_paths(self, img_paths):
        # Define label paths as a function of image paths
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    def _make_file(self, path, ext_list):
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ext_list)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'{"prefix"}No images found'
            return im_files
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}') from e

    def __len__(self):
        return len(self.im_files)

    '''
    return:
    img: float, [0,1.], shape: [640,640,3] 
    labels: RaggedTensor, float shapeL [Nti, 5] Where Nti varies between images 
    masks:  float32 shape:[h/4,w/4], pixels values: 0: nomask. smalest mask with largest pixel values.
    paths: Tensor, string. image path on disk
    shapes: Tensor, float32, shape: [3,2] [[h,w], [ h / h0, w / w0], [padh, padw]]

     tf.constant(self.im_files[index]), shapes)
    '''

    def __getitem__(self, index):
        """
        This is the main method for dataset entries construction. It produces a dataset entry according to index
        self.im_files list. if mosaic 4 config is true, the entry is constructed using 3 more randomly selected samples.

        :param index: index points to an image entry in self.im_files list
        :type index: int
        :return:
         img: shape: [self.imgsz[0], self.imgsz[1],3], resized image, float, normalized to 1.
         labels: Per image object entry with [cls,x,y,w,h] where bbox normalized. Ragged tensor, since entries nt
         varies. shape: [nt,5].
         masks: shape: [h/4,w/4], pixels' val in ranges 1:numof masks, takes smallest idx if overlap. 0 if non mask.
         files: self.im_files[index] i.e src file path (in mosaic too-only 1 src returned). Tensor, str.
         shapes:  [(h0,w0),(h1/w0,w1/w0),(padh,padw)], all zeros if mosaic. shape:[3,2],float
         """
        mosaic = random.random() < self.mosaic  # randmoly select mosaic mode, unless self.mosaic is 0 or 1.
        # why is_segment_ragged needed: in case of mosaic or augment true, all processed segments are interpolated to 1000
        # points. Otherwise, segment is a ragged tensor shape: [nt,(npolygons),2], where npolygons differs, that's why
        # ragged. However, before feeding to cv2.polly() polygon-by-polygon, must convert to tensor otherwise crash. So,
        # is_segment_ragged is used, as otherwise code can't tell if tensor is ragged and needs conversion to tensor.

        is_segment_ragged = False  # False if augment or mosaic where nof vertices interpolated and so is uniform to all.
        # otherwise nof vertices in segments variessegments are padded to same nof vertices
        if self.augment and mosaic:
            img, labels, segments = self.load_mosaic(index)
            shapes = tf.zeros([3, 2], float)  # for mAP rescaling. Dummy same shape (keep generator's spec) for mosaic
        else:
            (img, (h0, w0), (h1, w1), pad) = self.decode_resize(index, padding=True)
            shapes = tf.constant(((float(h0), float(w0)), (h1 / h0, w1 / w0), pad))  # for mAP rescaling.
            segments = tf.ragged.constant(
                self.segments[index])  # image_i segments ragged tensor: "shape": [nti,v_ij,2], vij:vertices in obj_j
            padw, padh = pad[0], pad[1]
            # map loops on all segments, scale normalized coordibnates to fit mage scaling:
            segments = tf.map_fn(fn=lambda t: self.xyn2xy(t, w1, h1, padw, padh), elems=segments,
                                 fn_output_signature=tf.RaggedTensorSpec(shape=[None, None], dtype=tf.float32,
                                                                         ragged_rank=1))
            labels = self.xywhn2xyxy(self.labels[index], w1, h1, padw, padh)

            if self.augment:
                img, labels, segments = self.random_perspective(img,
                                                                labels,
                                                                segments,
                                                                degrees=self.hyp['degrees'],
                                                                translate=self.hyp['translate'],
                                                                scale=self.hyp['scale'],
                                                                shear=self.hyp['shear'],
                                                                perspective=self.hyp['perspective'],
                                                                border=[0, 0]
                                                                )  # border to remove
            else:
                is_segment_ragged = True

        if segments.shape[0]:
            if self.overlap:# produce a single mask per image, with a color per target's mask:
                masks, sorted_index = self.polygons2masks_overlap(segments, self.imgsz, self.downsample_ratio, is_segment_ragged)
                labels = tf.gather(labels, sorted_index, axis=0)  # follow masks sorted order
            else:# produce a mask per each target, mask pixels always 1.
                masks = self.polygons2masks_non_overlap(segments, self.imgsz, color=1, downsample_ratio=self.downsample_ratio, is_ragged=is_segment_ragged)
        else: # create zeros mask with shape image.shape/4
            masks = tf.fill([img.shape[0] // self.downsample_ratio, img.shape[1] // self.downsample_ratio], 0).astype(
                tf.float32)  # np.zeros(img_size, dtype=np.uint8)

        labels = xyxy2xywhn(labels, w=640, h=640, clip=True, eps=1e-3)  # return xywh normalized
        if self.augment:
            img, labels, masks = self.augmentation(img, labels, masks)
            img = img.astype(tf.float32) / 255
        # set ragged labels tensor - support different nof labels per image:
        labels = tf.RaggedTensor.from_tensor(labels)  # [nt,5], nt: nof objects in current image
        return img, labels, masks, tf.constant(self.im_files[index]), shapes

    def iter(self):
        for i in self.indices:
            yield self[i]

    def decode_resize(self, index, preserve_aspect_ratio=True, padding=False):
        filename = self.im_files[index]
        img_orig = tf.io.read_file(filename)

        img0 = tf.image.decode_image(img_orig, channels=3).astype(tf.float32) / 255
        img_resized = img0  # init
        r = self.imgsz[0] / max(img0.shape[:2])  # ratio, + note- imgsz is of a square...
        padh = padw = 0
        if r != 1:  # don't resize if h or w equals  self.imgsz
            img_resized = tf.image.resize(img0, self.imgsz, preserve_aspect_ratio=True)
        resized_shape = img_resized.shape[:2]
        if padding:
            padh = int((self.imgsz[1] - img_resized.shape[0]) / 2)
            padw = int((self.imgsz[0] - img_resized.shape[1]) / 2)
            # pad with grey color:
            paddings = tf.constant([[padh, self.imgsz[1] - img_resized.shape[0] - padh],
                                    [padw, self.imgsz[0] - img_resized.shape[1] - padw], [0, 0]])
            img_resized = tf.pad(img_resized, paddings, "CONSTANT", constant_values=114)
        return (
            img_resized, img0.shape[:2], resized_shape,
            (padw, padh))  # pad is 0 by def while aspect ratio not preserved

    def scatter_img_to_mosaic(self, dst_img, src_img, dst_xy):
        """
        Place a n image in the mosaic-4 tensor
        :param dst_img: 2w*2h*3ch 4mosaic dst img
        :param src_img:
        :type src_img:
        :param dst_xy:
        :type dst_xy:
        :return:
        :rtype:
        """
        y_range = tf.range(dst_xy[2], dst_xy[3])[..., None]
        y_ind = tf.tile(y_range, [1, dst_xy[1] - dst_xy[0]])
        x_range = tf.range(dst_xy[0], dst_xy[1])[None]
        x_ind = tf.tile(x_range, [dst_xy[3] - dst_xy[2], 1])
        indices = tf.squeeze(tf.concat([y_ind[..., None], x_ind[..., None]], axis=-1))
        dst = tf.tensor_scatter_nd_update(
            dst_img, indices, src_img
        )
        return dst

    def xywhn2xyxy(self, x, w, h, padw=0, padh=0):
        #     """
        #      transform scale and align bboxes: xywh to xyxy, scaled to image size, shift by padw,padh to location in mosaic
        #     :param x: xywh normalized bboxes
        #     :type x: float array, shape: [nboxes,4]
        #     :param w: dest image width
        #     :type w: int
        #     :param h: dest image height
        #     :type h: int
        #     :param padw: shift of src image left end from mosaic left end
        #     :type padw: float ]
        #     :param padh: shift of src image upper end from mosaic upper end
        #     :type padh: float
        #     :return: scaled bboxes in xyxy coords, aligned to shifts in mosaic
        #     :rtype: float array, shape: [nboxes, 4]
        #     """
        #     # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        xmin = tf.math.multiply(float(w), (x[..., 1:2] - x[..., 3:4] / 2)) + float(padw)  # top left x
        ymin = tf.math.multiply(float(h), (x[..., 2:3] - x[..., 4:5] / 2)) + float(padh)  # top left y
        xmax = tf.math.multiply(float(w), (x[..., 1:2] + x[..., 3:4] / 2)) + float(padw)  # bottom right x
        ymax = tf.math.multiply(float(h), (x[..., 2:3] + x[..., 4:5] / 2)) + float(padh)  # bottom right y
        y = tf.concat([x[..., 0:1], xmin, ymin, xmax, ymax], axis=-1, name='concat')
        return y

    def xyn2xy(self, x, w, h, padw=0, padh=0):
        # Convert normalized segments into pixel segments, shape (n,2)
        x = x.to_tensor()
        xcoord = tf.math.multiply(float(w), x[:, 0:1]) + float(padw)  # x coords - resized and shifted by pad val
        ycoord = tf.math.multiply(float(h), x[:, 1:2]) + float(padh)  # y coords - resized and shifted by pad val
        y = tf.concat(
            [xcoord, ycoord], axis=-1, name='stack'
        )
        y = tf.RaggedTensor.from_tensor(y)

        return y

    def affaine_transform(self, img, M):
        # img = cv2.warpAffine(img.numpy(), M[:2].numpy(), dsize=(1280, 1280), borderValue=(114, 114, 114))
        img = cv2.warpAffine(np.asarray(img), M[:2].numpy(), dsize=(640, 640),
                             borderValue=(114. / 255, 114. / 255, 114. / 255))  # grey

        # img = tf.keras.preprocessing.image.apply_affine_transform(img,theta=0,tx=0,ty=0,shear=0,zx=1,zy=1,row_axis=0,col_axis=1,channel_axis=2,fill_mode='nearest',cval=0.0,order=1 )
        return img

    def resample_segments(self, ninterp, seg_coords):
        seg_coords = seg_coords.to_tensor()[..., 0:]
        seg_coords = tf.concat([seg_coords, seg_coords[0:1, :]], axis=0)  # close polygon's loop before interpolation
        x_ref_max = seg_coords.shape[0] - 1  # x max
        x = tf.linspace(0., x_ref_max, ninterp).astype(tf.float32)  # n interpolation points. n points array

        # interpolate polygon's to N points - loop on x & y
        segment = [tfp.math.interp_regular_1d_grid(
            x=x,  # N points range
            x_ref_min=0,
            x_ref_max=x_ref_max,  # shape0,  # tf.constant(len(s)),
            y_ref=seg_coords[..., idx],
            axis=-1,
            fill_value='constant_extension',
            fill_value_below=None,
            fill_value_above=None,
            grid_regularizing_transform=None,
            name='interp_segment'
        ) for idx in range(2)]
        segment = tf.concat([segment], axis=0)
        segment = tf.reshape(segment, [2, -1])
        segment = tf.transpose(segment)

        segment = tf.concat([segment, tf.ones([ninterp, 1], dtype=tf.float32)], axis=-1)
        return segment

    def random_perspective(self, im,
                           targets=(),
                           segments=(),
                           degrees=10,
                           translate=.1,
                           scale=.1,
                           shear=10,
                           perspective=0.0,
                           border=(0, 0)):
        # targets = [cls, xyxy]
        # random.seed(0)
        # tf.random.set_seed(
        #     0
        # )  # ronen todo!!
        height = im.shape[0] + border[0] * 2  # shape(h,w,c)
        width = im.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

        # Perspective
        presp = tf.random.uniform([2], -perspective, perspective, dtype=tf.float32)
        P = tf.tensor_scatter_nd_update(tf.eye(3), [[2, 0], [2, 1]], presp)  # x perspective (about y)

        # Rotation and Scale
        a = tf.random.uniform((), -degrees, degrees, dtype=tf.float32)
        s = tf.random.uniform((), 1 - scale, 1 + scale, dtype=tf.float32)
        R = [[s * tf.math.cos(a), s * tf.math.sin(a), 0], [- s * tf.math.sin(a), s * tf.math.cos(a), 0], [0, 0, 1]]
        # Shear
        shearval = tf.math.tan(tf.random.uniform([2], -shear, shear, dtype=tf.float32) * math.pi / 180)  # x shear (deg)
        S = tf.tensor_scatter_nd_update(tf.eye(3), [[0, 1], [1, 0]], shearval)  # x perspective (about y)

        # Translation
        transn = tf.random.uniform([2], 0.5 - translate, 0.5 + translate) * [width, height]
        T = tf.tensor_scatter_nd_update(tf.eye(3), [[0, 2], [1, 2]], transn)  # x perspective (about y)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        if (border[0] != 0) or (border[1] != 0) or tf.math.reduce_any(M != tf.eye(3)):  # if image changed...
            if perspective:
                im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                im = tf.py_function(self.affaine_transform, [im, M], Tout=tf.float32)  # img shape[1280,1280,3] M:3x3
        if True:  # if n: # Todo clean this
            ninterp = 1000
            # reample & add homogeneous coords & ragged->tensor (map_fn needed since segments ragged):
            # Note: before resample, segments are ragged (variable npoints per segment), accordingly tf.map_fn required:
            segments = tf.map_fn(fn=lambda segment: self.resample_segments(ninterp, segment), elems=segments,
                                 fn_output_signature=tf.TensorSpec(shape=[1000, 3], dtype=tf.float32,
                                                                   ))
            segments = tf.matmul(segments, tf.transpose(M).astype(tf.float32))  # affine transform
            segments = tf.gather(segments, [0, 1], axis=-1)

            bboxes = segments2boxes_exclude_outbound_points(segments)
        indices = box_candidates(box1=tf.transpose(targets[..., 1:]) * s, box2=tf.transpose(bboxes),
                                 area_thr=0.01)
        bboxes = bboxes[indices]

        targets = targets[indices]
        bboxes = tf.concat([targets[:, 0:1], bboxes], axis=-1)  # [cls, bboxes]
        segments = segments[indices]
        return im, bboxes, segments

    def load_mosaic(self, index, ):  # filenames, size, y_labels, y_segments):
        # labels4, segments4 = [], []
        segments4 = None
        labels4 = None
        # randomly select mosaic center:

        xc = tf.random.uniform((), -self.mosaic_border[0], 2 * self.imgsz[0] + self.mosaic_border[0], dtype=tf.int32)
        yc = tf.random.uniform((), -self.mosaic_border[1], 2 * self.imgsz[1] + self.mosaic_border[1], dtype=tf.int32)

        indices = random.choices(self.indices, k=3)  # 3 additional image indices
        indices.insert(0, index)
        if self.debug:  # determine mosaic
            yc, xc = 496, 642
            indices = [0, 1, 2, 3]

        img4 = tf.fill(
            (self.imgsz[0] * 2, self.imgsz[1] * 2, 3), 114 / 255
        )  # gray background

        w, h = self.imgsz[0], self.imgsz[1]
        # arrange mosaic 4:
        # for idx in range(4):
        for idx, index in enumerate(indices):
            img, _, _, _ = self.decode_resize(index)

            if idx == 0:  # top left mosaic dest zone,  bottom-right aligned src image fraction:
                x1a, y1a, x2a, y2a = tf.math.maximum(xc - w, 0), tf.math.maximum(yc - h,
                                                                                 0), xc, yc  # xmin, ymin, xmax, ymax
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                        y2a - y1a), w, h  # xmin, ymin, xmax, ymax: src image fraction
            elif idx == 1:  # top right mosaic dest zone, bottom-left aligned src image fraction:
                x1a, y1a, x2a, y2a = xc, tf.math.maximum(yc - h, 0), tf.math.minimum(xc + w, w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), tf.math.minimum(w, x2a - x1a), h  # src image fraction
            elif idx == 2:  # bottom left mosaic dest zone, top-right aligned src image fraction:
                x1a, y1a, x2a, y2a = tf.math.maximum(xc - w, 0), yc, xc, tf.math.minimum(w * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, tf.math.minimum(y2a - y1a,
                                                                            h)  # src image fraction: aligned right-up
            elif idx == 3:  # bottom right mosaic dest zone, top-left aligned src image fraction:
                x1a, y1a, x2a, y2a = xc, yc, tf.math.minimum(xc + w, w * 2), tf.math.minimum(w * 2, yc + h)  #
                x1b, y1b, x2b, y2b = 0, 0, tf.math.minimum(w, x2a - x1a), tf.math.minimum(y2a - y1a,
                                                                                          h)  # src image fraction
            else:
                raise Exception('Too many images assigned for Mosaic-4')

            img4 = self.scatter_img_to_mosaic(dst_img=img4, src_img=img[y1b:y2b, x1b:x2b], dst_xy=(x1a, x2a, y1a, y2a))
            padw = x1a - x1b  # shift of src scattered image from mosaic left end. Used for bbox and segment alignment.
            padh = y1a - y1b  # shift of src scattered image from mosaic top end. Used for bbox and segment alignment.

            # arrange boxes & segments: scale normalized coords and shift location to mosaic zone by padw, padh:
            y_l = self.xywhn2xyxy(self.labels[index], w, h, padw, padh)
            # map_fn since segments is a ragged tensor:
            segments = tf.ragged.constant(self.segments[index])
            y_s = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=segments,
                            fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
                                                                    ragged_rank=1))
            # concat 4 mosaic elements together. idx=0 is the first concat element:
            labels4 = tf.cond(tf.equal(idx, 0), true_fn=lambda: y_l, false_fn=lambda: tf.concat([labels4, y_l], axis=0))
            segments4 = tf.cond(segments4 is None, true_fn=lambda: y_s,
                                false_fn=lambda: tf.concat([segments4, y_s], axis=0))

        clipped_bboxes = tf.clip_by_value(
            labels4[:, 1:], 0, 2 * float(w), name='labels4'
        )
        labels4 = tf.concat([labels4[..., 0:1], clipped_bboxes], axis=-1)

        segments4 = tf.clip_by_value(
            segments4, 0, 2 * 640, name='segments4'  # todo 640
        )

        img4, labels4, segments4 = self.random_perspective(img4,
                                                           labels4,
                                                           segments4,
                                                           degrees=self.hyp['degrees'],
                                                           translate=self.hyp['translate'],
                                                           scale=self.hyp['scale'],
                                                           shear=self.hyp['shear'],
                                                           perspective=self.hyp['perspective'],
                                                           border=self.mosaic_border)  # border to remove

        return img4, labels4, segments4

    def polygons2mask(self, is_ragged, img_size, polygon):
        """
        Converts a single polygon to a masks.
        :param is_ragged: indicates if input segments is a ragged tensor, bool
        :param img_size: The image size used to produce the mask image.  noramlly (640,640), tuple(2), int
        :param polygon: input polygon, shape: [1, N_vertices,2], either a ragged tensor or a tensor
        :return: mask, shape

        :rtype:
        """
        """
        Args:
            :is_ragged:
            :img_size (tuple): The image size.
            :polygon [1, npoints, 2]
            :color:
            :downsample_ratio
        """

        # polygon = tf.cond(is_ragged, true_fn=lambda:polygon, false_fn=lambda: polygon)
        # init allzeros mask
        mask = np.zeros(img_size, dtype=np.uint8)

        if is_ragged:
            polygon = polygon.to_tensor()[..., 0:]
        polygon = np.array(polygon)
        cv2.fillPoly(mask, polygon, color=1)

        return mask  # nparray, uint8, shape: [img_size] typically [640,640]

    def polygons2masks_non_overlap(self, segments, img_size, color, downsample_ratio=1, is_ragged=False):
        """
        Converts input polygon segments to masks. Produces a mask image per each segment, mask pixels set to '1'.
        :param segments: either a ragged tensor, shape: [b,None,2], or a tensor [b,ns,2], Latter if images' segments iterpolated to a common size,
        :param img_size: The image size, used to produce mask image. Normally 640*640, tuple(2), int
        :param color:
        :param downsample_ratio: Downsampled masks pattern wrt image. Normally 4, int
        :param is_ragged: indicates if input segments is a ragged tensor, bool
        :return: masks: a mask per input segment, ragged tensor of size: [nt, 160,160],  float
        """
        segments = tf.cast(segments, tf.int32)
        masks = tf.map_fn(fn=lambda segment: tf.py_function(self.polygons2mask, [is_ragged, img_size, segment[None]],
                                                                Tout=tf.float32), elems=segments,
                              fn_output_signature=tf.TensorSpec(shape=[640, 640], dtype=tf.float32))

        nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)  # downsample masks by 4
        masks = tf.squeeze(tf.image.resize(masks[..., None], [nh, nw]), axis=3)  # masks shape: [nl, 160, 160]
        return tf.RaggedTensor.from_tensor(masks) # ragged!!! fro no overlap!!

    def polygons2masks_overlap(self, segments, size, downsample_ratio, is_ragged):
        """
        Converts input polygon segments to masks. Produces a common mask image per all segments, assigning different
        pixel colors accordingly. Overlapped pixels are assigned to the smallest area mask.

        :param segments: either a ragged tensor, shape: [b,None,2], or a tensor [b,ns,2], Latter if images' segments iterpolated to a common size,
        :param size: The image size, used to produce mask image. Normally 640*640, tuple(2), int
        :param downsample_ratio: Downsampled masks pattern wrt image. Normally 4, int
        :param is_ragged: indicates if input segments is a ragged tensor, bool
        :return: masks: a single mask image, common to all input segments, segregated by pixel colors. float
        """
        color = 1  # default value 1 is later modifed to a color per mask
        segments = tf.cast(segments, tf.int32)
        # run polygons2mask for all segments by tf.map_fn runs . py_function is needed to call cv2.fillpoly in graph mode
        masks = tf.map_fn(fn=lambda segment: tf.py_function(self.polygons2mask, [is_ragged, size, segment[None]],
                                                            Tout=tf.float32), elems=segments,
                          fn_output_signature=tf.TensorSpec(shape=[640, 640], dtype=tf.float32))
        # Merge downsampled masks after sorting by mask size and coloring:
        nh, nw = (size[0] // downsample_ratio, size[1] // downsample_ratio)  # downsample masks by 4
        masks = tf.squeeze(tf.image.resize(masks[..., None], [nh, nw]), axis=3)  # masks shape: [nl, 160, 160]
        # sort masks by area.  reason: to select smallest area mask if masks overlap
        areas = tf.math.reduce_sum(masks, axis=[1, 2])  # shape: [nl]
        sorted_index = tf.argsort(areas, axis=-1, direction='DESCENDING', stable=False, name=None)  # shape: [nl]
        masks = tf.gather(masks, sorted_index, axis=0)  # sort masks by areas shape: [nl]
        # color masks by index, before merge: 1 for larger, nl to smallest. 0 remains no mask:
        mask_colors = tf.range(1, len(sorted_index) + 1, dtype=tf.float32)
        masks = tf.math.multiply(masks, tf.reshape(mask_colors, [-1, 1, 1]))  # set color values to mask pixels
        masks = tf.reduce_max(masks, axis=0)  # merge overlaps: keep max color value  (i.e. smallest area mask)
        return masks, sorted_index
