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


from utils.tf_general import xyxy2xywhn, segments2bboxes_batch
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
                              -imgsz[1] // 2]  # mosaic center placed randomly at [-border, 2 * imgsz + border]
        self.imgsz = imgsz
        # self.augment, self.degrees, self.translate, self.scale, self.shear, self.perspective=augment, degrees, translate, scale, shear, perspective
        self.augment = augment
        self.downsample_ratio = mask_ratio  # yolo training requires downsampled by 4 mask

        self.augmentation = Augmentation(hsv_h=hyp["hsv_h"], hsv_s=hyp["hsv_s"], hsv_v=hyp["hsv_v"],
                                         flipud=hyp["flipud"], fliplr=hyp["fliplr"])

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
            nt = len(lb) # nof targets
            if nt:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:,
                        2:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 2:][lb[:, 2:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nt:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{warning_msg_prefix}WARNING ⚠️ {im_file}: {nt - len(i)} duplicate labels removed'
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
            # set raggged  as objects' segments sizes differ. (unlike augment mode, where segments interpolated to 1000)
            segments = tf.ragged.constant(
                self.segments[index])  # image_i segments ragged tensor: "shape": [nti,v_ij,2], vij: object j vertices
            padw, padh = pad[0], pad[1]
            # map loops on all segments, scale normalized coordibnates to fit mage scaling:
            segments = tf.map_fn(fn=lambda t: self.xyn2xy(t, w1, h1, padw, padh), elems=segments,
                                 fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
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
                                                                    )
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

        labels = xyxy2xywhn(labels, w=640, h=640, clip=True, eps=1e-3)  # normalize xywh by image size wxh
        if self.augment:
            img, labels, masks = self.augmentation(img, labels, masks)
            img = img.astype(tf.float32) / 255
        # set ragged labels tensor. Reason: needed to pack all images [nti,5] tensors, where nti nof targets in image i
        labels = tf.RaggedTensor.from_tensor(labels)  # [nt,5], nt: nof objects in current image
        return img, labels, masks, tf.constant(self.im_files[index]), shapes

    def iter(self):
        for i in self.indices:
            yield self[i]

    def decode_resize(self, index, preserve_aspect_ratio=True, padding=False):
        """
        Reads, decodes and resizes an image from file.

        :param index: index to list of images files path, int
        :param preserve_aspect_ratio: If True, max(w,h) resized to imgsz. Bool
        :param padding: Relevant if preserve_aspect_ratio=True. Bool
        :return:
        :rtype:
        """
        filename = self.im_files[index]
        img_orig = tf.io.read_file(filename)
        # note: format result: [height,width,ch]
        img0 = tf.image.decode_image(img_orig, channels=3).astype(tf.float32) / 255 # read format:  [height,width,ch]
        resized_img = img0  # init - to be resized
        r = self.imgsz[0] / max(img0.shape[:2])  # ratio, Note: assumed squared target imgsz
        padh = padw = 0
        if r != 1:  # don't resize if h or w equals  self.imgsz
            resized_img = tf.image.resize(img0, self.imgsz, preserve_aspect_ratio=preserve_aspect_ratio) # shape: h,w,ch
        resized_shape = resized_img.shape[:2]
        if padding:
            padh = int((self.imgsz[1] - resized_img.shape[0]) / 2) # note: resized_img shape: [h,w,ch]
            padw = int((self.imgsz[0] - resized_img.shape[1]) / 2)

            resized_img = tf.image.pad_to_bounding_box(
                resized_img, padh, padw, self.imgsz[1], self.imgsz[0]
            )

        return (
            resized_img, img0.shape[:2], resized_shape,
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

    def xywhn2xyxy(self, x, w, h, padw=0, padh=0, width=640, height=640):
        """
        Scale normalized (xc,yc,w,h) bbox to (xmin,ymin, xma,ymax) to real size bbox padded to uniform size
        :param 5 entry array of clas id and normalized bbox: [classId, xc,yc,w,h], float
        :param w: width of scaled bbox, float
        :param h: height of scaled bbox, float
        :param padw: padding for bbox width to fit
        :param padh:
        :type padh:
        :return:
        :rtype:
        """
        xmin = tf.math.multiply(float(w), (x[..., 1:2] - x[..., 3:4] / 2)) + float(padw)  # top left x
        ymin = tf.math.multiply(float(h), (x[..., 2:3] - x[..., 4:5] / 2)) + float(padh)  # top left y
        xmax = tf.math.multiply(float(w), (x[..., 1:2] + x[..., 3:4] / 2)) + float(padw)  # bottom right x
        ymax = tf.math.multiply(float(h), (x[..., 2:3] + x[..., 4:5] / 2)) + float(padh)  # bottom right y
        y = tf.concat([x[..., 0:1], xmin, ymin, xmax, ymax], axis=-1, name='concat') # [class,x,y,x,y]
        return y

    def xyn2xy(self, x, w, h, padw=0, padh=0):
        """
        Sclae normalized segments coordinates
        :param x: segment coordibnates. (can probably be ragged). shape: [n_vertices, 2] (None, 3] if ragged
        :param w: width
        :param h: height
        :param padw: pad width, for coords shift
        :type padh: pad height, for coords shift
        :rtype: scaled segments coords
        """
        # Convert normalized segments into pixel segments, shape (n,2)
        x = x.to_tensor()
        xcoord=tf.gather(x, 0, axis=1)[...,None]
        ycoord =tf.gather(x, 1, axis=1)[...,None]
        xcoord = tf.math.multiply(float(w), xcoord) + float(padw)  # x coords - resized and shifted by pad val
        ycoord = tf.math.multiply(float(h), ycoord) + float(padh)  # y coords - resized and shifted by pad val
        y = tf.concat(
            [xcoord, ycoord], axis=-1, name='stack'
        )

        y = tf.RaggedTensor.from_tensor(y)

        return y

    def perspective_transform(self, img, M, dsize):
        """
        A wrapper for cv2.warpPerspective, which should be invoked by map_fn, to call a cv2 function by a tf.data.Dataset
        pipeline.

        :param img: input 3 channels image, float tensor
        :param M: Transform Homogenouse matrix, shape: [3,3]: [[a00, a01,a02][a1,0,a11,a12][0,0,1]], float tensor
        :param dsize: size of Affaine transform output. obtained by cropping image to fit desired dimensions. int
        :return: img: transformed image, shape: [dsize[0], dsize[1], 3], float

        :rtype:
        """

        img = cv2.warpPerspective(img, M, dsize=(dsize[0], dsize[1]), borderValue=(114, 114, 114))
        return img

    def affaine_transform(self, img, M, dsize):
        """
        A wrapper for cv2.warpAffine, which should be invoked by map_fn, to call a cv2 function by a tf.data.Dataset
        pipeline.

        :param img: input 3 channels image, float tensor
        :param M: Transform Homogenouse matrix, shape: [3,3]: [[a00, a01,a02][a1,0,a11,a12][0,0,1]], float tensor
        :param dsize: size of Affaine transform output. obtained by cropping image to fit desired dimensions. int
        :return: img: transformed image, shape: [dsize[0], dsize[1], 3], float

        :rtype:
        """

        img = cv2.warpAffine(np.asarray(img), M[:2].numpy(), dsize=dsize.numpy(),
                             borderValue=(114. / 255, 114. / 255, 114. / 255))  # grey

        # img = tf.keras.preprocessing.image.apply_affine_transform(img,theta=0,tx=0,ty=0,shear=0,zx=1,zy=1,row_axis=0,col_axis=1,channel_axis=2,fill_mode='nearest',cval=0.0,order=1 )
        return img

    def resample_segments(self, seg_coords, ninterp):
        """

        :param seg_coords:
        :type seg_coords:
        :param ninterp:
        :type ninterp:
        :return:
        :rtype:
        """
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
                           border=(0, 0),
                           upsample=1000  # segments vertices interpolation value
                           ):
        """
        0. Create 5 transform matrices: Cent, Perspective, Rotate, Shear, Translate
        1. Performs either perspective or affaine transform. Result is cropped to image's size-relevant for mosaic mode.
        2. Apply transform on upsampled segments.
        3. Produce bboxes from transformed segments.
        4. Filter out target according to bbox thresholding criteria.

        :param im: input image, shape: [h,w,3], tf.float32
        :param targets: target labels, each a 5 words entries: [class, bbox], shape: [nt,5], tf.float32
        :param segments: target segments polygons. Ragged tensor shape: [nt, None,2], nt:nof image's target, dim1: nof
                segment vertices (varies per target), 2 (coords per vertex (x,y). ragged tensor , tf.float32
        :param degrees: [-degrees,degrees] is the range of uniform random rotation value pick, float, degrees.
        :param translate: [0.5-translate, 0.5+translate] is the range of uniform random translation (mult by w or h).
        :param scale:[1-scale,1+scale] is the range of uniform random uniform scaling value
        :param shear: [-shear,shear] is the range of uniform random shearing value pick, float.
        :param perspective: [-perspective,perspective-] is the range of uniform random shearing value pick, float.
        :param border: border for cropping the 2*2 expanded mosaic image. Output is cropped to (border
        :type border: margins added by mosaic expansion. Will be cropped here. in mosaic4: [-imgo//2,-img0//2],
            in REGULAR non-mosaic MODE: [0,0]:
        :param upsample: Upsampled nof interpolated segments vertices, common to all segment. Typically 1000, int
        :return:
            im: transformed, cropped to image size (in mosaic mode). shape:[h-2*border[0],w-2*border[1],3],tf.float
            targets: transform matching bboxes, entries are [class, bbox],  shape: [nt,5]
            segments: transformed segments, upsampled to a uniform size, so changed from ragged to regular tensors
        """
        # 0. Create 5 transform matrices: Cent, Perspective, Rotate, Shear, Translate
        # Subtract borders offsets to set output size. Relevant to mosaic expanded image, otherwise borders are 0s:
        height = im.shape[0] + border[0] * 2
        width = im.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        # C translation combined with T give [-320,-320] offset translation for mosaic4 or [0,0] for regular non-mosaic:
        C[0, 2] = -im.shape[1] / 2  # x translation by half (pixels)
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
        M = T @ S @ R @ P @ C  # shape: [3,3], order of operations (right to left) is IMPORTANT

        # 1. Performs either perspective or affaine transform. Result is cropped to image's size-relevant for mosaic mode.

        # Transform if either M is not identity OR borders should be cropped (i.e. mosaic):
        if (border[0] != 0) or (border[1] != 0) or tf.math.reduce_any(M != tf.eye(3)):  # if image changed...
            if perspective:
                #  img shape: [1280,1280,3], Out shape: [width, height,3], obtained by image cropping:
                im = tf.py_function(self.perspective_transform, [im, M, (width, height)], Tout=tf.float32)
            else:  # affine
                #  img shape: [1280,1280,3], Out shape: [width, height,3], obtained by image cropping:
                im = tf.py_function(self.affaine_transform, [im, M, (width, height)], Tout=tf.float32)
        n = len(targets)
        if n:
            # 2. Apply transform on upsampled segments.

            # reample & add homogeneous coords before transformation.
            # Notes:
            # 1. map_fn needed since segments are ragged.
            # 2. Out segments are upsampled to uniform size, so converted to regular tensors
            segments = tf.map_fn(fn=lambda segment: self.resample_segments(segment, upsample), elems=segments,
                                 fn_output_signature=tf.TensorSpec(shape=[upsample, 3], dtype=tf.float32,
                                                                   ))# in shape: [nt,None,2], out shape:[nt, upsample,3]
            segments = tf.matmul(segments, tf.transpose(M).astype(tf.float32)) # affine transform. shape:[nt,upsample,3]
            segments = tf.gather(segments, [0, 1], axis=-1) # From homogenouse to normal: discard bottom all 1's row
            #  3. Produce bboxes from transformed segments:
            bboxes = segments2bboxes_batch(segments)

            # bboxes = segment2box(segments)


            # 4. Filter out target according to bbox thresholding criteria:
            indices = box_candidates(box1=tf.transpose(targets[..., 1:]) * s, box2=tf.transpose(bboxes),
                                 area_thr=0.01)
            bboxes = bboxes[indices]

            targets = targets[indices]
            targets = tf.concat([targets[:, 0:1], bboxes], axis=-1)  # [cls, bboxes]
            segments = segments[indices]
        return im, targets, segments

    def load_mosaic(self, index, ):  # filenames, size, y_labels, y_segments):

        # 1. Mosaic Setup:
        # 1.1 Randomly pick mosaic central ref point:
        xc = tf.random.uniform((), -self.mosaic_border[0], 2 * self.imgsz[0] + self.mosaic_border[0], dtype=tf.int32)
        yc = tf.random.uniform((), -self.mosaic_border[1], 2 * self.imgsz[1] + self.mosaic_border[1], dtype=tf.int32)
        # 1.2. Randomly pick 3 more images from dataset's image list:
        indices = random.choices(self.indices, k=3)  # 3 additional image indices
        indices.insert(0, index)
        if self.debug:  # determine mosaic
            yc, xc = 496, 642
            indices = [0, 0, 0, 0]
        # 1.3. construct empty 4 x size mosaic image template:
        img4 = tf.fill(
            (self.imgsz[0] * 2, self.imgsz[1] * 2, 3), 114 / 255
        )  # gray background
        # 2. loop on 4 images, arrange mosaic4:
        for idx, index in enumerate(indices):
            # 2.1 load image
            img, _, (h, w), _ = self.decode_resize(index)
            # 2.2 place image in mosaic4 quarter:
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
            img4 = self.scatter_img_to_mosaic(dst_img=img4, src_img=img[y1b:y2b, x1b:x2b,:], dst_xy=(x1a, x2a, y1a, y2a))
            # 2.3  scale bbox and segmentsand, adapt coords to mosaic structure placement shift:
            padw = x1a - x1b  # image's x offset from mosaic's axis origin
            padh = y1a - y1b  # image's y offset from mosaic's axis origin

            # Scale bbox coordinates to fit in mosaic structure: result::[cls,xmin,ymin,xmax,ymax]:
            y_l = self.xywhn2xyxy(self.labels[index], w, h, padw, padh,self.imgsz[0], self.imgsz[1])
            # Convert segments from list to ragged tensor. Note: images segments non-uniform size requires ragged tanser
            segments = tf.ragged.constant(self.segments[index])
            # Scale segments coordinates to fit in mosaic structure: result::[nt, None, 2]:
            y_s = tf.map_fn(fn=lambda t: self.xyn2xy(t, w, h, padw, padh), elems=segments,
                            fn_output_signature=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32,
                                                                    ragged_rank=1))
            # 2.4 Concat current 4*image's labels with other mosaic elements. if idx=0, (1st element), don't concat:
            labels4 = tf.cond(tf.equal(idx, 0), true_fn=lambda: y_l, false_fn=lambda: tf.concat([labels4, y_l], axis=0))
            # 2.5 Concat current 4*image's segments with other mosaic elements. if idx=0, (1st element), don't concat:
            segments4 = tf.cond(tf.equal(idx, 0), true_fn=lambda: y_s,
                                false_fn=lambda: tf.concat([segments4, y_s], axis=0))
        # 3. clip labels and segments to mosaic boundaries:
        clipped_bboxes = tf.clip_by_value(
            labels4[:, 1:], 0,  2 * self.imgsz[0], name='labels4'
        )
        segments4 = tf.clip_by_value(
            segments4, 0, 2 * self.imgsz[0], name='segments4'
        )
        # 4. reconstruct labels by concat class and bbox:
        labels4 = tf.concat([labels4[..., 0:1], clipped_bboxes], axis=-1)

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
        masks = tf.squeeze(tf.image.resize(masks[..., None], [nh, nw]), axis=3)  # masks shape: [nt, 160, 160]
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
        masks = tf.squeeze(tf.image.resize(masks[..., None], [nh, nw]), axis=3)  # masks shape: [nt, 160, 160]
        # DESCENDING sort masks by area.  reason: smaller mask area will get larger color, so it will survive if overlap
        areas = tf.math.reduce_sum(masks, axis=[1, 2])  # shape: [nt]
        sorted_index = tf.argsort(areas, axis=-1, direction='DESCENDING', stable=False, name=None)  # shape: [nt]
        masks = tf.gather(masks, sorted_index, axis=0)  # sort masks by areas shape: [nt, 160,160]
        # color masks by index, before merge: 1 for larger, nt to smallest. 0 remains no mask:
        mask_colors = tf.range(1, len(sorted_index) + 1, dtype=tf.float32) # masks colors range is 1:nt

        masks = tf.math.multiply(masks, tf.reshape(mask_colors, [-1, 1, 1]))  # set color values to mask pixels
        masks = tf.reduce_max(masks, axis=0)  # merge overlaps: keep max color value (i.e. smaller mask areas)
        return masks, sorted_index
