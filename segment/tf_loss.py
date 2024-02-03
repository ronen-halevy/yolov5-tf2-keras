# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

if __name__ == '__main__':
    import os
    import sys
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# import torch
# import torch.nn as nn
from utils.tf_general import xywh2xyxy
from utils.segment.tf_general import crop_mask

from utils.tf_metrics import bbox_iou
# from utils.torch_utils import de_parallel
import tensorflow as tf

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(tf.keras.layers.Layer):  # nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def call(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = tf.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    # fl_gamma - focal loss gamma
    # box_lg, obj_lg, cls_lg - box, obj and class loss gain
    # anchor_t - anchor multiple thresh
    def __init__(self, na, nl, nc, nm, stride, grids, anchors, overlap, fl_gamma, box_lg, obj_lg, cls_lg, anchor_t,
                 autobalance=False, label_smoothing=0.0):
        """
        :param na: number of anchors, 3, int
        :param nl: number of grid layers, 3, int
        :param nc: number of classes, int
        :param nm: number of predictd masks. 32, int
        :param : model strides.  currently [8,16,32], n/a in default configuration. float
        :param anchors: all anchors, shape: [nl,na,2], tyep: float
        :param fl_gamma: focal loss gamma, type: float
        :param box_lg: box loss gain, type: float
        :param obj_lg: obj loss gain, type: float
        :param cls_lg: class loss gain, type: float
        :param anchor_t: max threshold for anchor to bbox w,h ratio or w,h to anchor ratio, type float
        :param overlap: targets' mask overlap. if overlap color masks separately by indices,   threshold for anchor to bbox w,h ratio or w,h to anchor ratio, type bool
        :param autobalance: max threshold for anchor to bbox w,h ratio or w,h to anchor ratio, type bool
        :param label_smoothing:
        """
        self.na = na  # number of anchors
        self.nc = nc  # number of classes
        self.nl = nl  # number of layers
        self.nm = nm  # number of masks

        # Define criteria
        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        # Non-probability values so logits=True:
        BCEobj = tf.losses.BinaryCrossentropy(
            from_logits=True)  # with SUM_OVER_BATCH_SIZE reduction: sum/(nof elements)
        BCEcls = tf.losses.BinaryCrossentropy(from_logits=True)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)  # positive, negative BCE targets

        # Focal loss
        # g = h['fl_gamma']  # focal loss gamma
        if fl_gamma > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, fl_gamma), FocalLoss(BCEobj, fl_gamma)
        # balance adjustments loss by incrementing larger layers' loss, as those may overfit earlier:
        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(stride).index(16) if autobalance else 0  # stride index for autobalance. n/a in default config
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance

        self.anchors = anchors
        self.overlap = overlap
        self.box_lg = box_lg  # box loss gain
        self.obj_lg = obj_lg  # obj loss gain
        self.cls_lg = cls_lg  # class loss gain
        self.anchor_t = anchor_t
        self.grids = grids

    # @tf.function(input_signature=(
    #         tf.TensorSpec(shape=[None, None, None, None, None], dtype=tf.float32, name='p'),
    #         tf.TensorSpec(shape=[None, 4], dtype=tf.int32, name='indices'),
    #         tf.TensorSpec(shape=[None, 2], dtype=tf.float32, name='anchors'),
    #         tf.TensorSpec(shape=[None, 4], dtype=tf.float32, name='xywhn'),
    #         tf.TensorSpec(shape=[None, 4], dtype=tf.float32, name='tboxi'),
    #         tf.TensorSpec(shape=[None], dtype=tf.float32, name='tcls'),
    #         tf.TensorSpec(shape=[None], dtype=tf.float32, name='tidxs'),
    #         tf.TensorSpec(shape=[None, 32, 160, 160], dtype=tf.float32, name='proto'),
    #         tf.TensorSpec(shape=[1], dtype=tf.float32, name='lbox'),
    #         tf.TensorSpec(shape=[1], dtype=tf.float32, name='lseg'),
    #         tf.TensorSpec(shape=[1], dtype=tf.float32, name='lobj'),
    #         tf.TensorSpec(shape=[1], dtype=tf.float32, name='lcls'),
    #         tf.TensorSpec(shape=[None, 160, 160], dtype=tf.float32, name='masks'),
    #         tf.TensorSpec(shape=[], dtype=tf.float32, name='mask_h'),
    #         tf.TensorSpec(shape=[], dtype=tf.float32, name='mask_w'),
    #         tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32, name='tobj'),
    #         tf.TensorSpec(shape=[], dtype=tf.float32, name='balancei'),
    #         tf.TensorSpec(shape=[], dtype=tf.bool, name='anytargets'),
    # )
    # )  # (reduce_retracing=True)
    # @tf.function
    def calc_loss(self, pi, indicesi, anchorsi, xywhni, tboxi, tclsi, tidxsi, proto, lbox, lseg, lobj, lcls, masks,
              mask_h, mask_w, tobj, balancei, anytargets):
        """
        Calc loss for a single prediction grid layer
        :param pi: model prediction output shape: [b,na,gyi,gxi,4+1+nc+nm], gyi,gxi =[[80,80],[40,40],[20,20]]
        :param indicesi:  [batch ind,anchor ind, box center], shape: [nt,4]
        :param anchorsi: layer's anchors, shape: [na,2]
        :param xywhni: normalized bboxes coords, bbox format: xc,yc,w,h, shape: [ntl,4]  w ntl nof detections in layer (l=0-2)
        :param tboxi: x,y,w,h where x,y are offset from grid square corner: [ntl,4]
        :param tclsi: targets classes, shape: [nt1]
        :param tidxsi: runnig indices of target in image. shape: [ntl]
        :param proto:  model's proto output. tf.float32, shape: [nm, 160,160]
        :param lbox:  bbox loss accumulator, shape[1]
        :param lseg: segmentation loss accumulator, shape[1]
        :param lobj: obj loss accumulator, shape[1]
        :param lcls: class loss accumulator, shape[1]
        :param masks: masks (true labels), shape: [b,h/4,w/4],tf.float32
        :param mask_h:h/4
        :param mask_w:w/4
        :param tobj: obj true value, shape: [b,na,gyi,gxi], gyi,gxi =[[80,80],[40,40],[20,20]]
        :param balancei: balance factor for tobj, per grid layer, float
        :param anytargets: indication of valid true targets, bool

        :return:
            :lbox:  bbox loss accumulator, shape[1]
            : lseg: segmentation loss accumulator, shape[1]
            : lobj: obj loss accumulator, shape[1]
            : lcls: class loss accumulator, shape[1]
        """

        # step 4.1: take indices of targets, to fetch matching preds with:
        b, a, gj, gi = tf.unstack(indicesi, 4,
                                  axis=1)  # indicesi unstacked to 4: imgid, matched_anchor, grid location. shape:[nti,4]
        # ff = tf.split(indices[i], 4)
        # tobj = tf.zeros(pi.shape[:4], dtype=pi.dtype)  # init target obj with all 0s shape:[b,na,gy,gx]
        n = b.shape[0]  # number of targets
        if anytargets:
            # step 4.2: extract relקvant preds by targets indices. shapes: pxy,pwh:[Nti,2] pcls:[Nti,nc] pmask:[Nti,nm]

            pxy, pwh, _, pcls, pmask = tf.split(pi[b.astype(tf.int32), a.astype(tf.int32), gj, gi],
                                                (2, 2, 1, self.nc, self.nm ), 1)
            # step 4.3: calc box loss as 1-mean(iou(pbox,tbox))
            pxy = tf.sigmoid(pxy) * 2 - 0.5  # xy  coords adapted according to yolo's formulas
            pwh = (tf.sigmoid(pwh) * 2) ** 2 * anchorsi  # wh coords  adapted according to yolo's formulas:
            pbox = tf.concat((pxy.astype(tf.half), pwh.astype(tf.half)), 1)  # predicted box
            iou = tf.squeeze(bbox_iou(pbox, tboxi.astype(tf.half), CIoU=True))  # iou(prediction, target). shpe:[Nti]
            lbox += (1.0 - iou).mean()  # lbox as a mean iou of all candidate layer's objects, shape:[]

            # step 4.4: prepare tobj for lobj. tobj=max(iou) of all
            iou = tf.maximum(iou, 0).astype(tobj.dtype)  # clamp to min 0. (tbd: iou always positive)
            if self.sort_obj_iou:  # False by default
                j = iou.argsort()
                b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
            if self.gr < 1:  # default 1, otherwise, modify iou
                iou = (1.0 - self.gr) + self.gr * iou
            index = tf.transpose([b.astype(tf.int32), a.astype(tf.int32), gj, gi])  # tobj place idx. shape:[Nti,4]
            tobj = tf.tensor_scatter_nd_update(tobj, index,
                                               iou)  # scatter ious: tobj[b,a,gy,gx]=ioui, index shape:[Nt,4], iou shape: [Nt]

            # step 4.5: calc class loss by Binary Cross Entropy. Only in multi class case.
            if self.nc > 1:  # cls loss (only if multiple classes)
                # create [nt, nc] one_hot class array:
                t = tf.one_hot(indices=tclsi.astype(tf.int32), depth=pcls.shape[1])
                lcls += self.BCEcls(t, pcls)  # BCE, with SUM_OVER_BATCH_SIZE reduction: sum/(nof elements)
                # lcls +=tf.math.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(t, pcls,pos_weight=tf.constant(1.0 )))

            # step 4.6: calc mask loss as a mean of objects lossess, which are mean of all mask pixels BCE loss:
            # if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample mask by 4. Default: skip already d-sampled
            #     masks = tf.image.resize(masks, (mask_h, mask_w), method='nearest')[0]
            marea = tf.math.reduce_prod(xywhni[:, 2:], axis=1)  # normed target areas for mean calc. shape:[nti]
            # convert xywhn->xyxy, downsampled by 4, for bbox. cropping. shape: [nt,4]:
            mxyxy = xywh2xyxy(xywhni * tf.stack([mask_w, mask_h, mask_w, mask_h]))
            for bi in tf.unique(b)[0]:  # loop on images and sum lseg. unique(b) reduces all image's common targets
                j = b == bi  # mask targets with current iteration's image index bi shape:[nti], bool
                # In overlap object's mask pixels are colored by a per image index, otherwise index is batch global
                if self.overlap:  # Note: tidxs -1based targets indices in image (if overlap) or global
                    # Convert image's single mask to nti targets masks. Modify pixel vals from object index to 1s:
                    ltidxs = tidxsi  # current layer tidxs, shape [nt_layer,1]
                    imtixds = ltidxs[j]  # current image tidxs shape [ntcurrent,1], values: 0:ntcurrent
                    imtixds = tf.reshape(imtixds, [-1, 1, 1])  # shape:  [ntcurrent, 1,1]
                    immask = masks[bi.astype(tf.int32)][None]  # image mask. pix vals: 0:ntcurrent  shape: [1,160,160]
                    mask_gti = tf.where(immask == imtixds, 1.0, 0.0)  # shape: [ntcurrent,160,160]
                else:
                    # Take current image's objects masks. (no overlap, already mask per object, not colored):
                    # masks=tf.reshape(masks,[-1, masks.shape[2]/4, masks.shape[2]/4])
                    masks = tf.reshape(masks, [-1, 160, 160])
                    ltidxs = tidxsi  # current layer tidxs, shape [nt_layer,1]
                    lmasks = tf.gather(masks, ltidxs.astype(tf.int32))
                    mask_gti = lmasks[j]
                # acc image's lseg:
                lseg += self.single_mask_loss(mask_gti, pmask[j], proto[bi.astype(tf.int32)], mxyxy[j], marea[j])
        # 4.7 obj loss calc:
        obji = self.BCEobj(tobj, pi[..., 4])  # with SUM_OVER_BATCH_SIZE reduction: sum/(nof elements)
        lobj += obji * balancei  # increment loss on larger layers (80*80 vs 40*40 vs 20*20)
        # if self.autobalance:  # False by default (adjust lobj balance factor)
        #     self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        return lbox, lseg, lobj, lcls

    # @tf.function
    def __call__(self, preds, targets, masks):
        """
        Calc batch loss
        :param preds: model output. 2 tupple: preds[0]: list[3] per grid layer,shape:[b,na,gy,gx,4+1+nc+nm], gs=80,40,20
        preds[1]: proto of masks, shape:[b,nm,h/4,w/4] where currently nm=32,w,h=640
        :param targets: dataset labels. shape: [nt,6], where an entry consists of [imidx+cls+xywh],tf.float32
        :param masks: masks labels, shape: [b,h/4,w/4],tf.float32
        :return:
        1. loss sum: lbox+lobj+lcls+lseg
        2. concatenated loss: (lbox, lseg, lobj, lcls) shape: [b,4]
        """
        # step 1: unpack preds:
        p, proto = preds
        # p = [preds[0], preds[1], preds[2]]
        # proto = preds[3]
        bs, nm, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        # step 2: zero 3 loss accumulators:
        lcls = tf.zeros([1])  # class loss
        lbox = tf.zeros([1])  # box loss
        lobj = tf.zeros([1])  # object loss
        lseg = tf.zeros([1])  # segment loss
        # step 3: build targetst as entries for loss computation in 3 layers. build_targets normally expands nt.
        # tcls, tbox, indices, anchors, tidxs, xywhn are list[3] of tensor shape [Nti] where Nti is an expanded nof
        # targets. `indices` is list[3] of tuple:4 entries each of shape: [Nti].
        bsize = p[0].shape[0]
        tcls, tbox, indices, anchors, tidxs, xywhn = self.build_targets(targets, bsize,
                                                                        self.grids)  # each a list[nl], entry per layer

        # step 4: loop on 3 layers, calc and accumulate losses:
        for i, pi in enumerate(p):  # loop on 3 layer grids. accumulate 4 losses.
            tobj = tf.zeros(pi.shape[:4], dtype=pi.dtype)  # init target obj with all 0s shape:[b,na,gy,gx]
            lbox, lseg, lobj, lcls = self.calc_loss(pi, indices[i], anchors[i], xywhn[i], tbox[i], tcls[i], tidxs[i], proto,
                                                lbox, lseg, lobj, lcls, masks,
                                                mask_h, mask_w, tobj, self.balance[i], indices[i].shape[0]>0)

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        # mul by loss gains:
        lbox *= self.box_lg
        lobj *= self.obj_lg
        lcls *= self.cls_lg
        lseg *= self.box_lg / bs  # summed on batch images loop, here do average
        loss = lbox + lobj + lcls + lseg
        return loss * bs, tf.concat((lbox, lseg, lobj, lcls), axis=-1)

    # @tf.function
    def single_mask_loss(self, gt_mask, pmask, proto, xyxy, area):
        """ Description Calc mask loss as the mean of all input objects masks losses calculated separately.
        Each object mask loss is the mean of its mask pixels losses  calculated by BinaryCrossentropy,
        cropped by target bounding boxes.
        :param gt_mask: nti tmasks, pixels 1 or 0, tf.float32 shape:[nti,h/4,w/4]
        :param pmask: pred tensor's mask fields, tf.float32, shape: [nti,nm], where nm=32
        :param proto: model's proto output.   tf.float32, shape: [nm, 160,160]
        :param xyxy: bbox xyxy format, downsampled by 4, to math mask's scale. shape: [nti,4]
        :param area: targets' bboxes areas (normalized).  tf.float32, shape: [nti]
        :return: mask loss tf.float32, shape: []
        """
        # 1. produce pred mask as a product of pmask and proto:
        # 1a. reshape proto: ->[nm,h/4*w/4]. 1b. mult: [nti,nm]@[nm,h/4*w/4]->[nti,h/4*w/4] 1c. reshape:[nti,160,160]
        pred_mask = tf.reshape(pmask @ tf.reshape(proto, (self.nm, -1)), [-1, *proto.shape[1:]])
        # 2. Calc per pixels loss by BinaryCrossentropy:
        bse = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')  # setup BinaryCrossentropy
        mloss = bse(gt_mask[..., None], pred_mask[..., None])  # pixels mask loss. shape: [nti,160,160]
        # 3. Calc per object mask loss as a mean of object's pixels loss:
        targets_mask_loss = tf.math.reduce_mean(crop_mask(mloss, xyxy), axis=[1, 2])  # shape[nti]
        # 4. Calc mask loss as a mean of objects' mask losses, each divided by its area to equalize effect on mean:
        mask_loss = tf.math.reduce_mean(targets_mask_loss / area)  # shape: []
        # debug:
        # mask_loss = tf.reduce_mean(tf.square(gt_mask - pred_mask)/ tf.reshape(area,[-1,1,1]))

        return mask_loss

    # @tf.function
    def anchors_thresholding(self, t, anchors):
        """
        Threshold bbox width and height to anchors ratio. Filter out target entries with ratio above threshold.
        Envoked by build targets method
        :param t:  targets bbox coords scaled to grid dimenssions. shape(na,nt,8), dim 2: [imidx+cls+xywh+ai+ti], float
        :param anchors: anchors of a single grid layer, shape: [na,2], th.int32
        :return: t: survived targets, shape(na,nt_filtered,8),float
        """
        r = (t[..., 4:6] / anchors[:, None, :].astype(tf.float32))  # wh/anchors ratio. shape: [na, nt,2]
        j = tf.math.reduce_max(tf.math.maximum(r, 1 / r),
                               axis=-1) < self.anchor_t  # thresh bbox width to anchor ratio, bool shape: [na, nt]
        t = t[j]  # filter out unmatched to anchors targets. shape:  [nt, 8] where nt changed to nt_filtered
        return t

    # @tf.function
    def duplicate_bbox(self, t, gxy, grid_dims):
        """
        Duplicate bboxes to adjacent grid bboxes.

        :param t:  targets bbox coords scaled to grid dimenssions. shape(na,nt,8), dim 2: [imidx+cls+xywh+ai+ti], float

        :param t: targets, shape: [nt,8]  dim 1: [imidx+cls+xywh+ai+ti]
        :param gxy: grid scaled bbox centers, shape: [nt,2], float
        :param grid_dims: used to find bbox center dist from right/bottom edge. shape: [2], in yolov5: [80,80] or [40,40] or [20,20]
        :return:
            t: duplicated (unmodified) targets . shape: [nt_dup, 8], float, dim 1: [imidx+cls+xywh+ai+ti]
            offsets: tbox centers offsets wrt orig tbox. vals set:[0,0],[.5,0],[0,.5],[-.5,0],[0,-.5],shape:[nt_dup, 2]
        """

        g = 0.5  # max pred bbox center bias due to yolo operator-explaination follows
        # offsets to related neighbours:
        off = tf.constant(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ], dtype=tf.float32
           ) * g  # offsets
        # 2.b duplicate targets to adjacent grid squares. reason: xy preds transformed to px,y=sigmoid()*2-0.5,
        # i.e. -0.5<=pxy<=1.5, so pred may cross over square border, but still let be candidates to match target
        # If center in l/up/r/down halfs, marked j,k,l,m respectively, dup to one of 4 neighbors accordingly.
        # then add offset to duplicated entries to place those in adjacent grid squares.
        # gxy = t[:, 2:4]  # take bbox centers to determine entry duplication. shape: [nt,2]
        # dup to left/up if x/y in left/up half & gxy>1 i.e. a none left/up edge with adjacent squares.

        # left/up cpprds: ((gxy % 1 <  0.5) & (gxy > 1)).T:
        left_upper_coords = ((tf.math.less(tf.math.floormod(gxy, tf.constant(1.0)), tf.constant(g)))
                     & (tf.math.greater(gxy, tf.constant(1.)))).T  # bool, shape: j:[nt], k:[nt]
        j, k = tf.split(left_upper_coords, 2, axis=0, num=None, name='split')  # bool, shape: j:[nt], k:[nt]

        gxi = grid_dims - gxy  # inverse, i.e. offsets from box centers to squares' right/down ends. shape: [nt,2]
        # right/bottom coords: ((gxi % 1 < 0.5) & (gxi > 1)).T, gxi>1 means none edge, i.e. has neighbors:
        right_bottom_coords = tf.transpose((gxi % 1 < g) & (gxi > 1))
        l, m = tf.split(right_bottom_coords, 2, axis=0, num=None, name='split')  # bool, shape: l:[nt], m:[nt]
        # entries dup indications: center (always 1),4 adjacents if true:
        j = tf.concat((tf.ones_like(j), j, k, l, m),
                      axis=0)  # shape:[5,nt]: central (allways valid so akk 1s), and 4 offsets
        t = tf.tile(t[None], (5, 1, 1))  # tile by 5 - max duplication of each entry. shape: [5,nt, 8]
        t = t[j]  # filter valid entries' duplicates . shape: [valid dup nt, 8]
        offsets = tf.zeros_like(gxy)[None] + off[:, None]  # broadcast add 5 offsets to square. shape:[5,nt.2]
        offsets = offsets[j]  # filter valid offsets . shape: [valid dup nt, 2]
        return t, offsets
    # @tf.function
    def build_targets_per_layer(self, shape, targets,anchors):
        """

        :param grids: grid's shape - [80,80] or [40,40] or [20,20] for layers 0-2, int32
        :param targets:  # shape:[na, nt, imidx+cls+xywh+ai+ti]
        :param anchors: shape: [na,2]
        :return:
                ind: [batch ind,anchor ind, box center], shape: [nt,4]
                tbox: [x,y,w,h] x,y offsets from  squares corner, [nt,4]
                tcls: class. list size: [nt]
                tidxs: target indices - target index in image, shape: [nt]
                xywhn: xywh normalized shape: [nt, 4]
                ai: anchor indices. shape: [nt]

        :rtype:
        """
        # 2.a scale tbbox to grid, and threshold wh to annchor ratio:
        # shape = grids[i]  # anchors scale, shape: [na,2], grids[i]: [gy[i],gx[i]], shape[2]
        # update gain columns 2,3,4,5 by grid dims gsx[i],gsy[i] where gs are [[80,80],[40,40],[20,20]] for i=0:2
        gain = tf.concat([tf.ones([2]), shape[[1, 0, 1, 0]].astype(tf.float32), tf.ones([2])],
                         axis=0)  # [1,1,gy,gx,gy,gx,1,1]
        # gain = tf.tensor_scatter_nd_update(gain, [[2],[3],[4],[5]], tf.constant(shape)[[1, 0, 1, 0]].astype(tf.float32))
        # scale targets normalized bbox to grid dimensions, to math pred scales:
        t = tf.math.multiply(targets, gain)  # scale targets bbox coords to grid scale. shape(na,nt,8)
        if targets.shape[0]: # if  targets
            #  filtering targets by bbox-width to anchor ratio thresholding:
            t = self.anchors_thresholding(t, anchors)
            # 2.b duplicate targets to adjacent grid squares. reason: xy preds transformed to px,y=sigmoid()*2-0.5,
            gxy = t[:, 2:4]  # take bbox centers to determine entry duplication. shape: [nt,2]
            t, offsets = self.duplicate_bbox(t, gxy, gain[[2, 3]])
        else:
            t = targets[0]  # take a single dummy target entry
            offsets = 0

        bi, cls, gxy, gwh, ai, tidx = tf.split(t, [1, 1, 2, 2, 1, 1],
                                               axis=-1)  # split 8 words: (image, class), (grid_xy), (grid_wh), (anch_idx, ti_idx)

        gij = (gxy - offsets).astype(
            tf.int32)  # grid's square left corners. gij=gxy-offs giving left corner of grid square, shape: [nt,2]
        gij = tf.clip_by_value(gij, [0, 0],
                               [shape[0] - 1, shape[1] - 1])  # clip grid indices to grid bounderies, shape: [nt,2]
        ind = tf.concat([bi.astype(tf.int32), ai.astype(tf.int32), gij],
                        axis=1)  # [batch ind,anchor ind, box center], shape: [nt,4]
        tbox = tf.concat((gxy - gij.astype(tf.float32), gwh), 1)  # [x,y,w,h] x,y offsets from  squares corner
        ai = tf.squeeze(ai).astype(tf.int32)  # anchor indices. list.size: 3. shape: [nt]
        tcls = tf.squeeze(cls, axis=1)  # class. list size: [nt]
        tidxs = tf.squeeze(tidx, axis=1)  # target indices - target index in image, shape: [nt,1]=>[nt]
        xywhn = tf.concat((gxy, gwh), 1) / gain[2:6]  # xywh normalized shape: [nt, 4]

        return ind, tbox, tcls,tidxs,xywhn, ai
    # @tf.function
    def build_targets(self, targets, batch_size, grids):
        """
        Description: Arrange target dataset as entries for loss computation. Note that nof targets ar enormally expanded
        to match preds in neighbour grid squares, as explained.
        :param targets: batch dataset labels. tf.float32 tensor. shape:[nt,6], entry:imidx+cls+xywh
        :param batch_size: num of samples in batch. Output is structured accordingly
        :param grids: sizes of layers grids, calculated as iimage_size/strides, giving [[80,80],[40,40],[20,20]]

        :return:
        tcls: targets classes. list[3] per 3 grid layers. shapes: [[nt0], [nt1], [nt2]], nti: nof targets in layer i
        tbox: x,y,w,h where x,y are offset from grid square corner, for loss calc. list[3] per 3 grid layers. shapes: [[nt0,4],[nt1,4],[nt2,4]]
        indices: per layer list[3] of 4-tuple entries, pointing to targets' attributes: [b,a,gyi,gxi], each a tensor,
        shape [nti] nti:ntargets in layer i, b: image index in batch, a: matched anchor index, gyi,gxi: grid coords
        anch: selected anchor pairs pre target. list[3] per 3 grid layers.shapes: [[nt0,2], [nt1,2], [nt2,2]], float
        tidxs: runnig indices of target in image. list[3] per 3 grid layers.  shapes:  [[nt0], [nt1], [nt2]]
        xywhn: normalized targets bboxes. list[3] per 3 grid layers. shapes: [[nt0,4], [nt1,4], [nt2,4]]
        :rtype:
        """
        # step 1: dup targets na times, needed for loss per anchor. concat targets with ai (anchor idx) & ti (target idx)
        na, nt = self.na, targets.shape[0]  # nof anchors, nof targets in batch
        # tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], [] # init result lists

        # 1a. prepare ai- anchor indices of target in batch. shape:[na,nt], a row per anchor index
        ai = tf.tile(tf.reshape(tf.range(na, dtype=tf.float32), (na, 1)), [1, nt])

        # 1.b prepare ti-target index within image(s): in mask overlap mode, ti runs per image, otherwise, index is global. Example: 2 images,2 & 3
        # objects. ti=[[1,2,1,2,3],[1,2,1,2,3],[1,2,1,2,3]] if overlap, [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]] otherwise
        if self.overlap:
            # in mask overlap mode, ti used as mask colors,values 1:ti:
            ti = []  # target list of np entries. each holds na dups of range(nti), nti: nof objs in ith sample. shape: [na,nti]
            for idx in range(batch_size):  # loop on preds in batch,
                num = tf.math.reduce_sum((targets[:, 0:1] == idx).astype(tf.float32))  # nof all targets in image idx
                # entry shape:(na, nti). 1 based range(num) entries
                ti.append(tf.tile(tf.range(start=1, limit=num+1, dtype=tf.float32)[None], [na, 1]) ) #  shape: [na,nti]

            #  # concat list.
            ti = tf.concat(ti, axis=1)  # shape:(na, nt), nt nof all batch targets.
        else:  # no overlap: ti holds flat nt indices, where nt nof obj targets in the batch # shape: [na, nt]
            ti = tf.tile(tf.range(nt, dtype=tf.float32)[None], [na, 1])  #
        # 1.c duplicate targets na times:
        ttpa = tf.tile(targets[None], (na, 1, 1))  # tile targets per anchors. shape: [na, nt, 6]
        # 1.d  concat targets, ai and ti. shape:[na, nt,8]
        targets = tf.concat((ttpa, ai[..., None], ti[..., None]), 2)  # shape:[na, nt, imidx+cls+xywh+ai+ti]

        tcls, tbox, indices, anchors, tidxs, xywhn = [], [], [], [], [], []  # init result lists

        # step 2: loop on layers, append layer's target to lists
        for i in range(self.nl):
            ind, tboxl, tclsl,tidxsl,xywhnl, ai=self.build_targets_per_layer(grids[i], targets, self.anchors[i])

            indices.append(ind)
            tbox.append(tboxl)  # [x,y,w,h] x,y offsets from  squares corner
            anchors.append(
                self.anchors[i][ai])  # anchor indices. list.size: 3. shape: [nt]
            tcls.append(tclsl)  # class. list size: [nt]
            tidxs.append(tidxsl)  # target indices - target index in image, shape: [nt,1]=>[nt]
            xywhn.append(xywhnl)  # xywh normalized shape: [nt, 4]
        return tcls, tbox, indices, anchors, tidxs, xywhn  # arranged target values, each a list[nl]

    # @tf.function


# if __name__ == '__main__':
# for debug and testing:
def main():
    # hyp, na, nl, nc, nm, anchors
    na, nl, nc, nm = 3, 3, 80, 32
    box: 0.05  # box loss gain
    cls: 0.5  # cls loss gain
    cls_pw: 1.0  # cls BCELoss positive_weight
    obj: 1.0  # obj loss gain (scale with pixels)
    obj_pw: 1.0  # obj BCELoss positive_weight
    iou_t: 0.20  # IoU training threshold
    anchor_t: 4.0  # max threshold of anchor to bbox w,h ratio or w,h to anchor ratio

    box_lg, obj_lg, cls_lg, anchor_t = 0.005, 1.0, 0.5, 4  # box, obj &cls loss gains, anchor_t
    anchors_cfg = [[10, 13, 16, 30, 33, 23],  # P3/8
                   [30, 61, 62, 45, 59, 119],  # P4/16
                   [116, 90, 156, 198, 373, 326]]  # P5/32
    anchors = tf.reshape(anchors_cfg, [3, -1, 2])
    stride = [8, 16, 32]
    anchors = (anchors / tf.reshape(stride, (-1, 1, 1))).astype(tf.float32)
    fl_gamma = 0
    grids = tf.constant([[80, 80], [40, 40], [20, 20]])
    overlap = True
    loss = ComputeLoss(na, nl, nc, nm, stride, grids, anchors, overlap, fl_gamma, box_lg, obj_lg, cls_lg, anchor_t,
                       autobalance=False)

    b = 2  # batch

    p0 = tf.ones([b, nl, 80, 80, 5 + nc + nm])
    p1 = tf.ones([b, nl, 40, 40, 5 + nc + nm])
    p2 = tf.ones([b, nl, 20, 20, 5 + nc + nm])
    p = [p0, p1, p2]

    proto = tf.ones([b, nm, 160, 160], dtype=tf.float32)
    pred = [p, proto]

    nt = 10
    targets1 = tf.ones([int(nt / 2), 6], dtype=tf.float32)
    targets2 = tf.zeros([int(nt / 2), 6], dtype=tf.float32)
    targets = tf.concat([targets1, targets2], axis=0)

    masks = tf.ones([b, 160, 160], dtype=tf.float32)
    tot_loss, closs = loss(pred, targets, masks)

    return tot_loss, closs


if __name__ == '__main__':
    # FILE = Path(__file__).resolve()
    # ROOT = FILE.parents[1]  # YOLOv5 root directory
    # if str(ROOT) not in sys.path:
    #     sys.path.append(str(ROOT))  # add ROOT to PATH
    # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    tftot_loss, tfcloss = main()
