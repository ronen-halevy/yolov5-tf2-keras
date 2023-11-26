# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""





if __name__ == '__main__':
    import os
    import platform
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

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior() # allows running NumPy code, accelerated by TensorFlow


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps



class FocalLoss(tf.keras.layers.Layer):#nn.Module):
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

    def __init__(self, na,nl,nc,nm,stride,anchors, overlap,fl_gamma, box_lg, obj_lg, cls_lg, anchor_t, autobalance=False, label_smoothing=0.0):
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
        self.na = na # number of anchors
        self.nc = nc  # number of classes
        self.nl = nl  # number of layers
        self.nm = nm  # number of masks

        # Define criteria
        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = tf.losses.BinaryCrossentropy(from_logits=True)
        BCEcls = tf.losses.BinaryCrossentropy(from_logits=True)

        # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)  # positive, negative BCE targets

        # Focal loss
        # g = h['fl_gamma']  # focal loss gamma
        if fl_gamma > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, fl_gamma), FocalLoss(BCEobj, fl_gamma)
        # balance adjustments loss by incrementing larger layers' loss, as those may overfit earlier:
        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(stride).index(16) if autobalance else 0  # stride index for autobalance. n/a in default config
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0,  autobalance

        self.anchors = anchors
        self.overlap=overlap
        self.box_lg=box_lg # box loss gain
        self.obj_lg=obj_lg # obj loss gain
        self.cls_lg=cls_lg # class loss gain
        self.anchor_t=anchor_t

    def __call__(self, preds, targets, masks):
        """
        Calc batch loss
        :param preds: model output. 2 tupple: preds[0]: list[3] per grid layer,shape:[b,na,gs,gs,4+1+nc+nm], gs=80,40,20
        preds[1]: proto of masks, shape:[b,nm,h/4,w/4] where currently nm=32,w,h=640
        :param targets: dataset labels. shape: [nt,6], where an entry consists of [imidx+cls+xywh]
        :type targets: tf.float32
        :param masks: masks labels, shape: [b,h/4,w/4]
        :type masks: tf.float32
        :return:
        1. loss sum: lbox+lobj+lcls+lseg
        2. concatenated loss: (lbox, lseg, lobj, lcls) shape: [b,4]
        """

        p, proto = preds
        bs, nm, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width

        lcls = tf.zeros([1])  # class loss
        lbox = tf.zeros([1])  # box loss
        lobj = tf.zeros([1])  # object loss
        lseg = tf.zeros([1])  # segment loss
        tcls, tbox, indices, anchors, tidxs, xywhn = self.build_targets(p, targets)  # targets entry, each is list[nl]

        # Losses
        for i, pi in enumerate(p):  # loop on preds' 3 layers, calc & accumulate 4 losses types per each
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = tf.zeros(pi.shape[:4], dtype=pi.dtype)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls, pmask = tf.split(pi[b.astype(tf.int32), a.astype(tf.int32), gj, gi], (2, 2, 1, self.nc, nm), 1)
                # Box regression
                pxy = tf.sigmoid(pxy) * 2 - 0.5
                pwh = (tf.sigmoid(pwh) * 2) ** 2 * anchors[i]
                pbox = tf.concat((pxy, pwh), 1)  # predicted box
                iou = tf.squeeze(bbox_iou(pbox, tbox[i], CIoU=True))  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                # Objectness
                iou = tf.maximum(iou, 0).astype(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                # tobj[b, a, gj, gi] = iou  # iou ratio
                index = tf.transpose([b.astype(tf.int32), a.astype(tf.int32), gj, gi] )
                tobj= tf.tensor_scatter_nd_update(tobj,index, iou)

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # create [nt, nc] one_hot class array:
                    t= tf.one_hot(indices=tcls[i].astype(tf.int32), depth=pcls.shape[1])
                    lcls += self.BCEcls( t, pcls)  # BCE

                # Mask regression
                if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                    masks = tf.image.resize(masks, (mask_h, mask_w), method='nearest')[0]
                marea = tf.math.reduce_prod(xywhn[i][:, 2:], axis=1)  # mask width, height normalized
                mxyxy = xywh2xyxy(xywhn[i] * tf.constant([mask_w, mask_h, mask_w, mask_h]))
                for bi in tf.unique(b)[0]:
                    j = b == bi  # matching index
                    if self.overlap:
                        mask_gti = tf.where(masks[bi.astype(tf.int32)][None] == tf.reshape(tidxs[i][j], [-1, 1, 1]), 1.0, 0.0)

                    else:
                        mask_gti = masks[tidxs[i]][j]
                    lseg += self.single_mask_loss(mask_gti, pmask[j], proto[bi.astype(tf.int32)], mxyxy[j], marea[j])

            obji = self.BCEobj(tobj, pi[..., 4])

            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.box_lg
        lobj *= self.obj_lg
        lcls *= self.cls_lg
        lseg *= self.box_lg / bs
        loss = lbox + lobj + lcls + lseg
        return loss * bs, tf.concat((lbox, lseg, lobj, lcls), axis=-1)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        pred_mask = tf.reshape(pred @ tf.reshape(proto, (self.nm, -1)),[ -1, *proto.shape[1:]])  # (n,32) @ (32,80,80) -> (n,80,80)
        bse = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
        loss=bse(gt_mask[...,None], pred_mask[...,None])

        return tf.math.reduce_mean(tf.math.reduce_mean(crop_mask(loss, xyxy), axis=[1,2]) / area)


    def build_targets(self, p, targets):
        """
        Description: Arrange target dataset as entries for loss computation
        :param p: preds, (for batch and grid sizes only). list[3],shape:[b,na,gs,4+1+nc+nm],gs=[[80,80],[40,40],[20,20]]
        :param targets: dataset labels for rearrangemnt . tf.float32 tensor. shape:[nt,6], entry:imidx+cls+xywh
        :return:
        tcls: targets classes. list[3] per 3 grid layers. shapes: [[nt0], [nt1], [nt2]], nti: nof targets in layer i
        tbox: x,y,w,h where x,y are offset from grid square corner, for loss calc. list[3] per 3 grid layers. shapes: [[nt0,4],[nt1,4],[nt2,4]]
        indices: grid indices of targets. list[3] per 3 grid layers.shapes: [[nt0,4],[nt1,4],[nt2,4]]
        anch: selected anchor pairs pre target. list[3] per 3 grid layers.shapes: [[nt0,2], [nt1,2], [nt2,2]]
        tidxs: runnig indices of target in image. list[3] per 3 grid layers.  shapes:  [[nt0], [nt1], [nt2]]
        xywhn: normalized targets bboxes. list[3] per 3 grid layers. shapes: [[nt0,4], [nt1,4], [nt2,4]]
        :rtype:
        """
        # step 1: dup targets na times, needed for loss per anchor. concat targets with ai anchor idx & ti target idx
        na, nt = self.na, targets.shape[0]  # nof anchors, nof targets in batch
        tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], [] # init result lists

        gain = tf.ones([8]) # gain scales box coords to grid space coords. shape [8] values: [1,1,gs,gs,gs,gs,1,1]
        # 1a. prepare ai, anchor indices for target in batch. shape:[na,nt], a row per anchor index
        ai = tf.tile(tf.reshape(tf.range(na, dtype= tf.float32),(na, 1)),[1,nt])

        # 1.b prepare ti: if mask overlap mode, ti runs per image, otherwise, ti is global. Example: 2 images,2 & 3
        # objects. ti=[[1,2,1,2,3],[1,2,1,2,3],[1,2,1,2,3]] if overlap, [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]] otherwise
        if self.overlap:
            ti = [] # target list of np entries. each holds na dups of range(nti), nti: nof objs in ith sample. shape: [na,nti]
            for idx in range( p[0].shape[0]):# loop on preds in batch,
                num =tf.math.reduce_sum ( (targets[:, 0:1] == idx).astype(tf.float32)) # nof all targets in image idx
                ti.append(tf.tile(tf.range(num, dtype=tf.float32 )[None], [na,1]) + 1) #entry shape:(na, nti), +1 for 1 based entries
            #  # concat list.
            ti = tf.concat(ti, axis=1) # shape:(na, nt), nt nof all batch targets.
        else:# no overlap: ti holds flat nt indices, where nt nof obj targets in the batch # shape: [na, nt]
            ti = tf.tile(tf.range(nt, dtype=tf.float32)[None], [na, 1])#
        # 1.c duplicate targets na times:
        ttpa = tf.tile(targets[None], (na, 1,1)) # tile targets per anchors. shape: [na, nt, 6]
        targets = tf.concat((ttpa, ai[..., None], ti[..., None]), 2) # concat targets, ai and ti idx. shape:[na, nt,8]
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
        # step 2: loop on layers, append layer's target to lists

        for i in range(self.nl):
            # 2.a match targets to anchors: scale box to grid scale, then drop targets if box wh to anchor ratio (or its
            # inverse) is above threshold, current thresh is 4.
            anchors, shape = self.anchors[i], p[i].shape # anchors scale i, p[i].shape:[b,na,gy[i],gx[i],cls+xywh+nc+nm]
            # update gain columns 2,3,4,5 by grid dims gsx[i],gsy[i] where gs are [[80,80],[40,40],[20,20]] for i=0:2
            gain = tf.tensor_scatter_nd_update(gain, [[2],[3],[4],[5]], tf.constant(shape)[[3, 2, 3, 2]].astype(tf.float32))
            # scale targets normalized bbox to grid dimensions, to math pred scales:
            t = tf.math.multiply(targets, gain)  # scale targets coordinates to grid scale. shape(3,nt,8)
            if nt:
                #  match targets to anchors by Limit ratio between wh to anchor by to max thresh:
                r = (t[..., 4:6]/  anchors[:,None,:].astype(tf.float32) )# wh/anchors ratio. shape: [na, nt,2]
                j = tf.math.reduce_max(tf.math.maximum(r, 1 / r), axis=-1) < self.anchor_t  # compare, bool shape: [na, nt]
                t = t[j]  # filter out unmatched to anchors targets. shape:  [nt, 8] where nt changed to nt_filtered
                # 2.b duplicate targets to adjacent grid squares. reason: xy preds transformed to px,y=sigmoid()*2-0.5,
                # i.e. -0.5<=pxy<=1.5, so pred may cross over square border, but still let be candidates to match target
                # If center in l/up/r/down halfs, marked j,k,l,m respectively, dup to one of 4 neighbors accordingly.
                # then add offset to duplicated entries to place those in adjacent grid squares.
                gxy = t[:, 2:4]  # take bbox centers to determine entry duplication. shape: [nt,2]
                # dup to left/up if x/y in left/up half & gxy>1 i.e. a none left/up edge with adjacent squares.
                j, k = ((gxy % 1 < g) & (gxy > 1)).T # bool, shape: j:[nt], k:[nt]
                gxi = gain[[2, 3]] - gxy  # inverse: offsets from box center to square's right/down ends. shape: [nt,2]
                # dup to right/bottom if x/y in r/bottom half & gxi>1 i.e a none r/bottom edge with adjacent squares:
                l, m = ((gxi % 1 < g) & (gxi > 1)).T # bool, shape: l:[nt], m:[nt]
                # entries dup indications: center (always 1),4 adjacents if true:
                j = tf.stack((tf.ones_like(j), j, k, l, m)) # shape:[5,nt]
                t = tf.tile(t[None], (5, 1, 1))[j] # tile by 5 and filter valid. shape: [valid dup nt, 8]
                offsets = (tf.zeros_like(gxy)[None] + off[:, None])[j] # offsets wrt orig square. shpe:[valid dup nt, 2]
            else:
                t = targets[0] # take a single dummy target entry
                offsets = 0

            bc, gxy, gwh, ati = tf.split(t, 4, axis=-1)  # (image, class), grid xy, grid wh, anchors
            (a,tidx), (b, c) =  tf.transpose(ati), tf.transpose(bc)  # anchors, image, class
            gij = (gxy - offsets).astype(tf.int32) # gij=gxy-offs giving left corner of grid square
            gij = tf.clip_by_value(gij,[0,0], [shape[2] - 1,shape[3] - 1] )
            gi, gj = gij.T  # grid indices
            indices.append((b, a, gj, gi)) #  gj,gi target indices to grid squares
            # tbox entry, [xc,yc,w,h] where xc,yc are offsets from grid squares corner:
            tbox.append(tf.concat((gxy - gij.astype(tf.float32), gwh), 1)) # [x,y,w,h] x,y offsets from  squares corner
            anch.append(anchors[a.astype(tf.int32)])   # anchor indices. list.size: 3. shape: [nt]
            tcls.append(c)  # class. list size: [nt]
            tidxs.append(tidx) # target indices, i.e. running count of target in image shape: [nt]
            xywhn.append(tf.concat((gxy, gwh), 1) / gain[2:6])  # xywh normalized shape: [nt, 4]
        return tcls, tbox, indices, anch, tidxs, xywhn # arranged target values, each a list[nl]


# if __name__ == '__main__':
# for debug and testing:
def main():
    # hyp, na, nl, nc, nm, anchors
    na, nl, nc, nm =3,3,80,32
    box: 0.05  # box loss gain
    cls: 0.5  # cls loss gain
    cls_pw: 1.0  # cls BCELoss positive_weight
    obj: 1.0  # obj loss gain (scale with pixels)
    obj_pw: 1.0  # obj BCELoss positive_weight
    iou_t: 0.20  # IoU training threshold
    anchor_t: 4.0  # max threshold of anchor to bbox w,h ratio or w,h to anchor ratio

    box_lg, obj_lg, cls_lg, anchor_t= 0.005, 1.0, 0.5 , 4 # box, obj &cls loss gains, anchor_t
    anchors_cfg= [[10, 13, 16, 30, 33, 23],  # P3/8
     [30, 61, 62, 45, 59, 119],  # P4/16
     [116, 90, 156, 198, 373, 326]]  # P5/32
    anchors = tf.reshape(anchors_cfg, [3, -1, 2])
    stride=[8,16,32]
    anchors = (anchors / tf.reshape(stride, (-1, 1, 1))).astype(tf.float32)
    fl_gamma=0
    loss = ComputeLoss( na,nl,nc,nm,anchors, fl_gamma, box_lg, obj_lg, cls_lg, anchor_t,  autobalance=False)

    b=2 # batch

    p0=tf.ones([b,nl,80,80,5+nc+nm])
    p1=tf.ones([b,nl,40,40,5+nc+nm])
    p2=tf.ones([b,nl,20,20,5+nc+nm])
    p=[p0,p1,p2]

    proto =tf.ones([b,nm, 160,160], dtype=tf.float32)
    pred=[p,proto]

    nt=10
    targets1 = tf.ones([int(nt/2), 6], dtype=tf.float32)
    targets2 = tf.zeros([int(nt/2), 6], dtype=tf.float32)
    targets = tf.concat([targets1, targets2], axis=0)

    masks=tf.ones([b, 160, 160], dtype=tf.float32)
    tot_loss, closs = loss(pred, targets, masks)

    return tot_loss, closs

if __name__ == '__main__':

    # FILE = Path(__file__).resolve()
    # ROOT = FILE.parents[1]  # YOLOv5 root directory
    # if str(ROOT) not in sys.path:
    #     sys.path.append(str(ROOT))  # add ROOT to PATH
    # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    tftot_loss, tfcloss = main()
