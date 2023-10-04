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



np_config.enable_numpy_behavior()
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

    def __init__(self, na,nl,nc,nm,anchors, fl_gamma, box_lg, obj_lg, cls_lg, anchor_t, autobalance=False, label_smoothing=0.0):
        self.na = na # number of anchors
        self.nc = nc  # number of classes
        self.nl = nl  # number of layers
        self.nm = nm  # number of masks


        # device = next(model.parameters()).device  # get model device
        # h = hyp  # hyperparameters

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

        # m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0,  autobalance
        # self.na = m.na  # number of anchors
        # self.nc = m.nc  # number of classes
        # self.nl = m.nl  # number of layers
        self.anchors = anchors
        self.overlap=True # Todo
        # self.device = device
        self.box_lg=box_lg # box loss gain
        self.obj_lg=obj_lg # obj loss gain
        self.cls_lg=cls_lg # class loss gain
        self.anchor_t=anchor_t


    def __call__(self, preds, targets, masks):  # predictions, targets
        p, proto = preds
        bs, nm, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width

        lcls = tf.zeros([1])  # class loss
        lbox = tf.zeros([1])   # box loss
        lobj = tf.zeros([1])   # object loss
        lseg = tf.zeros([1])
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        tcls, tbox, indices, anchors, tidxs, xywhn = self.build_targets(p, targets)  # targets


        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = tf.zeros(pi.shape[:4], dtype=pi.dtype)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls, pmask = tf.split(pi[tf.cast(b, tf.int32), tf.cast(a, tf.int32), gj, gi], (2, 2, 1, self.nc, nm), 1)
                # Regression
                pxy = tf.sigmoid(pxy) * 2 - 0.5
                pwh = (tf.sigmoid(pwh) * 2) ** 2 * anchors[i]
                pbox = tf.concat((pxy, pwh), 1)  # predicted box
                iou = tf.squeeze(bbox_iou(pbox, tbox[i], CIoU=True))  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                # Objectness
                iou = tf.cast(tf.maximum(iou, 0), tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                # tobj[b, a, gj, gi] = iou  # iou ratio
                index = tf.transpose([tf.cast(b, tf.int32), tf.cast(a, tf.int32), gj, gi] )
                tobj= tf.tensor_scatter_nd_update(tobj,index, iou)

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # create [nt, nc] one_hot class array:
                    t= tf.one_hot(indices=tf.cast(tcls[i], tf.int32), depth=pcls.shape[1])
                    lcls += self.BCEcls( t, pcls)  # BCE

                # Mask regression
                if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                    masks = tf.image.resize(masks, (mask_h, mask_w), method='nearest')[0]
                marea = tf.math.reduce_prod(xywhn[i][:, 2:], axis=1)  # mask width, height normalized
                mxyxy = xywh2xyxy(xywhn[i] * tf.constant([mask_w, mask_h, mask_w, mask_h]))
                for bi in tf.unique(b)[0]:
                    j = b == bi  # matching index
                    if self.overlap:
                        mask_gti = tf.where(masks[tf.cast(bi, tf.int32)][None] == tf.reshape(tidxs[i][j], [-1, 1, 1]), 1.0, 0.0)

                    else:
                        mask_gti = masks[tidxs[i]][j]
                    lseg += self.single_mask_loss(mask_gti, pmask[j], proto[tf.cast(bi, tf.int32)], mxyxy[j], marea[j])

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
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
        return loss * bs, tf.concat((lbox, lobj, lcls, lseg), axis=-1)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        pred_mask = tf.reshape(pred @ tf.reshape(proto, (self.nm, -1)),[ -1, *proto.shape[1:]])  # (n,32) @ (32,80,80) -> (n,80,80)
        bse = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
        loss=bse(gt_mask[...,None], pred_mask[...,None])

        return tf.math.reduce_mean(tf.math.reduce_mean(crop_mask(loss, xyxy), axis=[1,2]) / area)


    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, num of targets in images batch
        tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], []

        gain = tf.ones([8])  # normalized to gridspace gain

        ai = tf.tile(tf.reshape(tf.range(na, dtype= tf.float32), (na, 1)), [1,nt]) # anchor index. shape: [na, nt]

        # objects overlap assign objects pixels by a per class index. CPixels common to multi assigned by greater index
        if self.overlap:
            batch = p[0].shape[0] # images in batch
            ti = []
            for idx in range(batch):# loop on batch to create targets index range per image
                num =tf.math.reduce_sum ( tf.cast(targets[:, 0:1] == idx, tf.float32)) # targets in image
                ti.append(tf.tile(tf.range(num, dtype=tf.float32 )[None], [na,1]) + 1)  #entries shape:(na, num)
            ti = tf.concat(ti, axis=1)  # e.g. batch=2 ,3&1 targets:[[1,2,3,1], [1,2,3,1], [1,2,3,1]],shape:(na, nt)
        else:# no overlap: flat ti indices, e.g. batch=2, 4&1 targets: [[1,2,3,4], [1,2,3,4], [1,2,3,4]], shape:(na, nt)
            ti = tf.tile(tf.range(nt, dtype=tf.float32)[None], [na, 1]) # shape: [na, nt]

        ttpa = tf.tile(targets[None], (na, 1,1)) # tile targets per anchors. shape: [na, nt, 6]
        targets = tf.concat((ttpa, ai[..., None], ti[..., None]), 2)  # concat targets, ai and ti idx. shape:[na, nt,8]

        g = 0.5  # bias
        off = tf.constant(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ], dtype=tf.float32
           ) * g  # offsets
        # loop on layers i.e. 3 output grids (80*80,40*40,20*20), calc target samples per each:
        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape # take layer's anchors and grid. p[i] shape: [na,gy,gx]
            # gain from tf.ones([8]) to  [1, 1, gs, gs, gs, gs, 1, 1]  where gs is [80,40,20] for i=0:2
            gain = tf.tensor_scatter_nd_update(gain, [[2],[3],[4],[5]], tf.cast(tf.constant(shape)[[3, 2, 3, 2]], tf.float32))
            # gain[2:6] =

            # Match targets to anchors: resize xywh to grid size:
            t = tf.math.multiply(targets, gain)  # shape(3,nt,8)
            if nt:
                # Limit ratio between gt boxes and anchors to within threshold (x4 ratio). Otherwise exclude gt box from
                # unmatching anchor's boxes records.:
                # Matches:

                r = (t[..., 4:6]/  tf.cast(anchors[:,None,:], tf.float32) )# wh/anchors ratio. shape: [na, nt,2]
                j = tf.math.reduce_max(tf.math.maximum(r, 1 / r), axis=-1) < self.anchor_t  # compare, bool shape: [na, nt]
                t = t[j]  # filter out unmatched to anchors targets. shape:  [nt, 8] where nt changed to nt_filtered

                # idea of offset: pred bbox center trainsformation is pxy.sigmoid()*2-0.5 giving 0.5<=pxy<=1.5 so preds
                # may transform to adjacent gid squares-upper half of l,u and lower hald of r,d, and need ti be matched
                # with gt boxes located there. Accordingly, gt boxes set is duplicated 5.  j, k,l,m is true if boxe
                # center in the left, upp, right, down hald of the square respectively.
                #

                # Offsets
                gxy = t[:, 2:4]  # grid xy. shape: [nt,2]
                # from tensorflow.python.ops.numpy_ops import np_config
                # np_config.enable_numpy_behavior()
                # inverse: offsets from square right/down ends. shape: [nt,2]:
                gxi = gain[[2, 3]] - gxy  # inverse
                #j,k true if located in left/up half part of square, & gxy>1 False if in l/up edge square. shape:[nt]:
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                #l,m true if located in right/low half part of square, & gxi>1 False if in r/l edge square ( shape:[nt]:
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = tf.stack((tf.ones_like(j), j, k, l, m)) # ind to 5 adjacent squares. shape:[5,nt]

                # target coords tiled to the 5 squares, and then filtered by j to relevant squares:
                t = tf.tile(t[None], (5, 1, 1))[j] # tile by 5 and filter valid. shape: [valid dup nt, 8]

                offsets = (tf.zeros_like(gxy)[None] + off[:, None])[j] # gt indices offs. shpe:  [valid dup nt, 2]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, ati = tf.split(t, 4, axis=-1)  # (image, class), grid xy, grid wh, anchors

            (a,tidx), (b, c) =  tf.transpose(ati), tf.transpose(bc)  # anchors, image, class

            gij = tf.cast((gxy - offsets), tf.int32) # gij=gxy-offs giving left corner of grid square
            gij = tf.clip_by_value(gij,[0,0], [shape[2] - 1,shape[3] - 1] )

            gi, gj = gij.T  # grid indices

            # Append  b,a,gj,gi:
            indices.append((b, a, gj, gi))

            # xc,yc,w,h where xc,yc is offset from grid squares corner:
            tbox.append(tf.concat((gxy - tf.cast(gij, tf.float32), gwh), 1)) # list.size: 3. shape: [nt, 4]

            anch.append(anchors[tf.cast(a, tf.int32)])   # anchor indices. list.size: 3. shape: [nt]

            tcls.append(c)  # class. list.size: 3. shape: nt
            tidxs.append(tidx) # target indices, i.e. running count of target in image shape: [nt]
            xywhn.append(tf.concat((gxy, gwh), 1) / gain[2:6])  # xywh normalized shape: [nt, 4]

        return tcls, tbox, indices, anch, tidxs, xywhn


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
    anchor_t: 4.0  # anchor-multiple threshold

    box_lg, obj_lg, cls_lg, anchor_t= 0.005, 1.0, 0.5 , 4
    anchors_cfg= [[10, 13, 16, 30, 33, 23],  # P3/8
     [30, 61, 62, 45, 59, 119],  # P4/16
     [116, 90, 156, 198, 373, 326]]  # P5/32
    anchors = tf.reshape(anchors_cfg, [3, -1, 2])
    stride=[8,16,32]
    anchors = tf.cast(anchors / tf.reshape(stride, (-1, 1, 1)), tf.float32)
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
