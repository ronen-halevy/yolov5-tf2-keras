# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

# import torch
# import torch.nn as nn
from ..tf_general import xywh2xyxy
from .tf_general import crop_mask

from utils.tf_metrics import bbox_iou
from utils.torch_utils import de_parallel
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
    def __init__(self, hyp, na,nl,nc,nm,anchors, autobalance=False):
        self.na = na # number of anchors
        self.nc = nc  # number of classes
        self.nl = nl  # number of layers
        self.nm = nm  # number of masks


        # device = next(model.parameters()).device  # get model device
        h = hyp  # hyperparameters

        # Define criteria
        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = tf.losses.BinaryCrossentropy(from_logits=True)
        BCEcls = tf.losses.BinaryCrossentropy(from_logits=True)

        # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        # self.na = m.na  # number of anchors
        # self.nc = m.nc  # number of classes
        # self.nl = m.nl  # number of layers
        self.anchors = anchors
        self.overlap=True # Todo
        # self.device = device

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
                                            # .split((2, 2, 1, self.nc), 1))  # target-subset of predictions

                # Regression
                pxy = tf.sigmoid(pxy) * 2 - 0.5
                pwh = (tf.sigmoid(pwh) * 2) ** 2 * anchors[i]
                pbox = tf.concat((pxy, pwh), 1)  # predicted box
                iou = tf.squeeze(bbox_iou(pbox, tbox[i], CIoU=True))  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = tf.cast(tf.minimum(iou, 0), tobj.dtype)

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
                    lcls += self.BCEcls(pcls, t)  # BCE

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

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size
        loss = lbox + lobj + lcls + lseg

        print('loss', loss.numpy() * bs, lbox.numpy(), lobj.numpy(), lcls.numpy(), lseg.numpy())
        return loss* bs, tf.concat((lbox, lobj, lcls), axis=-1)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        pred_mask = tf.reshape(pred @ tf.reshape(proto, (self.nm, -1)),[ -1, *proto.shape[1:]])  # (n,32) @ (32,80,80) -> (n,80,80)
        bse = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
        loss=bse(pred_mask[...,None], gt_mask[...,None])
        return tf.math.reduce_mean(tf.math.reduce_mean(crop_mask(loss, xyxy), axis=[1,2]) / area)
        # return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()

        # return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()

    def build_targets(self, p, targets):
        # targets = targets.to_tensor() # todo

        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, num of targets in images batch
        tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], []

        gain = tf.ones([8])  # normalized to gridspace gain
        # ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        ai = tf.tile(tf.reshape(tf.range(na, dtype= tf.float32), (na, 1)), [1,nt]) # anchor index. shape: [na, nt]

        # objects overlap assign objects pixels by a per class index. CPixels common to multi assigned by greater index
        if self.overlap:
            batch = p[0].shape[0] # images in batch
            ti = []
            for idx in range(batch):# loop on batch to create targets index range per image
                num =tf.math.reduce_sum ( tf.cast(targets[:, 0:1] == idx, tf.float32)) # targets in image
                ti.append(tf.tile(tf.range(num, dtype=tf.float32 )[None], [na,1]) + 1)  #entries shape:(na, num)
            ti = tf.concat(ti, axis=1)  # e.g. batch=2 with 2&1 targets: [[1,2,1], [1,2,1], [1,2,1]], shape:(na, nt)
        else:# no overlap: flat ti indices, e.g. batch=2 with 2&1 targets: [[1,2,3], [1,2,3], [1,2,3]], shape:(na, nt)
            ti = tf.tile(tf.range(nt, dtype=tf.float32)[None], [na, 1])

        ttpa = tf.tile(targets[None], (na, 1,1)) # tile targets per anchors. shape: [na, nt, 6]
        targets = tf.concat((ttpa, ai[..., None], ti[..., None]), 2)  # concat anchor idx and ti idx. shape: [na, nt,8]

        g = 0.5  # bias
        # off = torch.tensor(
        #     [
        #         [0, 0],
        #         [1, 0],
        #         [0, 1],
        #         [-1, 0],
        #         [0, -1],  # j,k,l,m
        #         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        #     ],
        #     device=self.device).float() * g  # offsets

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
            # gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            # xyxy_gain = tf.tile(tf.slice(shape, [2],[2]), [2]) # todo check
            # from tensorflow.python.ops.numpy_ops import np_config
            # np_config.enable_numpy_behavior()
            # xyxy_gain = tf.constant(shape)[[3, 2, 3, 2]]  # xyxy gain
            # gain = tf.concat([gain[0:2],tf.cast(tf.constant(shape)[[3, 2, 3, 2]], tf.float32), gain[6:]], axis=0)# xyxy gain
            # gain = tf.concat([gain[0:2],tf.cast(tf.constant(shape)[[3, 2, 3, 2]], tf.float32), gain[6:]], axis=0)# xyxy gain
            # gain: [1, 1, gs, gs, gs, gs, 1, 1] where gs is in 80,40,20 according to prediction layer
            gain = tf.tensor_scatter_nd_update(gain, [[2],[3],[4],[5]], tf.cast(tf.constant(shape)[[3, 2, 3, 2]], tf.float32))
            # gain[2:6] =


            # Match targets to anchors: resize xywh to grid size:
            t = tf.math.multiply(targets, gain)  # shape(3,nt,8)
            if nt:
                # Limit ratio between gt boxes and anchors to within threshold (x4 ratio). Otherwise exclude gt box from
                # unmatching anchor's boxes records.:
                # Matches:

                # r = tf.math.divide(t[..., 4:6],  tf.cast(anchors[:,None,:], tf.float32) )# wh/anchors ratio. shape: [na, nt,2]
                r = (t[..., 4:6]/  tf.cast(anchors[:,None,:], tf.float32) )# wh/anchors ratio. shape: [na, nt,2]

                j = tf.math.reduce_max(tf.math.maximum(r, 1 / r), axis=-1) < self.hyp['anchor_t']  # compare, bool shape: [na, nt]

                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
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
                # gxi = tf.slice(gain, [2] ,[2]) - gxy  # inverse: offsets from square upper ends. shape: [nt,2]
                gxi = gain[[2, 3]] - gxy  # inverse

                j, k = ((gxy % 1 < g) & (gxy > 1)).T # j,k true if gt in l,up half. gxy<=1 l,u, edge square. shape:[nt].
                l, m = ((gxi % 1 < g) & (gxi > 1)).T# l,m true if gt in r,low half. gxi<=1 r,l, edge square. shape:[nt].
                j = tf.stack((tf.ones_like(j), j, k, l, m)) # 5 validity ind to main &4 adjacent squares. shape:[5,nt]
                t = tf.tile(t[None], (5, 1, 1))[j] # tile by 5 and filter valid. shape: [valid dup nt, 8]

                # t = tf.tile((5, 1, 1))[j] # tile by 5 and filter valid. shape: [valid dup nt, 8]

                offsets = (tf.zeros_like(gxy)[None] + off[:, None])[j] # gt indices offs. shpe:  [valid dup nt, 2]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, ati = tf.split(t, 4, axis=-1)  # (image, class), grid xy, grid wh, anchors
            (a,tidx), (b, c) =  tf.transpose(ati), tf.transpose(bc)  # anchors, image, class
            gij = tf.cast((gxy - offsets), tf.int32) # gij=gxy-offs giving left corner of grid square
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, tf.clip_by_value(gj,0, (shape[2] - 1)), tf.clip_by_value(gi ,0, shape[3] - 1)))  # image, anchor, grid

            tbox.append(tf.concat((gxy - tf.cast(gij, tf.float32), gwh), 1))  # xy modified : gxy - gij i.e. box_center-grid square indices
            # anch.append(anchors[a])  # anchors
            # anch.append(tf.gather(anchors, [tf.cast(a, tf.int32)], axis=0))   # anchors
            anch.append(anchors[tf.cast(a, tf.int32)])  # anchors

            tcls.append(c)  # class
            tidxs.append(tidx)
            xywhn.append(tf.concat((gxy, gwh), 1) / gain[2:6])  # xywh normalized


        return tcls, tbox, indices, anch, tidxs, xywhn

