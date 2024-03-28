# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
AutoAnchor utils
"""

import random

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

from utils import TryExcept
from utils.general import LOGGER, TQDM_BAR_FORMAT, colorstr

PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m_anchors, m_stride):
    """
    Check anchors order against stride orders: Both order should be ascending or descending. Reasons: smaller stride
    head expected to be better for small objects, and the larger strides to larger. Sort anchors sizes accordingly.

    :param m:
    :type m:
    :return:
    :rtype:
    """
    a=tf.math.reduce_prod(m_anchors, axis=2)
    a = tf.math.reduce_mean(a,axis=-1)
    a = tf.reshape(a, [-1]) # mean anchor area per output layer
    da = a[-1] - a[0]  # delta of 3rd amd 1st layers anchors
    ds = m_stride[-1] - m_stride[0]  # delta of 3rd amd 1st layers strides
    if da and (tf.math.sign(da) != tf.math.sign (ds)):  # stride and anchor deltas should have same sign. Otherwise flip order
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m_anchors = tf.reverse(m_anchors, axis=0)
    return m_anchors


@TryExcept(f'{PREFIX}ERROR')
def check_anchors(dataset, m_stride, m_anchors, thr=4.0, imgsz=640):
    """
    Check anchor fit to dataset, recompute if necessary

    :param dataset: Used attributes: labels - only wh extracted, shapes-to scale wh.
    :param m_stride: strides to heads. list(3).

    :type m_stride:
    :param m_anchors: Value for check. shape: [nl,na,2]. Assumed in anchors are scaled by stride.
    :param thr: threshold used by fitness for anchors-dataset ratio.  float
    :param imgsz: model's input image shape. Used to scaled wh for anchors matching. scalar.
    :return:
        bpr: best possible recall i.e. portion of dataset entries for which best anchor ratio is above threshold, float
        aat: anchors above thresh, i.e. mean number of anchors which ratio to a dataset entry is above threshold, float
    :rtype:
    """
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.ones([shapes.shape[0], 1]) # np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale

    wh = tf.constant(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)]))#.float()  # wh
    def metric(k):  # compute metric
        """
        Calculates bpr - a metric to detemine fitness pf anchors set to given dataset. bpr is the portion of dataset's
        entries which ratio between either w & h to best fitting anchor, is inside boundary threshold, i.e.
        less than 4 and greater than 0.25.

        :param k: anchors scaled by strides. shape: [Nl*Na, 2] Nl: nof layers (normally 3), Na: anchors per layer (3)
        :return:
        aat: mean number (on all dataset entries) of anchors, which ratio to the dataset entry is above threshold (1/thr)
        bpr: best possible recall: portion of dataset entries for which best anchor ratio is within thr boundary.
        :rtype:
        """
        r = wh[:, None] / k[None] # shape: [Nentries,Nanchors,2], Nentries: number of data entries

        """
         # Pick min(r,1/r)
        * Note:* Here ratio is taken as min(r, 1/r), accordingly 1/thr is used as min thresh, and best ratio is the max.
        Alternatively, picking r=max(r,1/r) would lead to taking thr as max thresh, and best ratio would be the min.
        """
        x = tf.math.minimum(r, 1 / r)  # shape:[Ne.Na,2]
        x = tf.math.reduce_min(x, axis=2)# both w&h should comply to thresh. so pick worse of each couple  i.e. min ratio.  shape: [Nd,Na]
        # x hold ratios between all data and anchor entries. Pick the best data entry to anchor for each data entry
        best = tf.math.reduce_max(x, axis=1)  # shape: [Ns]
        # aat: anchors above thresh, i.e. mean number of anchors which ratio (x) to a dataset entry is above threshold:
        aat =  x > 1 / thr # Find thresh passing ratios, each data entry vs each anchor, Bool ,shape: [Nd.Na]
        # sum nof threshold passeing data-anchor ratios per data entry over all anchors:
        aat = tf.math.reduce_sum(tf.cast(aat, tf.float32), axis=1) # shape:[Nd]
        aat= tf.math.reduce_mean(aat,axis=0) # mean number of passed ratios over all data entries, float, scalar

        # bpr: best possible recall i.e. portion of dataset entries for which best anchor ratio is above threshold:
        bpr = tf.math.reduce_mean(tf.cast(best > 1 / thr, tf.float32), axis=0 ) # float scalr
        return bpr, aat

    stride = tf.reshape(m_stride, [-1,1,1]) # tf.reshape(stride, [-1,1,1]) # m.stride.to(m_anchors.device).view(-1, 1, 1)  # model strides
    anchors = m_anchors * stride  # current anchors
    # bpr, aat = metric(anchors.cpu().view(-1, 2))
    bpr, aat = metric(tf.reshape(anchors, [-1, 2]))

    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f'{s}Current anchors are a good fit to dataset ✅')
    else:
        LOGGER.info(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...')
        # na = m_anchors.numel() // 2  # number of anchors
        na = int(tf.size(m_anchors) // 2)  # number of anchors

        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            # m_anchors[:] = tf.reshape(anchors, m_anchors.shape) #.clone().view_as(m_anchors)
            m_anchors = tf.reshape(anchors, m_anchors.shape) #.clone().view_as(m_anchors)

            m_anchors=check_anchor_order(m_anchors, stride)  # must be in pixel-space (not grid-space)
            s = f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future:\n{m_anchors})'
            m_anchors /= stride # scale by strides to head
        else:
            s = f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
        LOGGER.info(s)
    return m_anchors


def kmean_anchors(dataset, n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans
    # random.seed(42)
    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics # todo merge metric methods
        """

        :param k:
        :type k:
        :param wh:
        :type wh:
        :return:
        x: ratios between each data entry and each anchor. shape: [Nd,Na]
        best: best data-anchor ratio for each data entry, shape: [Nd]
        :rtype:
        """
        r = wh[:, None] / k[None]
        x = tf.math.minimum(r, 1 / r).min(2)  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        # ss=tf.math.reduce_max (x, axis=1)
        return x, tf.math.reduce_max (x, axis=1)  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        """
        Fitness function:
        :param k: Anchors shape [Na,2]
        :return:
        fitness: Mean of best data entry-anchors threshold which pass threshold., scalar, float
        :rtype:
        """
        _, best = metric(tf.constant(k, dtype=tf.float32), wh) # best's shape:[Nd] Nd-dataset length
        return tf.math.reduce_mean(best * tf.cast(best > thr, tf.float32))  # fitness

    def print_results(k, verbose=True):
        """

        :param k: anchors array shape:[Na,2],  Na is normally 9
        :param verbose:
        :return:
        :rtype:
            k: sorted anchors(in value is not modification). shape:[Na,2]
        """
        k = k[np.argsort(k.prod(1))]  # sort anchor_w*anchor_h, wsmall to large
        x, best = metric(k, wh0)
        bpr = tf.math.reduce_mean(tf.cast(best > thr, tf.float32), axis=0)
        aat =  tf.math.reduce_mean(tf.cast(x > thr,tf.float32)) * n  # best possible recall, anch > thr

        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={ tf.math.reduce_mean(x):.3f}/{tf.math.reduce_mean(best):.3f}-mean/best, ' \
            f'past_thr-mean:={tf.math.reduce_mean(tf.gather_nd(x,tf.where(x > thr))):.3f}, candidate anchors: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    # if isinstance(dataset, str):  # *.yaml file
    #     with open(dataset, errors='ignore') as f:
    #         data_dict = yaml.safe_load(f)  # model dict
    #     from utils.dataloaders import LoadImagesAndLabels
    #     dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True) # scale sizes preserve aspect ratio, [Nd,2]
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # scale wh by scaled shape sizes

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size')
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # filter > 2 pixels


    # Kmeans init
    try:
        LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening shape:[2]
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
        # k=np.array([[ 11.85394845,  20.3769149 ], [ 34.77631254, 105.40426016], # todo remove debug
        #             [127.99777593, 143.31180036],
        #             [152.25140568, 221.02507565], [342.80499394, 369.42629671],
        #             [377.54238487, 392.93821182], [397.74617696, 450.82521723],
        #             [467.27809889, 507.96653015], [593.97202782, 597.89695867]])
    wh, wh0 = (tf.constant(x, dtype=tf.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format=TQDM_BAR_FORMAT)  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)
