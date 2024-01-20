import tensorflow as tf

def flatten_btargets(b_targets, shape0):
    """
     Flatten batched images targets tensors to a flat targets tensors, i.e. shape: [b, None, 5] to [nt,6]
     To preserve targets to image relation, bidx-imnage index,is concatenated to entries, which are now 6 words wide.
    :param b_targets: A ragged tensor with src targers. shape:[bi,None,5], where dim[1] holds nti, and dim2: [cls+xywh]
    :return:
    :rtype: Resultanned flatten target tensor, shape: [Nt,6] where Nt is nof targets in batch, 6 words: [imgid+cls+xywh]
    """
    targets = tf.reshape(b_targets.flat_values, [-1, 5])  # flatten ragged tensor shape: [bnt,5]
    # generate idx - target indices of related image:
    idx = tf.range(shape0)[..., None]  # image indices. shape: [b]

    def generate_imidx(targets, idx):
        """
        Produces image idx with size of nof image's targets.
        :param target: targets of image i, shape: [nti,5], float.
        :param idx: index in batch of image i,  tf.int32
        :return: A tensor with image duplicated for all image's targets. shape: [nti], tf.int32
        :rtype:
        """
        imidx = tf.fill([tf.shape(targets)[0]], idx)
        return imidx

    def get_ncnt(targets):
        """
        Produces image idx with size of nof image's targets.
        :param target: targets of image i, shape: [nti,5], float.
        :param idx: index in batch of image i,  tf.int32
        :return: A tensor with image duplicated for all image's targets. shape: [nti], tf.int32
        :rtype:
        """
        imidx = tf.shape(targets)[0]
        return imidx
    im_cnt_targets=[]
    for tar in b_targets:
        im_cnt_targets.append(tar.shape[0])
    # targets_cnt = tf.map_fn(fn=lambda t: get_ncnt(t), elems=(b_targets),
    #                   fn_output_signature=tf.int32)

    def fill_indices(imidxs, idx):
        """
        Produces image idx with size of nof image's targets.
        :param target: targets of image i, shape: [nti,5], float.
        :param idx: index in batch of image i,  tf.int32
        :return: A tensor with image duplicated for all image's targets. shape: [nti], tf.int32
        :rtype:
        """
        imidx = tf.fill([imidxs], idx)

        return imidx
    acc = [0]
    summ = 0
    for cnt in im_cnt_targets:
        summ+=cnt
        acc.append(summ)
    imidxs=tf.ragged.row_splits_to_segment_ids(
        acc, name=None, out_type=None
    )
    # imidxs = tf.map_fn(fn=lambda t: fill_indices(t[0], t[1]), elems=(targets_cnt, idx),
    #                   fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32))

    # imidxs=tf.ragged.range(imidxs)
    # generate imidxs - image index for each target. a ragged tensor, shape: [bi, None], int32
    # idx=tf.reshape(idx, [-1,1]) # does this line prevent unexplained error? TBD
    # imidxs = tf.map_fn(fn=lambda t: generate_imidx(t[0], t[1]), elems=(b_targets, idx),
    #                   fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32))
    # flatten indices. shape [bnt], i.e. nof targets in batch
    # imidxs = imidxs.flat_values
    # concat imidxs to target. result shape: [bnt, 6]
    imidxs=imidxs[..., None].astype(tf.float32)
    targets = tf.concat([imidxs, targets], axis=-1)

    # imidxs = []
    # for idx, target in enumerate(b_targets):
    #     imidx = tf.fill([target.shape[0]], idx)
    #     imidxs.append(imidx)
    # imidxs = tf.concat(imidxs,axis=0)
    # targets = tf.concat([imidxs[...,None].astype(tf.float32), targets], axis=-1)


    return targets
