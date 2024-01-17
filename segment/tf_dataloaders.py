import tensorflow as tf
import math


from tf_data_reader import LoadImagesAndLabelsAndMasks

class DataLoader(object):

    def __call__(self, data_path, batch_size, imgsz, mask_ratio, mosaic, augment, hyp, overlap):
        """
        Creates generator dataset.
        :param data_path:
        :type data_path:
        :param batch_size:
        :type batch_size:
        :param imgsz:
        :type imgsz:
        :param mask_ratio:
        :type mask_ratio:
        :param mosaic:
        :type mosaic:
        :param augment:
        :type augment:
        :param hyp:
        :type hyp:
        :return:
        :rtype:
        """
        dataset =  LoadImagesAndLabelsAndMasks(data_path, imgsz, mask_ratio, mosaic, augment, hyp, overlap) #iterate by __getitem__
        if overlap: # single mask per image:
            dataset_loader = tf.data.Dataset.from_generator(dataset.iter,
                                                 output_signature=(
                                                     tf.TensorSpec(shape=[imgsz[0], imgsz[1], 3], dtype=tf.float32, ),
                                                     tf.RaggedTensorSpec(shape=[None, 5], dtype=tf.float32,
                                                                         ragged_rank=1),
                                                     tf.TensorSpec(shape=[160, 160], dtype=tf.float32),
                                                     tf.TensorSpec(shape=(), dtype=tf.string),
                                                                   tf.TensorSpec(shape=[3,2], dtype=tf.float32)
                                                 )
                                                 )
        else: # a mask per target, nti masks per image i, deploying a ragged tensor accordingly.
            dataset_loader = tf.data.Dataset.from_generator(dataset.iter,
                                                            output_signature=(
                                                                tf.TensorSpec(shape=[imgsz[0], imgsz[1], 3],
                                                                              dtype=tf.float32, ),
                                                                tf.RaggedTensorSpec(shape=[None, 5], dtype=tf.float32,
                                                                                    ragged_rank=1),
                                                                tf.RaggedTensorSpec(shape=[None, 160, 160], dtype=tf.float32,  ragged_rank=1),
                                                                tf.TensorSpec(shape=(), dtype=tf.string),
                                                                tf.TensorSpec(shape=[3, 2], dtype=tf.float32)
                                                            )
                                                            )

        dataset_loader=dataset_loader.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE) # batch dataset

        nb = math.ceil( len(dataset)/batch_size) # returns nof batch separately
        return dataset_loader, tf.concat(dataset.labels, 0), nb # labels tensor - returned for debug

#

if __name__ == '__main__':

    data_path_ = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train'
    imgsz_ = [640, 640]
    mosaic_ = True
    hyp_ = '../data/hyps/hyp.scratch-low.yaml'
    with open(hyp_, errors='ignore') as f_:
        hyp__ = yaml.safe_load(f_)  # load hyps dict

    degrees_, translate_, scale_, shear_, perspective_ = hyp__['degrees'],hyp__['translate'],hyp__['scale'],hyp__['shear'],hyp__['perspective']
    hgain, sgain, vgain, flipud, fliplr =hyp__['hsv_h'],hyp__['hsv_s'],hyp__['hsv_v'],hyp__['flipud'],hyp__['fliplr']
    augment_=False
    batch_size_=2
    mask_ratio_ = 4
    # dataset_loader = create_dataloader(data_path, batch_size, imgsz, mask_ratio, mosaic, augment, degrees, translate, hyp)
    create_dataloader=DataLoader()
    dataset_loader_ = create_dataloader(data_path_, batch_size_, imgsz_, mask_ratio_, mosaic_, augment_, hyp_)


