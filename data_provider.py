import os
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
dataset_data_provider = slim.dataset_data_provider
dataset = slim.dataset
queues = slim.queues
gfile = tf.gfile
from scipy.misc import imsave

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tf_utils import *
from utils import *
from glob import glob
import cv2

# regard as only test for one raw
def load_batch_real(dataset_dir, crop_size, format = '*.npy', burst_length = 7):
    file_names = glob(os.path.join(dataset_dir, format))
    file_names = sorted(file_names)
    assert(len(file_names) == burst_length)
    patches_data, size_list = crop_in_order(file_names, save_dir=None, crop_size=crop_size, with_data = True)
    patches = np.stack(patches_data, axis = -1)
    # return (b ,w ,h , burst_length), ('w:' 'h:')
    return patches, size_list[0]


# the training dats has the bayer pattern of RGGB
def load_batch(dataset_dir, batch_size, select_ch, patches_per_img = 2, burst_length = 7, repeats =1, height = 128, width= 128, min_queue = 8,
                            to_shift = 1, upscale = 1, jitter=1, smalljitter = 1, shuffle = True, keep_size = False, ):

    file_names = glob(os.path.join(dataset_dir, '*.png'))
    file_names = sorted(file_names)
    file_name_queue = tf.train.string_input_producer(file_names, shuffle = shuffle)
    _, img_file = tf.WholeFileReader().read(file_name_queue)
    img_pure = tf.image.decode_png(img_file)

    # select_ch is None: just raddom crop from img_pure
    # else: select channel and resize
    if select_ch is None:
        img = tf.random_crop(img_pure, size = [height, width, 1])
    else:
        # 20 need to be n times of 4 (RGGB)
        img_pure = tf.image.crop_to_bounding_box(img_pure, 0,0, height * 20, width*20)

        img = select_raw(img_pure, select_ch)
        if not keep_size:
            img = tf.random_crop(img_pure, size = [height, width, 1])

    height_next, width_next = img.get_shape().as_list()[0], img.get_shape().as_list()[1]
    patches = make_stack_hqjitter((tf.cast(img, tf.float32) / 255.),
                                  height_next, width_next, patches_per_img, burst_length, to_shift, upscale, jitter)
    unique = batch_size // repeats

    if shuffle:
        patches = tf.train.shuffle_batch(
            [patches],
            batch_size=unique,
            num_threads=2,
            capacity=min_queue + 3 * batch_size,
            enqueue_many=True,
            min_after_dequeue=min_queue)
    else:
        patches = tf.train.batch(
            [patches],
            batch_size=unique,
            num_threads=2,
            capacity=min_queue + 3 * batch_size,
            enqueue_many=True,)

    print('PATCHES =================', patches.get_shape().as_list())

    patches = make_batch_hqjitter(patches, burst_length, batch_size,
                                  repeats, height_next, width_next, to_shift, upscale, jitter, smalljitter)
    print ('after make_batch_hqjitter: ', patches.shape)
    return patches


if __name__ == '__main__':
    dataset_dir = './real_test'
    crop_size = 128
    patches, size = load_batch_real(dataset_dir, crop_size=crop_size)
    print (patches.shape)
