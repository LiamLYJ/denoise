import os
import tensorflow as tf
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

def load_batch(dataset_dir, batch_size, patches_per_img = 2, burst_length = 7, repeats =1, height = 128, width= 128, min_queue = 8,
                            to_shift = 1, upscale = 1, jitter=1, smalljitter = 1, shuffle = True):

    file_names = glob(os.path.join(dataset_dir, '*.png'))
    file_name_queue = tf.train.string_input_producer(file_names)
    _, img_file = tf.WholeFileReader().read(file_name_queue)
    img_pure = tf.image.decode_png(img_file)
    img = tf.random_crop(img_pure, size = [height, width, 1])

    patches = make_stack_hqjitter((tf.cast(img, tf.float32) / 255.),
                                  height, width, patches_per_img, burst_length, to_shift, upscale, jitter)

    unique = batch_size // repeats
    patches = tf.train.shuffle_batch(
        [patches],
        batch_size=unique,
        num_threads=2,
        capacity=min_queue + 3 * batch_size,
        enqueue_many=True,
        min_after_dequeue=min_queue)

    print('PATCHES =================', patches.get_shape().as_list())

    patches = make_batch_hqjitter(patches, burst_length, batch_size,
                                  repeats, height, width, to_shift, upscale, jitter, smalljitter)
    print ('after make_batch_hqjitter: ', patches.shape)

    return patches


if __name__ == '__main__':
    pass
