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


# the training dats has the bayer pattern of RGGB
def load_batch(dataset_dir, batch_size, select_ch, patches_per_img = 2, burst_length = 7, repeats =1, height = 128, width= 128, min_queue = 8,
                            to_shift = 1, upscale = 1, jitter=1, smalljitter = 1, tile_scale = 4, shuffle = True, keep_size = False, ):

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
        # default is crop image_pure with scale of 16 /4 (RGGB) = 5
        # img_pure = tf.image.crop_to_bounding_box(img_pure, 0,0, height * 20, width*20)
        img_pure = tf.random_crop(img_pure, size = [height *4 * tile_scale, width * 4 *tile_scale, 1])

        img = select_raw(img_pure, select_ch)
        if not keep_size:
            # img = tf.image.resize_images(img, size = [height, width], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # use the careful design downsampling
            img = tile_resize(img, tile_scale, is_up=False, not_batch = True)

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
    pass
