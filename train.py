import os
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
app = tf.app
flags = tf.flags
gfile = tf.gfile

import net
from tf_utils import *
from data_provider import *
from demosaic_utils import *

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_integer(
    'patch_size', 128, 'The height/width of images in each batch.')

flags.DEFINE_string('train_log_dir', './logs_sony/',
                    'Directory where to write training.')

flags.DEFINE_string('dataset_dir_train', './data/sony/train/', '')
flags.DEFINE_string('dataset_dir_val', './data/sony/val/', '')

flags.DEFINE_string('val_head', '100M;','val_file start with')

flags.DEFINE_float('learning_rate', .0001, 'The learning rate')

flags.DEFINE_float('anneal', .9998, 'Anneal rate')

flags.DEFINE_integer('max_number_of_steps', 100000000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer('final_K', 5, 'size of filter')
# flags.DEFINE_integer('final_K', 1, 'size of filter')
flags.DEFINE_integer('final_W', 3, 'size of output channel')
flags.DEFINE_integer('burst_length', 3, 'size of input channel')
flags.DEFINE_integer('use_noise', 1,
                     '1/0 use noise.')

flags.DEFINE_integer('save_iter', 500, 'save iter inter')
flags.DEFINE_integer('sum_iter', 5, 'sum iter inter')

lags.DEFINE_boolean('h_flip', True, 'horizontal fip')
flags.DEFINE_boolean('v_flip', True, 'vertical fip')
flags.DEFINE_float('rotate', 60.0, 'rotate angle')
flags.DEFINE_float('crop_prob', 0.5, 'crop_probability')
flags.DEFINE_float('crop_min_percent', 0.3, 'crop min percent' )
flags.DEFINE_float('crop_max_percent', 1.0, 'crop max percent' )

flags.DEFINE_float('mixup', 0.0, 'mix up for data augmentation')
flags.DEFINE_string('layer_type', 'singlestd',
                    'Layers in singlestd.')


FLAGS = flags.FLAGS

def train(FLAGS):
    batch_size = FLAGS.batch_size
    height = width = FLAGS.patch_size
    final_W = FLAGS.final_W
    final_K = FLAGS.final_K
    burst_length = FLAGS.burst_length
    dataset_dir_train = os.path.join(FLAGS.dataset_dir_train)
    dataset_dir_val = os.path.join(FLAGS.dataset_dir_val)
    dataset_file_name_train = FLAGS.dataset_file_name_train
    dataset_file_name_val = FLAGS.dataset_file_name_val
    burst_length = FLAGS.burst_length

    demosaic_truth = data_provider.load_batch(dataset_dir = dataset_dir_train, patches_per_img = 2, min_queue=2,
                                    burst_length = burst_length, batch_size=batch_size, repeats=2,
                                    height = height, width = width, to_shift = 1., upscale = 4, jitter = 16, smalljitter = 2)
                                    )

    # input_stack_val, gt_image_val = data_provider.load_batch(dataset_dir = dataset_dir_val, patches_per_img = 2, min_queue=2,
    #                                 burst_length = burst_length, batch_size=batch_size, repeats=2,
    #                                 height = height, width = width, to_shift = 1., upscale = 4, jitter = 16, smalljitter = 2)
    #                                 )

    sig_read = tf.pow(10., tf.random_uniform(
        [batch, 1, 1, 1], -3., -1.5))
    sig_shot = tf.pow(10., tf.random_uniform(
        [batch, 1, 1, 1], -2., -1.))

    truth_all = demosaic_truth
    dec = demosaic_truth
    noisy_, _ = add_read_shot_tf(dec, sig_read, sig_shot)

    print ('NOISY', noisy_.get_shape().as_list())
    print ('DT2', demosaic_truth.get_shape().as_list())

    # noisy = tf.clip_by_value(noisy_, 0.0, 1.0)
    noisy = noisy_

    # choose first frame as groundtruth
    demosaic_truth = demosaic_truth[...,0]
    print ('DT3', demosaic_truth.get_shape().as_list())

    dt = demosaic_truth
    nt = noisy

    sig_read = tf.tile(
        sig_read, [1, tf.shape(noisy)[1], tf.shape(noisy)[2], 1])
    sig_shot = tf.tile(
        sig_shot, [1, tf.shape(noisy)[1], tf.shape(noisy)[2], 1])
    sig_tower = tf.concat([sig_shot, sig_read], axis=-1)
    print ('sig_read shape: ', sig_read.shape)
    print ('sig_shot shape: ', sig_shot.shape)
    print ('sig_tower shape: ', sig_tower.shape)

    noisy = tf.placeholder_with_default(
        noisy, [None, None, None, BURST_LENGTH], name='noisy')
    dt = tf.placeholder_with_default(dt, [None, None, None], name='dt')
    sig_tower = tf.placeholder_with_default(
        (sig_tower), [None, None, None, 2], name='sig_tower')

    tf.add_to_collection('inputs', noisy)
    tf.add_to_collection('inputs', dt)
    tf.add_to_collection('inputs', sig_tower)
    print ('Added to collection')

    sig_shot = sig_tower[..., 0:1]
    sig_read = sig_tower[..., 1:2]

    sig_read_single_std = tf.sqrt(
        sig_read**2 + tf.maximum(0., noisy[..., 0:1]) * sig_shot**2)
    sig_read_dual_params = tf.concat([sig_read, sig_shot], axis=-1)
    sig_read_empty = tf.zeros_like(noisy[..., 0:0])

    sig_reads = {
        'singlestd': sig_read_single_std,
        'dualparams': sig_read_dual_params,
        'empty': sig_read_empty
    }

    sig_read = sig_reads[FLAGS.layer_type]

    noisy_sig = tf.concat([noisy, sig_read], axis=-1)
    with tf.variable_scope('generator'):
        if FLAGS.patch_size == 128:
            N_size = 3
        else:
            N_size = 2
        filts = net.convolve_net(input_stack = noisy_sig, final_K, final_W, ch0=64,
                                 N=N_size, D=3,
                      scope='get_filted', separable=False, bonus=False)

    gs = tf.Variable(0, name='global_step', trainable=False)
    predict_image = convolve(noisy, filts, final_K, final_W)

    anneal = FLAGS.anneal
    if anneal > 0:
        per_layer = convolve_per_layer(noisy, filts, final_K, final_W)
        for ii in range(burst_length):
            itmd = per_layer[..., ii] * burst_length


    # compute loss
    losses = []
    predict_image_srgb = sRGBforward(predict_image)
    gt_image_srgb = sRGBforward(gt_image)
    img_loss = FLAGS.img_loss_weight * basic_img_loss(gt_image_srgb, predict_image_srgb)
    losses.append(img_loss)

    slim.losses.add_loss(tf.reduce_sum(tf.stack(losses)))
    total_loss = slim.losses.get_total_loss()

    # check val loss
    predict_image_val = convolve(input_stack_val, filts, final_K, final_W)
    predict_image_val_srgb = sRGBforward(predict_image_val)
    gt_image_val_srgb = sRGBforward(gt_image_val)
    val_loss = basic_img_loss(gt_image_val_srgb, predict_image_val_srgb)

    # summaies
    input_image_sum = tf.summary.image('input_image', input_image)
    gt_image_sum = tf.summary.image('gt_image', gt_image)
    predict_image_sum = tf.summary.image('predict_image', predict_image)
    total_loss_sum = tf.summary.scalar('total_loss', total_loss)
    img_loss_sum = tf.summary.scalar('img_loss', img_loss)

    sum_total = tf.summary.merge_all()
    sum_val = tf.summary.scalar('val_loss', val_loss)

    # optimizer
    g_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step_g = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(
            total_loss, global_step=gs, var_list=g_vars)

    max_steps = FLAGS.max_number_of_steps

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        print ('Initializers variables')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        writer_train = tf.summary.FileWriter(os.path.join(FLAGS.train_log_dir,'train'), sess.graph)
        writer_val = tf.summary.FileWriter(os.path.join(FLAGS.train_log_dir,'val'), sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(max_to_keep=None)

        ckpt_path = tf.train.latest_checkpoint(FLAGS.train_log_dir)
        if ckpt_path is not None:
            print ('Restoring from', ckpt_path)
            saver.restore(sess, ckpt_path)


        for i_step in range(max_steps):
            _, loss, i, sum_total_ = sess.run([train_step_g, total_loss, gs, sum_total])
            if i_step % 5 == 0:
                print ('Step', i, 'loss =', loss)

            if i % FLAGS.save_iter == 0:
                print ('Saving ckpt at step', i)
                saver.save(sess, FLAGS.train_log_dir + 'model.ckpt', global_step=i)
                sum_val_ = sess.run(sum_val)
                writer_val.add_summary(sum_val_, i)

            if i % FLAGS.sum_iter == 0:
                writer_train.add_summary(sum_total_, i)
                print ('summary saved')

        coord.request_stop()
        coord.join(threads)


def main(_):
    if not gfile.Exists(FLAGS.train_log_dir):
        gfile.MakeDirs(FLAGS.train_log_dir)
    if not gfile.Exists(os.path.join(FLAGS.train_log_dir, 'train')):
        gfile.MakeDirs(os.path.join(FLAGS.train_log_dir, 'train'))
    if not gfile.Exists(os.path.join(FLAGS.train_log_dir, 'val')):
        gfile.MakeDirs(os.path.join(FLAGS.train_log_dir, 'val'))

    train(FLAGS)


if __name__ == '__main__':
    app.run()
