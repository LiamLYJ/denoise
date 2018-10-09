import os
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
app = tf.app
flags = tf.flags
gfile = tf.gfile

import net
from tf_utils import *
import data_provider

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_integer(
    'patch_size', 128, 'The height/width of images in each batch.')

flags.DEFINE_string('train_log_dir', './logs_sony/',
                    'Directory where to write training.')
flags.DEFINE_string('dataset_dir', './data/sony/', '')

flags.DEFINE_string('val_head', '100M;','val_file start with')

flags.DEFINE_float('learning_rate', .0001, 'The learning rate')

flags.DEFINE_float('anneal', .9998, 'Anneal rate')

flags.DEFINE_integer('max_number_of_steps', 100000000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer('final_K', 5, 'size of filter')
# flags.DEFINE_integer('final_K', 1, 'size of filter')
flags.DEFINE_integer('final_W', 3, 'size of output channel')
flags.DEFINE_integer('burst_size', 3, 'size of input channel')
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


FLAGS = flags.FLAGS

def train(FLAGS):
    batch_size = FLAGS.batch_size
    height = width = FLAGS.patch_size
    final_W = FLAGS.final_W
    final_K = FLAGS.final_K
    burst_size = FLAGS.burst_size
    dataset_dir = os.path.join(FLAGS.dataset_dir)
    dataset_file_name_train = FLAGS.dataset_file_name_train
    dataset_file_name_val = FLAGS.dataset_file_name_val

    input_stack, gt_image = data_provider.load_batch( ,
                                    val_head = 'None')

    input_stack_val, gt_image_val = data_provider.load_batch( , val_head = FLAGS.val_head)


    with tf.variable_scope('generator'):
        if FLAGS.patch_size == 128:
            N_size = 3
        else:
            N_size = 2
        filts = net.convolve_net(input_stack, final_K, final_W, ch0=64,
                                 N=N_size, D=3,
                      scope='get_filted', separable=False, bonus=False)
    gs = tf.Variable(0, name='global_step', trainable=False)

    predict_image = convolve(input_stack, filts, final_K, final_W)

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
