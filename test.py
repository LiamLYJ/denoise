import os
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
app = tf.app
flags = tf.flags
gfile = tf.gfile
from scipy.misc import imsave
from glob import glob

import net
from tf_utils import *
import data_provider
from demosaic_utils import *
import utils

flags.DEFINE_integer('patch_size', 128, 'The height/width of images in each batch.')

flags.DEFINE_string('ckpt_dir', './logs/',
                    'Directory where to write training.')

flags.DEFINE_string('dataset_dir', './data/sony/val/', 'where the data is ')
flags.DEFINE_integer('iter_num', 10, 'how many iter to run in test, in not use_fully_crop mode')

flags.DEFINE_string('mode', 'fast', 'normal, fast, fully_crop')
flags.DEFINE_string('itmd_dir', './itmd_save', 'itermediate path for saveing crop input data')
flags.DEFINE_integer('collect_num', 1, 'how many number to collect')

flags.DEFINE_integer('final_K', 5, 'size of filter')
flags.DEFINE_integer('final_W', 1, 'size of output channel')
flags.DEFINE_integer('burst_length', 7, 'size of input channel')
flags.DEFINE_integer('tile_scale', 4, 'scale of raw and input channel wise')

flags.DEFINE_string('layer_type', 'singlestd', 'Layers in singlestd.')
flags.DEFINE_string('save_path', './save_path', '')

flags.DEFINE_float('read_noise', 0.000000483, 'read noise from noise profile')
flags.DEFINE_float('shot_noise', 0.00059, 'shot noise from noise profile')

#choose specific channel
flags.DEFINE_string('select_ch', 'R', 'choose which channel to process')

FLAGS = flags.FLAGS

def test(FLAGS):
    mode = FLAGS.mode
    batch = 1
    height = width = FLAGS.patch_size
    final_W = FLAGS.final_W
    final_K = FLAGS.final_K
    burst_length = FLAGS.burst_length
    dataset_dir = os.path.join(FLAGS.dataset_dir)
    burst_length = FLAGS.burst_length
    ckpt_dir = FLAGS.ckpt_dir
    select_ch = FLAGS.select_ch
    tile_scale = FLAGS.tile_scale

    size_list = []
    tmp_dataset_dir = dataset_dir
    keep_size = False
    if mode is 'fully_crop':
        assert (FLAGS.itmd_dir is not None)
        if not gfile.Exists(FLAGS.itmd_dir):
            gfile.MakeDirs(FLAGS.itmd_dir)
        assert (FLAGS.collect_num > 0)
        # gerneate crop samples for one big input raw
        file_names = glob(os.path.join(dataset_dir, '*.png'))
        file_names = sorted(file_names)[0:FLAGS.collect_num]
        # size_list : every one with box_h, and box_w
        size_list = utils.crop_in_order(file_names, FLAGS.itmd_dir, FLAGS.patch_size)
        tmp_dataset_dir = FLAGS.itmd_dir
    if mode is 'fast':
        assert (not FLAGS.select_ch is None)
        keep_size = True

    truth = data_provider.load_batch(dataset_dir = tmp_dataset_dir, batch_size=1, select_ch=select_ch,
                                    patches_per_img = 1, min_queue=1,
                                    burst_length = burst_length, tile_scale=tile_scale, repeats=1, height = height,
                                    width = width, to_shift = 1., upscale = 1, jitter = 16, smalljitter = 2, shuffle = False,
                                    keep_size = keep_size,
                                    )
    truth = tf.reduce_mean(truth, axis= -2)

    # use noise randomly
    # sig_read = tf.pow(10., tf.random_uniform(
    #     [batch, 1, 1, 1], -3., -1.5))
    # sig_shot = tf.pow(10., tf.random_uniform(
    #     [batch, 1, 1, 1], -2., -1.))
    # noisy, _ = add_read_shot_tf(truth, sig_read, sig_shot)

    sig_read = FLAGS.read_noise * tf.ones([batch, 1, 1, 1])
    sig_shot = FLAGS.shot_noise * tf.ones([batch, 1, 1, 1])

    noisy = add_read_shot_tf(truth, sig_read, sig_shot, use_profile = True)
    noisy = tf.placeholder_with_default(
        noisy, [batch, None, None, burst_length], name='noisy')
    # take the referrene frame
    truth = truth[...,0]
    truth = tf.placeholder_with_default(truth, [batch, None, None], name='dt')

    # prepare for concating feed with noise level
    sig_read = tf.tile(
        sig_read, [1, tf.shape(noisy)[1], tf.shape(noisy)[2], 1])
    sig_shot = tf.tile(
        sig_shot, [1, tf.shape(noisy)[1], tf.shape(noisy)[2], 1])
    sig_tower = tf.concat([sig_shot, sig_read], axis=-1)
    sig_tower = tf.concat([sig_shot, sig_read], axis=-1)
    sig_tower = tf.placeholder_with_default(
        (sig_tower), [None, None, None, 2], name='sig_tower')

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
    sig_read = sig_reads['singlestd']

    # forward the net
    dnet = 'dnet-'
    demosaic = {}
    filters = {}

    with tf.variable_scope('generator'):
        noisy_sig = tf.concat([noisy,sig_read], axis = -1)
        key = dnet + 's1'
        if FLAGS.patch_size == 128:
            N_size = 3
        else:
            N_size = 2

        # in mode fast: save original big image
        if mode is 'fast':
            # only allow for final_K == 1
            # assert (final_K == 1)
            # save the original size of input
            ori_noisy = tf.identity(noisy)
            noisy = tile_resize(noisy, tile_scale, is_up =False)
            noisy_sig = tile_resize(noisy_sig, tile_scale, is_up=False)

        filts = net.convolve_net(input_stack = noisy_sig, noisy_input=noisy, final_K = final_K, final_W = final_W, ch0=64,
                                 N=N_size, D=3,
                      scope='get_filted', separable=False, bonus=False)

    if mode is 'fast':
        filts_big = tile_resize(filts, tile_scale, is_up = True)
        predict_image = convolve(ori_noisy, filts_big, final_K, final_W)
    else:
        predict_image = convolve(noisy, filts, final_K, final_W)
    demosaic[key] = predict_image
    demosaic[key] = demosaic[key][..., 0]

    sref = tf_ssim(tf.expand_dims(demosaic[key],axis = -1),
                   tf.expand_dims(truth, axis = -1))

    sref_results = []
    with tf.Session() as sess:
        saver = tf.train.Saver()
        tmp_ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
        saver.restore(sess, tmp_ckpt_path)
        print ('succes load model')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if mode is 'fully_crop':
            for i_index, i_file in enumerate(file_names):
                cur_name = i_file.split('/')[-1][:-4]
                cur_num = size_list[i_index]['w'] * size_list[i_index]['h']
                cur_after_filt = []
                cur_noisy_input = []
                cur_gt = []
                cur_ssim = []
                for i_step in range(cur_num):
                    after_filt,noisy_input, ground_truth, sref_result = sess.run(
                                    [demosaic[key], noisy[...,0], truth, sref])
                    cur_ssim.append(sref_result)
                    cur_after_filt.append(after_filt)
                    cur_noisy_input.append(noisy_input)
                    cur_gt.append(ground_truth)
                    if i_step % 10 == 0:
                        print ('processing')
                cur_size = size_list[i_index]
                # assemble fragmentation into one big image
                big_after_filt, big_noisy_input, big_gt = utils.assem_in_order([cur_after_filt, cur_noisy_input, cur_gt], cur_size)
                imsave(os.path.join(FLAGS.save_path, cur_name + '_gt.png'), big_gt)
                imsave(os.path.join(FLAGS.save_path, cur_name + '_input.png'), big_noisy_input)
                imsave(os.path.join(FLAGS.save_path, cur_name + '_after_filt.png'), big_after_filt)
                print ('ssim for %s'%(cur_name), np.mean(cur_ssim))

        elif mode is 'normal':
            # just choose random crop from
            for i_step in range(FLAGS.iter_num):
                after_filt,noisy_input, ground_truth, sref_result = sess.run(
                                [demosaic[key], noisy[...,0], truth, sref])
                # print ('after_filt shape: ', after_filt.shape)
                # print ('gt shape: ', ground_truth.shape)
                imsave(os.path.join(FLAGS.save_path,'%04d_'%(i_step) + '_gt.png'), ground_truth[0])
                imsave(os.path.join(FLAGS.save_path, '%04d_'%(i_step) + '_after_filt.png'), after_filt[0])
                imsave(os.path.join(FLAGS.save_path, '%04d_'%(i_step) + '_input.png'), noisy_input[0])
                sref_results.append(sref_result)
                print ('ssim: ', sref_result)

        elif mode is 'fast':
            for i_step in range(FLAGS.iter_num):
                big_filted, big_noisy_input, big_gt, sref_result = sess.run(
                    [demosaic[key], ori_noisy[..., 0], truth, sref])
                imsave(os.path.join(FLAGS.save_path,'%04d_'%(i_step) + '_gt.png'), big_gt[0])
                imsave(os.path.join(FLAGS.save_path, '%04d_'%(i_step) + '_after_filt.png'), big_filted[0])
                imsave(os.path.join(FLAGS.save_path, '%04d_'%(i_step) + '_input.png'), big_noisy_input[0])
                sref_results.append(sref_result)
                print ('ssim: ', sref_result)
        else:
            raise ValueError('wrong flag mode: %s'%(mode))
    coord.request_stop()
    coord.join(threads)


def main(_):
    if not gfile.Exists(FLAGS.save_path):
        gfile.MakeDirs(FLAGS.save_path)
    test(FLAGS)

if __name__ == '__main__':
    app.run()
