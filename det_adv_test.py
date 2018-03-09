from __future__ import print_function

import os
import cv2
import sys
import ipdb
import math
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, 'models/')

from utilities import *
from det_adv_net import DET_ADV_NET

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'True for training time and flase for test')
flags.DEFINE_boolean('bn', True, 'Whether use batch normalization or not')
flags.DEFINE_boolean('use_ms', True, 'Whether use multi-scale or not')


def run_testing(batch_size, lr):

    model_filename  = 'det_adv_new_ms_grad_k=3_m=1_leaky_concat'
    model_save_dir  = './ckpt/' + model_filename
    pred_save_dir   = './test/' + model_filename
    test_model_name = model_filename + '-33000'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # load data from hd5 file
    Ih_data, Ib_data = load_from_hdf5('data/IH_test2.h5', 'data/IB_test2.h5')
    print('Data loading done.')

    # normalize data
    Ih_data, Ih_max = normalize(Ih_data)
    print('Normalizing Ih done.')
    Ib_data, Ib_max = normalize(Ib_data)
    print('Normalizing Ib done.')

    # gaussian filtering
    Ih_data_gauss = gaussian_filer_batch(Ih_data)

    # build model graph
    N = Ih_data.shape[0]
    img_sz = Ih_data.shape[1:]
    with tf.device('/gpu:0'):
        global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                trainable=False
                )
        det_adv   = DET_ADV_NET(img_sz)
        train_gen = tf.train.AdamOptimizer(lr).minimize(det_adv.gen_loss, var_list=det_adv.g_vars, global_step=global_step)
        train_gen_l1  = tf.train.AdamOptimizer(lr).minimize(det_adv.l1_loss, var_list=det_adv.g_vars, global_step=global_step)
        train_gen_adv = tf.train.AdamOptimizer(lr).minimize(det_adv.gen_loss, var_list=det_adv.g_vars, global_step=global_step)
        train_dis = tf.train.AdamOptimizer(lr).minimize(det_adv.dis_adv_loss, var_list=det_adv.d_vars, global_step=global_step)

    # open a training session
    sess = tf.InteractiveSession()

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # load checkpoint
    last_step = 0
    print('[*] Loading checkpoint ...')
    saver = tf.train.Saver(max_to_keep=10)
    model = os.path.join(model_save_dir, test_model_name)
    saver.restore(sess, model)
    print('[*] Loading success: %s!'%model)
    last_step = sess.run(global_step)

    # testing
    num_iter = int(math.ceil(N / float(batch_size)))
    last_epoch = last_step / (int(math.ceil(4059 / float(6))))

    pred_save_dir = os.path.join(pred_save_dir, 'epoch=%d'%last_epoch)
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)

    print("\n\n\nStart testing model trained %d epoch(s)... %d iterations in total\n\n\n" %(last_epoch, num_iter))
    idx_list = range(N)
    for it in range(num_iter):
        idx = idx_list[it*batch_size:min((it+1)*batch_size,N)]
        Ih_batch, Ib_batch = Ih_data[idx], Ib_data[idx]
        Ih_gauss_batch = Ih_data_gauss[idx]

        Ib_hat, dis_adv_loss, gen_adv_loss, gen_loss, l1_loss = \
               sess.run([det_adv.Ib_hat[0], det_adv.dis_adv_loss, det_adv.gen_adv_loss, det_adv.gen_loss, det_adv.l1_loss], \
                         feed_dict={det_adv.Ih: Ih_batch, det_adv.Ib: Ib_batch})

        print("Sample [%d/%d] dis_loss=%.8f, gen_loss=%.8f, gen_adv_loss=%.8f, l1_loss=%.8f" \
                %(it+1, num_iter, dis_adv_loss, gen_loss, gen_adv_loss, l1_loss))   

        # save transferred image
        Ih     = normalize_to_jpeg(denormalize(Ih_data[idx[0]], Ih_max[idx[0]]))
        Ib     = normalize_to_jpeg(denormalize(Ib_data[idx[0]], Ib_max[idx[0]]))
        Ib_hat = normalize_to_jpeg(denormalize(Ib_hat, Ib_max[idx[0]]))
        cv2.imwrite(os.path.join(pred_save_dir, '%07d.jpg'%(it+1)), \
            montage([Ih[:,:,0],Ib[:,:,0],Ib_hat[:,:,0]], [1,3]))


def main(unused):
    batch_size = 1
    lr = 1e-4
    run_testing(batch_size, lr)


if __name__ == '__main__':
    tf.app.run(main)