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
flags.DEFINE_boolean('train', True, 'True for training time and flase for test')
flags.DEFINE_boolean('bn', True, 'Whether use batch normalization or not')
flags.DEFINE_boolean('use_ms', True, 'Whether use multi-scale or not')


def run_training(num_epoch, batch_size, lr, k, m):

    model_filename = 'det_adv_w=1000_k=3_m=1'
    model_save_dir = './ckpt/'   + model_filename
    pred_save_dir  = './output/' + model_filename
    logs_save_dir  = './logs/'   + model_filename
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)
    if not os.path.exists(logs_save_dir):
        os.makedirs(logs_save_dir)

    # load data from hd5 file
    Ih_data, Ih_max, Ib_data, Ib_max = \
            load_from_hdf5('data/IH_train3_norm.h5', 'data/IB_train3_norm.h5', True)
    Ih_val_data, Ib_val_data = \
            load_from_hdf5('data/IH_validation3.h5', 'data/IB_validation3.h5', False)
    print('Data loading done.')

    # normalize data

    Ih_val_data, Ih_val_max = normalize(Ih_val_data)
    print('Normalizing Ih done.')
    Ib_val_data, Ib_val_max = normalize(Ib_val_data)
    print('Normalizing Ib done.')

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

    # create summary writer
    tf.summary.scalar('l1_loss', det_adv.l1_loss)
    tf.summary.scalar('gen_adv_loss', det_adv.gen_adv_loss)
    tf.summary.scalar('gen_loss', det_adv.gen_loss)
    tf.summary.scalar('dis_loss', det_adv.dis_adv_loss)
    tf.summary.scalar('dis_real_loss', det_adv.dis_real_loss)
    tf.summary.scalar('dis_fake_loss', det_adv.dis_fake_loss)
    merged = tf.summary.merge_all()
    sum_writer = tf.summary.FileWriter(logs_save_dir, sess.graph)

    # load checkpoint
    last_step = 0
    print('[*] Loading checkpoint ...')
    model = tf.train.latest_checkpoint(model_save_dir)
    saver = tf.train.Saver(max_to_keep=10)
    if model is not None:
        saver.restore(sess, model)
        print('[*] Loading success: %s!'%model)
        last_step = sess.run(global_step)
    else:
        print('[!] Loading failed ...')

    # training
    train_loss, val_loss = np.zeros((num_epoch,4)), np.zeros((num_epoch,4))
    gen_cnt, dis_cnt = k, m
    num_iter = int(math.ceil(N / float(batch_size)))
    print("Start training ... %d iterations per epoch" %num_iter)
    last_epoch = last_step / num_iter

    try:
        for i in range(int(last_epoch), num_epoch):
            rand_idx = np.random.permutation(N)
            for it in range(num_iter):
                idx = sorted(rand_idx[it*batch_size:min((it+1)*batch_size,N)].tolist())
                Ih_batch, Ib_batch = Ih_data[idx], Ib_data[idx]
                if gen_cnt > 0:
                    if gen_cnt == k and it % 5 == 0: print("Training Generator:")
                    train_step = train_gen
                    gen_cnt -= 1
                    if gen_cnt == 0: dis_cnt = m
                elif dis_cnt > 0:
                    if dis_cnt == m and it % 5 == 0: print("Training Discriminator:")
                    train_step = train_dis
                    dis_cnt -= 1
                    if dis_cnt == 0: gen_cnt = k

                _, dis_adv_loss, gen_adv_loss, gen_loss, l1_loss = \
                   sess.run([train_step, det_adv.dis_adv_loss, det_adv.gen_adv_loss, det_adv.gen_loss, det_adv.l1_loss], \
                             feed_dict={det_adv.Ih: Ih_batch, det_adv.Ib: Ib_batch})

                if it % 5 == 0:
                    print("\tEpoch [%d/%d] Iter [%d/%d] dis_loss=%.8f, gen_loss=%.8f, gen_adv_loss=%.8f, l1_loss=%.8f" \
                           %(i+1, num_epoch, it+1, num_iter, dis_adv_loss, gen_loss, gen_adv_loss, l1_loss))

            Ib_hat, dis_adv_loss, gen_adv_loss, gen_loss, l1_loss, summary = \
                    sess.run([det_adv.Ib_hat[0], det_adv.dis_adv_loss, det_adv.gen_adv_loss, det_adv.gen_loss, det_adv.l1_loss, merged], \
                              feed_dict={det_adv.Ih: Ih_batch, det_adv.Ib: Ib_batch})
            print("Epoch [%d/%d] dis_loss=%.8f, gen_loss=%.8f, gen_adv_loss=%.8f, l1_loss=%.8f" \
                    %(i+1, num_epoch, dis_adv_loss, gen_loss, gen_adv_loss, l1_loss))
            train_loss[i] = [dis_adv_loss, gen_adv_loss, gen_loss, l1_loss]
            sum_writer.add_summary(summary, i)

            if (i+1) % 50 == 0:
                # save model
                saver.save(sess, os.path.join(model_save_dir, model_filename), global_step=global_step)      

            # save transferred image
            Ih     = normalize_to_jpeg(denormalize(Ih_data[idx[0]], Ih_max[idx[0]]))
            Ib     = normalize_to_jpeg(denormalize(Ib_data[idx[0]], Ib_max[idx[0]]))
            Ib_hat = normalize_to_jpeg(denormalize(Ib_hat, Ib_max[idx[0]]))
            cv2.imwrite(os.path.join(pred_save_dir, '%07d.jpg'%(i+1)), \
                montage([Ih[:,:,0],Ib[:,:,0],Ib_hat[:,:,0]], [1,3]))

            # evaluate on validation set
            N_val = Ih_val_data.shape[0]
            num_iter_val = int(math.ceil(N_val / float(batch_size)))
            for it in range(num_iter_val):
                idx = range(it*batch_size,min((it+1)*batch_size,N_val))
                Ih_batch_val, Ib_batch_val = Ih_val_data[idx], Ib_val_data[idx]
                dis_adv_loss, gen_adv_loss, gen_loss, l1_loss = \
                    sess.run([det_adv.dis_adv_loss, det_adv.gen_adv_loss, det_adv.gen_loss, det_adv.l1_loss], \
                              feed_dict={det_adv.Ih: Ih_batch_val, det_adv.Ib: Ib_batch_val})
                val_loss[i] += [dis_adv_loss, gen_adv_loss, gen_loss, l1_loss]
            val_loss[i] /= float(num_iter_val)
            print("Validation: dis_loss=%.8f, gen_loss=%.8f, gen_adv_loss=%.8f, l1_loss=%.8f" \
                    %(val_loss[i][0], val_loss[i][1], val_loss[i][2], val_loss[i][3]))

    except KeyboardInterrupt:
        pass

    # save loss
    np.save(os.path.join(logs_save_dir, 'train_loss.npy'), train_loss)
    np.save(os.path.join(logs_save_dir, 'val_loss.npy'), val_loss)

    # save model
    saver.save(sess, os.path.join(model_save_dir, model_filename), global_step=global_step) 


def main(unused):
    num_epoch = 1000
    batch_size = 3
    lr = 1e-4
    k  = 3  # num of iterations for generator to train
    m  = 1  # num of iterations for discriminator to train
    run_training(num_epoch, batch_size, lr, k, m)


if __name__ == '__main__':
    tf.app.run(main)