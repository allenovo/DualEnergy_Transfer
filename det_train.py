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
from det_net import DET_ADV_NET


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'True for training time and flase for test')
flags.DEFINE_boolean('bn', True, 'Whether use batch normalization or not')
flags.DEFINE_boolean('use_ms', True, 'Whether use multi-scale or not')


def run_training(num_epoch, batch_size, lr):

    model_filename = 'det_no_adv_new'
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
    Ih_data, Ib_data = \
            load_from_hdf5('data/IH_train2.h5', 'data/IB_train2.h5', False)
    print('Data loading done.')

    # normalize data
    Ih_data, Ih_max = normalize(Ih_data)
    print('Normalizing Ih done.')
    Ib_data, Ib_max = normalize(Ib_data)
    print('Normalizing Ib done.')
    # Ib_data = normalize(Ib_data_raw, Ih_max)

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
        train_gen_l1  = tf.train.AdamOptimizer(lr).minimize(det_adv.l1_loss, global_step=global_step)

    # open a training session
    sess = tf.InteractiveSession()

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # create summary writer
    tf.summary.scalar('l1_loss', det_adv.l1_loss)
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
    num_iter = int(math.ceil(N / float(batch_size)))
    print("Start training ... %d iterations per epoch" %num_iter)
    last_epoch = last_step / num_iter
    for i in range(int(last_epoch),num_epoch):
        rand_idx = np.random.permutation(N)
        for it in range(num_iter):
            idx = sorted(rand_idx[it*batch_size:min((it+1)*batch_size,N)].tolist())
            Ih_batch, Ib_batch = Ih_data[idx], Ib_data[idx]
            train_step = train_gen_l1

            _, l1_loss, summary = sess.run([train_step, det_adv.l1_loss, merged], \
                                                feed_dict={det_adv.Ih: Ih_batch, det_adv.Ib: Ib_batch})

            if it % 5 == 0:
                print("\tEpoch [%d/%d] Iter [%d/%d] l1_loss=%.8f"%(i+1, num_epoch, it+1, num_iter, l1_loss))

        Ib_hat, l1_loss, summary = sess.run([det_adv.Ib_hat[0], det_adv.l1_loss, merged], \
                                                feed_dict={det_adv.Ih: Ih_batch, det_adv.Ib: Ib_batch})
        print("Epoch [%d/%d] l1_loss=%.8f" %(i+1, num_epoch, l1_loss))
        sum_writer.add_summary(summary, i)

        # save transferred image
        Ih     = normalize_to_jpeg(denormalize(Ih_data[idx[0]], Ih_max[idx[0]]))
        Ib     = normalize_to_jpeg(denormalize(Ib_data[idx[0]], Ib_max[idx[0]]))
        Ib_hat = normalize_to_jpeg(denormalize(Ib_hat, Ib_max[idx[0]]))
        cv2.imwrite(os.path.join(pred_save_dir, '%07d.jpg'%(i+1)), \
            montage([Ih[:,:,0],Ib[:,:,0],Ib_hat[:,:,0]], [1,3]))

        if (i+1) % 50 == 0:
            # save model
            saver.save(sess, os.path.join(model_save_dir, model_filename), global_step=global_step)  

    # save model
    saver.save(sess, os.path.join(model_save_dir, model_filename))


def main(unused):
    num_epoch = 500
    batch_size = 4
    lr = 1e-4
    k  = 1  # num of iterations for generator to train
    m  = 1  # num of iterations for discriminator to train
    run_training(num_epoch, batch_size, lr)


if __name__ == '__main__':
    tf.app.run(main)