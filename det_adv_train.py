from __future__ import print_function
import os
import math
import h5py
import numpy as np
import tensorflow as tf
import scipy.misc as sm
import matplotlib.pyplot as plt

from utilities import *
from det_adv_net import DET_ADV_NET

def run_training(num_epoch, batch_size, lr, k):

    model_filename = 'det_adv'
    model_save_dir = './ckpt/' + model_filename
    pred_save_dir  = './output/' + model_filename
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)

    # load data from hd5 file
    Ih_data, Ib_data = load(...) # TODO! load data as (n, h, w, c)

    # build model graph
    N = Ih_data.shape[0]
    img_sz = Ih_data.shape[1:2]
    det_adv = DET_ADV_NET(img_sz)
    train_gen = tf.train.AdamOptimizer(lr).minimize(det_adv.gen_loss)
    train_dis = tf.train.AdamOptimizer(lr).minimize(det_adv.dis_loss)

    # open a training session
    sess = tf.InteractiveSession()

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # training
    cnt = k
    num_iter = int(math.ceil(N / batch_size))
    print("Start training ... %d iterations per epoch" %num_iter)
    for i in range(num_epoch):
        rand_idx = np.random.permutation(N)
        for it in range(num_iter):
            idx = rand_idx[it*batch_size:(it+1)*batch_size]
            Ih_batch, Ib_batch = Ih_data[idx], Ib_data[idx]
            if k == 0:
                train_step = train_gen
                k = cnt
                print("Training Generator:")
            else:
                train_step = train_dis
                k -= 1
                print("Training Discriminator:")
            _, dis_adv_loss, gen_adv_loss, gen_loss, l1_loss = \
                sess.run([train_step, det_adv.dis_adv_loss, det_adv.gen_adv_loss, det_adv.gen_loss, det_adv.l1_loss], \
                          feed_dict={det_adv.Ih: Ih_batch, det_adv.Ib: Ib_batch})
            if it % 1 == 0:
                print("\tIter [%d/%d] dis_loss=%.6f, gen_loss=%.6f, gen_adv_loss=%.6f, l1_loss=%.6f" \
                    %(it+1, num_iter, dis_adv_loss, gen_adv_loss, gen_loss, l1_loss))

        Ib_hat, dis_adv_loss, gen_adv_loss, gen_loss, l1_loss = \
            sess.run([det_adv.Ib_hat[rand_idx[0]], det_adv.dis_adv_loss, det_adv.gen_adv_loss, det_adv.gen_loss, det_adv.l1_loss], \
                      feed_dict={det_adv.Ih: Ih_data[rand_idx[0]], det_adv.Ib: Ib_data[rand_idx[0]]})
        print("Epoch [%d/%d] dis_loss=%.6f, gen_loss=%.6f, gen_adv_loss=%.6f, l1_loss=%.6f" \
                    %(it+1, num_iter, dis_adv_loss, gen_adv_loss, gen_loss, l1_loss))

        # save transferred image
        Ib = denormalize(Ib_data[rand_idx[0]])  # TODO! implement denormalize function in utilities
        Ib_hat = denormalize(Ib_hat)
        sm.imsave(os.path.join(pred_save_dir, '%07d.jpg'%(i+1)), montage([Ib,Ib_hat], [1,2]))

    # save model
    saver = tf.train.Saver(max_to_keep=10)
    saver.save(sess, os.path.join(model_save_dir, model_filename))


def main():
    num_epoch = 100
    batch_size = 1
    lr = 1e-3
    k  = 3  # num of iterations for discriminator to train
    run_training(num_epoch, batch_size, lr, k)


if __name__ == '__main__':
    main()