import numpy as np
import tensorflow as tf


class DET_ADV_NET:

    def __init__(self, img_sz):
        self.l1_adv_weight = 1
        self.__build_graph__(img_sz)


    def __build_graph__(self, img_sz):
        
        Ih = tf.placeholder(tf.float32, [None]+[i for i in img_sz]) # High energy images
        Ib = tf.placeholder(tf.float32, [None]+[i for i in img_sz]) # Bone images

        # transfer high energy image to bone image
        Ib_hat = self.__transfer__(Ih)

        # discriminator
        torf_Ib     = self.__discriminator__(Ih, Ib, img_sz)
        torf_Ib_hat = self.__discriminator__(Ih, Ib_hat, img_sz)

        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(Ib - Ib_hat), axis=[1,2,3])

        # adversarial loss
        # for discriminator:
        dis_adv_loss = tf.reduce_mean(-tf.log(torf_Ib)-tf.log(1-torf_Ib_hat))
        # for generator:
        gen_adv_loss = tf.reduce_mean(-tf.log(torf_Ib_hat))

        # combination generator loss
        gen_loss = gen_adv_loss + self.l1_adv_weight * l1_loss

        self.l1_loss, self.dis_adv_loss, self.gen_adv_loss, self.gen_loss = \
            l1_loss, dis_adv_loss, gen_adv_loss, gen_loss


    def __transfer__(self, Ih):
        # encoder (refer to U-Net)
        # encodes high energy images to latent space
        z, res_info = self.__encoder__(Ih)

        # decoder (refer to U-Net)
        # transfer latent feature to bone images
        Ib_hat = self.__decoder__(z, img_sz, res_info)

        return Ib_hat


    def __encoder__(self, Ih):
        with tf.variable_scope('encoder', reuse=False):
            layers = [64, 128, 256, 512, 512, 512, 512]
            conv1 = tf.layers.conv2d(inputs=Ih, filters=layers[0], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=layers[1], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=layers[2], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            conv4 = tf.layers.conv2d(inputs=conv3, filters=layers[3], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            conv5 = tf.layers.conv2d(inputs=conv4, filters=layers[4], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            conv6 = tf.layers.conv2d(inputs=conv5, filters=layers[5], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            z     = tf.layers.conv2d(inputs=conv6, filters=layers[6], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            res_info = [conv1, conv2, conv3, conv4, conv5, conv6]

        return z, res_info


    def __decoder__(self, z, img_sz, res_info):
        with tf.variable_scope('decoder', reuse=False):
            layers = [img_sz[2], 64, 128, 256, 512, 512, 512][::-1]
            conv1 = tf.layers.conv2d_transpose(inputs=z, filters=layers[0], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.relu)
            conv2 = tf.layers.conv2d_transpose(inputs=tf.concat([conv1,res_info[-1]],3), filters=layers[1], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.relu)
            conv3 = tf.layers.conv2d_transpose(inputs=tf.concat([conv1,res_info[-2]],3), filters=layers[2], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.relu)
            conv4 = tf.layers.conv2d_transpose(inputs=tf.concat([conv1,res_info[-3]],3), filters=layers[3], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.relu)
            conv5 = tf.layers.conv2d_transpose(inputs=tf.concat([conv1,res_info[-4]],3), filters=layers[4], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.relu)
            conv6 = tf.layers.conv2d_transpose(inputs=tf.concat([conv1,res_info[-5]],3), filters=layers[5], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.relu)
            Ib_hat = tf.layers.conv2d_transpose(inputs=tf.concat([conv1,res_info[-6]],3), filters=layers[6], kernel_size=[4,4],
                            strides=(2,2), padding="valid", activation=tf.nn.relu)

            return Ib_hat


    def __discriminator__(self, Ih, Ib, img_sz):
        rate = [4, 4]   # num of patches for row and col
        assert(img_sz[0]%rate[0] == 0 and img_sz[1]%rate[1] == 0)
        patch_size = [1, img_sz[0]/rate[0], img_sz[1]/rate[1], 1]
        layers = [64, 64, 128, 128, 256, 512, 1]

        # extract patches from large image and send into discriminator as a batch
        Ih_patches = tf.extract_image_patches(Ih, ksizes=patch_size, strides=patch_size)
        Ih_patches = tf.reshape(Ih_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]])
        Ib_patches = tf.extract_image_patches(Ib, ksizes=patch_size, strides=patch_size)
        Ib_patches = tf.reshape(Ib_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]])

        with tf.variable_scope('discriminator', reuse=False):
            Ih_Ib = tf.concat([Ih_patches, Ib_patches], 3)
            conv1 = tf.layers.conv2d(inputs=Ih_Ib, filters=layers[0], kernel_size=[4,4],
                                strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=layers[1], kernel_size=[4,4],
                                strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=layers[2], kernel_size=[4,4],
                                strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            conv4 = tf.layers.conv2d(inputs=conv3, filters=layers[3], kernel_size=[4,4],
                                strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            conv5 = tf.layers.conv2d(inputs=conv4, filters=layers[4], kernel_size=[4,4],
                                strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            conv6 = tf.layers.conv2d(inputs=conv5, filters=layers[5], kernel_size=[4,4],
                                strides=(2,2), padding="valid", activation=tf.nn.leaky_relu)
            torf  = tf.layers.conv2d(inputs=conv6, filters=layers[6], kernel_size=[4,4],
                                strides=(2,2), padding="valid", activation=tf.sigmoid)

        # reshape results from the same image back to the same batch
        torf = tf.reduce_mean(tf.reshape(torf, [-1, rate[0]*rate[1]]), 1)

        return torf



