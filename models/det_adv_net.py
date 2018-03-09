import ipdb
import math
import numpy as np
import tensorflow as tf
import scipy.ndimage.filters as fi

from utilities import batch_norm


class DET_ADV_NET:  # Dual Energy Transfer with Adversarial Loss

    def __init__(self, img_sz, train=True):
        self.gen_adv_weight = 1
        self.gen_l1_weight  = 1000
        self.train = train

        self.__build_graph__(img_sz)


    def __build_graph__(self, img_sz):
        
        Ih = tf.placeholder(tf.float32, [None]+[i for i in img_sz]) # High energy images
        Ib = tf.placeholder(tf.float32, [None]+[i for i in img_sz]) # Bone images

        # transfer high energy image to bone image
        Ib_hat, pyramid = self.__transfer__(Ih, img_sz)

        # discriminator
        torf_Ib, logits_Ib         = self.__discriminator__(Ih, Ib, img_sz)
        torf_Ib_hat, logits_Ib_hat = self.__discriminator__(Ih, Ib_hat, img_sz, reuse=True)

        # adversarial loss
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(logits_Ib, tf.ones_like(torf_Ib)))
        d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(logits_Ib_hat, tf.zeros_like(torf_Ib_hat)))
        # for generator:
        g_adv_loss  = tf.reduce_mean(sigmoid_cross_entropy_with_logits(logits_Ib_hat, tf.ones_like(torf_Ib_hat)))
        # for discriminator:      
        d_loss = d_loss_real + d_loss_fake

        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(Ib_hat - Ib))

        # combination generator loss
        gen_loss = self.gen_adv_weight * g_adv_loss +  self.gen_l1_weight * l1_loss
 
        self.Ih, self.Ib, self.Ib_hat, self.pyramid = Ih, Ib, Ib_hat, pyramid
        self.torf_Ib, self.torf_Ib_hat = torf_Ib, torf_Ib_hat
        self.logits_Ib, self.logits_Ib_hat = logits_Ib, logits_Ib_hat
        self.l1_loss, self.dis_adv_loss, self.dis_real_loss, self.dis_fake_loss, \
            self.gen_adv_loss, self.gen_loss = \
            l1_loss, d_loss, d_loss_real, d_loss_fake, g_adv_loss, gen_loss

        # separate generator and discriminator variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'transfer' in var.name]


    def __transfer__(self, Ih, img_sz):
        with tf.variable_scope('transfer', reuse=False):
            # encoder (refer to U-Net)
            # encodes high energy images to latent space
            z, res_info = self.__encoder__(Ih)

            # decoder (refer to U-Net)
            # transfer latent feature to bone images
            Ib_hat, pyramid = self.__decoder__(z, img_sz, res_info)

        return Ib_hat, pyramid


    def __encoder__(self, Ih):
        with tf.variable_scope('encoder', reuse=False):
            '''
            layers = [64, 128, 256, 512, 512, 512, 512]
            conv1 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=Ih, filters=layers[0], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv2 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv1, filters=layers[1], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv3 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv2, filters=layers[2], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv4 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv3, filters=layers[3], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv5 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv4, filters=layers[4], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv6 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv5, filters=layers[5], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            z     = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv6, filters=layers[6], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            res_info = [conv1, conv2, conv3, conv4, conv5, conv6]
            '''

            layers = [64, 128, 256, 512, 512, 512, 512]
            conv1 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=Ih, filters=layers[0], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv2 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv1, filters=layers[1], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv3 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv2, filters=layers[2], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv4 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv3, filters=layers[3], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv5 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv4, filters=layers[4], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv6 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv5, filters=layers[5], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            z     = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv6, filters=layers[6], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            res_info = [conv1, conv2, conv3, conv4, conv5, conv6]

        return z, res_info


    def __downscale_decoder__(self, feature, img_sz):
        feature_sz = feature.get_shape().as_list()[1]
        with tf.variable_scope('downscale_decoder_' + str(feature_sz), reuse=False):
            ch_dim = feature.get_shape().as_list()[-1]
            conv1  = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=feature, filters=ch_dim/2, kernel_size=[5,5],
                            strides=(1,1), padding="same", activation=None)))
            conv2  = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv1, filters=ch_dim/4, kernel_size=[5,5],
                            strides=(1,1), padding="same", activation=None)))
            ds_img = tf.layers.conv2d(inputs=conv2, filters=img_sz[-1], kernel_size=[1,1],
                            strides=(1,1), padding="same", activation=None)
            # ds_img = tf.layers.conv2d(inputs=feature, filters=img_sz[-1], kernel_size=[3,3],
            #                 strides=(1,1), padding="same", activation=None)

        return ds_img


    def __decoder__(self, z, img_sz, res_info):
        pyramid = []
        with tf.variable_scope('decoder', reuse=False):
            '''
            layers = [img_sz[2], 64, 128, 256, 512, 512, 512][::-1]
            conv1 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=z, filters=layers[0], kernel_size=[4,4],
                            strides=(2,2), padding="same", activation=None)))
            conv2 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv1+res_info[-1], filters=layers[1], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv3 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv2+res_info[-2], filters=layers[2], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))

            conv4 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv3+res_info[-3], filters=layers[3], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv4_img = self.__downscale_decoder__(conv4, img_sz)

            conv5 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv4+res_info[-4], filters=layers[4], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv5_img = self.__downscale_decoder__(conv5, img_sz)

            conv6 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv5+res_info[-5], filters=layers[5], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv6_img = self.__downscale_decoder__(conv6, img_sz)

            Ib_hat = tf.layers.conv2d_transpose(inputs=conv6+res_info[-6], filters=layers[6], kernel_size=[5,5],
                            strides=(2,2), padding="same")
            Ib_hat = tf.tanh(Ib_hat + tf.image.resize_images(conv6_img,img_sz[:2]) + \
                        tf.image.resize_images(conv5_img,img_sz[:2]) + tf.image.resize_images(conv4_img,img_sz[:2]))
            pyramid = [Ib_hat, conv6_img, conv5_img, conv4_img]
            '''
            use_ms = tf.app.flags.FLAGS.use_ms
            layers = [img_sz[2], 64, 128, 256, 512, 512, 512][::-1]
            conv1 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d_transpose(inputs=z, filters=layers[0], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv1 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv1, filters=layers[0], kernel_size=[5,5],
                            strides=(1,1), padding="same", activation=None)))

            conv2 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d_transpose(inputs=tf.concat([conv1,res_info[-1]], axis=3), filters=layers[1], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv2 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv2, filters=layers[1], kernel_size=[5,5],
                            strides=(1,1), padding="same", activation=None)))

            conv3 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d_transpose(inputs=tf.concat([conv2,res_info[-2]], axis=3), filters=layers[2], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv3 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv3, filters=layers[2], kernel_size=[5,5],
                            strides=(1,1), padding="same", activation=None)))

            conv4 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d_transpose(inputs=tf.concat([conv3,res_info[-3]], axis=3), filters=layers[3], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv4 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv4, filters=layers[3], kernel_size=[5,5],
                            strides=(1,1), padding="same", activation=None)))

            conv5 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d_transpose(inputs=tf.concat([conv4,res_info[-4]], axis=3), filters=layers[4], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv5 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv5, filters=layers[4], kernel_size=[5,5],
                            strides=(1,1), padding="same", activation=None)))

            conv6 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d_transpose(inputs=tf.concat([conv5,res_info[-5]], axis=3), filters=layers[5], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)))
            conv6 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv6, filters=layers[5], kernel_size=[5,5],
                            strides=(1,1), padding="same", activation=None)))

            Ib_hat = tf.layers.conv2d_transpose(inputs=conv6+res_info[-6], filters=layers[6], kernel_size=[5,5],
                            strides=(2,2), padding="same", activation=None)

            if use_ms:
                conv4_img = self.__downscale_decoder__(conv4, img_sz)
                conv5_img = self.__downscale_decoder__(conv5, img_sz)
                conv6_img = self.__downscale_decoder__(conv6, img_sz)
                Ib_hat    = Ib_hat + tf.image.resize_images(conv4_img,img_sz[:2]) \
                                   + tf.image.resize_images(conv5_img,img_sz[:2]) \
                                   + tf.image.resize_images(conv6_img,img_sz[:2])
                pyramid = [Ib_hat, conv4_img, conv5_img, conv6_img]

            Ib_hat = tf.tanh(Ib_hat)

            return Ib_hat, pyramid


    def __discriminator__(self, Ih, Ib, img_sz, reuse=False):
        rate = [8, 8]   # num of patches for row and col
        assert(img_sz[0]%rate[0] == 0 and img_sz[1]%rate[1] == 0)
        patch_size = [1, img_sz[0]/rate[0], img_sz[1]/rate[1], 1]
        layers = [64, 128, 256, 512]

        # concatenate gradients
        x_weight = tf.reshape(tf.constant([[+1,0,-1],[+2,0,-2],[+1,0,-1]], tf.float32), [3, 3, 1, 1])
        y_weight = tf.reshape(tf.constant([[+1,+2,+1],[0,0,0],[-1,-2,-1]], tf.float32), [3, 3, 1, 1])

        Ih_diff_x = tf.nn.conv2d(Ih, x_weight, [1, 1, 1, 1], 'SAME')
        Ih_diff_y = tf.nn.conv2d(Ih, y_weight, [1, 1, 1, 1], 'SAME')
        Ib_diff_x = tf.nn.conv2d(Ib, x_weight, [1, 1, 1, 1], 'SAME')
        Ib_diff_y = tf.nn.conv2d(Ib, y_weight, [1, 1, 1, 1], 'SAME')

        Ih_diff = tf.concat([Ih_diff_x, Ih_diff_y], 3)
        Ib_diff = tf.concat([Ib_diff_x, Ib_diff_y], 3)

        # extract patches from large image and send into discriminator as a batch
        Ih_patches = tf.extract_image_patches(Ih_diff, ksizes=patch_size, strides=patch_size, rates=[1,1,1,1], padding="VALID")
        Ih_patches = tf.reshape(Ih_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]*2])
        Ib_patches = tf.extract_image_patches(Ib_diff, ksizes=patch_size, strides=patch_size, rates=[1,1,1,1], padding="VALID")
        Ib_patches = tf.reshape(Ib_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]*2])

        # extract patches from large image and send into discriminator as a batch
        # Ih_patches = tf.extract_image_patches(Ih, ksizes=patch_size, strides=patch_size, rates=[1,1,1,1], padding="VALID")
        # Ih_patches = tf.reshape(Ih_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]])
        # Ib_patches = tf.extract_image_patches(Ib, ksizes=patch_size, strides=patch_size, rates=[1,1,1,1], padding="VALID")
        # Ib_patches = tf.reshape(Ib_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]])

        with tf.variable_scope('discriminator', reuse=reuse):
            Ih_Ib = tf.concat([Ih_patches, Ib_patches], 3)
            conv1 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=Ih_Ib, filters=layers[0], kernel_size=[5,5],
                                strides=(2,2), padding="same", activation=None)))
            conv2 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv1, filters=layers[1], kernel_size=[5,5],
                                strides=(2,2), padding="same", activation=None)))
            conv3 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv2, filters=layers[2], kernel_size=[5,5],
                                strides=(2,2), padding="same", activation=None)))
            conv4 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv3, filters=layers[3], kernel_size=[5,5],
                                strides=(2,2), padding="same", activation=None)))
            conv5 = tf.layers.dense(tf.contrib.layers.flatten(conv4), 1)

        # reshape results from the same image back to the same batch
        logits = tf.reduce_mean(tf.reshape(conv5, [-1, rate[0]*rate[1]]), 1)
        torf   = tf.nn.sigmoid(logits)

        return torf, logits



# class DET_ADV_NET:  # Dual Energy Transfer with Adversarial Loss

#     def __init__(self, img_sz, train=True):
#         self.gen_adv_weight = 1
#         self.gen_l1_weight  = 1000
#         self.train = train

#         self.__build_graph__(img_sz)


#     def __build_graph__(self, img_sz):
        
#         Ih = tf.placeholder(tf.float32, [None]+[i for i in img_sz]) # High energy images
#         Ib = tf.placeholder(tf.float32, [None]+[i for i in img_sz]) # Bone images

#         Ih_gauss = tf.placeholder(tf.float32, [None]+[i for i in img_sz]) # High energy images
#         self.Ih_hfreq = Ih - Ih_gauss

#         # transfer high energy image to bone image
#         Ib_hat, pyramid = self.__transfer__(tf.concat([Ih, self.Ih_hfreq], 3), img_sz)

#         # discriminator
#         torf_Ib, logits_Ib         = self.__discriminator__(Ih, Ib, img_sz)
#         torf_Ib_hat, logits_Ib_hat = self.__discriminator__(Ih, Ib_hat, img_sz, reuse=True)

#         # adversarial loss
#         def sigmoid_cross_entropy_with_logits(x, y):
#             try:
#                 return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
#             except:
#                 return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

#         d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(logits_Ib, tf.ones_like(torf_Ib)))
#         d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(logits_Ib_hat, tf.zeros_like(torf_Ib_hat)))
#         # for generator:
#         g_adv_loss  = tf.reduce_mean(sigmoid_cross_entropy_with_logits(logits_Ib_hat, tf.ones_like(torf_Ib_hat)))
#         # for discriminator:      
#         d_loss = d_loss_real + d_loss_fake

#         # L1 loss
#         l1_loss = tf.reduce_mean(tf.abs(Ib_hat - Ib))

#         # combination generator loss
#         gen_loss = self.gen_adv_weight * g_adv_loss +  self.gen_l1_weight * l1_loss
 
#         self.Ih, self.Ib, self.Ib_hat, self.pyramid = Ih, Ib, Ib_hat, pyramid
#         self.Ih_gauss = Ih_gauss
#         self.torf_Ib, self.torf_Ib_hat = torf_Ib, torf_Ib_hat
#         self.logits_Ib, self.logits_Ib_hat = logits_Ib, logits_Ib_hat
#         self.l1_loss, self.dis_adv_loss, self.dis_real_loss, self.dis_fake_loss, \
#             self.gen_adv_loss, self.gen_loss = \
#             l1_loss, d_loss, d_loss_real, d_loss_fake, g_adv_loss, gen_loss

#         # separate generator and discriminator variables
#         t_vars = tf.trainable_variables()
#         self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
#         self.g_vars = [var for var in t_vars if 'transfer' in var.name]


#     def __transfer__(self, Ih, img_sz):
#         with tf.variable_scope('transfer', reuse=False):
#             # encoder (refer to U-Net)
#             # encodes high energy images to latent space
#             z, res_info = self.__encoder__(Ih)

#             # decoder (refer to U-Net)
#             # transfer latent feature to bone images
#             Ib_hat, pyramid = self.__decoder__(z, img_sz, res_info)

#         return Ib_hat, pyramid


#     def __encoder__(self, Ih):
#         with tf.variable_scope('encoder', reuse=False):
#             '''
#             layers = [64, 128, 256, 512, 512, 512, 512]
#             conv1 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=Ih, filters=layers[0], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv2 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv1, filters=layers[1], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv3 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv2, filters=layers[2], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv4 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv3, filters=layers[3], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv5 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv4, filters=layers[4], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv6 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv5, filters=layers[5], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             z     = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv6, filters=layers[6], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             res_info = [conv1, conv2, conv3, conv4, conv5, conv6]
#             '''

#             layers = [64, 128, 256, 512, 512, 512, 512]
#             conv1 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=Ih, filters=layers[0], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv2 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv1, filters=layers[1], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv3 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv2, filters=layers[2], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv4 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv3, filters=layers[3], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv5 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv4, filters=layers[4], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv6 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv5, filters=layers[5], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             z     = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv6, filters=layers[6], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             res_info = [conv1, conv2, conv3, conv4, conv5, conv6]

#         return z, res_info


#     def __downscale_decoder__(self, feature, img_sz):
#         feature_sz = feature.get_shape().as_list()[1]
#         with tf.variable_scope('downscale_decoder_' + str(feature_sz), reuse=False):
#             ch_dim = feature.get_shape().as_list()[-1]
#             conv1  = tf.nn.relu(batch_norm(tf.layers.conv2d(inputs=feature, filters=ch_dim/2, kernel_size=[5,5],
#                             strides=(1,1), padding="same", activation=None)))
#             conv2  = tf.nn.relu(batch_norm(tf.layers.conv2d(inputs=conv1, filters=ch_dim/4, kernel_size=[5,5],
#                             strides=(1,1), padding="same", activation=None)))
#             ds_img = tf.layers.conv2d(inputs=conv2, filters=img_sz[-1], kernel_size=[1,1],
#                             strides=(1,1), padding="same", activation=None)
#             # ds_img = tf.layers.conv2d(inputs=feature, filters=img_sz[-1], kernel_size=[3,3],
#             #                 strides=(1,1), padding="same", activation=None)

#         return ds_img


#     def __decoder__(self, z, img_sz, res_info):
#         pyramid = []
#         with tf.variable_scope('decoder', reuse=False):
#             '''
#             layers = [img_sz[2], 64, 128, 256, 512, 512, 512][::-1]
#             conv1 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=z, filters=layers[0], kernel_size=[4,4],
#                             strides=(2,2), padding="same", activation=None)))
#             conv2 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv1+res_info[-1], filters=layers[1], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv3 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv2+res_info[-2], filters=layers[2], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))

#             conv4 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv3+res_info[-3], filters=layers[3], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv4_img = self.__downscale_decoder__(conv4, img_sz)

#             conv5 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv4+res_info[-4], filters=layers[4], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv5_img = self.__downscale_decoder__(conv5, img_sz)

#             conv6 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv5+res_info[-5], filters=layers[5], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv6_img = self.__downscale_decoder__(conv6, img_sz)

#             Ib_hat = tf.layers.conv2d_transpose(inputs=conv6+res_info[-6], filters=layers[6], kernel_size=[5,5],
#                             strides=(2,2), padding="same")
#             Ib_hat = tf.tanh(Ib_hat + tf.image.resize_images(conv6_img,img_sz[:2]) + \
#                         tf.image.resize_images(conv5_img,img_sz[:2]) + tf.image.resize_images(conv4_img,img_sz[:2]))
#             pyramid = [Ib_hat, conv6_img, conv5_img, conv4_img]
#             '''
#             use_ms = tf.app.flags.FLAGS.use_ms
#             layers = [img_sz[2], 64, 128, 256, 512, 512, 512][::-1]
#             conv1 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=z, filters=layers[0], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv1 = tf.nn.relu(batch_norm(tf.layers.conv2d(inputs=conv1, filters=layers[0], kernel_size=[5,5],
#                             strides=(1,1), padding="same", activation=None)))

#             conv2 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv1+res_info[-1], filters=layers[1], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv2 = tf.nn.relu(batch_norm(tf.layers.conv2d(inputs=conv2, filters=layers[1], kernel_size=[5,5],
#                             strides=(1,1), padding="same", activation=None)))

#             conv3 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv2+res_info[-2], filters=layers[2], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv3 = tf.nn.relu(batch_norm(tf.layers.conv2d(inputs=conv3, filters=layers[2], kernel_size=[5,5],
#                             strides=(1,1), padding="same", activation=None)))

#             conv4 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv3+res_info[-3], filters=layers[3], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv4 = tf.nn.relu(batch_norm(tf.layers.conv2d(inputs=conv4, filters=layers[3], kernel_size=[5,5],
#                             strides=(1,1), padding="same", activation=None)))

#             conv5 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv4+res_info[-4], filters=layers[4], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv5 = tf.nn.relu(batch_norm(tf.layers.conv2d(inputs=conv5, filters=layers[4], kernel_size=[5,5],
#                             strides=(1,1), padding="same", activation=None)))

#             conv6 = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(inputs=conv5+res_info[-5], filters=layers[5], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)))
#             conv6 = tf.nn.relu(batch_norm(tf.layers.conv2d(inputs=conv6, filters=layers[5], kernel_size=[5,5],
#                             strides=(1,1), padding="same", activation=None)))

#             Ib_hat = tf.layers.conv2d_transpose(inputs=conv6+res_info[-6], filters=layers[6], kernel_size=[5,5],
#                             strides=(2,2), padding="same", activation=None)

#             if use_ms:
#                 conv4_img = self.__downscale_decoder__(conv4, img_sz)
#                 conv5_img = self.__downscale_decoder__(conv5, img_sz)
#                 conv6_img = self.__downscale_decoder__(conv6, img_sz)
#                 Ib_hat    = Ib_hat + tf.image.resize_images(conv4_img,img_sz[:2]) \
#                                    + tf.image.resize_images(conv5_img,img_sz[:2]) \
#                                    + tf.image.resize_images(conv6_img,img_sz[:2])
#                 pyramid = [Ib_hat, conv4_img, conv5_img, conv6_img]

#             Ib_hat = tf.tanh(Ib_hat)

#             return Ib_hat, pyramid


#     def __discriminator__(self, Ih, Ib, img_sz, reuse=False):
#         rate = [8, 8]   # num of patches for row and col
#         assert(img_sz[0]%rate[0] == 0 and img_sz[1]%rate[1] == 0)
#         patch_size = [1, img_sz[0]/rate[0], img_sz[1]/rate[1], 1]
#         layers = [64, 128, 256, 512]

#         # concatenate gradients
#         # x_weight = tf.reshape(tf.constant([[+1,0,-1],[+2,0,-2],[+1,0,-1]], tf.float32), [3, 3, 1, 1])
#         # y_weight = tf.reshape(tf.constant([[+1,+2,+1],[0,0,0],[-1,-2,-1]], tf.float32), [3, 3, 1, 1])

#         # Ih_diff_x = tf.nn.conv2d(Ih, x_weight, [1, 1, 1, 1], 'SAME')
#         # Ih_diff_y = tf.nn.conv2d(Ih, y_weight, [1, 1, 1, 1], 'SAME')
#         # Ib_diff_x = tf.nn.conv2d(Ib, x_weight, [1, 1, 1, 1], 'SAME')
#         # Ib_diff_y = tf.nn.conv2d(Ib, y_weight, [1, 1, 1, 1], 'SAME')

#         # Ih_diff = tf.concat([Ih, Ih_diff_x, Ih_diff_y], 3)
#         # Ib_diff = tf.concat([Ib, Ib_diff_x, Ib_diff_y], 3)

#         # # extract patches from large image and send into discriminator as a batch
#         # Ih_patches = tf.extract_image_patches(Ih_diff, ksizes=patch_size, strides=patch_size, rates=[1,1,1,1], padding="VALID")
#         # Ih_patches = tf.reshape(Ih_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]*3])
#         # Ib_patches = tf.extract_image_patches(Ib_diff, ksizes=patch_size, strides=patch_size, rates=[1,1,1,1], padding="VALID")
#         # Ib_patches = tf.reshape(Ib_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]*3])

#         # concatenate high freq images
#         # extract patches from large image and send into discriminator as a batch
#         Ih_patches = tf.extract_image_patches(self.Ih_hfreq, ksizes=patch_size, strides=patch_size, rates=[1,1,1,1], padding="VALID")
#         Ih_patches = tf.reshape(Ih_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]])
#         Ib_patches = tf.extract_image_patches(Ib, ksizes=patch_size, strides=patch_size, rates=[1,1,1,1], padding="VALID")
#         Ib_patches = tf.reshape(Ib_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]])

#         # extract patches from large image and send into discriminator as a batch
#         # Ih_patches = tf.extract_image_patches(Ih, ksizes=patch_size, strides=patch_size, rates=[1,1,1,1], padding="VALID")
#         # Ih_patches = tf.reshape(Ih_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]])
#         # Ib_patches = tf.extract_image_patches(Ib, ksizes=patch_size, strides=patch_size, rates=[1,1,1,1], padding="VALID")
#         # Ib_patches = tf.reshape(Ib_patches, [-1, patch_size[1], patch_size[2], img_sz[-1]])

#         with tf.variable_scope('discriminator', reuse=reuse):
#             Ih_Ib = tf.concat([Ih_patches, Ib_patches], 3)
#             conv1 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=Ih_Ib, filters=layers[0], kernel_size=[5,5],
#                                 strides=(2,2), padding="same", activation=None)))
#             conv2 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv1, filters=layers[1], kernel_size=[5,5],
#                                 strides=(2,2), padding="same", activation=None)))
#             conv3 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv2, filters=layers[2], kernel_size=[5,5],
#                                 strides=(2,2), padding="same", activation=None)))
#             conv4 = tf.nn.leaky_relu(batch_norm(tf.layers.conv2d(inputs=conv3, filters=layers[3], kernel_size=[5,5],
#                                 strides=(2,2), padding="same", activation=None)))
#             conv5 = tf.layers.dense(tf.contrib.layers.flatten(conv4), 1)

#         # reshape results from the same image back to the same batch
#         logits = tf.reduce_mean(tf.reshape(conv5, [-1, rate[0]*rate[1]]), 1)
#         torf   = tf.nn.sigmoid(logits)

#         return torf, logits


