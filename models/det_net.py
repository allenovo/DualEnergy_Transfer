import ipdb
import numpy as np
import tensorflow as tf

from utilities import batch_norm


class DET_ADV_NET:  # Dual Energy Transfer with Adversarial Loss

    def __init__(self, img_sz):
        self.gen_adv_weight = 1
        self.gen_l1_weight  = 1
        self.__build_graph__(img_sz)


    def __build_graph__(self, img_sz):
        
        Ih = tf.placeholder(tf.float32, [None]+[i for i in img_sz]) # High energy images
        Ib = tf.placeholder(tf.float32, [None]+[i for i in img_sz]) # Bone images

        # transfer high energy image to bone image
        Ib_hat, pyramid = self.__transfer__(Ih, img_sz)

        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(Ib_hat - Ib))
 
        self.Ih, self.Ib, self.Ib_hat, self.pyramid = Ih, Ib, Ib_hat, pyramid
        self.l1_loss = l1_loss


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
            conv1  = tf.nn.relu(batch_norm(tf.layers.conv2d(inputs=feature, filters=ch_dim/2, kernel_size=[5,5],
                            strides=(1,1), padding="same", activation=None)))
            conv2  = tf.nn.relu(batch_norm(tf.layers.conv2d(inputs=conv1, filters=ch_dim/4, kernel_size=[5,5],
                            strides=(1,1), padding="same", activation=None)))
            ds_img = tf.layers.conv2d(inputs=conv2, filters=img_sz[-1], kernel_size=[1,1],
                            strides=(1,1), padding="same", activation=None)
            # ds_img = tf.layers.conv2d(inputs=feature, filters=img_sz[-1], kernel_size=[3,3],
            #                  strides=(1,1), padding="same", activation=None)

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

