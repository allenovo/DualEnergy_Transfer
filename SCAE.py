import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as fc
from cost_layer import *

DTYPE = tf.float32
latent_sz = 256

class SCAE:

    def __init__(self):
        self._build_graph_()

    def _weight_variable(self, name, shape):
        return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))

    def _bias_variable(self, name, shape):
        return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0, dtype=DTYPE))

    def _build_graph_(self):
        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, [None, 128, 128, 64, 1], name='x_input')
            y = tf.placeholder(tf.float32, [None, 128, 128, 64, 1], name='y_input')
            input_layer = x

        ### ENCODER (parametrization of approximate posterior q(z|x))
        with tf.variable_scope('level1_EC', reuse=None) as scope:  # LEVEL 1 for Encoder
            prev_layer = input_layer

            # conv3d1 (generate 128x128x64 with x16)
            in_filters = 1
            out_filters = 16
            kernel = self._weight_variable('weights1_l1', [3, 3, 3, in_filters, out_filters])
            conv1_l1 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l1', [out_filters])
            conv1_l1_out = tf.nn.bias_add(conv1_l1, biases)
            conv1_l1_out = tf.layers.batch_normalization(conv1_l1_out)
            conv1_l1_out = tf.nn.relu(conv1_l1_out)

            # conv3d2 (generate 128x128x64 with x16)
            in_filters = 16
            out_filters = 16
            kernel = self._weight_variable('weights2_l1', [3, 3, 3, in_filters, out_filters])
            conv2_l1 = tf.nn.conv3d(conv1_l1_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l1', [out_filters])
            conv2_l1_out = tf.nn.bias_add(conv2_l1, biases)
            conv2_l1_out = tf.layers.batch_normalization(conv2_l1_out)
            conv2_l1_out = tf.nn.relu(conv2_l1_out)

            level1_out = conv2_l1_out

            # Down MP for level 2 (generate 64x64x32 with x64)
            level2_in = tf.nn.max_pool3d(level1_out, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')


        with tf.variable_scope('level2_EC') as scope:  # LEVEL 2 for Encoder
            prev_layer = level2_in

            # conv3d1 (generate 64x64x32 with x32)
            in_filters = 16
            out_filters = 32
            kernel = self._weight_variable('weights1_l2', [3, 3, 3, in_filters, out_filters])
            conv1_l2 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l2', [out_filters])
            conv1_l2_out = tf.nn.bias_add(conv1_l2, biases)
            conv1_l2_out = tf.layers.batch_normalization(conv1_l2_out)
            conv1_l2_out = tf.nn.relu(conv1_l2_out)

            # conv3d2 (generate 64x64x32 with x32)
            in_filters = 32
            out_filters = 32
            kernel = self._weight_variable('weights2_l2', [3, 3, 3, in_filters, out_filters])
            conv2_l2 = tf.nn.conv3d(conv1_l2_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l2', [out_filters])
            conv2_l2_out = tf.nn.bias_add(conv2_l2, biases)
            conv2_l2_out = tf.layers.batch_normalization(conv2_l2_out)
            conv2_l2_out = tf.nn.relu(conv2_l2_out)

            level2_out = conv2_l2_out

            # Down MP for level 3 (generate 32x32x16 with x32)
            level3_in = tf.nn.max_pool3d(level2_out, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')


        with tf.variable_scope('level3_EC') as scope:  # LEVEL 3 for Encoder
            prev_layer = level3_in

            # conv3d1 (generate 32x32x16 with x64)
            in_filters = 32
            out_filters = 64
            kernel = self._weight_variable('weights1_l3', [3, 3, 3, in_filters, out_filters])
            conv1_l3 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l3', [out_filters])
            conv1_l3_out = tf.nn.bias_add(conv1_l3, biases)
            conv1_l3_out = tf.layers.batch_normalization(conv1_l3_out)
            conv1_l3_out = tf.nn.relu(conv1_l3_out)

            # conv3d2 (generate 32x32x16 with x64)
            in_filters = 64
            out_filters = 64
            kernel = self._weight_variable('weights2_l3', [3, 3, 3, in_filters, out_filters])
            conv2_l3 = tf.nn.conv3d(conv1_l3_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l3', [out_filters])
            conv2_l3_out = tf.nn.bias_add(conv2_l3, biases)
            conv2_l3_out = tf.layers.batch_normalization(conv2_l3_out)
            conv2_l3_out = tf.nn.relu(conv2_l3_out)

            level3_out = conv2_l3_out

            # Down MP for level 4 (generate 16x16x8 with x64)
            level4_in = tf.nn.max_pool3d(level3_out, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')


        with tf.variable_scope('level4_EC') as scope:  # LEVEL 4 for Encoder
            prev_layer = level4_in

            # conv3d1 (generate 16x16x8 with x128)
            in_filters = 64
            out_filters = 128
            kernel = self._weight_variable('weights1_l4', [3, 3, 3, in_filters, out_filters])
            conv1_l4 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l4', [out_filters])
            conv1_l4_out = tf.nn.bias_add(conv1_l4, biases)
            conv1_l4_out = tf.layers.batch_normalization(conv1_l4_out)
            conv1_l4_out = tf.nn.relu(conv1_l4_out)

            # fully connected layer (16x16x8 with x128 -> 1x26200 -> 1x512)
            flat1_out = tf.contrib.layers.flatten(conv1_l4_out)
            z = fc(flat1_out, 512, activation_fn=tf.nn.sigmoid)

        ### DECODER (mirror similar structure of the encoder)
        ## fully connected layer for level4 in (1x512 -> 1x104800 -> 16x16x8 with x128)
        with tf.variable_scope('latent_to_3D') as scope:  # latent space to 3D
            level4_in = fc(z, 16*16*8*128, activation_fn=tf.nn.sigmoid)
            level4_in = tf.reshape(level4_in, (tf.shape(level4_in)[0], 16, 16, 8, 128))


        with tf.variable_scope('level4_DC') as scope:   # LEVEL 4 for Decoder
            prev_layer = level4_in

            # conv3d1 (generate 16x16x8 with x128)
            in_filters = 128
            out_filters = 128
            kernel = self._weight_variable('weights1_l4', [3, 3, 3, in_filters, out_filters])
            conv1_l4 = tf.nn.conv3d(prev_layer, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l4', [out_filters])
            conv1_l4_out = tf.nn.bias_add(conv1_l4, biases)
            conv1_l4_out = tf.layers.batch_normalization(conv1_l4_out)
            conv1_l4_out = tf.nn.relu(conv1_l4_out)

            level4_out = conv1_l4_out

            # Up conv for level 3 (generate 32x32x16 with x64)
            in_filters = 128
            out_filters = 64
            out_shape = [tf.shape(level4_out)[0], 32, 32, 16, 64]
            kernel = self._weight_variable('weights2_l4', [2, 2, 2, out_filters, in_filters])
            conv3_l4 = tf.nn.conv3d_transpose(level4_out, kernel, output_shape=out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
            biases = self._bias_variable('biases2_l4', [out_filters])
            conv3_l4_out = tf.nn.bias_add(conv3_l4, biases)
            level3_in = tf.nn.relu(conv3_l4_out)

        with tf.variable_scope('level3_DC') as scope:  # LEVEL 3 for Decoder
            prev_layer = level3_in

            # element-wise sum prev_layers & level3_out
            sum1_l3 = tf.add(prev_layer, level3_out)
            level3_sum = tf.nn.relu(sum1_l3)

            # conv3d1 (generate 32x32x16 with x64)
            in_filters = 64
            out_filters = 64
            kernel = self._weight_variable('weights1_l3', [3, 3, 3, in_filters, out_filters])
            conv1_l3 = tf.nn.conv3d(level3_sum, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l3', [out_filters])
            conv1_l3_out = tf.nn.bias_add(conv1_l3, biases)
            conv1_l3_out = tf.layers.batch_normalization(conv1_l3_out)
            conv1_l3_out = tf.nn.relu(conv1_l3_out)

            # conv3d2 (generate 32x32x16 with x64)
            in_filters = 64
            out_filters = 64
            kernel = self._weight_variable('weights2_l3', [3, 3, 3, in_filters, out_filters])
            conv2_l3 = tf.nn.conv3d(conv1_l3_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l3', [out_filters])
            conv2_l3_out = tf.nn.bias_add(conv2_l3, biases)
            conv2_l3_out = tf.layers.batch_normalization(conv2_l3_out)
            conv2_l3_out = tf.nn.relu(conv2_l3_out)

            level3_out = conv2_l3_out

            # Up conv for level 2 (generate 64x64x32 with x32)
            in_filters = 64
            out_filters = 32
            out_shape = [tf.shape(level3_out)[0], 64, 64, 32, 32]
            kernel = self._weight_variable('weights3_l3', [2, 2, 2, out_filters, in_filters])
            conv3_l3 = tf.nn.conv3d_transpose(level3_out, kernel, output_shape=out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
            biases = self._bias_variable('biases3_l3', [out_filters])
            conv3_l3_out = tf.nn.bias_add(conv3_l3, biases)
            level2_in = tf.nn.relu(conv3_l3_out)

        with tf.variable_scope('level2_DC') as scope:  # LEVEL 2 for Decoder
            prev_layer = level2_in

            # element-wise sum prev_layers & level2_out
            sum1_l2 = tf.add(prev_layer, level2_out)
            level2_sum = tf.nn.relu(sum1_l2)

            # conv3d1 (generate 64x64x32 with x32)
            in_filters = 32
            out_filters = 32
            kernel = self._weight_variable('weights1_l2', [3, 3, 3, in_filters, out_filters])
            conv1_l2 = tf.nn.conv3d(level2_sum, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l2', [out_filters])
            conv1_l2_out = tf.nn.bias_add(conv1_l2, biases)
            conv1_l2_out = tf.layers.batch_normalization(conv1_l2_out)
            conv1_l2_out = tf.nn.relu(conv1_l2_out)

            # conv3d2 (generate 64x64x32 with x32)
            in_filters = 32
            out_filters = 32
            kernel = self._weight_variable('weights2_l2', [3, 3, 3, in_filters, out_filters])
            conv2_l2 = tf.nn.conv3d(conv1_l2_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l2', [out_filters])
            conv2_l2_out = tf.nn.bias_add(conv2_l2, biases)
            conv2_l2_out = tf.layers.batch_normalization(conv2_l2_out)
            conv2_l2_out = tf.nn.relu(conv2_l2_out)

            level2_out = conv2_l2_out

            # Up conv for level 1 (generate 128x128x64 with x16)
            in_filters = 32
            out_filters = 16
            out_shape = [tf.shape(level2_out)[0], 128, 128, 64, 16]
            kernel = self._weight_variable('weights3_l2', [2, 2, 2, out_filters, in_filters])
            conv2_l2 = tf.nn.conv3d_transpose(level2_out, kernel, output_shape=out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
            biases = self._bias_variable('biases2_l3', [out_filters])
            conv2_l2_out = tf.nn.bias_add(conv2_l2, biases)
            level1_in = tf.nn.relu(conv2_l2_out)

        with tf.variable_scope('level1_DC') as scope:  # LEVEL 1 for Decoder
            prev_layer = level1_in

            # element-wise sum prev_layers & level1_out
            sum1_l1 = tf.add(prev_layer, level1_out)
            level1_sum = tf.nn.relu(sum1_l1)

            # conv3d1 (generate 128x128x64 with x16)
            in_filters = 16
            out_filters = 16
            kernel = self._weight_variable('weights1_l1', [3, 3, 3, in_filters, out_filters])
            conv1_l1 = tf.nn.conv3d(level1_sum, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases1_l1', [out_filters])
            conv1_l1_out = tf.nn.bias_add(conv1_l1, biases)
            conv1_l1_out = tf.layers.batch_normalization(conv1_l1_out)
            conv1_l1_out = tf.nn.relu(conv1_l1_out)

            # conv3d2 (generate 128x128x64 with x16)
            in_filters = 16
            out_filters = 16
            kernel = self._weight_variable('weights2_l1', [3, 3, 3, in_filters, out_filters])
            conv2_l1 = tf.nn.conv3d(conv1_l1_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases2_l1', [out_filters])
            conv2_l1_out = tf.nn.bias_add(conv2_l1, biases)
            conv2_l1_out = tf.layers.batch_normalization(conv2_l1_out)
            conv2_l1_out = tf.nn.relu(conv2_l1_out)

            # conv3d3 (generate 128x128x64 with x1)
            in_filters = 16
            out_filters = 1
            kernel = self._weight_variable('weights3_l1', [3, 3, 3, in_filters, out_filters])
            conv3_l1 = tf.nn.conv3d(conv2_l1_out, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
            biases = self._bias_variable('biases3_l1', [out_filters])
            conv3_l1_out = tf.nn.bias_add(conv3_l1, biases)

            label_recon = conv3_l1_out

        # Label prediction loss (Soft Dice Coe)
        # label_loss = tf.losses.sigmoid_cross_entropy(y, label_recon)
        label_loss = 1 - dice_soft_coe(label_recon, y)

        self.z = z
        self.x = x
        self.y = y
        self.label_recon = label_recon
        self.label_loss = label_loss



        # # loss: negative of Evidence Lower BOund (ELBO)
        # # 1. KL-divergence: KL(q(z|x)||p(z))
        # # (divergence between two multi-variate normal distribution, please refer to Wiki)
        # kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.exp(z_log_sigma_sq) + tf.square(z_mu) - 1 - z_log_sigma_sq, axis=1))

        # # 2. Likelihood: p(x|z)
        # # also called as reconstruction loss
        # # we parametrized it with binary cross-entropy loss as MNIST contains binary images
        # eps = 1e-10  # add small number to avoid log(0.0)
        # recon_loss = tf.reduce_mean(-tf.reduce_sum(x * tf.log(eps + x_recon) + (1 - x) * tf.log(1 - x_recon + eps), axis=1))
        # total_loss = kl_loss + recon_loss

        # self.z = z
        # self.total_loss, self.recon_loss, self.kl_loss = total_loss, recon_loss, kl_loss
        # self.x = x
        # self.x_recon = x_recon
