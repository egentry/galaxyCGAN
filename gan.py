import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import os
import time

from misc import transform_0_1

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['figure.figsize'] = np.array((10, 6))*.6


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", padding="SAME"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        conv = tf.nn.conv2d(input_, w,
                            strides=[1, d_h, d_w, 1],
                            padding=padding)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2,
             name="deconv2d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w,
                                            output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w,
                                    output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


class CGAN(object):
    model_name = "CGAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 image_size, X_img, y_conditional,
                 y_for_visualization_samples,
                 checkpoint_dir, result_dir, log_dir,
                 d_learning_rate=0.0002,
                 relative_learning_rate=10,
                 loss_weighting=50,
                 ):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.image_size = image_size
        self.X_img = X_img
        self.y_conditional = y_conditional
        self.y_for_visualization_samples = y_for_visualization_samples
        self.loss_weighting = loss_weighting

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            raise NotImplementedError
#             # parameters
#             self.input_height = 28
#             self.input_width = 28
#             self.output_height = 28
#             self.output_width = 28

#             self.z_dim = z_dim         # dimension of noise-vector
#             self.c_dim = 1

#             # train
#             self.learning_rate = 0.0002
#             self.beta1 = 0.5

#             # test
#             self.sample_num = 64  # number of generated images to be saved

#             # load mnist
#             self.data_X, self.data_y = load_mnist(self.dataset_name)

#             # get number of batches for a single epoch
#             self.num_batches = len(self.data_X) // self.batch_size

        elif "galaxy" in dataset_name:
                        # parameters
            self.input_height  = self.image_size
            self.input_width   = self.image_size
            self.output_height = self.image_size
            self.output_width  = self.image_size

            self.z_dim = z_dim         # dimension of noise-vector
            self.y_dim = y_conditional.shape[1]  # number of conditional input variables
            self.c_dim = self.X_img.shape[-1]  # channel dim

            # train
            self.learning_rate = d_learning_rate
            self.beta1 = 0.5
            # relative_learning_rate = g learning rate / d learning rate
            self.relative_learning_rate = relative_learning_rate

            # test
            self.sample_num = 16  # number of generated images to be saved

            # load data
            self.data_X = self.X_img
            self.data_y = self.y_conditional

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size

        else:
            raise NotImplementedError

    def generator(self, z, y, is_training=True, reuse=False):
        # Originally started with infoGAN-like architecture
        # but that lead to strong checkerboard artifacts
        # given the 2 deconv layers with stride=5x5

        # so now I'm trying a resize approach
        # (see: https://distill.pub/2016/deconv-checkerboard/)

        with tf.variable_scope("generator", reuse=reuse):
            inputs = tf.concat(axis=1, values=[z, y])
            net = tf.nn.relu(bn(linear(inputs, 1024, scope='g_fc1'),
                                is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 16 * 16, scope='g_fc2'),
                                is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 16, 16, 128])

            post_conv_size = 32
            kernel_size = 4
            padding = "VALID"
            if padding == "SAME":
                pre_conv_size = post_conv_size
            else:
                pre_conv_size = post_conv_size + (kernel_size-1)

            net = tf.nn.relu(
                bn(conv2d(
                    tf.image.resize_bilinear(net, [pre_conv_size]*2),
                    64,
                    k_h=kernel_size, k_w=kernel_size, d_h=1, d_w=1,
                    name="g_rc3",
                    padding=padding,
                    ),
                   is_training=is_training,
                   scope="g_bn3",
                   )
            )

            post_conv_size = self.output_height
            kernel_size = 4
            padding = "VALID"
            if padding == "SAME":
                pre_conv_size = post_conv_size
            else:
                pre_conv_size = post_conv_size + (kernel_size-1)

            out = conv2d(
                    tf.image.resize_bilinear(net, [pre_conv_size]*2),
                    self.c_dim,
                    k_h=kernel_size, k_w=kernel_size, d_h=1, d_w=1,
                    name="g_rc4",
                    padding=padding,
                )

            return out

    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is based on infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture (roughly): (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        #
        # Note: we've now combined the real and fake images into one
        # big batch. See https://github.com/openai/improved-gan/issues/11
        with tf.variable_scope("discriminator", reuse=reuse):
            joint_batch_size = self.batch_size*2

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            last_num_kernels = 128
            net_post_conv = lrelu(bn(conv2d(net, last_num_kernels, 4, 4, 2, 2,
                                            name='d_conv2'),
                                     is_training=is_training, scope='d_bn2'))

            net_flattened = tf.reshape(net_post_conv, [joint_batch_size, -1])

            # This part computes the predicted y values
            net_y = lrelu(bn(linear(net_flattened, 1024, scope='d_fc3'),
                          is_training=is_training, scope='d_bn3'))
            out_y = linear(net_y, self.y_dim, scope='d_fc4')

            # This part computes the minibatch discrimination score
            # See arxiv.org/abs/1606.03498 for details
            # (Salimans et al. 2016: Improved Techniques for Training GANs)

            # In particular this part is based on the code from:
            # github.com/openai/improved-gan/blob/master/imagenet/discriminator.py

            dim_per_kernel = 5
            x = linear(net_flattened, last_num_kernels * dim_per_kernel,
                       scope="d_mbd")
            activation = tf.reshape(x,
                                    [joint_batch_size,
                                     last_num_kernels,
                                     dim_per_kernel],
                                    )

            big = np.zeros((joint_batch_size, joint_batch_size),
                           dtype='float32')
            big += np.eye(joint_batch_size)
            big = tf.expand_dims(big, 1)

            abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3)
                                           - tf.expand_dims(tf.transpose(activation, [1, 2, 0]),
                                                            0)),
                                    2)
            mask = 1. - big
            masked = tf.exp(-abs_dif) * mask

            def half(tens, second):
                m, n, _ = tens.get_shape()
                m = int(m)
                n = int(n)
                return tf.slice(tens, [0, 0, second * self.batch_size], [m, n, self.batch_size])

            f1 = tf.reduce_sum(half(masked, 0), 2) / tf.reduce_sum(half(mask, 0))
            f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))
            minibatch_features = tf.concat([f1, f2], 1)

            net_with_minibatch = tf.concat([net_flattened, minibatch_features], 1)

            out_logit = linear(net_with_minibatch, 1, scope='d_fc_mb')

            # split up outputs by generator and real data
            out_y_on_data = tf.slice(out_y, [0, 0], [self.batch_size, self.y_dim])
            out_y_on_generator = tf.slice(out_y, [self.batch_size, 0], [self.batch_size, self.y_dim])

            out_logit_on_data = tf.slice(out_logit, [0, 0], [self.batch_size, 1])
            out_logit_on_generator = tf.slice(out_logit,
                                              [self.batch_size, 0],
                                              [self.batch_size, 1])

            outputs = [out_y_on_data, out_y_on_generator,
                       out_logit_on_data, out_logit_on_generator,
                       net_y
                       ]

            return outputs

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        batch_size = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [batch_size] + image_dims,
                                     name='real_images')

        self.inputs_joint = tf.placeholder(tf.float32,
                                           [2*batch_size] + image_dims,
                                           name='joint_images')

        # noises
        self.z = tf.placeholder(tf.float32, [batch_size, self.z_dim], name='z')

        # conditionals
        self.y = tf.placeholder(tf.float32, [batch_size, self.y_dim], name='y')

        """ Loss Function """

        # output of D for real AND fake images
        G = self.generator(self.z, self.y, is_training=True, reuse=False)

        joint_inputs = tf.concat([self.inputs, G], 0)
        D_y_real, D_y_fake, D_logit_real, D_logit_fake, _ = self.discriminator(joint_inputs, is_training=True,
                                                                               reuse=False,
                                                                               )

        # get loss for discriminator
        # # First start with the l2 loss
        d_loss_y_real = tf.nn.l2_loss(tf.subtract(D_y_real, self.y))
        d_loss_y_fake = tf.nn.l2_loss(tf.subtract(D_y_fake, self.y))

        d_loss_real = d_loss_y_real
        d_loss_fake = d_loss_y_fake

        self.d_loss_y = tf.maximum(tf.subtract(d_loss_y_real, d_loss_y_fake),
                                   tf.constant(0.))

        # # Next calculate the minibatch discriminator loss
        d_loss_logit_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                    labels=tf.ones_like(D_logit_real)))
        d_loss_logit_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                    labels=tf.zeros_like(D_logit_fake)))

        self.d_loss = self.d_loss_y \
             + self.loss_weighting * (d_loss_logit_real + d_loss_logit_fake)

        # get loss for generator
        # # First start with the l2 loss
        g_loss_y = d_loss_y_fake

        # # Next calculate the minibatch discriminator loss
        g_loss_logit = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                    labels=tf.ones_like(D_logit_fake)))

        self.g_loss = g_loss_y + self.loss_weighting * g_loss_logit

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                             .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*self.relative_learning_rate, beta1=self.beta1) \
                             .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, self.y,
                                          is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def predictor(self, *args, **kwargs):
        raise NotImplementedError("`predictor()` no longer exists - replaced by `dicriminator()`")

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run(session=self.sess)

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir,
                                                         self.model_name),
                                            self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                batch_y = self.data_y[idx*self.batch_size:(idx+1)*self.batch_size]

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.inputs: batch_images,
                                                                  self.z: batch_z,
                                                                  self.y: batch_y})
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                       feed_dict={self.inputs: batch_images,
                                                                  self.z: batch_z,
                                                                  self.y: batch_y})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss),
                      end="",
                      )
                if idx != self.num_batches-1:
                    print("\r", end="")
                else:
                    print("")

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    label = 'train_{:02d}_{:04d}'.format(epoch, idx)
                    self.visualize_results(epoch, label,
                                           self.y_for_visualization_samples)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch, "test_at_end",
                                   self.y_for_visualization_samples)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def generate_samples(self, y_sample, z_sample=None):
        if z_sample is None:
            z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample,
                                                             self.y: y_sample})

        return samples

    def visualize_results(self, epoch, label, y_values):

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        y_sample = np.vstack([y_values]*self.batch_size)

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample,
                                                             self.y: y_sample})

        samples = samples[:self.sample_num]

        self.save_images(samples, epoch, label)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir,
                                      self.model_dir,
                                      self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name+'.model'),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir,
                                      self.model_dir,
                                      self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,
                               os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def plot(self, samples):
        """ assumes samples is of shape [n_img, img_x, img_y, num_channels]
        where n_img is assumed to be a perfect square"""
        assert(samples.shape[0]**.5 % 1 == 0)
        fig_per_side = int(samples.shape[0]**.5)
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(fig_per_side, fig_per_side)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(transform_0_1(
                sample.reshape(self.output_height,
                               self.output_width,
                               self.c_dim)
            ))
        return fig

    def save_images(self, samples, epoch, label):
        fig = self.plot(samples)

        plot_dir = os.path.join(self.result_dir, self.model_dir)
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        plot_filename = os.path.join(plot_dir, self.model_name + '_epoch%03d' % epoch + '_{}.png'.format(label))
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig)
