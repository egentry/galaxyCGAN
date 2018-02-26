import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import os
import time

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['figure.figsize'] = np.array((10, 6))*.6


def transform_0_1(X):
    return (X - X.min())/(X.max() - X.min())


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
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


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


class CGAN(object):
    model_name = "CGAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 image_size, X_img, y_conditional,
                 checkpoint_dir, result_dir, log_dir):
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

        elif dataset_name == "galaxy":
                        # parameters
            self.input_height  = self.image_size
            self.input_width   = self.image_size
            self.output_height = self.image_size
            self.output_width  = self.image_size

            self.z_dim = z_dim         # dimension of noise-vector
            self.y_dim = 2  # number of conditional input variables
            self.c_dim = 3  # channel dim

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 16  # number of generated images to be saved

            # load mnist
            self.data_X = self.X_img
            self.data_y = self.y_conditional

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size

        else:
            raise NotImplementedError

    def discriminator(self, x, y, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = tf.concat(axis=1, values=[net, y]) # this is custom; maybe wrong?
            net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            out_logit = linear(net, 1, scope='d_fc4')
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit, net

    def generator(self, z, y, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S

        with tf.variable_scope("generator", reuse=reuse):
            inputs = tf.concat(axis=1, values=[z,y])
            net = tf.nn.relu(bn(linear(inputs, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 5 * 5, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 5, 5, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, self.output_height//2, self.output_height//2, 64],
                            k_h=4, k_w=4, d_h=5, d_w=5,
                            name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, self.output_height, self.output_height, self.c_dim],
                                         k_h=4, k_w=4, d_h=2, d_w=2, name='g_dc4'))

            return out

#         with tf.variable_scope("generator", reuse=reuse):
#             inputs = tf.concat(axis=1, values=[z, y])
#             G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
#             G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
#             G_prob = tf.nn.sigmoid(G_log_prob)

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        batch_size = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [batch_size] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [batch_size, self.z_dim], name='z')

        # conditionals
        self.y = tf.placeholder(tf.float32, [batch_size, self.y_dim], name='y')


        """ Loss Function """

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, self.y, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, self.y, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                             .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                             .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run(session=self.sess)

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

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
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict={self.z: batch_z,
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
                    self.visualize_results(epoch, label)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch, "test_at_end")

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch, label, y_values=[.15, 8.5]):

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        y_sample = np.vstack([y_values]*self.batch_size)

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample,
                                                             self.y: y_sample,})

        samples = samples[:self.sample_num]

        self.save_images(samples, epoch, label)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
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
                sample.reshape(self.output_height, self.output_width, self.c_dim)
            ))
        return fig

    def save_images(self, samples, epoch, label):
        fig = self.plot(samples)

        plot_dir = os.path.join(self.result_dir, self.model_dir)
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        plot_filename = os.path.join(plot_dir, self.model_name + '_epoch%03d' % epoch + '_{}.png'.format(label))
        plt.savefig(plot_filename, bbox_inches='tight')
