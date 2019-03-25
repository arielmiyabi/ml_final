import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import matplotlib.gridspec as gridspec
import os
import random

z_dim = 100
i = 0
batch_size = 32
epochs = 200

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=0.02)
def sample_z(m, n):
    return np.random.uniform(-1, 1, size=[m, n])

def generator(z, is_training):
    def fc_layer(inputs, in_size, out_size):
        # Weights = tf.Variable(xavier_init([in_size, out_size]))
        # biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        outputs = tf.layers.dense(inputs, out_size, kernel_initializer = tf.random_normal_initializer(stddev=0.02))
        return outputs

    def deconv2d(layer_input, filters, k_size, output_shape):
        # layer_input = tf.image.resize_images(layer_input, output_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # mask = tf.Variable(xavier_init([k_size, k_size, layer_input.get_shape().as_list()[-1], filters]))
        # bias = tf.Variable(xavier_init([filters]))
        u = tf.image.resize_images(layer_input, output_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        outputs = tf.layers.conv2d(u, filters, kernel_size = k_size, strides = (1, 1), padding = 'same'
                    , kernel_initializer = tf.truncated_normal_initializer(stddev=0.02))
        return outputs

    with tf.variable_scope('Generator'):
        G_1 = tf.reshape(fc_layer(z, z_dim, 4 * 4 * 512), [-1, 4, 4, 512])
        G_1 = tf.nn.leaky_relu(G_1)

        G_2 = deconv2d(G_1, 512, 5, (8, 8))
        G_2 = tf.layers.batch_normalization(G_2, momentum=0.9, training = is_training)
        G_2 = tf.nn.leaky_relu(G_2)

        G_3 = deconv2d(G_2, 256, 5, (16, 16))
        G_3 = tf.layers.batch_normalization(G_3, momentum=0.9, training = is_training)
        G_3 = tf.nn.leaky_relu(G_3)

        G_4 = deconv2d(G_3, 128, 5, (32, 32))
        G_4 = tf.layers.batch_normalization(G_4, momentum=0.9, training = is_training)
        G_4 = tf.nn.leaky_relu(G_4)

        G_5 = deconv2d(G_4, 64, 5, (64, 64))
        G_5 = tf.nn.leaky_relu(G_5)

        # G_6 = deconv2d(G_5, 32, 5, (64, 64))
        # G_6 = tf.nn.leaky_relu(G_6)

        G_5 = tf.layers.conv2d(G_5, 3, kernel_size = 5, strides = (1, 1), padding = 'same', kernel_initializer = tf.truncated_normal_initializer(stddev=0.02))
        # G_5 = tf.nn.leaky_relu(G_5)
        outputs = tf.nn.sigmoid(G_5)

    return outputs


def discriminator(x, is_training, reuse = False):
    def fc_layer(inputs, in_size, out_size):
        outputs = tf.layers.dense(inputs, out_size, kernel_initializer = tf.random_normal_initializer(stddev=0.02))
        return outputs

    def conv2d(layer_input, filters, k_size):
        # mask = tf.Variable(xavier_init([k_size, k_size, layer_input.get_shape().as_list()[-1], filters]))
        # bias = tf.Variable(xavier_init([filters]))
        outputs = tf.layers.conv2d(layer_input, filters, kernel_size = k_size, strides = (2, 2), padding = 'same'
            , kernel_initializer = tf.truncated_normal_initializer(stddev=0.02))
        return outputs

    with tf.variable_scope('Discriminator') as scope:
        if reuse:
           scope.reuse_variables()
        # x = tf.print(x, [tf.shape(x)], message = 'x')
        D_1 = conv2d(x, 64, 5)
        D_1 = tf.nn.relu(D_1)

        D_2 = conv2d(D_1, 128, 5)
        D_2 = tf.layers.batch_normalization(D_2, momentum=0.9, training = is_training)
        D_2 = tf.nn.relu(D_2)

        D_3 = conv2d(D_2, 256, 5)
        D_3 = tf.layers.batch_normalization(D_3, momentum=0.9, training = is_training)
        D_3 = tf.nn.relu(D_3)

        D_4 = conv2d(D_3, 512, 5)
        D_4 = tf.layers.batch_normalization(D_4, momentum=0.9, training = is_training)
        D_4 = tf.nn.relu(D_4)
        # D_5 = tf.print(D_4, [tf.shape(D_4)], message = 'D_4')

        D_5 = tf.reduce_mean(D_4, [1, 2])
        D_5 = tf.nn.dropout(D_5, 0.5)
        # D_5 = tf.print(D_5, [tf.shape(D_5)], message = 'D_5')
        D_6 = fc_layer(D_5, 512, 1)
        outputs = tf.nn.sigmoid(D_6)

        return outputs, D_6

def load_min_batch(batch_size=1, img_res=(64, 64)):
    # train_path = glob('./edge_blur/train_blur_data/*')
    # label_path = glob('./edge_blur/train_blur_label/*')
    train_path = glob('./celebA_HQ_128/*')
    batch_index = []
    count = 0
    num = len(train_path)

    while count < batch_size:
        index = random.randint(0, num - 1)
        if index in batch_index:
            continue
        batch_index.append(index)
        count += 1

    imgs_B = []
    for i in batch_index:
        img_B = cv2.cvtColor(cv2.imread(train_path[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_B = cv2.resize(img_B, img_res).reshape((img_res[0], img_res[1], 3))
        imgs_B.append(img_B)

    imgs_B = np.array(imgs_B) / 255.0

    return imgs_B

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        # sample = (sample + 1.0) / 2
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(64, 64, 3))

    return fig

if __name__ == '__main__':
    z = tf.placeholder(tf.float32, shape=[None, z_dim])
    X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    training_ = tf.placeholder(tf.bool)

    g_sample = generator(z, training_)
    d_real, d_logit_real = discriminator(X, training_, reuse=False)
    d_fake, d_logit_fake = discriminator(g_sample, training_, reuse=True)

    # real_label = tf.constant([1, 0])
    # fake_label = tf.constant([0, 1])

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

    learning_rate = 0.0002
    beta = 0.5

    D_solver = tf.train.AdamOptimizer(learning_rate = 0.0001, beta1 = beta).minimize(
        D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
    G_solver = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = beta).minimize(
        G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())



    saver = tf.train.Saver()
    for it in range(20000):
        label_batch = load_min_batch(batch_size)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: label_batch, training_: True, z: sample_z(batch_size, z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={z: sample_z(batch_size, z_dim), training_: True})
        if it % 200 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()

            samples = sess.run(g_sample, feed_dict={z: sample_z(2, z_dim), training_: False})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
            # save_path = saver.save(sess, './model/model_' + str(it) + '.ckpt')
