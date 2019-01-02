import tensorflow as tf
import numpy as np
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import os

# Hyper Parameters
EPOCHS = 100
BATCH_SIZE = 64
EPS = 1e-12
lr = 0.0002
beta1 = 0.5
imgsize = 256


def load_min_batch(batch_size=1, img_res=(imgsize, imgsize)):
    train_path = glob('./data/train_blur_data/*')
    label_path = glob('./data/train_blur_label/*')
    batch_index = []
    count = 0
    num = len(train_path)

    while count < batch_size:
        index = random.randint(0, num - 1)
        if index in batch_index:
            continue
        batch_index.append(index)
        count += 1

    imgs_A, imgs_B = [], []
    for i in batch_index:
        img_B = cv.imread(train_path[i], cv.IMREAD_GRAYSCALE)
        img_B = cv.resize(img_B, img_res)
        img_B = np.reshape(img_B, [imgsize, imgsize, 1]) / 255.
        imgs_B.append(img_B)

        img_A = cv.imread(label_path[i], cv.IMREAD_GRAYSCALE)
        img_A = cv.resize(img_A, img_res)
        img_A = np.reshape(img_A, [imgsize, imgsize, 1]) / 255.
        imgs_A.append(img_A)

    return imgs_A, imgs_B
def load_data(batch_size=1):
    data = glob("./data/train_blur_data/*")
    label = glob("./data/train_blur_label/*")
    batch_data = np.random.choice(data, size=batch_size)
    batch_label = np.random.choice(label, size=batch_size)

    edge = []
    input_sketch = []
    for filename in batch_data:
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)/255
        img = cv.resize(img, (imgsize, imgsize))
        img = np.reshape(img, [imgsize, imgsize, 1])
        input_sketch.append(img)
    for filename in batch_label:
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)/255
        img = cv.resize(img, (imgsize, imgsize))
        img = np.reshape(img, [imgsize, imgsize, 1])
        edge.append(img)

    edge = np.array(edge)
    input_sketch = np.array(input_sketch)

    return edge, input_sketch
def load_batch(batch_size=1):
    data = glob("./data/train_blur_data/*")
    label = glob("./data/train_blur_label/*")
    n_batches = int(len(data) / batch_size)
    for i in range(n_batches-1):
        edge = []
        input_sketch = []
        for filename in data[i:i+batch_size]:
            img = cv.imread(filename, cv.IMREAD_GRAYSCALE)/255.0
            img = cv.resize(img, (imgsize, imgsize))
            img = np.reshape(img, [imgsize, imgsize, 1])
            input_sketch.append(img)
        for filename in label[i:i+batch_size]:
            img = cv.imread(filename, cv.IMREAD_GRAYSCALE)/255.0
            img = cv.resize(img, (imgsize, imgsize))
            img = np.reshape(img, [imgsize, imgsize, 1])
            edge.append(img)
        edge = np.array(edge)
        input_sketch = np.array(input_sketch)
        yield edge.astype(np.float32), input_sketch.astype(np.float32)

def add_conv_layer(INPUT, kernel_size, f_size=4, strides=(2, 2)):
    G = tf.layers.conv2d(INPUT, kernel_size, f_size, strides=strides, padding = 'same', activation=tf.nn.relu)
    OUT = tf.layers.batch_normalization(G)
    return OUT

def add_deconv_layer(INPUT,crop, cat  ,kernel_size, f_size=4):
    U = tf.layers.conv2d_transpose(INPUT, kernel_size, f_size, strides=(2, 2), padding="same", activation=tf.nn.relu)
    U = tf.layers.batch_normalization(U)
    if cat:
        U = tf.concat([U,crop], axis=3)
    return U

def add_disconv_layer(INPUT, kernel_size, f_size=4, strides=(2, 2)):
    INPUT = tf.pad(INPUT, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    G = tf.layers.conv2d(INPUT, kernel_size, f_size, strides=strides, padding = 'valid', activation=tf.nn.relu)
    OUT = tf.layers.batch_normalization(G)
    return OUT

def generator(x):
    with tf.variable_scope("generator"):
        filters = 64
        G_in = x
        G_l1 = tf.layers.conv2d(G_in, filters, 4, strides=(2, 2), padding = 'same', activation=tf.nn.relu)
        G_l2 = add_conv_layer(G_l1, filters*2)
        G_l3 = add_conv_layer(G_l2, filters*4)
        G_l4 = add_conv_layer(G_l3, filters*8)
        G_l5 = add_conv_layer(G_l4, filters*8)
        G_l6 = add_conv_layer(G_l5, filters*8)
        G_l7 = add_conv_layer(G_l6, filters*8)

        print ("gen conv")
        print (G_in.get_shape())
        print (G_l1.get_shape())
        print (G_l2.get_shape())
        print (G_l3.get_shape())
        print (G_l4.get_shape())
        print (G_l5.get_shape())
        print (G_l6.get_shape())
        print (G_l7.get_shape())

        U_1 = add_deconv_layer(G_l7, G_l6, True, filters*8)
        U_2 = add_deconv_layer(U_1, G_l5, True, filters*8)
        U_3 = add_deconv_layer(U_2, G_l4, True, filters*8)
        U_4 = add_deconv_layer(U_3, G_l3, True , filters*4)
        U_5 = add_deconv_layer(U_4, G_l2, True , filters*2)
        U_6 = add_deconv_layer(U_5, G_l1, True , filters)

        print("gen deconv")
        print (U_6.get_shape())
        print (U_1.get_shape())
        print (U_2.get_shape())
        print (U_3.get_shape())
        print (U_4.get_shape())
        print (U_5.get_shape())
        U_out = tf.layers.conv2d_transpose(U_6, 1, 4,strides=(2, 2), padding="same", activation=tf.nn.tanh)
        print (U_out.get_shape())
    return U_out

def discriminator(x, y, flag = False):
    with tf.variable_scope("discriminator", reuse=flag):
        filters = 64
        print ("edge, sketch")
        print (x.get_shape())
        print (y.get_shape())
        D_in = tf.concat([x,y], axis=3)
        D_1 = add_disconv_layer(D_in, filters)
        D_2 = add_disconv_layer(D_1, filters*2)
        D_3 = add_disconv_layer(D_2, filters*4)
        D_4 = add_disconv_layer(D_3, filters*8, strides=(1, 1))

        padded_input = tf.pad(D_4, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        convolved = tf.layers.conv2d(padded_input, 1, 4 ,strides=(1, 1),  padding = 'valid')
        output = tf.sigmoid(convolved)

        print ("dis conv")
        print (D_in.get_shape())
        print (D_1.get_shape())
        print (D_2.get_shape())
        print (D_3.get_shape())
        print (D_4.get_shape())
        print (output.get_shape())

    return output

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        f_sample = np.asarray(sample).flatten()
        normalized = (f_sample-min(f_sample))/(max(f_sample)-min(f_sample))
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(normalized.reshape(imgsize, imgsize), cmap='Greys_r')

    return fig
if __name__ == '__main__':
    Edge = tf.placeholder(tf.float32, shape=[None, imgsize, imgsize, 1])
    Input_sketch = tf.placeholder(tf.float32, shape=[None, imgsize, imgsize, 1])

    G_fake = generator(Input_sketch)
    D_real = discriminator(Edge, Input_sketch, False)
    D_fake = discriminator(G_fake, Input_sketch, True)


    discrim_loss = tf.reduce_mean(-(tf.log(D_real + EPS) + tf.log(1 - D_fake + EPS)))
    gen_loss_GAN = tf.reduce_mean(-tf.log(D_fake + EPS))
    gen_loss_L1 = tf.reduce_mean(tf.abs(Edge - G_fake))
    gen_loss = gen_loss_GAN + gen_loss_L1

    discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
    discrim_optim = tf.train.AdamOptimizer(lr, beta1)
    discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
    discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    gen_optim = tf.train.AdamOptimizer(lr, beta1)
    gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
    gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    saver = tf.train.Saver()
    sess = tf.Session()

    writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)
    sess.run(tf.global_variables_initializer())
    start_time = datetime.datetime.now()

    os.makedirs('result/train', exist_ok=True)
    os.makedirs('model/model-', exist_ok=True)

    for epoch in range(EPOCHS):
        for batch_i, (edge, input_sketch) in enumerate(load_batch(BATCH_SIZE)):
            _, D_loss_curr = sess.run([discrim_train, discrim_loss], feed_dict={Edge: edge, Input_sketch: input_sketch})
            _, G_loss_curr = sess.run([gen_train, gen_loss], feed_dict={Edge: edge, Input_sketch: input_sketch})
            elapsed_time = datetime.datetime.now() - start_time
            print('Epoch %d batch %d D_loss is %.5f G_loss is %.5f. time: %s'%(epoch+1, batch_i, D_loss_curr, G_loss_curr, elapsed_time))
        modelname = "model/model-/" + str(epoch) + "_model"
        saver.save(sess, modelname, global_step=epoch)
        test_edge, test_sketch = load_data(2)
        result = sess.run(G_fake, feed_dict={Input_sketch: test_sketch})
        fig = plot(result)
        plt.savefig('result/train/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
