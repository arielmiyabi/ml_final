import tensorflow as tf
import numpy as np
import os
import cv2 as cv
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
from glob import glob
import matplotlib.pyplot as plt


class sketch2edge():
    def __init__(self):
        # dataset
        self.sketch = []
        self.edge = []
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.filters = 64
        self.n_batches = 0

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Build the generator
        self.generator = self.generator()

        # Build and compile the discriminator
        self.discriminator = self.descriminator()
        self.discriminator.compile(loss='mse',
            optimizer=Adam(0.0002, 0.5),
            metrics=['accuracy'])

        # Input sketch and their conditioning images(edge)
        input_sketch = Input(shape=self.img_shape)
        edge = Input(shape=self.img_shape)
        fake_sketch = self.generator(input_sketch)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_sketch, input_sketch])

        self.combined = Model(inputs=[edge, input_sketch], outputs=[valid, fake_sketch])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=Adam(0.0002, 0.5))




    def generator(self):
        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.filters, bn=False)
        d2 = conv2d(d1, self.filters*2)
        d3 = conv2d(d2, self.filters*4)
        d4 = conv2d(d3, self.filters*8)
        d5 = conv2d(d4, self.filters*8)
        d6 = conv2d(d5, self.filters*8)
        d7 = conv2d(d6, self.filters*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.filters*8)
        u2 = deconv2d(u1, d5, self.filters*8)
        u3 = deconv2d(u2, d4, self.filters*8)
        u4 = deconv2d(u3, d3, self.filters*4)
        u5 = deconv2d(u4, d2, self.filters*2)
        u6 = deconv2d(u5, d1, self.filters)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)


    def descriminator(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        edge = Input(shape=self.img_shape)
        input_sketch = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([edge, input_sketch])

        d1 = d_layer(combined_imgs, self.filters, bn=False)
        d2 = d_layer(d1, self.filters*2)
        d3 = d_layer(d2, self.filters*4)
        d4 = d_layer(d3, self.filters*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([edge, input_sketch], validity)

    def load_batch(self, batch_size=1):
        data = glob("./new_train_test_data/train_edge_data/*")
        label = glob("./new_train_test_data/train_edge_label/*")
        self.n_batches = int(len(data) / batch_size)
        for i in range(self.n_batches-1):
            edge = []
            input_sketch = []
            for filename in data[i:i+batch_size]:
                img = cv.imread(filename, cv.IMREAD_COLOR)/255
                img = cv.resize(img, (128, 128))
                input_sketch.append(img)
            for filename in label[i:i+batch_size]:
                img = cv.imread(filename, cv.IMREAD_COLOR)/255
                img = cv.resize(img, (128, 128))
                edge.append(img)
            edge = np.array(edge)
            input_sketch = np.array(input_sketch)
            yield edge, input_sketch
    def load_data(self, batch_size=1):
        data = glob("./new_train_test_data/test_edge_data/*")
        label = glob("./new_train_test_data/test_edge_label/*")
        batch_data = np.random.choice(data, size=batch_size)
        batch_label = np.random.choice(label, size=batch_size)

        edge = []
        input_sketch = []
        for filename in batch_data:
            img = cv.imread(filename, cv.IMREAD_COLOR)/255
            img = cv.resize(img, (128, 128))
            input_sketch.append(img)
        for filename in batch_label:
            img = cv.imread(filename, cv.IMREAD_COLOR)/255
            img = cv.resize(img, (128, 128))
            edge.append(img)

        edge = np.array(edge)
        input_sketch = np.array(input_sketch)

        return edge, input_sketch
    def sample_images(self, epoch, batch_i):
        os.makedirs('result/train', exist_ok=True)
        r, c = 3, 3

        edge, input_sketch = self.load_data(batch_size=3)
        fake_A = self.generator.predict(input_sketch)

        gen_imgs = np.concatenate([input_sketch, fake_A, edge])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['input_sketch', 'Generated', 'edge']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("result/train/%d_%d.png" % (epoch, batch_i))
        plt.close()
    def train(self, epochs, batch_size=1, sample_interval=50):
        os.makedirs('model', exist_ok=True)
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (edge, input_sketch) in enumerate(self.load_batch(batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_sketch = self.generator.predict(input_sketch)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([edge, input_sketch], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_sketch, input_sketch], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([edge, input_sketch], [valid, edge])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
            if ((epoch+1)%20 == 0):
                modelname = "model/" + str(epoch) + "_model.h5"
                self.combined.save(modelname)
if __name__ == '__main__':
    print ("strat")
    gan = sketch2edge()
    gan.train(epochs=100, batch_size=32, sample_interval=200)
