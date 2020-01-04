# -*- coding=UTF-8 -*-

import tensorflow as tf

from tensorflow.keras import layers


class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()
        self.fc = layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(100,))
        self.bn1 = layers.BatchNormalization()

        self.up2 = layers.Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.up3 = layers.Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()

        self.up4 = layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.bn4 = layers.BatchNormalization()

        self.up5 = layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)


    def call(self, inputs, training=None, mask=None):
        x = self.fc(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        # print("Shape: {}".format(x.shape))

        x = layers.Reshape((4, 4, 1024))(x)

        x = self.up2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        # print("Shape: {}".format(x.shape))

        x = self.up3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        # print("Shape: {}".format(x.shape))

        x = self.up4(x)
        x = self.bn4(x)
        x = tf.nn.relu(x)
        # print("Shape: {}".format(x.shape))

        x = self.up5(x)
        x = tf.nn.tanh(x)
        # print("Shape: {}".format(x.shape))

        return x


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')

        self.conv2 = layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same')
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same')
        self.bn4 = layers.BatchNormalization()

        self.fc = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = tf.nn.leaky_relu(x)
        # x = layers.Dropout(0.3)(x)
        # print("Shape: {}".format(x.shape))

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.leaky_relu(x)
        # x = layers.Dropout(0.3)(x)
        # print("Shape: {}".format(x.shape))

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.leaky_relu(x)
        # x = layers.Dropout(0.3)(x)
        # print("Shape: {}".format(x.shape))

        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.leaky_relu(x)
        # x = layers.Dropout(0.3)(x)
        # print("Shape: {}".format(x.shape))

        x = layers.Flatten()(x)

        x = self.fc(x)
        # print("Shape: {}".format(x.shape))

        return x
