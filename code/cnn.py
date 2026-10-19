from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from manual_convolution import ManualConv2d
from base_model import CifarModel

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class CNN(CifarModel):
    def __init__(self, classes):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(CNN, self).__init__()

        # Initialize all hyperparameters
        self.loss_list = []
        self.batch_size = 64
        self.input_width = 32
        self.input_height = 32
        self.image_channels = 3
        self.num_classes = len(classes)

        self.hidden_layer_size = 320

        self.epsilon = 1e-3  # this is used for batch normalization only!

        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2))

        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2))

        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(self.num_classes)

        self.manual_conv = ManualConv2d(
            filter_shape=[3, 3, 3, 32],
            strides=[1, 1, 1, 1],
            padding="SAME",
            use_bias=True,
            trainable=False          # <-- important: prevents the gradient warning
        )

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, num_classes)
        """
        if is_testing:
            self.manual_conv.set_weights(self.conv1.kernel, self.conv1.bias)
            x = self.manual_conv(inputs)
        else:
            x = self.conv1(inputs)
        x = self.bn1(x, training=not is_testing)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=not is_testing)
        x = tf.nn.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=not is_testing)
        x = tf.nn.relu(x)

        x = self.flatten(x)
        x = self.dense1(x)
        if not is_testing:
            x = self.dropout(x)
        return self.dense2(x)

            