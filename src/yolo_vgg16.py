# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""YOLO-VGG16 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from config import model_config
import joblib
import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from yolo_model import YoloModel

class YoloVGG16Model(YoloModel):
  def __init__(self, mc):
    mc.LEAKY_COEF = 0 # Set the leaky coefficient to 0 for VGG16

    YoloModel.__init__(self, mc)

    self._add_inference_graph()
    self._add_loss_graph()
    self._add_train_graph()
    self._add_viz_graph()

  def _add_inference_graph(self):
    """Build the VGG-16 model."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    with tf.variable_scope('conv1') as scope:
      conv1_1 = self._conv_layer(
          'conv1_1', self.image_input, filters=64, size=3, stride=1, freeze=True)
      conv1_2 = self._conv_layer(
          'conv1_2', conv1_1, filters=64, size=3, stride=1, freeze=True)
      pool1 = self._pooling_layer(
          'pool1', conv1_2, size=2, stride=2)

    with tf.variable_scope('conv2') as scope:
      conv2_1 = self._conv_layer(
          'conv2_1', pool1, filters=128, size=3, stride=1, freeze=True)
      conv2_2 = self._conv_layer(
          'conv2_2', conv2_1, filters=128, size=3, stride=1, freeze=True)
      pool2 = self._pooling_layer(
          'pool2', conv2_2, size=2, stride=2)

    with tf.variable_scope('conv3') as scope:
      conv3_1 = self._conv_layer(
          'conv3_1', pool2, filters=256, size=3, stride=1)
      conv3_2 = self._conv_layer(
          'conv3_2', conv3_1, filters=256, size=3, stride=1)
      conv3_3 = self._conv_layer(
          'conv3_3', conv3_2, filters=256, size=3, stride=1)
      pool3 = self._pooling_layer(
          'pool3', conv3_3, size=2, stride=2)

    with tf.variable_scope('conv4') as scope:
      conv4_1 = self._conv_layer(
          'conv4_1', pool3, filters=512, size=3, stride=1)
      conv4_2 = self._conv_layer(
          'conv4_2', conv4_1, filters=512, size=3, stride=1)
      conv4_3 = self._conv_layer(
          'conv4_3', conv4_2, filters=512, size=3, stride=1)
      pool4 = self._pooling_layer(
          'pool4', conv4_3, size=2, stride=2)

    with tf.variable_scope('conv5') as scope:
      conv5_1 = self._conv_layer(
          'conv5_1', pool4, filters=512, size=3, stride=1)
      conv5_2 = self._conv_layer(
          'conv5_2', conv5_1, filters=512, size=3, stride=1)
      conv5_3 = self._conv_layer(
          'conv5_3', conv5_2, filters=512, size=3, stride=1)
      pool5 = self._pooling_layer(
          'pool5', conv5_3, size=2, stride=2)

    with tf.variable_scope('fc') as scope:
      fc6 = self._fc_layer('fc6', pool5, 4096, flatten=True)
      dropout6 = tf.nn.dropout(fc6, self.keep_prob, name='drop6')
      fc7 = self._fc_layer('fc7', dropout6, 4096)
      dropout7 = tf.nn.dropout(fc7, self.keep_prob, name='drop7')

    num_output = mc.GWIDTH * mc.GHEIGHT * (mc.CLASSES + (1 + 4))
    preds = self._fc_layer('output', dropout7, num_output, activation=True,
                           stddev=0.001)

    self.preds = preds
