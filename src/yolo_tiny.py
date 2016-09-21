# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""YOLO-tiny."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from config import model_config
import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from yolo_model import YoloModel

class YoloTinyModel(YoloModel):
  def __init__(self, mc):
    YoloModel.__init__(self, mc)

    self._add_inference_graph()
    self._add_loss_graph()
    self._add_train_graph()

  def _add_inference_graph(self):
    """Build the YOLO-tiny model."""

    mc = self.mc
  
    conv1 = self._conv_layer(
        layer_name='conv1', inputs=self.image_input, filters=16, size=3,
        stride=1)
    pool1 = self._pooling_layer(
        layer_name='pool1', inputs=conv1, size=2, stride=2)
  
    conv2 = self._conv_layer(
        layer_name='conv2', inputs=pool1, filters=32, size=3, stride=1)
    pool2 = self._pooling_layer(
        layer_name='pool2', inputs=conv2, size=2, stride=2)
  
    conv3 = self._conv_layer(
        layer_name='conv3', inputs=pool2, filters=64, size=3, stride=1)
    pool3 = self._pooling_layer(
        layer_name='pool3', inputs=conv3, size=2, stride=2)
  
    conv4 = self._conv_layer(
        layer_name='conv4', inputs=pool3, filters=128, size=3, stride=1)
    pool4 = self._pooling_layer(
        layer_name='pool4', inputs=conv4, size=2, stride=2)
  
    conv5 = self._conv_layer(
        layer_name='conv5', inputs=pool4, filters=256, size=3, stride=1)
    pool5 = self._pooling_layer(
        layer_name='pool5', inputs=conv5, size=2, stride=2)
  
    conv6 = self._conv_layer(
        layer_name='conv6', inputs=pool5, filters=512, size=3, stride=1)
    pool6 = self._pooling_layer(
        layer_name='pool6', inputs=conv6, size=2, stride=2)
  
    conv7 = self._conv_layer(
        layer_name='conv7', inputs=pool6, filters=1024, size=3, stride=1)
    conv8 = self._conv_layer(
        layer_name='conv8', inputs=conv7, filters=1024, size=3, stride=1)
    conv9 = self._conv_layer(
        layer_name='conv9', inputs=conv8, filters=1024, size=3, stride=1)
  
    fc10 = self._fc_layer('fc10', conv9, 256, flatten=True, activation=True)
    fc11 = self._fc_layer('fc11', fc10, 4096, flatten=False, activation=True)
    dropout12 = tf.nn.dropout(fc11, self.keep_prob, name='dropout12')
  
    num_output = mc.GWIDTH * mc.GHEIGHT * (mc.CLASSES + (1 + 4))
    preds = self._fc_layer(
        'output', dropout12, num_output, flatten=False, activation=False)
  
    self.preds = preds
