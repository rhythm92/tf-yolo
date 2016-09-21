# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""YOLO model base class."""

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


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss):
  """Add summaries for losses in Yolo.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

class YoloModel:
  def __init__(self, mc):
    self.mc = mc

    # with tf.variable_scope('model_input') as scope:
    # image batch input
    self.image_input = tf.placeholder(
        tf.float32, [self.mc.BATCH_SIZE, self.mc.IMAGE_HEIGHT, self.mc.IMAGE_WIDTH, 3],
        name='image_input'
    )
    # a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
    # 1.0 in evaluation phase
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # A tensor where each value is either {0, 1}. Used to filter which part of
    # the network output goes to the loss.
    self.input_mask = tf.placeholder(
        tf.float32, [self.mc.BATCH_SIZE, self.mc.GWIDTH, self.mc.GHEIGHT],
        name='input_mask'
    )
    # Tensor used to represent bounding boxes.
    self.bbox_input = tf.placeholder(
        tf.float32, [self.mc.BATCH_SIZE, self.mc.GWIDTH, self.mc.GHEIGHT, 4],
        name='bbox_input'
    )
    # Tensor used to represent confidence scores. 
    self.conf_input = tf.placeholder(
        tf.float32, [self.mc.BATCH_SIZE, self.mc.GWIDTH, self.mc.GHEIGHT],
        name='confidence_score_input'
    )
    # Tensor used to represent labels
    self.labels = tf.placeholder(
        tf.float32, 
        [self.mc.BATCH_SIZE, self.mc.GWIDTH, self.mc.GHEIGHT, mc.CLASSES],
        name='labels'
    )

  def _add_inference_graph(self):
    raise NotImplementedError

  def _add_loss_graph(self):
    mc = self.mc

    # interprret inference outputs
    with tf.variable_scope('interpret_output') as scope:
      preds = self.preds

      num_class_probs = mc.GWIDTH*mc.GHEIGHT*mc.CLASSES
      pred_class_probs = tf.reshape(
          tf.slice(preds, [0, 0], [-1, num_class_probs]),
          [mc.BATCH_SIZE, mc.GWIDTH, mc.GHEIGHT, mc.CLASSES],
          name='pred_class_probs'
      )
      num_confidence_scores = mc.GWIDTH*mc.GHEIGHT
      pred_conf = tf.reshape(
          tf.slice(preds, [0, num_class_probs], [-1, num_confidence_scores]),
          [mc.BATCH_SIZE, mc.GWIDTH, mc.GHEIGHT],
          name='pred_confidence_score'
      )
      pred_boxes = tf.reshape(
          tf.slice(preds, [0, num_class_probs+num_confidence_scores], [-1, -1]),
          [mc.BATCH_SIZE, mc.GWIDTH, mc.GHEIGHT, 4],
          name='pred_bbox'
      )

    with tf.variable_scope('class_regression') as scope:
      cls_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.square( 
                  pred_class_probs 
                  * tf.reshape(
                        self.input_mask, 
                        [mc.BATCH_SIZE, mc.GWIDTH, mc.GHEIGHT, 1])
                  - self.labels
              ),
              reduction_indices=[1,2,3]
          ),
          name='class_loss'
      )
      tf.add_to_collection('losses', cls_loss)

    with tf.variable_scope('confidence_score_regression') as scope:
      conf_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.square(
                  (pred_conf - self.conf_input)
                  * (self.input_mask + (1 - self.input_mask)*mc.LOSS_COEF_NOOBJ)
              ),
              reduction_indices=[1,2]
          ),
          name='confidence_loss'
      )
      tf.add_to_collection('losses', conf_loss)

    with tf.variable_scope('bounding_box_regression') as scope:
      bbox_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.square(
                  (pred_boxes - self.bbox_input) 
                  * tf.reshape(
                        self.input_mask, 
                        [mc.BATCH_SIZE, mc.GWIDTH, mc.GHEIGHT, 1])
                  * mc.LOSS_COEF_BBOX 
              ),
              reduction_indices=[1,2,3]
          ),
          name='bbox_loss'
      )
      tf.add_to_collection('losses', bbox_loss)

    # add above losses as well as weight decay losses to form the total loss
    self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')


  def _add_train_graph(self):
    mc = self.mc

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                    self.global_step,
                                    mc.DECAY_STEPS,
                                    mc.LR_DECAY_FACTOR,
                                    staircase=True)

    tf.scalar_summary('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(self.loss)

    with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.GradientDescentOptimizer(lr)
      grads = opt.compute_gradients(self.loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads:
      if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        mc.MOVING_AVERAGE_DECAY, self.global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      self.train_op = tf.no_op(name='train')

  def _conv_layer(
      self, layer_name, inputs, filters, size, stride, alpha=None):
    mc = self.mc
    if not alpha:
      alpha = mc.LEAKY_COEF
    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]
      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters], stddev=0.1,
          wd=mc.WEIGHT_DECAY)
      biases = _variable_on_cpu('biases', [filters], tf.constant_initializer(0.1))
  
      conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME',
          name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
  
      return tf.maximum(alpha*conv_bias, conv_bias)
  
  def _pooling_layer(
      self, layer_name, inputs, size, stride):
    with tf.variable_scope(layer_name) as scope:
      return tf.nn.max_pool(inputs, 
                            ksize=[1, size, size, 1], 
                            strides=[1, stride, stride, 1],
                            padding='SAME')
  
  def _fc_layer(
      self, layer_name, inputs, hiddens, flatten=False, activation=True,
      alpha=None):
    mc = self.mc
    if not alpha:
      alpha = mc.LEAKY_COEF
    with tf.variable_scope(layer_name) as scope:
      input_shape = inputs.get_shape().as_list()
      if flatten:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs = tf.reshape(inputs, [-1, dim])
        # TODO(bichen): figure out which is the right way to reshape/transpose
        # data.
        # inputs = tf.reshape(
        #     tf.transpose(inputs, (0, 3, 1, 2)),
        #     [-1, dim]
        # )
      else:
        dim = input_shape[1]
  
      # TODO(bichen): add weight decay
      weight = _variable_with_weight_decay('weights', shape=[dim, hiddens],
                                           stddev=0.1, wd=mc.WEIGHT_DECAY)
      biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(0.1))
  
      outputs = tf.nn.bias_add(tf.matmul(inputs, weight), biases)
      if activation:
        return tf.maximum(alpha*outputs, outputs)
      else:
        return outputs

  def interpret_prediction(self, preds):
    mc = self.mc

    num_class_probs = mc.GWIDTH*mc.GHEIGHT*mc.CLASSES
    class_probs = np.reshape(preds[:num_class_probs],
        (mc.GWIDTH, mc.GHEIGHT, mc.CLASSES))

    num_confidence_scores = mc.GWIDTH*mc.GHEIGHT
    confidence = np.reshape(
        preds[num_class_probs:num_class_probs+num_confidence_scores],
        (mc.GWIDTH, mc.GHEIGHT))

    boxes = np.reshape(
        preds[num_class_probs+num_confidence_scores:],
        (mc.GWIDTH, mc.GHEIGHT, 4))

    x_offset = np.transpose(
        np.reshape(
            np.array([np.arange(mc.GWIDTH)]*mc.GHEIGHT),
            (mc.GHEIGHT, mc.GWIDTH)),
        (1, 0)
    )
    boxes[:,:,0] += x_offset
    boxes[:,:,0] *= mc.IMAGE_WIDTH/mc.GWIDTH

    y_offset = np.reshape(
            np.array([np.arange(mc.GHEIGHT)]*mc.GWIDTH),
            (mc.GWIDTH, mc.GHEIGHT)
    )
    boxes[:,:,1] += y_offset
    boxes[:,:,1] *= mc.IMAGE_HEIGHT/mc.GHEIGHT

    boxes[:,:,2:] = boxes[:,:,2:]**2

    probs = class_probs * np.reshape(confidence, (mc.GWEIGHT, mc.GHEIGHT, 1))

    max_probs = np.max(probs, axis=2)
    cls_idx = np.argmax(probs, axis=2)

    filter_probs_idx = np.array(max_probs>mc.PROB_THRESH, dtype='bool')
    filter_boxes_idx = np.nonzero(filter_probs_idx)

    boxes_filtered = boxes[
        filter_boxes_idx[0], filter_boxes_idx[1]]
    probs_filtered = max_probs[filter_probs_idx]
    cls_idx_filtered = cls_idx[filter_probs_idx]

    keep = util.NMS(boxes_filtered, probs_filtered. mc.NMS_THRESH)

    boxes_filtered = boxes_filtered[keep]
    probs_filtered = probs_filtered[keep]
    cls_idx_filtered = cls_idx_filtered[keep]

    result = []
    for i in range(len(boxes_filtered)):
      result.append([mc.CLASS_NAMES[cls_idx_filtered[i]], 
                     boxes_filtered[i][0], boxes_filtered[i][1],
                     boxes_filtered[i][2], boxes_filtered[i][3],
                     probs_filtered[i]])

    return result
