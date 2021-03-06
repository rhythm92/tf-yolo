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


def _activation_summary(x, layer_name):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tf.histogram_summary(
      'activation_summary/'+layer_name + '/activations', x)
  tf.scalar_summary(
      'activation_summary/'+layer_name + '/sparsity', tf.nn.zero_fraction(x))
  tf.scalar_summary(
      'activation_summary/'+layer_name + '/average', tf.reduce_mean(x))
  tf.scalar_summary(
      'activation_summary/'+layer_name + '/max', tf.reduce_max(x))
  tf.scalar_summary(
      'activation_summary/'+layer_name + '/min', tf.reduce_min(x))

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

def _variable_on_cpu(name, shape, initializer, trainable=True):
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
    if not callable(initializer):
      var = tf.get_variable(name, initializer=initializer, trainable=trainable)
    else:
      var = tf.get_variable(
          name, shape, initializer=initializer, dtype=dtype,
          trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, initializer=None,
                                trainable=True):
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
  if initializer is None:
    initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
  var = _variable_on_cpu(name, shape, initializer, trainable)
  if wd is not None and trainable:
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
    # Tensor used to represent confidence scores, a.k.a, iou between predicted
    # bbox and gt bbox
    self.ious = tf.Variable(
        initial_value=np.zeros((self.mc.BATCH_SIZE, self.mc.GWIDTH, self.mc.GHEIGHT)),
        trainable=False, name='iou', dtype=tf.float32
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
      _activation_summary(pred_class_probs, 'pred_class_probs')
      num_confidence_scores = mc.GWIDTH*mc.GHEIGHT
      pred_conf = tf.reshape(
          tf.slice(preds, [0, num_class_probs], [-1, num_confidence_scores]),
          [mc.BATCH_SIZE, mc.GWIDTH, mc.GHEIGHT],
          name='pred_confidence_score'
      )
      _activation_summary(pred_conf, 'pred_conf')
      pred_boxes = tf.reshape(
          tf.slice(preds, [0, num_class_probs+num_confidence_scores], [-1, -1]),
          [mc.BATCH_SIZE, mc.GWIDTH, mc.GHEIGHT, 4],
          name='pred_bbox'
      )
      _activation_summary(pred_boxes, 'pred_bbox')

    with tf.variable_scope('IOU'):
      def tensor_bbox_transform(bbox):
        bbox = tf.unpack(bbox, axis=3)

        with tf.variable_scope('scale_wh') as scope:
          bbox[2] = tf.square(bbox[2]) * mc.GWIDTH
          bbox[3] = tf.square(bbox[3]) * mc.GHEIGHT

        with tf.variable_scope('bbox_transform') as scope:
          cx, cy, w, h = bbox
          bbox[0] = tf.identity(cx - w/2, name='xmin')
          bbox[1] = tf.identity(cy - h/2, name='ymin')
          bbox[2] = tf.identity(cx + w/2, name='xmax')
          bbox[3] = tf.identity(cy + h/2, name='ymax')

        return bbox

      def tensor_iou(bbox1, bbox2):
        with tf.variable_scope('intersection'):
          xmin = tf.maximum(bbox1[0], bbox2[0], name='xmin')
          xmax = tf.minimum(bbox1[2], bbox2[2], name='xmax')
          ymin = tf.maximum(bbox1[1], bbox2[1], name='ymin')
          ymax = tf.minimum(bbox1[3], bbox2[3], name='ymax')

          w = tf.maximum(0.0, xmax-xmin, name='inter_w')
          h = tf.maximum(0.0, ymax-ymin, name='inter_h')
          intersection = tf.mul(w, h, name='intersection')

        with tf.variable_scope('union'):
          w1 = tf.sub(bbox1[2], bbox1[0], name='w1')
          h1 = tf.sub(bbox1[3], bbox1[1], name='h1')
          w2 = tf.sub(bbox2[2], bbox2[0], name='w2')
          h2 = tf.sub(bbox2[3], bbox2[1], name='h2')

          union = tf.identity(
              w1*h1 + w2*h2 - intersection, name='union')

        return (intersection/(union+mc.EPSILON))*self.input_mask

      self.ious = self.ious.assign(
          tensor_iou(
              tensor_bbox_transform(pred_boxes),
              tensor_bbox_transform(self.bbox_input)
          )
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
      conf = self.ious if mc.USE_IOU_IN_CONF_REG else self.input_mask
      conf_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.square(
                  (pred_conf - conf)
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
    self.class_loss = cls_loss
    self.conf_loss = conf_loss
    self.bbox_loss = bbox_loss
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
      grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())

    with tf.variable_scope('clip_gradient') as scope:
      for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_norm(grad, 1), var)

    apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads_vars:
      if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        mc.MOVING_AVERAGE_DECAY, self.global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      self.train_op = tf.no_op(name='train')

  def _add_viz_graph(self):
    self.image_to_show = tf.placeholder(
        tf.float32, [None, self.mc.IMAGE_HEIGHT, self.mc.IMAGE_WIDTH, 3],
        name='image_to_show'
    )
    self.viz_op = tf.image_summary('sample_detection_results',
        self.image_to_show, collections='image_summary')

  def _conv_layer(
      self, layer_name, inputs, filters, size, stride, alpha=None, freeze=False,
      stddev=0.01):
    # TODO(bichen): add document for conv layer api
    mc = self.mc
    if not alpha:
      alpha = mc.LEAKY_COEF
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        use_pretrained_param = True
        kernel_val = cw[layer_name][0]
        bias_val = cw[layer_name][1]

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      kernel_init = tf.constant(
          np.transpose(kernel_val, [2,3,1,0]), dtype=tf.float32) \
          if use_pretrained_param else None
      bias_init = tf.constant(bias_val, dtype=tf.float32) \
          if use_pretrained_param else tf.constant_initializer(0.0)

      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters], stddev=stddev,
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

      biases = _variable_on_cpu('biases', [filters], bias_init, 
                                trainable=(not freeze))
  
      conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME',
          name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
  
      out = tf.maximum(alpha*conv_bias, conv_bias)
      _activation_summary(out, layer_name)
      return out
  
  def _pooling_layer(
      self, layer_name, inputs, size, stride):
    with tf.variable_scope(layer_name) as scope:
      return tf.nn.max_pool(inputs, 
                            ksize=[1, size, size, 1], 
                            strides=[1, stride, stride, 1],
                            padding='SAME')
  
  def _fc_layer(
      self, layer_name, inputs, hiddens, flatten=False, activation=True,
      alpha=None, stddev=0.01):
    # TODO(bichen): add document for fc layer api
    mc = self.mc
    if not alpha:
      alpha = mc.LEAKY_COEF

    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        use_pretrained_param = True
        kernel_val = cw[layer_name][0]
        bias_val = cw[layer_name][1]

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      input_shape = inputs.get_shape().as_list()
      if flatten:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs = tf.reshape(inputs, [-1, dim])
        if use_pretrained_param:
          try:
            kernel_val = np.reshape(
                np.transpose(
                    np.reshape(
                        kernel_val,
                        (hiddens, input_shape[3], input_shape[1], input_shape[2])
                    ),
                    (2, 3, 1, 0)
                ),
                (dim, -1)
            )
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            # Do not use pretrained parameter if shape doesn't match
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match,'
                   'Use random initialized parameter'.format(layer_name))
      else:
        dim = input_shape[1]
        if use_pretrained_param:
          kernel_val = np.transpose(kernel_val, (1,0))

      kernel_init = tf.constant(kernel_val, dtype=tf.float32) \
        if use_pretrained_param else None
      bias_init = tf.constant(bias_val, dtype=tf.float32) \
          if use_pretrained_param else tf.constant_initializer(0.0)

      weight = _variable_with_weight_decay(
          'weights', shape=[dim, hiddens], stddev=stddev, wd=mc.WEIGHT_DECAY,
          initializer=kernel_init)
      biases = _variable_on_cpu('biases', [hiddens], bias_init)
  
      outputs = tf.nn.bias_add(tf.matmul(inputs, weight), biases)
      if activation:
        outputs = tf.maximum(alpha*outputs, outputs)

      _activation_summary(outputs, layer_name)

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
            np.array([np.arange(mc.GWIDTH) + 0.5]*mc.GHEIGHT),
            (mc.GHEIGHT, mc.GWIDTH)),
        (1, 0)
    )
    boxes[:,:,0] += x_offset
    boxes[:,:,0] *= float(mc.IMAGE_WIDTH)/mc.GWIDTH

    y_offset = np.reshape(
            np.array([np.arange(mc.GHEIGHT) + 0.5]*mc.GWIDTH),
            (mc.GWIDTH, mc.GHEIGHT)
    )
    boxes[:,:,1] += y_offset
    boxes[:,:,1] *= float(mc.IMAGE_HEIGHT)/mc.GHEIGHT

    boxes[:,:,2] = boxes[:,:,2]**2 * float(mc.IMAGE_WIDTH)
    boxes[:,:,3] = boxes[:,:,3]**2 * float(mc.IMAGE_HEIGHT)

    probs = class_probs * np.reshape(confidence, (mc.GWIDTH, mc.GHEIGHT, 1))

    max_probs = np.max(probs, axis=2)
    cls_idx = np.argmax(probs, axis=2)

    filter_probs_mask = np.array(max_probs>mc.PROB_THRESH, dtype='bool')
    filter_boxes_idx = np.nonzero(filter_probs_mask)

    boxes_filtered = boxes[
        filter_boxes_idx[0], filter_boxes_idx[1]]
    probs_filtered = max_probs[filter_boxes_idx]
    cls_idx_filtered = cls_idx[filter_boxes_idx]

    keep = util.nms(boxes_filtered, probs_filtered, mc.NMS_THRESH)

    boxes_filtered = [boxes_filtered[i] for i, k in enumerate(keep) if k]
    probs_filtered = [probs_filtered[i] for i, k in enumerate(keep) if k]
    cls_idx_filtered = [cls_idx_filtered[i] for i, k in enumerate(keep) if k]

    # trim boxes if outside the image
    out_boxes = []
    for box in boxes_filtered:
      box = util.bbox_transform(box)

      box[0] = max(box[0], 0.0)
      box[1] = max(box[1], 0.0)
      box[2] = min(box[2], float(mc.IMAGE_WIDTH))
      box[3] = min(box[3], float(mc.IMAGE_HEIGHT))

      box = util.bbox_transform_inv(box)
      out_boxes.append(box)

    return cls_idx_filtered, out_boxes, probs_filtered
