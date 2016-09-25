# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Train YOLO on a single GPU"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import model_config
from imdb import pascal_voc
from util import sparse_to_dense, bgr_to_rgb, bbox_transform
from yolo_tiny import YoloTinyModel
from yolo_vgg16 import YoloVGG16Model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_path', 
                           '/home/eecs/bichen/Proj/YOLO/tf-yolo/data/VOC/VOCdevkit',
                           """Root directory of data""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for VOC data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """Only used for VOC data."""
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/yolo-vgg16/train',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('pretrained_model_path',
                           '/home/eecs/bichen/Proj/YOLO/tf-yolo/data/'
                           'VGG16/VGG_ILSVRC_16_layers_weights.pkl',
                            """Path to the pretrained model.""")


def _draw_box(im, box_list, label_list, color=(0,255,0)):
  for bbox, label in zip(box_list, label_list):

    xmin, ymin, xmax, ymax = [int(b) for b in bbox_transform(bbox)]
    # draw box
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 1)
    # draw label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, label, (xmin, ymax), font, 0.3, color, 1)

def _viz_prediction_result(model, images, bboxes, labels, preds):
  mc = model.mc

  for im, bbox_list, label_list, pred in zip(images, bboxes, labels, preds):
    # draw ground truth
    _draw_box(
        im, bbox_list,
        [mc.CLASS_NAMES[idx] for idx in label_list],
        (0, 255, 0))

    # draw prediction
    pred_cls_list, pred_bbox_list, probs = model.interpret_prediction(pred)
    _draw_box(
        im, pred_bbox_list,
        [mc.CLASS_NAMES[idx]+': (%.2f)'% prob for idx, prob in zip(pred_cls_list,
                                                                  probs)],
        (0,0,255))


def train():
  """Train YOLO"""
  with tf.Graph().as_default():
    mc = model_config()
    mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path

    imdb = pascal_voc(FLAGS.image_set, FLAGS.year, FLAGS.data_path, mc)
    model = YoloVGG16Model(mc)

    saver = tf.train.Saver(tf.all_variables())
    summary_op = tf.merge_all_summaries()
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # read batch input
      images, bboxes, labels, gidxes, orig_bboxes = imdb.read_batch()
      label_indices, bbox_indices, bbox_values, mask_indices = [], [], [], []
      for i in range(len(labels)):
        for j in range(len(labels[i])):
          label_indices.append(
              [i, gidxes[i][j][0], gidxes[i][j][1], labels[i][j]])
          bbox_indices.extend(
              [[i, gidxes[i][j][0], gidxes[i][j][1], k] for k in range(4)])
          bbox_values.extend(bboxes[i][j])
          mask_indices.append(
              [i, gidxes[i][j][0], gidxes[i][j][1]])

      feed_dict = {
          model.image_input: images,
          model.keep_prob: mc.KEEP_PROB,
          model.bbox_input: sparse_to_dense(
              bbox_indices, [mc.BATCH_SIZE, mc.GWIDTH, mc.GHEIGHT, 4],
              bbox_values),
          model.input_mask: sparse_to_dense(
              mask_indices, [mc.BATCH_SIZE, mc.GWIDTH, mc.GHEIGHT],
              [1.0]*len(mask_indices)),
          model.labels: sparse_to_dense(
              label_indices, 
              [mc.BATCH_SIZE, mc.GWIDTH, mc.GHEIGHT, mc.CLASSES],
              [1.0]*len(label_indices)),
      }

      if step % 100 == 0:
        op_list = [
            model.train_op, model.loss, summary_op, model.preds,
            model.conf_loss, model.bbox_loss, model.class_loss, model.ious
        ]
        _, loss_value, summary_str, preds, conf_loss, bbox_loss, class_loss, \
            ious = sess.run(op_list, feed_dict=feed_dict)

        print ('conf_loss: {}, bbox_loss: {}, class_loss: {}, '
               'max iou: {}'.format(conf_loss, bbox_loss, class_loss,
                                    np.max(ious)))
        preds = preds.tolist()
        _viz_prediction_result(model, images, orig_bboxes, labels, preds)
        images = bgr_to_rgb(images)
        viz_summary = sess.run(
            model.viz_op, feed_dict={model.image_to_show: images})

        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(viz_summary, step)
      else:
        _, loss_value = sess.run(
            [model.train_op, model.loss], feed_dict=feed_dict)

      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_images_per_step = mc.BATCH_SIZE
        images_per_sec = num_images_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             images_per_sec, sec_per_batch))
        sys.stdout.flush()

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
