# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configurations"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def model_config():
  cfg = edict()

  # Dataset used to train/val/test model. Now support PASCAL_VOC or KITTI
  cfg.DATASET = 'PASCAL_VOC'

  if cfg.DATASET == 'PASCAL_VOC':
    # object categories to classify
    cfg.CLASS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                       'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                       'sofa', 'train', 'tvmonitor')

  # number of categories to classify
  cfg.CLASSES = len(cfg.CLASS_NAMES)    

  # parameter used in leaky ReLU
  cfg.LEAKY_COEF = 0.1

  # Probability to keep a node in dropout
  cfg.KEEP_PROB = 0.5

  # grid width
  cfg.GWIDTH = 7

  # grid height
  cfg.GHEIGHT = 7

  # number of boxes per grid
  cfg.BOXES = 2

  # batch size
  cfg.BATCH_SIZE = 10

  # image width
  cfg.IMAGE_WIDTH = 224

  # image height
  cfg.IMAGE_HEIGHT = 224

  # Only plot boxes with probability higher than this threshold
  cfg.PROB_THRESH = 0.2

  # Bounding boxes with IOU larger than this are going to be removed
  cfg.NMS_THRESH = 0.5

  # Pixel mean values (BGR order) as a (1, 1, 3) array. Below is the BGR mean
  # of VGG16
  cfg.BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])

  # loss coefficients for no-obect confidence loss
  cfg.LOSS_COEF_NOOBJ = np.sqrt(0.5)

  # loss coefficients for bounding box regression
  cfg.LOSS_COEF_BBOX = np.sqrt(5.0)
                           
  # reduce step size after this many steps
  cfg.DECAY_STEPS = 20000

  # multiply the learning rate by this factor
  cfg.LR_DECAY_FACTOR = 0.1

  # learning rate
  cfg.LEARNING_RATE = 0.001

  # weight decay
  cfg.WEIGHT_DECAY = 0.0005

  # moving average decay, used to visualize parameter evolution
  cfg.MOVING_AVERAGE_DECAY = 0.9999

  # wether to load pre-trained model
  cfg.LOAD_PRETRAINED_MODEL = True

  # path to load the pre-trained model
  cfg.PRETRAINED_MODEL_PATH = ''

  # print log to console in debug mode
  cfg.DEBUG_MODE = False

  return cfg
