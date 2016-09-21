# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""The data base wrapper class"""

import cv2
import os 
import numpy as np
import xml.etree.ElementTree as ET

from config import model_config

class imdb(object):
  """Image database."""

  def __init__(self, name, mc):
    self._name = name
    self._classes = []
    self._image_index = []
    self.mc = mc

  @property
  def name(self):
    return self._name

  @property
  def classes(self):
    return self._classes

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def image_index(self):
    return self._image_index

class pascal_voc(imdb):
  def __init__(self, image_set, year, data_path, mc):
    imdb.__init__(self, 'voc_'+year+'_'+image_set, mc)
    self._year = year
    self._image_set = image_set
    self._data_root_path = data_path
    self._data_path = os.path.join(self._data_root_path, 'VOC' + self._year)
    self._classes = self.mc.CLASS_NAMES
    self._class_to_idx = dict(zip(self.classes, xrange(self.num_classes)))

    # a list of string indices of images in the directory
    self._image_idx = self._load_image_set_idx() 
    # a dict of image_idx -> [[x, y, w, h, cls_idx]]. x,y,w,h are not divided by
    # the image width and height
    self._rois = self._load_pascal_annotation()

    ## batch reader ##
    self._perm_idx = None
    self._cur_idx = 0
    # TODO(bichen): add a random seed as parameter
    self._shuffle_image_idx()

  def _load_image_set_idx(self):
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set+'.txt')
    assert os.path.exists(image_set_file), \
        'File does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
      image_idx = [x.strip() for x in f.readlines()]
    return image_idx

  def _image_path_at(self, idx):
    image_path = os.path.join(self._data_path, 'JPEGImages', idx+'.jpg')
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path


  def _load_pascal_annotation(self):
    idx2annotation = {}
    for index in self._image_idx:
      filename = os.path.join(self._data_path, 'Annotations', index+'.xml')
      tree = ET.parse(filename)
      objs = tree.findall('object')
      objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
      bboxes = []
      for obj in objs:
        bbox = obj.find('bndbox')
        xl = float(bbox.find('xmin').text)
        xr = float(bbox.find('xmax').text)
        yb = float(bbox.find('ymin').text)
        yt = float(bbox.find('ymax').text)

        x = (xl + xr)/2
        y = (yb + yt)/2
        w = xr - xl
        h = yt - yb

        cls = self._class_to_idx[obj.find('name').text.lower().strip()]
        bboxes.append([x, y, w, h, cls])

      idx2annotation[index] = bboxes

    return idx2annotation

  def _shuffle_image_idx(self):
    self._perm_idx = [self._image_idx[i] for i in
        np.random.permutation(np.arange(len(self._image_idx)))]
    self._cur_idx = 0

  def read_batch(self):
    """Read a batch of image and bounding box annotations.

    Returns:
      images: length batch_size list of arrays [width, height, 3]
      bboxes: length batch_size list of list of arrays [x,y,w,h]
      labels: length batch_size list of class index
      gidxes: length batch_size list of list of arrays [grid_idx_x, grid_idx_y]
    """
    mc = self.mc

    if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
      self._shuffle_image_idx()

    batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
    self._cur_idx += mc.BATCH_SIZE
    images = []
    bboxes = []
    labels = []
    gidxes = []
    for i in batch_idx:
      im = cv2.imread(self._image_path_at(i))
      im = im.astype(np.float32, copy=False)
      im -= mc.BGR_MEANS
      orig_h, orig_w, _ = im.shape
      im = cv2.resize(im, (mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH))

      # TODO(bichen): these reference center coordinates are baddly chosen.
      x_centers = np.arange(mc.GWIDTH)
      y_centers = np.arange(mc.GHEIGHT)

      # normalize bounding boxes
      bbox = self._rois[i]
      gidx = []
      cls_idx = []
      for b in bbox:
        b[0] *= mc.GWIDTH/mc.IMAGE_WIDTH
        gidx_x = np.argmin(abs(b[0] - x_centers))
        b[0] -= x_centers[gidx_x]

        b[1] *= mc.GHEIGHT/mc.IMAGE_HEIGHT
        gidx_y = np.argmin(abs(b[1] - y_centers))
        b[1] -= y_centers[gidx_y]

        b[2] = np.sqrt(b[2]/orig_w)
        b[3] = np.sqrt(b[3]/orig_h)

        cls_idx.append(b[4])
        gidx.append([gidx_x, gidx_y])

      images.append(im)
      bboxes.append(bbox)
      labels.append(cls_idx)
      gidxes.append(gidx)

    return images, bboxes, labels, gidxes
