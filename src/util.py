# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Utility functions used in tf-yolo"""

import numpy as np
from config import model_config

def iou(box1, box2):
  """Compute the Intersection-Over-Union of two given boxes.

  Args:
    box1: array of 4 elements [x, y, width, height].
    box2: same as above
  Returns:
    iou: a float number in range[0,1]. iou of the two boxes.
  """

  lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
      max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
  tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
      max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])

  if lr <= 0 or tb <= 0:
    return 0
  else:
    intersection = tb*lr
    union = box1[2]*box1[3]+box2[2]*box2[3]-intersection
    return intersection/union

def nms(boxes, probs, threshold):
  """Non-Maximum supression."""

  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)):
    if not keep[order[i]]:
      continue
    for j in range(i+1, len(order)):
      if iou(boxes[order[i]], boxes[order[j]]) > threshold:
        keep[order[j]] = False
  return keep

def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
  """Build a dense matrix from sparse representations.

  Args:
    sp_indices: A [0-2]-D array that contains the index to place values.
    shape: shape of the dense matrix.
    values: A {0,1}-D array where values corresponds to the index in each row of
    sp_indices.
    default_value: values to set for indices not specified in sp_indices.
  Return:
    A dense numpy N-D array with shape output_shape.
  """

  assert len(sp_indices) == len(values), \
      'Length of sp_indices is not equal to length of values'

  array = np.ones(output_shape) * default_value
  for idx, value in zip(sp_indices, values):
    array[tuple(idx)] = value
  return array

def bgr_to_rgb(ims):
  """Convert a list of images from BGR format to RGB format."""
  out = []
  for im in ims:
    out.append(im[:,:,::-1])
  return out

def bbox_transform(bbox):
  """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]."""

  bbox = [float(b) for b in bbox]
  xmin = bbox[0] - bbox[2]/2
  ymin = bbox[1] - bbox[3]/2
  xmax = bbox[0] + bbox[2]/2
  ymax = bbox[1] + bbox[3]/2

  return [xmin, ymin, xmax, ymax]

def bbox_transform_inv(bbox):
  """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]."""

  bbox = [float(b) for b in bbox]
  cx = (bbox[0] + bbox[2])/2
  cy = (bbox[1] + bbox[3])/2
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]

  return [cx, cy, w, h]
