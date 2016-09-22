# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Utility functions used in tf-yolo"""

import numpy as np
from config import model_config

def IOU(box1, box2):
  """Compute the Intersection-Over-Union of two given boxes.

  Args:
    box1: array of 4 elements [x, y, width, height].
    box2: same as above
  Returns:
    IOU: a float number in range[0,1]. IOU of the two boxes.
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

def NMS(boxes, probs, threshold):
  """Non-Maximum supression."""

  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)):
    if not keep[order[i]]:
      continue
    for j in range(i+1, len(order)):
      if IOU(boxes[order[i]], boxes[order[j]]) > threshold:
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
