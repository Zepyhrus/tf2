import os
import tensorflow as tf
import numpy as np

import random
from skimage import transform

from utils import load_sample, show_img, get_one


def _distort_img(
  img, size, ch=1, shuffleflag=True, cropflag=True,
  brightnessflag=True, contrastflag=True
):
  # Note that the input is different from 4-10
  distorted_img = tf.image.random_flip_left_right(img)

  if cropflag:
    s = tf.random.uniform((1, 2), int(size[0]*0.8), size[0], tf.int32)
    distorted_img = tf.image.random_crop(distorted_img, [s[0][0], s[0][0], ch])

  distorted_img = tf.image.random_flip_up_down(distorted_img)
  
  if brightnessflag:
    distorted_img = tf.image.random_brightness(distorted_img, max_delta=10)
  
  if contrastflag:
    distorted_img = tf.image.random_contrast(distorted_img, lower=0.2, upper=1.8)

  if shuffleflag:
    distorted_img = tf.random.shuffle(distorted_img)
  
  return distorted_img


def _norm_img(img, size, ch=1, flattenflag=False):
  img_decoded = img / 255.0

  if flattenflag:
    img_decoded = tf.reshape(img_decoded, [size[0]*size[1]*ch])
  
  return img_decoded


def _rotated(img):
  shift_y, shift_x = np.array(img.shape[:2], np.float32) / 2.
  tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(random.randint(0, 30)))
  tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
  tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
  return transform.wrap(img, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)

def _rotated_wrap(img):
  img_rotated = tf.py_function( _rotated, [img], [tf.float32] )

  return img_rotated[0]


def _random_rotated(img, label):
  a = tf.random.uniform([1], 0, 2, tf.int32)
  img_decoded = tf.cond(
    pred=tf.equal(tf.constant(0), a[0]), 
    true_fn=lambda: img,
    false_fn=lambda: _rotate_wrap(img)
  )

  return img_decoded, label


if __name__ == "__main__":
  (filenames, labels), lab = load_sample('data/mnist/train', shuffleflag=True)

  


