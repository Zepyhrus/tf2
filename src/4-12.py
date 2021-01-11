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

  return transform.warp(img.numpy(), (tf_shift + (tf_rotate + tf_shift_inv)).inverse)


def _rotate_wrap(img):
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

def dataset(directory, size, batch_size, random_rotated=False):
  (filenames, labels), _ = load_sample(directory, shuffleflag=False)

  def _parseone(filename, label):
    """ Reading and handle image """
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_decoded.set_shape([None, None, None])
    image_decoded = _distort_img(image_decoded, size)
    image_decoded = tf.image.resize(image_decoded, size)
    image_decoded = _norm_img(image_decoded, size)
    image_decoded = tf.cast(image_decoded, dtype=tf.float32)
    label = tf.cast( tf.reshape(label, []), dtype=tf.int32 )

    return image_decoded, label

  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  dataset = dataset.map(_parseone)

  if random_rotated:
    dataset = dataset.map(_random_rotated)
  
  dataset = dataset.batch(batch_size)

  return dataset


if __name__ == "__main__":
  save_dir = 'data/mnist'
  (filenames, labels), lab = load_sample(save_dir, shuffleflag=True)
  
  size = [28, 28]
  batch_size = 10

  tdataset2 = dataset(save_dir, size, batch_size, True)

  for step, value in enumerate(tdataset2):
    show_img(step, value[1].numpy(), np.asarray( value[0] * 255, np.uint8 ), 10)

  


