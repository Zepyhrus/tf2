import numpy as np


import tensorflow as tf
import tensorflow_addons as tfa
from skimage import transform


from utils import load_sample, get_one, show_img


tf.compat.v1.disable_v2_behavior()


def distort_image(
  image, size, ch=1, shuffleflag=False, cropflag=False,
  brightnessflag=False, contrastflag=False
):
  distorted_img =  tf.image.random_flip_left_right(image)

  if cropflag:
    s = tf.random.uniform((1, 2), int(size[0]*0.8), size[0], tf.int32)
    distorted_img = tf.image.random_crop(distorted_img, [s[0][0], s[0][0], ch])
  
  distorted_img = tf.image.random_flip_up_down(distorted_img)

  if brightnessflag:
    distorted_img = tf.image.random_contrast(distorted_img, lower=0.2, upper=1.8)

  if shuffleflag:
    distorted_img = tf.random.shuffle(distorted_img)

  return distorted_img


def _norm_image(image, size, ch=1, flatttenflag=False):
  image_decoded = image * 2 / 255.0 - 1 # normalize to [-1, 1]
  
  if flatttenflag:
    image_decoded = tf.reshape(image_decoded, [size[0] * size[1] * ch])
  
  return image_decoded


# def _rotated(image):
#   shift_y, shift_x = np.array(image.shape.as_list()[:2], np.float32) / 2.

#   tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(30))
#   tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
#   tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
  
#   return transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)


def _rotatedwrap(image):
  # image_rotated = tf.py_function(_rotated, [image], [tf.float64])
  # return tf.cast(image_rotated, tf.float32)[0]

  image_rotated = tfa.image.transform_ops.rotate(image, np.deg2rad(30))
  return tf.cast(image_rotated, tf.float32)



def _random_rotated30(image, label):
  a = tf.random.uniform([1], 0, 2, tf.int32)

  image_decoded = tf.cond(
    pred=tf.equal(tf.constant(0), a[0]),
    true_fn=lambda: image,
    false_fn=lambda: _rotatedwrap(image)
  )

  return image_decoded, label


def _parseone(filename, label):
  image_string = tf.io.read_file(filename=filename)
  image_decoded = tf.image.decode_image(image_string)
  image_decoded.set_shape([None, None, None])
  image_decoded = distort_image(image_decoded, size)
  image_decoded = tf.image.resize(image_decoded, size)
  image_decoded = _norm_image(image_decoded, size)
  image_decoded = tf.cast(image_decoded, dtype=tf.float32)
  label = tf.cast(tf.reshape(label, []), dtype=tf.float32)

  return image_decoded, label


def dataset(directory, size, batch_size, random_rotated=False):
  """parse dataset"""
  # load samples
  (filenames, labels), lab  = load_sample(directory, shuffleflag=True)
  print(filenames[:2], labels[:2], lab)
  
  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  dataset = dataset.map(_parseone)

  if random_rotated:
    dataset = dataset.map(_random_rotated30)

  dataset = dataset.batch(batch_size)

  return dataset


if __name__ == "__main__":
  sample_dir = 'data/mnist/train'
  size = [28, 28]
  batch_size = 10
  tdataset = dataset(sample_dir, size, batch_size)
  tdataset2 = dataset(sample_dir, size, batch_size, True).shuffle(100)

  print(tdataset.element_spec, tdataset2.element_spec)

  one_element1 = get_one(tdataset)
  one_element2 = get_one(tdataset2)

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.local_variables_initializer())
    sess.run(tf.compat.v1.global_variables_initializer())

    try:
      for step in range(10):
        value = sess.run(one_element1)
        value2 = sess.run(one_element2)

        show_img(step, value2[1], np.asarray(value2[0] * 255, np.uint8), 10)
    except tf.errors.OutOfRangeError:
      print('Done!!!')


