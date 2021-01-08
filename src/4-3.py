import tensorflow as tf
import os
import matplotlib.pyplot as plt

import numpy as np
from sklearn.utils import shuffle

tf.compat.v1.disable_v2_behavior()


def load_sample(sample_dir):
  print('loading sample dataset..')
  
  lfilenames = []
  labelsnames = []

  for dirpath, dirname, filenames in os.walk(sample_dir):
    for filename in filenames:
      filename_path = os.sep.join([dirpath, filename])
      
      lfilenames.append(filename_path)
      labelsnames.append( dirpath.split('/')[-1] )

  lab = list(sorted(set(labelsnames)))
  labdict = dict(zip( lab, list(range(len(lab))) ))

  labels = [labdict[i] for i in labelsnames]

  return shuffle((np.asarray(lfilenames), np.asarray(labels))), np.asarray(lab)


def get_batches(image, label, resize_w, resize_h, channels, batch_size):
  queue = tf.compat.v1.train.slice_input_producer([image, label]) # a input queue
  
  label = queue[1]
  image = tf.image.decode_bmp(
    tf.io.read_file(queue[0]),
    channels=channels
  ) # read image
  image = tf.image.resize_with_crop_or_pad(image, resize_w, resize_h) # resize
  image = tf.image.per_image_standardization(image=image) # standardization

  images_batch, labels_batch = tf.compat.v1.train.batch(
    [image, label],
    batch_size=batch_size,
    num_threads=64
  )
  images_batch = tf.cast(images_batch, tf.float32)

  return images_batch, labels_batch


def showresult(subplot, title, thisimg):
  p = plt.subplot(subplot)
  p.axis('off')

  p.imshow(np.reshape(thisimg, (28, 28)))
  p.set_title(title)


def showimg(index, label, img, ntop):
  plt.figure(figsize=(20, 10))
  plt.axis('off')

  ntop = min(ntop, 9)
  print(index)

  for i in range(ntop):
    showresult(100 + 10*ntop + 1 + i, label[i], img[i])
  
  plt.show()

if __name__ == "__main__":
  data_dir = 'data/mnist/train'

  (image, label), labelsnames = load_sample(data_dir)

  print(len(image), image[:2], len(label), label[:2])
  print(labelsnames[ label[:2] ], labelsnames)

  batch_size = 16
  image_batches, label_batches = get_batches(image, label, 28, 28, 1, batch_size)

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    coord = tf.train.Coordinator()

    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

    try:
      for step in np.arange(10):
        if coord.should_stop():
          break
      
        images, labels = sess.run([image_batches, label_batches]) # inject data
        showimg(step, labels, images, batch_size)
        print(labels)
    except tf.errors.OutOfRangeError:
      print('Done!!!')
    finally:
      coord.request_stop()
    
    coord.join(threads)

