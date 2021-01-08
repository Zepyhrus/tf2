import os
from os.path import join, split, basename

import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm

tf.compat.v1.disable_v2_behavior()

def load_sample(sample_dir, shuffle=True):
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

  if shuffle:
    return shuffle(
      (np.asarray(lfilenames), np.asarray(labels))
    ), np.asarray(lab)
  else:
    return (np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)


def make_tfrec(filenames, labels, target):
  writer = tf.io.TFRecordWriter(target)

  for i in tqdm(range(len(labels))):
    img = Image.open(filenames[i])
    img = img.resize((28, 28))
    img_raw = img.tobytes()
    example = tf.train.Example(
      features=tf.train.Features(
        feature={
          'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
          'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }
      )
    )

    writer.write(example.SerializeToString())
  writer.close()


def read_and_decode(filenames, flag='train', batch_size=3):
  if flag == 'train':
    filename_queue = tf.compat.v1.train.string_input_producer(filenames)
  else:
    filename_queue = tf.compat.v1.train.string_input_producer(
      filenames, num_epochs=1, shuffle=False
    )

  reader = tf.compat.v1.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = tf.io.parse_example(
    serialized=serialized_example,
    features={
      'label': tf.io.FixedLenFeature([], tf.int64),
      'img_raw': tf.io.FixedLenFeature([], tf.string)
    }
  )
  # 
  image = tf.io.decode_raw(features['img_raw'], tf.uint8)
  image = tf.reshape(image, [28, 28])

  label = tf.cast(features['label'], tf.int32)

  if flag == 'train':
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    img_batch, label_batch = tf.compat.v1.train.batch(
      [image, label], batch_size=batch_size, capacity=20
    )

    return img_batch, label_batch
  return image, label

if __name__ == "__main__":
  directory = 'data/mnist/test'

  (filenames, labels), _ = load_sample(directory, shuffle=False)

  print(filenames[:2], labels[:2], _)

  target = 'data/mnist_test.tfrecords'

  # make_tfrec(filenames, labels, target)

  image, label = read_and_decode([target], flag='test')

  # save image to file system
  save_path = 'show/'

  if tf.io.gfile.exists(save_path):
    tf.io.gfile.rmtree(save_path)
  tf.io.gfile.makedirs(save_path)

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.local_variables_initializer())  # initialize

    coord = tf.train.Coordinator()  # 
    threads = tf.compat.v1.train.start_queue_runners(coord=coord)
    myset = set([])

    try:
      i = 0
      while True:
        example, examplelab = sess.run([image, label])
        examplelab = str(examplelab)

        if examplelab not in myset:
          myset.add(examplelab)
          tf.io.gfile.makedirs(join(save_path, examplelab))
        
        img = Image.fromarray(example)
        img.save(join(save_path, examplelab, f'{i}.jpg'))
        print(i)
        i += 1
    except tf.errors.OutOfRangeError:
      print('Done Test -- epoch limit reached')
    finally:
      coord.request_stop()
      coord.join(threads)

      print('Stopped')

