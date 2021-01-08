import os
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


if __name__ == "__main__":
  directory = 'data/mnist/'

  (filenames, labels), _ = load_sample(directory, shuffle=False)

  print(filenames[:2], labels[:2], _)

  target = 'data/mnist.tfrecords'

  make_tfrec(filenames, labels, target)




