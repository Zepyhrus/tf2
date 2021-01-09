import os
from os.path import join, split, basename

import matplotlib.pyplot as plt

import random
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


def get_cost(x, y, W, b):
  # forward
  z = tf.cast(tf.multiply(np.asarray(x, dtype=np.float32), W) + b, dtype=tf.float32)

  cost = tf.reduce_mean( tf.square(y - z) )

  return cost


def moving_average(a, w=10):
  if len(a) < w:
    return a[:]
  
  return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


def get_one(dataset):
  iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

  one_element = iterator.get_next()

  return one_element


def load_sample(sample_dir, shuffleflag=True):
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

  if shuffleflag:
    lfilenames, labels = shuffle(lfilenames, labels)

  return (np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)


def show_result(subplot, title, thisimg):
  p = plt.subplot(subplot)
  p.axis('off')
  p.imshow(thisimg)
  p.set_title(title)

def show_img(index, label, img, ntop):
  plt.figure(figsize=(12, 6))
  plt.axis('off')
  ntop = min(ntop, 9)

  print(index)

  for i in range(ntop):
    show_result(100+10*ntop+1+i, label[i], img[i])
  plt.show()

