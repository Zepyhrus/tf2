import os
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm

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

if __name__ == "__main__":
  directory = 'data/mnist/'



