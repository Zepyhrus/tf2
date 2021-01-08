import os
from os.path import join, split, basename

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True

from PIL import Image


if __name__ == "__main__":
  train_images = 'data/train-images-idx3-ubyte'
  train_labels = 'data/train-labels-idx1-ubyte'
  test_images = 'data/t10k-images-idx3-ubyte'
  test_labels = 'data/t10k-labels-idx1-ubyte'


  with open(test_images, 'rb') as f:
    bi_images = f.read()

  with open(test_labels, 'rb') as f:
    bi_labels = f.read()
  
  total = 0
  for i in range(10000):
    st = 16 + i * 28 * 28
    ed = 16 + (i + 1) * 28 * 28
    
    img = np.frombuffer(bi_images[st:ed], dtype=np.uint8).reshape((28, 28))
    lab = bi_labels[8 + i]

    fd = f'data/mnist/test/{lab}'
    fl = f'{total}.bmp'

    if not os.path.exists(fd):
      os.makedirs(fd)

    Image.fromarray(img).save(join(fd, fl))
    total += 1

  plt.imshow(img)
  plt.show()

