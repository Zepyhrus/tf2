import os

import tensorflow as tf


import numpy as np
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True

tf.compat.v1.disable_v2_behavior()


def generate_data(batch_size=100):
  train_X = np.linspace(-batch_size/100, batch_size/100, batch_size)
  train_Y = 2 * train_X + np.random.randn(*train_X.shape)

  yield train_X, train_Y


if __name__ == "__main__":
  X_input = tf.compat.v1.placeholder('float', (None))
  Y_input = tf.compat.v1.placeholder('float', (None))

  # 
  training_epochs = 20

  with tf.compat.v1.Session() as sess:
    for epoch in range(training_epochs):
      for x, y in generate_data():
        xv, yv = sess.run([X_input, Y_input], feed_dict={X_input: x, Y_input: y})
        print(epoch, '| x.shape:', np.shape(xv), '| x[:3]: ', xv[:3])
        print(epoch, '| y.shape:', np.shape(yv), '| y[:3]: ', yv[:3])

  train_data = list(generate_data())[0]

  plt.plot(train_data[0], train_data[1], 'ro', label='Original data')
  plt.legend()
  plt.show()
