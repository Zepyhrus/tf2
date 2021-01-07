import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True


if __name__ == "__main__":
  print(tf.__version__)

  tf.compat.v1.disable_v2_behavior()

  # 
  train_X = np.linspace(-1, 1, 100)
  train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

  plt.plot(train_X, train_Y, 'ro', label='Original data')
  plt.legend()
  plt.show()

  # 
  X = tf.compat.v1.placeholder('float')
  Y = tf.compat.v1.placeholder('float')

  # 
  W = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
  b = tf.Variable(tf.zeros([1]), name='bias')

  # forward
  z = tf.multiply(X, W) + b