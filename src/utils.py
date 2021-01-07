import tensorflow as tf
import numpy as np




def get_cost(x, y, W, b):
  # forward
  z = tf.cast(tf.multiply(np.asarray(x, dtype=np.float32), W) + b, dtype=tf.float32)

  cost = tf.reduce_mean( tf.square(y - z) )

  return cost


def moving_average(a, w=10):
  if len(a) < w:
    return a[:]
  
  return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]
