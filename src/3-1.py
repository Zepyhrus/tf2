import tensorflow as tf
from tensorflow.python.ops import variable_scope

import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
  # generate simulate data
  train_X = np.linspace(-1, 1, 100)
  train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3


  # get dataset
  dataset = tf.data.Dataset.from_tensor_slices(
    (
      np.reshape(train_X, [-1, 1]),
      np.reshape(train_Y, [-1, 1])
    )
  )
  dataset = dataset.repeat().batch(5)

  # preparation
  global_step = tf.compat.v1.train.get_or_create_global_step()

  container = variable_scope.EagerVariableStore()

  learning_rate = 0.01
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

  def get_cost(x, y):
    # forward
    with container.as_default():
      z = tf.compat.v1.layers.dense(x, 1, name='l1')
    
    cost = tf.reduce_mean( input_tensor=tf.square(y - z) )

    return cost

  def grad(inputs, targets):
    with tf.GradientTape() as tape:
      loss_value = get_cost(inputs, targets)

    return tape.gradient(loss_value, container.trainable_variables())

  
  training_steps = 2000  # steps should change according to batchsize
  display_step = 100

  for step, value in enumerate(dataset):
    grads = grad(value[0], value[1])

    optimizer.apply_gradients(
      zip(grads, container.trainable_variables()),
      global_step=global_step
    )

    if step >= training_steps:
      break

    if (step+1) % display_step == 0:
      cost = get_cost(value[0], value[1])

      print('step:', step, 'cost=', cost.numpy())
  
  print('Finished!')
  print('cost=', cost.numpy())

  for i in container.trainable_variables():
    print(i.name, i.numpy())
  
