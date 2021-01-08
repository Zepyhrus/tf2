import os
from os.path import join, split, basename

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_v2_behavior()

def generate_data(datasize=100):
  train_X = np.linspace(-1, 1, datasize)
  train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

  return train_X, train_Y


if __name__ == "__main__":
  train_data =  generate_data()
  batch_size = 10

  def train_input_fn(train_data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(
      (train_data[0], train_data[1])
    )
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset
  
  plotdata = { 'batchsize': [], 'loss': [] }
  
  def moving_average(a, w=10):
    if len(a) < w:
      return a[:]
    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

  tf.compat.v1.reset_default_graph()

  features = tf.compat.v1.placeholder('float', [None])
  labels = tf.compat.v1.placeholder('float', [None])

  W = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
  b = tf.Variable(tf.zeros([1]), name='bias')
  predictions = tf.multiply(tf.cast(features, dtype=tf.float32), W) + b
  loss = tf.compat.v1.losses.mean_squared_error(
    labels=labels, predictions=predictions
  )

  global_step = tf.compat.v1.train.get_or_create_global_step()

  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
  train_op = optimizer.minimize(loss, global_step=global_step)

  saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1)

  # training parameters
  training_epochs = 5000
  display_step = 100

  dataset = train_input_fn(train_data, batch_size)
  one_element = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()  # get input

  with tf.compat.v1.Session() as sess:
    # restore
    savedir = 'myestimatormode/'
    kpt = tf.train.latest_checkpoint(savedir)
    print('kpt:', kpt)
    saver.restore(sess, kpt)

    # continue training
    while global_step.eval() < training_epochs:
      step = global_step.eval()
      
      x, y = sess.run(one_element)

      vloss, _ = sess.run([loss, train_op], feed_dict={features: x, labels: y})

      if step % display_step == 0:
        print('Epoch: ', step+1, 'cost = ', vloss)
        if vloss:
          plotdata['batchsize'].append(step)
          plotdata['loss'].append(vloss)
        
        saver.save(sess, join(savedir, 'linearmodel.ckpt'), global_step=global_step)
    print('Finished!')
    print('cost = ', vloss)

    saver.save(sess, join(savedir, 'linearmodel.ckpt'), global_step=global_step)

  plt.figure(1)
  plt.plot(plotdata['batchsize'], plotdata['loss'], 'b--')
  plt.xlabel('Minibatch number')
  plt.ylabel('loss')
  plt.title('Minibatch run vs. Training loss')
  plt.show()








