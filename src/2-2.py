import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import moving_average, get_cost



if __name__ == "__main__":
  # 
  train_X = np.linspace(-1, 1, 100)
  train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

  W = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='weight')
  b = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='bias')

  global_step = tf.compat.v1.train.get_or_create_global_step()

  learning_rate = 0.01

  # define optimizer
  optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)


  savedir = 'logeager/'

  saver = tf.compat.v1.train.Saver([W, b], max_to_keep=1) # saver

  kpt = tf.train.latest_checkpoint(savedir)
  if kpt != None:
    saver.restore(None, kpt)

  training_epochs = 5
  display_step = 10

  plotdata = { 'batchsize': [], 'loss': [] }

  # using GradientTape to track automatic differentiation
  while global_step / len(train_X) < training_epochs:
    step = int(global_step / len(train_X))

    with tf.GradientTape() as tape:
      cost_ = get_cost(train_X, train_Y, W, b)
    
    gradients = tape.gradient(target=cost_, sources=[W, b])

    optimizer.apply_gradients(zip(gradients, [W, b]), global_step)

    # logging
    if global_step % display_step == 0:
      print ('Epoch:', step, 'global step:', global_step.numpy(), 'cost=', cost_.numpy(), 'W=', W.numpy(), 'b=', b.numpy())
      
      plotdata['batchsize'].append(global_step.numpy())
      plotdata['loss'].append(cost_.numpy())
      
      saver.save(None, savedir+'linearmodel.ckpt', global_step=global_step)
    
  print('Finished!')
  saver.save(None, savedir+'linearmodel.ckpt', global_step=global_step)

  # show model
  plt.figure(1)
  plt.plot(train_X, train_Y, 'ro', label='Original data')
  plt.plot(train_X, W * train_X + b, label='Fitted line')
  
  plt.legend()
  
  plotdata['avgloss'] = moving_average(plotdata['loss'])
  plt.figure(2)
  plt.subplot(211)
  plt.plot(plotdata['batchsize'], plotdata['avgloss'], 'b--')
  plt.title('Minibatch run vs.Training loss')
  
  plt.subplot(212)
  plt.plot(plotdata['batchsize'], plotdata['loss'])

  plt.xlabel('Minibatch number')
  plt.ylabel('Loss')

  plt.show()






  




