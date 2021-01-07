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

  # plt.plot(train_X, train_Y, 'ro', label='Original data')
  # plt.legend()
  # plt.show()

  # 
  X = tf.compat.v1.placeholder('float')
  Y = tf.compat.v1.placeholder('float')

  # 
  W = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
  b = tf.Variable(tf.zeros([1]), name='bias')

  # forward
  z = tf.multiply(X, W) + b

  # backward
  global_step = tf.Variable(0, name='global_step', trainable=False)

  # optimization
  cost = tf.reduce_mean( tf.square(Y - z) )

  learning_rate = 0.01
  optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).\
    minimize(loss=cost, global_step=global_step)

  # initialize
  init = tf.compat.v1.global_variables_initializer()

  # learning parameters
  training_epochs = 20
  display_step = 2

  savedir = 'log/'
  saver = tf.compat.v1.train.Saver(
    tf.compat.v1.global_variables(),
    max_to_keep=1 # generate saver
  )

  plotdata = { 'batchsize': [], 'loss': [] }

  def moving_average(a, w=10):
    if len(a) < w:
      return a[:]
    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

  # Train model
  with tf.compat.v1.Session() as sess:
    sess.run(init)
    # kpt = tf.train.latest_checkpoint(savedir)
    # if kpt != None:
    #   saver.restore(sess, kpt)

    # 
    while global_step.eval() / len(train_X) < training_epochs:
      step = int( global_step.eval() / len(train_X) )
      for (x, y) in zip(train_X, train_Y):
        sess.run(optimizer, feed_dict={X: x, Y: y})

      # logging
      if step % display_step == 0:
        loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print ('Epoch:', step+1, 'cost=', loss, 'W=', sess.run(W), 'b=', sess.run(b))
        if not (loss == 'NA'):
          plotdata['batchsize'].append(global_step.eval())
          plotdata['loss'].append(loss)
        saver.save(sess, savedir+'linearmodel.ckpt', global_step=global_step)
    
    print(' Finished!')
    saver.save(sess, savedir+'linearmodel.ckpt', global_step=global_step)

    print('cost=', sess.run(cost, feed_dict={X: train_X, Y: train_Y}), 'W=', sess.run(W), 'b=', sess.run(b))

    # show model
    plt.figure(1)
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    
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


