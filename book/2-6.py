import tensorflow as tf
import numpy as np

if __name__ == "__main__":
  input_xs = np.random.rand(1000).astype(np.float32)
  input_ys = 3 * input_xs + 0.217 + np.random.randn(*input_xs.shape) * 0.1
  input_ys = input_ys.astype(np.float32)

  print(input_xs[:3], input_ys[:3])

  # model
  weight = tf.Variable(1., dtype=tf.float32, name='weight')
  bias = tf.Variable(1., dtype=tf.float32, name='bias')

  def model(xs):
    logits = tf.multiply(xs, weight) + bias

    return logits

  # optimizer
  opt = tf.optimizers.Adam(1e-1)

  
  # loss
  loss = tf.losses.MeanSquaredError()
  
  # dataset
  dataset = tf.data.Dataset.from_tensor_slices((input_xs, input_ys))
  dataset = dataset.shuffle(buffer_size=1000).batch(256).repeat(20)

  # training
  for xs, ys in dataset:
    with tf.GradientTape() as tape:
      _loss = loss(model(xs), ys)
    
    grads = tape.gradient(_loss, [weight, bias])
    opt.apply_gradients(zip(grads, [weight, bias]))
    print('Training loss is:', _loss.numpy())

  print(weight.numpy(), bias.numpy())


  




