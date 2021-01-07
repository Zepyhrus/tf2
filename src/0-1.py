import tensorflow as tf


@tf.function
def autograph(input_data):
  if tf.reduce_mean(input_data) > 0:
    return input_data
  else:
    return input_data // 2

if __name__ == "__main__":
  a = autograph(tf.constant([-6, 4]))
  b = autograph(tf.constant([6, -4]))

  print(a.numpy(), b.numpy())