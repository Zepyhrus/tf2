import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


def test_one_column():
  price = tf.feature_column.numeric_column('price')

  features = {'price': [[1.], [5.]]}
  net = tf.compat.v1.feature_column.input_layer(features, [price])

  with tf.compat.v1.Session() as sess:
    tt = sess.run(net)
    print(tt)


def test_reshaping():
  tf.compat.v1.reset_default_graph()
  price = tf.feature_column.numeric_column('price', shape=[1, 2])

  features = {'price': [[[1., 2.]], [[5., 6.]]]}
  features1 = {'price': [[3., 4.], [7., 8.]]}

  net = tf.compat.v1.feature_column.input_layer(features, price)
  net1 = tf.compat.v1.feature_column.input_layer(features1, price)

  with tf.compat.v1.Session() as sess:
    print(net.eval())
    print(net1.eval())


def test_column_order():
  tf.compat.v1.reset_default_graph()

  price_a = tf.feature_column.numeric_column('price_a')
  price_b = tf.feature_column.numeric_column('price_b')
  price_c = tf.feature_column.numeric_column('price_c')

  features = {
    'price_a': [[1.]],
    'price_c': [[4.]],
    'price_b': [[3.]],
  }

  net = tf.compat.v1.feature_column.input_layer(features, [price_c, price_a, price_b])

  with tf.compat.v1.Session() as sess:
    print(net.eval())



if __name__ == "__main__":
  test_one_column()

  test_reshaping()

  test_column_order()






