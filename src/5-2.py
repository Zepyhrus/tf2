import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

def test_numeric_cols_to_bucketized():
  price = tf.feature_column.numeric_column('price')

  price_bucketized = tf.feature_column.bucketized_column(
    price, boundaries=[3., 5.]
  )

  features = {
    'price': [[2.], [6.]]
  }

  net = tf.compat.v1.feature_column.input_layer(
    features, [price, price_bucketized]
  )

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(net.eval())


def test_numeric_cols_to_identity():
  tf.compat.v1.reset_default_graph()

  price = tf.feature_column.numeric_column('price')

  categorical_column = tf.feature_column.categorical_column_with_identity(
    'price', num_buckets=6
  )
  one_hot_style = tf.feature_column.indicator_column(categorical_column)
  features = {'price': [[2], [4]]}
  net = tf.compat.v1.feature_column.input_layer(features, [price, one_hot_style])

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    print(net.eval())


if __name__ == "__main__":
  test_numeric_cols_to_bucketized()

  test_numeric_cols_to_identity()
