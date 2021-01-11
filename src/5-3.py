import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.ops import lookup_ops

tf.compat.v1.disable_v2_behavior()


def test_categorical_cols_to_hash_bucket():
  tf.compat.v1.reset_default_graph()

  some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
    'sparse_feature', hash_bucket_size=5
  )

  builder = _LazyBuilder({
    'sparse_feature': [['a'], ['x']]
  })
  id_weight_pair = some_sparse_column._get_sparse_tensors(builder)

  with tf.compat.v1.Session() as sess:
    id_tensor_eval = id_weight_pair.id_tensor.eval()

    print('Sparse matrix: \n', id_tensor_eval)

    dense_decoded = tf.sparse.to_dense(
      id_tensor_eval, default_value=-1
    ).eval()
    print('Dense matricx: \n', dense_decoded)


def test_with_1d_sparse_tensor():
  tf.compat.v1.reset_default_graph()

  # mix
  body_style = tf.feature_column.categorical_column_with_vocabulary_list(
    'name', vocabulary_list=['anna', 'gary', 'bob'], num_oov_buckets=2
  )

  # dense matrix
  builder = _LazyBuilder({ 'name': ['gary', 'alsa', 'anna', 'alsa'] })

  # sparse matrix
  builder2 = _LazyBuilder({
    'name': tf.SparseTensor(
      indices = ((0,), (1,), (2,)),
      values = ('anna', 'gary', 'alsa'),
      dense_shape = (3,)
    )
  })

  id_weight_pair = body_style._get_sparse_tensors(builder)
  id_weight_pair2 = body_style._get_sparse_tensors(builder2)

  with tf.compat.v1.Session() as sess:
    sess.run(lookup_ops.tables_initializer())

    id_tensor_eval = id_weight_pair.id_tensor.eval()

    print('Sparse matrix 1: \n', id_tensor_eval)
    print('Sparse matrix 2: \n', id_weight_pair2.id_tensor.eval())

    dense_decoded = tf.sparse.to_dense(
      id_tensor_eval, default_value=-1
    ).eval()
    print('Dense matrix: \n', dense_decoded)


def test_categorical_cols_to_onehot():
  tf.compat.v1.reset_default_graph()

  some_sparse_column = tf.feature_column\
    .categorical_column_with_hash_bucket(
      'sparse_feature', hash_bucket_size=12
    )
  
  one_hot_style = tf.feature_column.indicator_column(
    some_sparse_column
  )

  features = {'sparse_feature': [['a'], ['a'], ['b'], ['c'], ['d'], ['e'], ['f']]}
  # input tensor
  net = tf.compat.v1.feature_column.input_layer(features, one_hot_style)

  with tf.compat.v1.Session() as sess:
    print(net.eval())


def test_categorical_cols_to_embedding():
  tf.compat.v1.reset_default_graph()

  some_sparse_column = tf.feature_column\
    .categorical_column_with_hash_bucket(
      'sparse_feature', hash_bucket_size=5
    )
  embedding_col = tf.feature_column.embedding_column(
    some_sparse_column, dimension=3
  )

  features = {
    'sparse_feature': [['a'], ['x']]
  }

  # get input tensor
  cols_to_vars = {}
  net = tf.compat.v1.feature_column.input_layer(
    features, embedding_col, cols_to_vars
  )

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    print(net.eval())


def test_order():
  tf.compat.v1.reset_default_graph()

  some_sparse_column = tf.feature_column\
    .categorical_column_with_hash_bucket(
      'asparse_feature', hash_bucket_size=5
    )
  
  numeric_col = tf.feature_column.numeric_column('numeric_col')
  embedding_col = tf.feature_column.embedding_column(
    some_sparse_column, dimension=3
  )
  one_hot_col = tf.feature_column.indicator_column(some_sparse_column)

  print(one_hot_col.name)
  print(embedding_col.name)
  print(numeric_col.name)

  features = {
    'numeric_col': [[3], [6]],
    'asparse_feature': [['a'], ['x']],
  }

  cols_to_vars = {}
  net = tf.compat.v1.feature_column.input_layer(
    features, [numeric_col, embedding_col, one_hot_col],
    cols_to_vars=cols_to_vars
  )

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(net.eval())


if __name__ == "__main__":
  test_categorical_cols_to_hash_bucket()

  test_with_1d_sparse_tensor()

  test_categorical_cols_to_onehot()

  test_categorical_cols_to_embedding()

  test_order()


