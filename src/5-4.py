import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.ops import lookup_ops

tf.compat.v1.disable_v2_behavior()


def test_crossed():
  a = tf.feature_column.numeric_column('a', dtype=tf.int32, shape=(2,))
  # convert sparse value
  b = tf.feature_column.bucketized_column(a, boundaries=(0, 1))

  crossed = tf.feature_column.crossed_column([b, 'c'], hash_bucket_size=5)

  builder = _LazyBuilder({
    'a': tf.constant(((-1., -1.5), (.5, 1.))),
    'c': tf.SparseTensor(
      indices=((0, 0), (1, 0), (1, 1)),
      values=['cA', 'cB', 'cC'],
      dense_shape=(2, 2)
    )
  })
  id_weight_pair = crossed._get_sparse_tensors(builder)

  with tf.compat.v1.Session() as sess2:
    id_tensor_eval = id_weight_pair.id_tensor.eval()
    print(id_tensor_eval)

    dense_decoded = tf.sparse.to_dense(
      id_tensor_eval, default_value=-1
    ).eval()
    print(dense_decoded)


if __name__ == "__main__":
  test_crossed()

