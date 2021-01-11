import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


def _get_initializer(embedding_dimension, embedding_values):
  def _initializer(shape, dtype, partition_info):
    return embedding_values
  return _initializer


if __name__ == "__main__":
  tf.compat.v1.reset_default_graph()

  vocabulary_size = 3
  sparse_input_a = tf.SparseTensor(
    indices=((0, 0), (1, 0), (1, 1)),
    values=(2, 0, 1),
    dense_shape=(2, 2)
  )

  sparse_input_b = tf.SparseTensor(
    indices=((0, 0), (1, 0), (1, 1)),
    values=(1, 2, 0),
    dense_shape=(2, 2)
  )

  embedding_dimension_a = 2
  embedding_values_a = (
    (1., 2.), # id 0
    (3., 4.), # id 1
    (5., 6.), # id 2
  )

  embedding_dimension_b = 3
  embedding_values_b = (
    (11., 12., 13.),  # id 0
    (14., 15., 16.),  # id 1
    (17., 18., 19.),  # id 2
  )

  categorical_column_a = tf.feature_column\
    .sequence_categorical_column_with_identity(
      key='a', num_buckets=vocabulary_size
    )
  embedding_column_a = tf.feature_column.embedding_column(
    categorical_column=categorical_column_a,
    dimension=embedding_dimension_a,
    initializer=_get_initializer(embedding_dimension_a, embedding_values_a)
  )

  categorical_column_b = tf.feature_column\
    .sequence_categorical_column_with_identity(
      key='b', num_buckets=vocabulary_size
    )
  embedding_column_b = tf.feature_column.embedding_column(
    categorical_column=categorical_column_b,
    dimension=embedding_dimension_b,
    initializer=_get_initializer(embedding_dimension_b, embedding_values_b)
  )

  # share column
  shared_embedding_columns = tf.feature_column.shared_embeddings(
    [categorical_column_b, categorical_column_a],
    dimension=embedding_dimension_a,
    initializer=_get_initializer(embedding_dimension_a, embedding_values_a)
  )

  # features
  features = {
    'a': sparse_input_a,
    'b': sparse_input_b,
  }
  sequence_feature_layer = tf.keras.experimental.SequenceFeatures(
    feature_columns=[embedding_column_b, embedding_column_a]
  )
  input_layer, sequence_length = sequence_feature_layer(features)
  
  sequence_feature_layer2 = tf.keras.experimental.SequenceFeatures(
    feature_columns=shared_embedding_columns
  )
  input_layer2, sequence_length2 = sequence_feature_layer2(features)

  # return tensor
  global_vars = tf.compat.v1.get_collection(
    tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
  )
  print([v.name for v in global_vars])


  with tf.compat.v1.train.MonitoredSession() as sess:
    print(global_vars[0].eval(session=sess))
    print(global_vars[1].eval(session=sess))
    print(global_vars[2].eval(session=sess))

    print(sequence_length.eval(session=sess))
    print(input_layer.eval(session=sess))
    print(sequence_length2.eval(session=sess))
    print(input_layer2.eval(session=sess))


