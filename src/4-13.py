import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


if __name__ == '__main__':
  dataset1 = tf.data.Dataset.from_tensor_slices( list(range(5)) )

  iterator1 = tf.compat.v1.data.Iterator.from_structure(
    tf.compat.v1.data.get_output_types(dataset1),
    tf.compat.v1.data.get_output_shapes(dataset1)
  )

  # iterator1 = tf.compat.v1.data.make_initializable_iterator(dataset1)

  one_element1 = iterator1.get_next()

  with tf.compat.v1.Session() as sess:
    sess.run( iterator1.make_initializer(dataset1) )

    for ii in range(2):
      while True:
        try:
          print(sess.run(one_element1))
        except tf.errors.OutOfRangeError:
          print('Done!!!')
          sess.run( iterator1.make_initializer(dataset1) )
          break






