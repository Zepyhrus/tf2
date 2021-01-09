import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


if __name__ == '__main__':
  dataset1 = tf.data.Dataset.from_tensor_slices( list(range(5)) )

  # NOTE: the following 2 methods need to be initialized 
  # while make_one_shot_iterator does not need
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

    print(sess.run(one_element1, {one_element1: 356}))

  
  dataset1 = tf.data.Dataset.from_tensor_slices( list(range(5)) )
  iterator1 = tf.compat.v1.data.make_one_shot_iterator(dataset1)


  dataset2 = tf.data.Dataset.from_tensor_slices( list(range(10, 20)) )
  iterator2 = tf.compat.v1.data.make_one_shot_iterator(dataset2)

  with tf.compat.v1.Session() as sess2:
    iterator_handle = sess2.run(iterator1.string_handle())
    iterator_handle2 = sess2.run(iterator2.string_handle())

    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iterator3 = tf.compat.v1.data.Iterator.from_string_handle(handle, iterator1.output_types)

    one_element3 = iterator3.get_next()
    print(sess2.run(one_element3, {handle: iterator_handle}))
    print(sess2.run(one_element3, {handle: iterator_handle2}))







