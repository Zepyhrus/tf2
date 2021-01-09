import tensorflow as tf
import numpy as np

tf.compat.v1.disable_v2_behavior()

def generate_data(data_size=100):
  train_X = np.linspace(-data_size/100, data_size/100, data_size)
  train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

  return train_X, train_Y


def get_one(dataset):
  iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

  one_element = iterator.get_next()

  return one_element


if __name__ == "__main__":
  X, Y = generate_data()


  dataset1 = tf.data.Dataset.from_tensor_slices( (X, Y) )

  dataset2 = tf.data.Dataset.from_tensor_slices({
    'x': X,
    'y': Y
  })

  batch_size = 10
  dataset3 = dataset1.repeat().batch(batch_size=batch_size)

  dataset4 = dataset2.map(
    lambda data: (data['x'], tf.cast(data['y'], tf.int32))
  )

  dataset5 = dataset1.shuffle(100)

  o1 = get_one(dataset1)
  o2 = get_one(dataset2)
  o3 = get_one(dataset3)
  o4 = get_one(dataset4)
  o5 = get_one(dataset5)

  # a simple demo 
  one_element = tf.compat.v1.data.make_one_shot_iterator(dataset1).get_next()

  with tf.compat.v1.Session() as sess:
    while True:
      try:
        _x, _y = sess.run(one_element)
      except KeyboardInterrupt:
        print('Keyboard interrupt')
        break
      except tf.errors.OutOfRangeError:
        print('Finished!')
        break


