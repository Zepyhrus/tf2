import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

Dataset = tf.data.Dataset


def parse_fn(x):
  print(x)

  return x


def get_one(dataset):
  iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

  one_element = iterator.get_next()

  return one_element


def showone(one_element, datasetname):
  print('{0:-^50}'.format(datasetname))

  for i in range(4):
    datav = sess.run(one_element)
    print(datav)





if __name__ == "__main__":
  ds = Dataset.list_files('testset/*.txt', shuffle=True).\
    interleave(
      lambda x: tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
      cycle_length=2,
      block_length=2
    )

  ones = get_one(ds)

  with tf.compat.v1.Session() as sess:
    showone(ones, 'dataset1')

  