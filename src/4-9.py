import tensorflow as tf
from utils import get_one

tf.compat.v1.disable_v2_behavior()


if __name__ == "__main__":
  dataset1 = tf.data.Dataset.from_tensor_slices( list(range(5)) )
  dataset1 = dataset1.shuffle(buffer_size=10).repeat()

  one_element1 = get_one(dataset1)

  with tf.compat.v1.Session() as sess:
    for i in range(20):
      print(sess.run(one_element1))







