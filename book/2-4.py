import tensorflow as tf
import numpy as np

def model(xs):
  outputs = tf.multiply(xs, 0.1)

  return outputs

if __name__ == "__main__":
  arr_list = np.arange(0, 100).astype(np.float32)
  
  shape = arr_list.shape
  dataset = tf.data.Dataset.from_tensor_slices(arr_list)
  dataset_iterator = dataset.shuffle(shape[0]).batch(10)

  for it in dataset_iterator:
    logits = model(it)

    print(logits)




