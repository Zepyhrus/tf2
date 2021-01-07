import tensorflow as tf



if __name__ == "__main__":
  gpu_device_name = tf.test.gpu_device_name()

  print(gpu_device_name)

  tf.test.is_gpu_available()



