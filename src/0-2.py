import tensorflow_datasets as tfds


if __name__ == "__main__":
  print(tfds.list_builders())

  # load datasets 
  ds_train, ds_test = tfds.load(name='mnist', split=['train', 'test'])

  ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)

  for features in ds_train.take(1):
    image, label = features['image'], features['label']




