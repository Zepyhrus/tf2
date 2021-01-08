import tensorflow_datasets as tfds

def get_iris_data():
  ds_train, *_ = tfds.load(name='iris', split=['train'])


  with open('iris.csv', 'w') as f:
    for i, ds in enumerate(ds_train):
      features = ds['features'].numpy()
      label = ds['label'].numpy()

      msg = f'{i},'
      for feature in features:
        msg += f'{feature:.1f},'
      msg += f'{label}\n'

      f.write(msg)

if __name__ == "__main__":
  pass
