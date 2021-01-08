import tensorflow as tf
import tensorflow_datasets as tfds

tf.compat.v1.disable_v2_behavior()

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


def read_data(file_queue):
  reader = tf.compat.v1.TextLineReader(skip_header_lines=1)
  key, value = reader.read(file_queue)

  defaults = [[0], [0.], [0.], [0.], [0.], [0]]
  csv_column = tf.io.decode_csv(records=value, record_defaults=defaults)

  feature_column = [i for i in csv_column[1:-1]]
  label_column = csv_column[-1]

  return tf.stack(feature_column), label_column


def create_pipeline(filename, batch_size, num_epochs=None):
  file_queue = tf.compat.v1.train.string_input_producer([filename], num_epochs=num_epochs)

  feature, label = read_data(file_queue)

  min_after_dequeue = 1000  # keep at least 1000 data
  capacity = min_after_dequeue + batch_size

  feature_batch, label_batch = tf.compat.v1.train.shuffle_batch(
    [feature, label],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue
  )

  return feature_batch, label_batch



if __name__ == "__main__":
  x_train_batch, y_train_batch = create_pipeline('iris.csv', 32, num_epochs=100)
  x_test, y_test = create_pipeline('iris.csv', 32)

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())  # this is necessary

    coord = tf.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(coord=coord)

    try:
      while True:
        if coord.should_stop():
          break
        
        example, label = sess.run([x_train_batch, y_train_batch])

        print('training data: ', example)
        print('training label: ', label)
    except tf.errors.OutOfRangeError:
      print('Done reading')
      example, label = sess.run([x_test, y_test])

      print('test data: ', example)
      print('test label: ', label)
    except KeyboardInterrupt:
      print('Terminated by keyboard')
    finally:
      coord.request_stop()

    coord.join(threads)
    sess.close()
