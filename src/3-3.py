import tensorflow as tf
import numpy as np

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# generate simulation data in memory
def generate_data(datasize=100):
  train_X = np.linspace(-1, 1, datasize)
  train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

  return train_X, train_Y


def train_input_fn(train_data, batch_size):
  # generate training dataset
  dataset = tf.data.Dataset.from_tensor_slices(
    (train_data[0], train_data[1])
  )
  dataset = dataset.shuffle(1000).repeat().batch(batch_size)

  return dataset


def eval_input_fn(data, labels, batch_size):
  # generate evaluation dataset
  assert batch_size, 'batch_size must not be None'

  inputs = (data, labels) if labels else data

  dataset = tf.data.Dataset.from_tensor_slices(inputs)
  dataset = dataset.batch(batch_size)

  return dataset


def my_model(features, labels, mode, params):
  # define model
  W = tf.Variable(tf.random.normal([1]), name='weight')
  b = tf.Variable(tf.zeros([1]), name='bias')

  # 
  predictions = tf.multiply(tf.cast(features, dtype=tf.float32), W) + b

  # prediction
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  loss = tf.compat.v1.losses.mean_squared_error(labels=labels, predictions=predictions)

  meanloss = tf.compat.v1.metrics.mean(loss)
  metrics = {'meanloss': meanloss}

  # evaluation
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

  # train
  assert mode == tf.estimator.ModeKeys.TRAIN
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['learning_rate'])
  train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_or_create_global_step())

  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
  train_data = generate_data()
  test_data = generate_data(20)

  batch_size = 10

  tf.compat.v1.reset_default_graph()  # clear all tensors in the graph
  

  # train model
  gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.667)
  session_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

  estimator = tf.estimator.Estimator(
    model_fn=my_model,
    model_dir='./myestimatormode',
    params={'learning_rate': 0.01},
    config=tf.estimator.RunConfig(session_config=session_config)
  )

  estimator.train(
    lambda: train_input_fn(train_data=train_data, batch_size=batch_size),
    steps=200
  )

  tf.compat.v1.logging.info('Finished!')

  # warm start
  warm_start_from = tf.estimator.WarmStartSettings(
    ckpt_to_initialize_from='./myestimatormode',
  )

  estimator2 = tf.estimator.Estimator(
    model_fn=my_model,
    model_dir='./myestimatormode2',
    warm_start_from=warm_start_from,
    params={'learning_rate': 0.1},
    config=tf.estimator.RunConfig(session_config=session_config)
  )

  estimator2.train(
    lambda: train_input_fn(train_data=train_data, batch_size=batch_size),
    steps=200
  )

  # evaluation on test dataset
  test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    test_data[0], test_data[1], batch_size=1, shuffle=True
  )

  train_metrics = estimator2.evaluate(input_fn=test_input_fn)
  print('train_metrics', train_metrics)

  # predict
  predictions = estimator2.predict(
    input_fn=lambda: eval_input_fn(test_data[0], None, batch_size)
  )
  print('predictions', list(predictions))

  # define input
  new_samples = np.array( [6.4, 3.2, 4.5, 1.5], dtype=np.float32 )
  predict_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    new_samples, num_epochs=1, batch_size=1, shuffle=False
  )
  predictions = list(estimator2.predict(input_fn=predict_input_fn))
  print(f'input, res: {new_samples} {predictions}')
  





