import os
from os.path import join, split, basename

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model

import random


class MyLayer(Layer):
  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)

  # define build
  def build(self, input_shape):
    shape = tf.TensorShape((input_shape[1], self.output_dim))
    # define trainable variables
    self.weight = self.add_weight(
      name='weight',
      shape=shape,
      initializer='uniform',
      trainable=True
    )
    super(MyLayer, self).build(input_shape)

  def call(self, inputs):
    return tf.matmul(inputs, self.weight)

  # 
  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.output_dim

    return tf.TensorShape(shape)

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    base_config['output_dim'] = self.output_dim

    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
  
  
if __name__ == "__main__":
  # train data
  x_train = np.linspace(0, 10, 100)
  y_train_random = -1 + 2 * np.random.random(*x_train.shape)
  y_train = 2 * x_train + y_train_random


  # test data
  x_test = np.linspace(10, 20, 10)
  y_test_random = -1 + 2 * np.random.random(*x_test.shape)
  y_test = 2 * x_test + y_test_random

  # predict
  x_predict = random.sample(range(20, 30), 10)

  # define the network
  inputs = Input(shape=(1, ))
  x = Dense(64, activation='relu')(inputs)
  x = MyLayer(64)(x)
  predictions = Dense(1)(x)

  # compile the model
  model = Model(inputs=inputs, outputs=predictions)
  model.compile(
    optimizer='rmsprop',
    loss='mse',
    metrics=['mae']
  )
  
  # train
  history = model.fit(
    x=x_train,
    y=y_train,
    epochs=100,
    batch_size=16
  )

  # test
  score = model.evaluate(
    x=x_test,
    y= y_test,
    batch_size=16
  )

  print('score: ', score)

  # predict
  y_predict = model.predict(x_predict)
  print('x_predict: ', x_predict)
  print('y_predict: ', y_predict)