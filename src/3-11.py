import numpy as np
import random
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model




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
  x = Dense(64, activation='relu')(x)
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

