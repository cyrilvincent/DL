from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2
X=cancer['data']
y=cancer['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import keras
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

class MeanSquaredError(tf.losses.Loss):
  def call(self, y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(math_ops.square(y_pred - y_true), axis=-1)


model = keras.Sequential([
    keras.layers.Dense(30, activation="relu",
                       input_shape=(X_train.shape[1],)),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
  ])

# model = keras.Sequential()
# model.add(keras.layers.Dense(30, activation=tf.nn.relu,
#                        input_shape=(X_train.shape[1],)))
# model.add(keras.layers.Dense(30, activation=tf.nn.relu))
# model.add(keras.layers.Dense(30, activation=tf.nn.relu))
# model.add(keras.layers.Dense(30, activation=tf.nn.relu))
# model.add(keras.layers.Dense(1))

#model.compile(loss="mse", optimizer="sgd")
#sgd = keras.optimizers.SGD(nesterov=True, lr=1e-5)
model.compile(loss=MeanSquaredError(), optimizer="adam")
model.summary()

history = model.fit(X_train, y_train, epochs=2000, class_weight={0: 1, 1: 5}) # Apporte un facteur 5 aux positifs
eval = model.evaluate(X_test, y_test)
print(eval)
model.save("cancer.h5")
