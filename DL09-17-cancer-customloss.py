from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2
X=cancer['data']
y=cancer['target']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)



import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

class MeanSquaredError(tf.losses.Loss):
  def call(self, y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(math_ops.square(y_pred - y_true), axis=-1)

class MeanAbsoluteError(tf.losses.Loss):
  def call(self, y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(math_ops.abs(y_pred - y_true), axis=-1)

class MinFalseNegativeError(tf.losses.Loss):
  def call(self, y_true, y_pred):
      y_pred = ops.convert_to_tensor(y_pred)
      y_true = math_ops.cast(y_true, y_pred.dtype)
      res = math_ops.abs((y_pred * y_true) * 5 + y_pred - y_true)
      return K.mean(res, axis=-1)

model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])

sgd = keras.optimizers.SGD(nesterov=True, lr=1e-4)
model.compile(loss=MinFalseNegativeError(), optimizer=sgd,metrics=['accuracy'])
model.summary()

history = model.fit(X, y, epochs=200, batch_size=10, validation_split=0.2)
eval = model.evaluate(X, y)
print(eval)

