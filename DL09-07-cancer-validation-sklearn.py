import tensorflow as tf
import numpy as np
tf.random.set_seed(1)
np.random.seed(1)

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2
X=cancer['data']
y=cancer['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8, test_size=0.2)



import tensorflow.keras as keras
model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X_train.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=10)
eval = model.evaluate(X_test, y_test)
print(eval)

