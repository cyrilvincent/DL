import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.model_selection as ms

tf.random.set_seed(1)
np.random.seed(1)

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop(["diagnosis"], 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
  ])

model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))
eval = model.evaluate(X_test, y_test)
print(eval)

