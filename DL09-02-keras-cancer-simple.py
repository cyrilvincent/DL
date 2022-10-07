import tensorflow.keras as keras
import tensorflow as tf
import pandas


tf.random.set_seed(1)

dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)
print(x.shape)

model = keras.Sequential([
    keras.layers.Dense(25, input_shape=(x.shape[1],)),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1)
  ])

model.compile(loss="mse", metrics="accuracy")
model.summary()

history = model.fit(x, y, epochs=100)

predicted = model.predict(x)
print(model.evaluate(x,y))

import numpy as np
print(predicted)
predicted = np.where(predicted > 0.5,1,0)
predicted = predicted.reshape(-1)

print(predicted)
errors = predicted - y
print(errors)
print(len(errors[errors == 0]) / len(y))


