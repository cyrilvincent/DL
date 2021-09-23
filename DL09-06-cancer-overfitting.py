import tensorflow.keras as keras
import tensorflow as tf
import pandas


tf.random.set_seed(1)

dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)

model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(x.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
model.summary()

hist = model.fit(x, y, epochs=200, batch_size=10)
eval = model.evaluate(x, y)
print(eval)

import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.plot([None] + hist.history['accuracy'], 'o-')
ax.legend(['Train accuracy'], loc = 0)
ax.set_title('Accuracy per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
plt.show()

