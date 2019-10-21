import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd

dataset = pd.read_csv("house/house.csv")
train_labels = dataset["loyer"].values
del dataset["loyer"]
train_data = dataset.values


model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])

model.compile(loss='mse',
              optimizer=tf.keras.optimizers.RMSprop(0.001),
              metrics=["mae"])

model.summary()

history = model.fit(train_data, train_labels, epochs=100)
result = model.evaluate(train_data, train_labels)
print(result)
