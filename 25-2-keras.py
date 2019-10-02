import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd

print(tf.VERSION)
print(keras.__version__)

dataset = pd.read_csv("house/house.csv")
print(dataset.shape) # 542 * 2
print(dataset.head())
train_labels = dataset["loyer"].values
del dataset["loyer"]
train_data = dataset.values
print(train_data[:10])
print(train_labels[:10])

# Normalize
mean = train_data.mean(axis=0) # Moyenne
print(mean)
std = train_data.std(axis=0) # Déviation standard std = sqrt(mean(abs(x - x.mean())**2)).
print(std)
train_data = (train_data - mean) / std
print(train_data[0])


# Model
model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

model.compile(loss='mse',
                optimizer=tf.train.RMSPropOptimizer(0.001),
                metrics=['mae'])

model.summary()

history = model.fit(train_data, train_labels, epochs=500,
                    validation_split=0.2)

import matplotlib.pyplot as plt
import numpy as np

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.show()

plot_history(history)

# Predict
# Normalement sur test_data normalisées
[loss, mae] = model.evaluate(train_data, train_labels)

print(mae)
