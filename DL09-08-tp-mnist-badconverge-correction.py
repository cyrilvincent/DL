import tensorflow.keras as keras

import numpy as np

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

sample = np.random.randint(60000, size=5000)
data = x_train[sample]
target = y_train[sample]

model = keras.Sequential([
    keras.layers.Dense(500, activation=tf.nn.relu, input_shape=(x_train.shape[1],)),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.relu),
  ])

model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
print(model.summary())
trained = model.fit(x_train, y_train, epochs=100, batch_size=10)


