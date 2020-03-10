import tensorflow.keras as keras
import tensorflow.compat.v1 as tf

import numpy as np

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']


# Set numeric type to float32 from uint8
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255


x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


sample = np.random.randint(60000, size=5000)
data = x_train[sample]
target = y_train[sample]


model = keras.Sequential([
    keras.layers.Dense(500, activation=tf.nn.relu, input_shape=(x_train.shape[1],)),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
  ])

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
trained = model.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_test, y_test))
print(model.summary())

