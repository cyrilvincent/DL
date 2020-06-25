import tensorflow.keras as keras
import tensorflow as tf

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
x_train = x_train[sample]
y_train = y_train[sample]

model = keras.Sequential([
    keras.layers.Dense(600, input_shape=(x_train.shape[1],)),
    keras.layers.Dense(400),
    keras.layers.Dense(200),
    keras.layers.Dense(100),
    keras.layers.Dense(10, activation=tf.nn.sigmoid),
  ])

model.compile(loss="categorical_crossentropy", metrics=['accuracy'])
trained = model.fit(x_train, y_train, epochs=5, batch_size=10,validation_data=(x_test, y_test))
print(model.summary())

predicted = model.predict(x_test)
import matplotlib.pyplot as plt
# Gestion des erreurs
# on récupère les données mal prédites
predicted = predicted.argmax(axis=1)

import sklearn.metrics
print(sklearn.metrics.confusion_matrix(y_test.argmax(axis=1),predicted))
print(sklearn.metrics.classification_report(y_test.argmax(axis=1),predicted))
