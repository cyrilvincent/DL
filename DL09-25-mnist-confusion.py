import tensorflow as tf
import sklearn.metrics

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

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.load_model("data/h5/mnist.h5")

predicted = model.predict(x_test)

import sklearn.metrics
print(sklearn.metrics.confusion_matrix(y_test.argmax(axis=1), predicted))
print(sklearn.metrics.classification_report(y_test.argmax(axis=1), predicted))
