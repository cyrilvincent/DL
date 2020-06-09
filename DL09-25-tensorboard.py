from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2
X=cancer['data']
y=cancer['target']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)



import tensorflow as tf
import tensorflow.keras as keras

model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X.shape[1],)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])

import datetime
logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.compile(loss="binary_crossentropy",metrics=['accuracy'])
model.summary()

history = model.fit(X, y, epochs=20, batch_size=10, validation_split=0.2,callbacks=[tensorboard_callback])
model.save("data/h5/cancer.h5")

