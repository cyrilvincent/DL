import tensorflow as tf
import tensorflow.keras as keras

model = keras.Sequential()
model.add(keras.layers.Dense(30, activation=tf.nn.relu,input_shape=(30,)))
model.add(keras.layers.Dense(30, activation="relu"))
model.add(keras.layers.Dense(30))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(1))

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

model.summary()





