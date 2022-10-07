import tensorflow as tf
import tensorflow.keras as keras

input = keras.layers.Input(shape=(30,))
x = keras.layers.Dense(30, activation=tf.nn.relu)(input)
x = keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dense(30)(x)
x = keras.layers.Activation("relu")(x)
x = keras.layers.Dense(1)(x)

model = keras.models.Model(inputs=input, outputs=x)




