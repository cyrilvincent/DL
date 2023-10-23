import tensorflow as tf

input = tf.keras.layers.Input(shape=(30,))
x = tf.keras.layers.Dense(30, activation=tf.nn.relu)(input)
x = tf.keras.layers.Dense(30, activation="relu")(x)
x = tf.keras.layers.Dense(30)(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model(inputs=input, outputs=x)




