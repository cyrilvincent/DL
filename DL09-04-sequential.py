import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(30,)),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(30),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(1)
  ])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu,input_shape=(30,)))
model.add(tf.keras.layers.Dense(30, activation="relu"))
model.add(tf.keras.layers.Dense(30))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dense(1))





