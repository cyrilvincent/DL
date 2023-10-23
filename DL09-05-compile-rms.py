import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu,input_shape=(30,)))
model.add(tf.keras.layers.Dense(30, activation="relu"))
model.add(tf.keras.layers.Dense(30))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

model.summary()





