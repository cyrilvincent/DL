import tensorflow.keras as keras
model = keras.Sequential()
model.add(keras.layers.Dense(3))
model.add(keras.layers.Dense(5))
model.add(keras.layers.Dense(2))

model = keras.Sequential()
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dense(60))
model.add(keras.layers.Dense(20))
model.add(keras.layers.Dense(5))
model.add(keras.layers.Dense(1))


