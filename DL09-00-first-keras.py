import tensorflow.keras as keras
model = keras.Sequential()
model.add(keras.layers.Dense(5))
model.add(keras.layers.Dense(5))
model.add(keras.layers.Dense(1))


