import tensorflow.keras as keras

model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

x = model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dense(30, activation="relu")(x)
x =  keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dense(3, activation="softmax")(x)

y = model.output
y = keras.layers.Flatten()(x)
y = keras.layers.Dense(50, activation="relu")(x)
y = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.models.Model(inputs=model.input, outputs=[x, y])