import tensorflow.keras as keras

model1 = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
model2 = keras.applications.resnet_v2.ResNet152V2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
model = keras.layers.Concatenate([model1, model2], axis=1)

x = model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dense(30, activation="relu")(x)
x =  keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dense(3, activation="softmax")(x)

model = keras.models.Model(inputs=(model1.input, model2.input), outputs=x)