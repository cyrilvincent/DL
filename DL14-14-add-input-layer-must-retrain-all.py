import tensorflow as tf
import tensorflow.keras as keras

model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

input = keras.layers.Conv2D(64, (3, 3), input_shape=(448, 448, 3), activation="relu", padding="same")
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(input)
x = model.layers[0](x)
model.layers[0].input_shape=(224,224,64)
model = keras.models.Model(inputs=input, outputs=model.outputs)

# MUST RETRAIN ALL
model.summary()



