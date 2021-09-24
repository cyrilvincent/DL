import tensorflow.keras as keras

model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

for layer in model.layers:
    layer.trainable = False

x = model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(10, activation="softmax")(x)

model = keras.models.Model(inputs=model.input, outputs=x)

model.summary()

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

trainset = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

batchSize = 8

trainGenerator = trainset.flow_from_directory(
        'data/state-farm-distracted-driver-detection/train-224',
        target_size=(224, 224),
        subset = 'training',
        class_mode="categorical",
        batch_size=batchSize)

validationGenerator = trainset.flow_from_directory(
        'data/state-farm-distracted-driver-detection/train-224',
        target_size=(224, 224),
        class_mode="categorical",
        subset = 'validation',
        batch_size=batchSize)

model.fit(
        trainGenerator,
        epochs=30,
        validation_data=validationGenerator,
)

model.save('data/state-farm-distracted-driver-detection/vgg16model.h5')




