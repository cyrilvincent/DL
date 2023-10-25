import tensorflow.keras as keras

model = keras.models.load_model('data/dogsvscats/vgg16model-small.h5')

for layer in model.layers[:-1]:
    layer.trainable = False

model.add(keras.layers.Dense(3))
model.add(keras.layers.Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])

trainset = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

batchSize = 16

trainGenerator = trainset.flow_from_directory(
        'data/dogsvscats/small/train',
        target_size=(224, 224),
        subset='training',
        class_mode="categorical",
        batch_size=batchSize)

validationGenerator = trainset.flow_from_directory(
        'data/dogsvscats/small/train',
        target_size=(224, 224),
        class_mode="categorical",
        subset = 'validation',
        batch_size=batchSize)


model.fit(
        trainGenerator,
        epochs=30,
        validation_data=validationGenerator,
)

model.save('data/dogsvscats/vgg16model-cows.h5')


