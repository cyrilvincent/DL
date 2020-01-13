import tensorflow.keras as keras

trainset = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validationset = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
batchSize = 16
nbSample = 287
validationSize = 260

trainGenerator = trainset.flow_from_directory(
        r'D:\CVC\ATP\Mesulog\191219_baseImages_tester\_BaseReference',
        target_size=(64, 64),
        color_mode="grayscale",
        batch_size=batchSize)

validationGenerator = validationset.flow_from_directory(
        r'D:\CVC\ATP\Mesulog\191219_baseImages_tester\validation\cyril',
        target_size=(64, 64),
        color_mode="grayscale",
        batch_size=batchSize)

#CNN
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(64, 64, 1)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 32, 32, 32

model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 16, 16, 32

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 8, 8, 64

#Dense
model.add(keras.layers.Flatten())
# 4096
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(6))
model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
        trainGenerator,
        steps_per_epoch=nbSample // batchSize,
        epochs=30,
        validation_data=validationGenerator,
        validation_steps=validationSize // batchSize
)
model.save('cnnmodel.h5')


