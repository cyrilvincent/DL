# Prendre data/state-farm-distracted-driver-detection/train-224/c0 et c1
# S'inspirer grandement du DL10-07-cyril-sequential

import tensorflow.keras as keras

def CNNCyril():
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3), padding="same"))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # 112, 112, 32

        model.add(keras.layers.Conv2D(64, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # 56, 56, 64

        model.add(keras.layers.Conv2D(128, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # 28, 28, 128

        model.add(keras.layers.Conv2D(128, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # 14, 14, 128

        #Dense
        model.add(keras.layers.Flatten())
        # 25088
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10))
        model.add(keras.layers.Activation('softmax'))
        return model

def train():
    model = CNNCyril()
    model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()

    trainset = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2,shear_range=0.2
        )

    batchSize = 4

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
            epochs=10,
            validation_data=validationGenerator,
            batch_size=batchSize
    )

    model.save('data/state-farm-distracted-driver-detection/cyrilmodel.h5')

if __name__ == '__main__':
    train()
