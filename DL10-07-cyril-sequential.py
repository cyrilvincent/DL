import tensorflow.keras as keras

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1))
        model.add(keras.layers.Activation('sigmoid'))
        return model

def train():
    model = CNNCyril()
    model.compile(loss='binary_crossentropy',
              metrics=['accuracy'])
    model.summary()

    trainset = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    batchSize = 16

    trainGenerator = trainset.flow_from_directory(
            'data/dogsvscats/large',
            target_size=(224, 224),
            subset = 'training',
            class_mode="binary",
            batch_size=batchSize)

    validationGenerator = trainset.flow_from_directory(
            'data/dogsvscats/large',
            target_size=(224, 224),
            class_mode="binary",
            subset = 'validation',
            batch_size=batchSize)

    model.fit(
            trainGenerator,
            epochs=30,
            validation_data=validationGenerator,
    )

    model.save('data/dogsvscats/cyrilmodel.h5')
    model.save_weights('data/dogsvscats/cyrilmodel-weights.h5')

    #30 * 15s 151ms/step - loss: 0.3309 - accuracy: 0.8512 - val_loss: 0.5017 - val_accuracy: 0.7750

if __name__ == '__main__':

    train()
    # model = keras.models.load_model("data/dogsvscats/cyrilmodel.h5")
    # img = keras.preprocessing.image.load_img("data/dogsvscats/small/validation/cats/cat.1000.jpg", target_size=(224, 224))
    # img = keras.preprocessing.image.img_to_array(img)
    # img *= 1. / 255
    # img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    # res = model.predict(img)[0][0] # image 0, output 0
    # s = "Dog"
    # if res < 0.5:
    #     s = "Cat"
    #     res = 1 - res
    # print(f"Prediction: {s} {res*100:.0f}%")





