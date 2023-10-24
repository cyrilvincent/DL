import tensorflow.keras as keras

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def Chollet():
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # 64, 64, 32

        model.add(keras.layers.Conv2D(32, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # 32, 32, 32

        model.add(keras.layers.Conv2D(64, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # 16, 16, 64

        #Dense
        model.add(keras.layers.Flatten())
        # 4096
        model.add(keras.layers.Dense(64))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1))
        model.add(keras.layers.Activation('sigmoid'))
        return model

def train():
    model = Chollet()
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.summary()
    #keras.utils.plot_model(model, to_file='data/h5/model.png', show_shapes=True, show_layer_names=True)

    trainset = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2,)

    batchSize = 16

    trainGenerator = trainset.flow_from_directory(
            'data/dogsvscats/train',
            target_size=(150, 150),
            subset = 'training',
            class_mode="binary",
            batch_size=batchSize)

    validationGenerator = trainset.flow_from_directory(
            'data/dogsvscats/train',
            target_size=(150, 150),
            class_mode="binary",
            subset = 'validation',
            batch_size=batchSize)

    model.fit(
            trainGenerator,
            epochs=2,
            validation_data=validationGenerator,
    )

    #model.save('data/dogsvscats/cholletmodel.h5')

    # 25 * 8s 81ms/step - loss: 0.4310 - accuracy: 0.8044 - val_loss: 0.5018 - val_accuracy: 0.7500

if __name__ == '__main__':

    train()
    model = keras.models.load_model("data/dogsvscats/cholletmodel.h5")
    img = keras.preprocessing.image.load_img("data/dogsvscats/small/validation/dogs/dog.1001.jpg", target_size=(150, 150))
    img = keras.preprocessing.image.img_to_array(img)
    img *= 1. / 255
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    res = model.predict(img)[0][0] # image 0, output 0
    s = "Dog"
    if res < 0.5:
        s = "Cat"
        res = 1 - res
    print(f"Prediction: {s} {res*100:.0f}%")





