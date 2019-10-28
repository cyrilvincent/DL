import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'dogsvscats/train'
validation_data_dir = 'dogsvscats/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 20
batch_size = 8

# train_data_dir = 'dogsvscats/small/train'
# validation_data_dir = 'dogsvscats/small/validation'
# nb_train_samples = 20
# nb_validation_samples = 4
# epochs = 1
# batch_size = 1


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('dogsvscats/vgg16-bottleneck-train.npy', 'wb'),
            bottleneck_features_train)
    np.savetext(open('dogsvscats/vgg16-bottleneck-train.csv', 'w'),
            bottleneck_features_train, delimiter=",", newline="\n")

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('dogsvscats/vgg16-bottleneck-validation.npy', 'wb'),
            bottleneck_features_validation)
    np.savetext(open('dogsvscats/vgg16-bottleneck-validation.csv', 'w'),
                bottleneck_features_train, delimiter=",", newline="\n")


def train_top_model():
    train_data = np.load(open('dogsvscats/vgg16-bottleneck-train.npy','rb'))
    #train_data = np.savetext(open('dogsvscats/vgg16-bottleneck-train.csv', 'r'), delimiter=",", newline="\n")
    train_labels = np.array(
        [0] * (int(nb_train_samples / 2)) + [1] * (int(nb_train_samples / 2))) #50% cat & 50% dog

    validation_data = np.load(open('dogsvscats/vgg16-bottleneck-validation.npy','rb'))
    # validation_data = np.savetext(open('dogsvscats/vgg16-bottleneck-validation.csv', 'r'), delimiter=",", newline="\n")
    validation_labels = np.array(
        [0] * (int(nb_validation_samples / 2)) + [1] * (int(nb_validation_samples / 2)))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(3, activation='softmax')) #sum outputs = 1

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights("dogsvscats/transfer-strategy-weights.h5")
    model.save("dogsvscats/transfer-strategy.h5")

save_bottlebeck_features()
train_top_model()