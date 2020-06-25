from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

def VGG_16():
    vis_input = Input(shape=(224,224,3), name="VGG16")
    x = ZeroPadding2D((1, 1))(vis_input)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu')(x)

    model = Model(input=vis_input, output=x)

    model.load_weights("vgg16_weights.h5")