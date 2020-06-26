# Prendre data/state-farm-distracted-driver-detection/train-224/c0 et c1
# S'inspirer grandement du DL10-07-cyril-sequential

import tensorflow.keras as keras

def CNNCyril():
        model = keras.models.Sequential()
        #TODO
        return model

def train():
    model = CNNCyril()
    #TODO Compile

    trainset = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2,
        )

    batchSize = 16

    trainGenerator = None #TODO

    validationGenerator = None #TODO

    #TODO fit

    model.save('data/state-farm-distracted-driver-detection/cyrilmodel.h5')

if __name__ == '__main__':
    train()
