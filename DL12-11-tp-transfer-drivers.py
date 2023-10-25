import tensorflow.keras as keras

model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

# model =

model.summary()

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

trainset = None #TODO

batchSize = 16

trainGenerator = trainset.flow_from_directory(
        'data/state-farm-distracted-driver-detection/train-224',
        #TODO
        )

validationGenerator = trainset.flow_from_directory(
        'data/state-farm-distracted-driver-detection/train-224',
        #TODO
        )

model.fit(
    #TODO
)

model.save('data/state-farm-distracted-driver-detection/vgg16model.h5')




