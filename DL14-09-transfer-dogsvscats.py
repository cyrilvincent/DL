import tensorflow as tf
import tensorflow.keras as keras

model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

for layer in model.layers[:25]:
    layer.trainable = False

x = model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dense(30, activation="relu")(x)
x =  keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.models.Model(inputs=model.input, outputs=x)

model.summary()

#keras.utils.plot_model(model, to_file='data/h5/model.png', show_shapes=True, show_layer_names=True)

model.compile(loss='binary_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])

trainset = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

batchSize = 16

trainGenerator = trainset.flow_from_directory(
        'data/dogsvscats/train',
        target_size=(224, 224),
        subset = 'training',
        class_mode="binary",
        batch_size=batchSize)

validationGenerator = trainset.flow_from_directory(
        'data/dogsvscats/validation',
        target_size=(224, 224),
        class_mode="binary",
        subset = 'validation',
        batch_size=batchSize)

checkpointer = keras.callbacks.ModelCheckpoint(filepath = 'data/h5/vgg16-dogsvscats-ckpt-{epoch:03d}-{accuracy:.3f}.h5',
                                                         monitor='accuracy'
                                                        )
model.fit(
        trainGenerator,
        epochs=30,
        validation_data=validationGenerator,
        callbacks=[checkpointer]
)

model.save('data/dogsvscats/vgg16model.h5')
model.save_weights('data/dogsvscats/vgg16model-weights.h5')



