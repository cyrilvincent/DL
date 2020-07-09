import tensorflow.keras as keras

checkpointer = keras.callbacks.ModelCheckpoint(filepath = 'data/h5/vgg16-ckpt-{epoch:03d}-{accuracy:.3f}.h5',
                                                         monitor='accuracy'
                                                        )
model = keras.applications.vgg16.VGG16()
model.fit(
    None, #"Replace by your data"
    callbacks=[checkpointer],
    )