import tensorflow.keras as keras

checkpointer = keras.callbacks.ModelCheckpoint(filepath = 'data/h5/vgg16-ckpt-{epoch:03d}-{accuracy:.3f}.h5',
                                                         monitor='accuracy'
                                                        )
model = keras.applications.vgg16.VGG16()
model.optimizer = keras.optimizers.SGD(nesterov=True, lr=1e-5)
model.fit(
    None, #"Replace by your data"
    callbacks=[checkpointer],
    )