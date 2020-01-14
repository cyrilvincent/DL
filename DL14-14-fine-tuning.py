import tensorflow.keras as keras

model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
model.compile(optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

checkpointer = keras.callbacks.ModelCheckpoint(filepath = 'data/h5/vgg16-ckpt-{epoch:03d}-{accuracy:.3f}.h5',
                                                         monitor='accuracy'
                                                        )
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min')
model.fit(
    None, #"Replace by your data"
    callbacks=[checkpointer, es],
    )