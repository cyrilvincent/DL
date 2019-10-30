

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks.callbacks import ModelCheckpoint

img_width, img_height = 150, 150

train_data_dir = 'wafer/img-train'
validation_data_dir = 'wafer/img-test'
nb_train_samples = 4000
nb_validation_samples = 1000
epochs = 20
batch_size = 8

model = applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape = (img_width,img_height,3))
print(model.summary())

newModel = Sequential()
for l in model.layers:
    newModel.add(l)
newModel.add(Flatten())
newModel.add(Dense(1024, activation='relu'))
newModel.add(Dropout(0.5))
newModel.add(Dense(1024, activation='relu'))
newModel.add(Dropout(0.5))
newModel.add(Dense(9, activation='softmax'))
model = newModel
model.build()
print(model.summary())

for layer in model.layers[:25]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
)

# test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

#validation_generator = test_datagen.flow_from_directory(
validation_generator = train_datagen.flow_from_directory(
    #validation_data_dir,
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

checkpointer = ModelCheckpoint(filepath = 'wafer/ts-ckpt-{epoch:03d}-{accuracy:.3f}.h5',
                               monitor='accuracy'
                               )

model.fit_generator(
    train_generator,
    #steps_per_epoch=nb_train_samples // batch_size,
    steps_per_epoch=int((nb_train_samples * 0.8) / batch_size),
    epochs=epochs,
    validation_data=validation_generator,
    #validation_steps=nb_validation_samples // batch_size
    validation_steps=int((nb_train_samples * 0.2) / batch_size),
    callbacks=[checkpointer],
)

model.save_weights("wafer/transfer-strategy-weights.h5")
model.save("wafer/transfer-strategy.h5")
