from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks.callbacks import ModelCheckpoint

img_width, img_height = 150, 150

train_data_dir = 'wafer/img-train'
validation_data_dir = 'wafer/img-test'
nb_train_samples = 4000
nb_validation_samples = 1000
epochs = 20
batch_size = 1

input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape)) #(150,150,32)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #(75,75,32)

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #(37,37,32)

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #(18,18,64)

model.add(Flatten()) #20736
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    #subset='training'
)

validation_generator = test_datagen.flow_from_directory(
#validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    #train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    #subset='validation'
)

checkpointer = ModelCheckpoint(filepath = 'wafer/fss-ckpt-{epoch:03d}-{val_accuracy:.3f}.h5',
                               monitor='val_accuracy'
                               )
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    #steps_per_epoch=int((nb_train_samples * 0.8) / batch_size),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    #validation_steps=int((nb_train_samples * 0.2) / batch_size),
    callbacks=[checkpointer]
)

model.save_weights('wafer/full-strategy-weights.h5')
model.save('wafer/full-strategy.h5')
