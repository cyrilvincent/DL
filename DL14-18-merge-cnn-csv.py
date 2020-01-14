

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, Concatenate

img_width, img_height = 150, 150

train_data_dir = '../data/dogsvscats/train'
validation_data_dir = '../data/dogsvscats/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

#model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape = (img_width,img_height,3))


model = applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape = (img_width,img_height,3))
print(model.summary())

for layer in model.layers[:25]:
    layer.trainable = False

model = Flatten()(model.output)
csvmodel = Input(shape=(2,))
model = Concatenate([model, csvmodel], axis=1) # Same row size !!
model = Dense(256, activation='relu')(model)
model = Dropout(0.5)(model)
model = Dense(1, activation='sigmoid')
# Add extra info from CSV with 2 columns

model.build()
print(model.summary())

for layer in model.layers[:25]:
    layer.trainable = False




# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
model.summary()

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples)

model.save_weights("dogsvscats/fine-tuning-weights.h5")
model.save("dogsvscats/fine-tuning.h5")
