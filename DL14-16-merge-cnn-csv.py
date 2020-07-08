

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

model = applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape = (img_width,img_height,3))
print(model.summary())

for layer in model.layers[:19]:
    layer.trainable = False

model = Flatten()(model.output)
csvmodel = Input(shape=(30,))
model = Concatenate([model, csvmodel], axis=1)
model = Dense(256, activation='relu')(model)
model = Dropout(0.5)(model)
model = Dense(1, activation='sigmoid')

model.build()
print(model.summary())

for layer in model.layers[:19]:
    layer.trainable = False
