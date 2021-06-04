from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras import applications

model = applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
newModel = Sequential()
for l in model.layers:
    newModel.add(l)
newModel.add(Flatten())
model = newModel
model.build()
print(model.summary())


