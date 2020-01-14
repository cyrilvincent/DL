import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
import json

nb_train_samples = 2000
batch_size = 20

datagen = ImageDataGenerator(rescale=1. / 255)

model = applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
newModel = Sequential()
for l in model.layers:
    newModel.add(l)
newModel.add(Flatten())
model = newModel
model.build()
print(model.summary())


generator = datagen.flow_from_directory(
    "data/state-farm-distracted-driver-detection/train-224",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_train = model.predict(generator)
with open(f"data/state-farm-distracted-driver-detection/train-224/db.json", "r") as f:
    db = json.loads(f.read())
print("Save CSV")
with open('data/state-farm-distracted-driver-detection/vgg16-bottleneck-train.large.csv','w') as f:
    for item in zip(db["data"],bottleneck_features_train):
        s = item[0]["path"]+"/"+item[0]["name"]+","
        s += item[0]["path"][-1]
        for i in item[1]:
            s+=f",{float(i)}"
        s += "\n"
        f.write(s)
        print(s)




