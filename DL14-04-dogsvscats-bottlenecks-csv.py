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
model.summary()


generator = datagen.flow_from_directory(
    "data/dogsvscats/train",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_train = model.predict(generator)
with open(f"data/dogsvscats/train/db.json", "r") as f:
    db = json.loads(f.read())
print("Save CSV")
with open('data/dogsvscats/vgg16-bottleneck-train.large.csv','w') as f:
    for item in zip(db["data"],bottleneck_features_train):
        s = item[0]["path"]+"/"+item[0]["name"]+","
        if "dog" in item[0]["name"]:
            s += "0"
        else:
            s+="1"
        for i in item[1]:
            s+=f",{float(i)}"
        s += "\n"
        f.write(s)
        print(s)
# 7*7*512 = 25088 features
# 25088 / 64 = 392
# Diviser l'importance feature par 392 pour trouver les meilleurs filtres



