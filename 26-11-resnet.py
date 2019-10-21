import keras
import keras.applications

model = keras.applications.resnet50.ResNet50(weights="imagenet")

print(len(model.layers))

print(model.summary())