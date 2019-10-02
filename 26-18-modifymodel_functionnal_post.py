import keras
import keras.applications

# Enleve les 2 derniers niveaux et en ajoute 3
model = keras.applications.inception_v3.InceptionV3(include_top=False, weights="imagenet")

print(len(model.layers))

print(model.output) # Dernier layer (celui avant GlobalAveragePooling2d)

layer = keras.layers.GlobalAveragePooling2D()(model.output)
layer = keras.layers.Dense(1024,activation='relu')(layer)
#Add as many dense layers / Fully Connected layers required
layer = keras.layers.Dense(10,activation='softmax')(layer)
model = keras.models.Model(model.input,layer)

print(len(model.layers))
for l in model.layers[:]:
    print(l)

# Entraine seulement les 3 derniers layers
for l in model.layers[:-3]:
    l.trainable=False