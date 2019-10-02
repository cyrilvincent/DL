import keras
import keras.applications

# Enleve les 2 derniers niveaux et en ajoute 3
model = keras.applications.inception_v3.InceptionV3(include_top=True, weights="imagenet")

# Mauvais exemple car dans un inception un layer peut avoir plusieurs parents ce qui est impossible à fire avec sequential
print(len(model.layers))
print(model.summary())

newModel = keras.models.Sequential()
#newModel.add(model.layers[1])
#newModel.layers[0].batch_input_shape = model.layers[0].input_shape
for l in model.layers[:-1]:
    newModel.add(l)
newModel.add(keras.layers.Dense(1024,activation='relu'))
newModel.add(keras.layers.Dense(10,activation='softmax'))
#newModel.layers[0].batch_input_shape = model.layers[0].input_shape
print(len(newModel.layers))
newModel.build()
print(newModel.summary())

# Entraine seulement les 3 derniers layers
for l in newModel.layers[:-3]:
    l.trainable=False