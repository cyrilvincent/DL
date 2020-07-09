import tensorflow.keras as keras

x = None
y = None

model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

model1 = model.output
model1 = keras.layers.Flatten()(model1)
model1 = keras.layers.Dense(30, activation="relu")(model1)
model1 = keras.layers.Dense(30, activation="relu")(model1)
model1 =  keras.layers.Dense(30, activation="relu")(model1)
model1 = keras.layers.Dense(3, activation="softmax")(model1)

model2 = model.output
model2 = keras.layers.Flatten()(model1)
model2 = keras.layers.Dense(50, activation="relu")(model1)
model2 = keras.layers.Dense(1, activation="sigmoid")(model1)

model = keras.models.Model(inputs=model.input, outputs=[model1, model2])

model.fit(x, y)
predicted = model.predict(x)
print(predicted[0]) #3 proba in a list
print(predicted[1][0]) #1 float
