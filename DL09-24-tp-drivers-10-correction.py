import pandas as pd
import tensorflow.keras as keras

print("Load CSV")
df = pd.read_csv("data/state-farm-distracted-driver-detection/vgg16-bottleneck-train.large.csv")
print(df.head())
x = df.iloc[:,2:]
y = df.iloc[:,1]
y = keras.utils.to_categorical(y)
print(y)

model = keras.Sequential()
model.add(keras.layers.Dense(512, input_shape=(x.shape[1],)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(256))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(x, y, epochs=50, batch_size=1, validation_split=0.2)
print(model.evaluate(x,y))

model.save("data/h5/drivers_10_mlp.h5")
