import pandas as pd

print("Load CSV")
df = pd.read_csv("data/state-farm-distracted-driver-detection/vgg16-bottleneck-train.small.csv")
print(df.head())
x = df.iloc[:,2:]
y = df.iloc[:,1]
x=x[y < 2]
y=y[y < 2]
print(y)

import tensorflow.keras as keras
model = keras.Sequential()
model.add(keras.layers.Dense(512, input_shape=(x.shape[1],)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(256))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(x, y, epochs=50, batch_size=10, validation_split=0.2)
print(model.evaluate(x,y))
