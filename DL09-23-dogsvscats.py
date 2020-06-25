import pandas as pd

print("Load CSV")
df = pd.read_csv("data/dogsvscats/vgg16-bottleneck-train.large.csv")
print(df.head())
x = df.iloc[:,2:]
y = df.iloc[:,1]

import tensorflow.keras as keras
model = keras.Sequential()
model.add(keras.layers.Dense(512, input_shape=(x.shape[1],)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(256))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=0.0001),
              metrics=['accuracy'])
model.summary()

history = model.fit(x, y, epochs=50, batch_size=10, validation_split=0.2)
eval = model.evaluate(x, y)
print(eval)