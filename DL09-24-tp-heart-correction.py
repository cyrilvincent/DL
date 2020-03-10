import pandas as pd

df = pd.read_csv("data/heartdisease/data.csv")
print(df.head())
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

import tensorflow.keras as keras
model = keras.Sequential()
model.add(keras.layers.Dense(20, input_shape=(x.shape[1],)))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(20))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])
model.summary()

history = model.fit(x, y, epochs=200, batch_size=1, validation_split=0.2)
