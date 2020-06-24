from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2
X=cancer['data']
y=cancer['target']

print(X)
print(y)

import tensorflow.keras as keras
model = keras.Sequential([
    keras.layers.Dense(30, input_shape=(X.shape[1],)),
    keras.layers.Dense(30),
    keras.layers.Dense(30),
    keras.layers.Dense(1)
  ])

model.compile(loss="mse", metrics="accuracy")
model.summary()

history = model.fit(X, y, epochs=200)

predicted = model.predict(X)
print(model.evaluate(X,y))

import numpy as np
predicted = np.where(predicted > 0.5,1,0)
predicted = predicted.reshape(-1)

print(predicted)
errors = predicted - y
print(errors)
print(len(errors[errors == 0]) / len(y))


