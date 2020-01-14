from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2
X=cancer['data']
y=cancer['target']

import tensorflow.keras as keras
model = keras.Sequential([
    keras.layers.Dense(30, input_shape=(X.shape[1],)),
    keras.layers.Dense(1)
  ])

model.compile(loss="mse")
model.summary()

history = model.fit(X, y, epochs=10)

