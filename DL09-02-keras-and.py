import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,0,0,1])

import tensorflow.keras as keras
model = keras.Sequential([
    keras.layers.Dense(4, input_shape=(X.shape[1],)),
    keras.layers.Dense(8),
    keras.layers.Dense(1)
  ])

model.compile(loss="mse")
model.summary()

history = model.fit(X, y, epochs=100, batch_size=1)

res = model.predict(X)
print(res)
print(res > 0.5)

