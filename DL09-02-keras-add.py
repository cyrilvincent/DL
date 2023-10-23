import numpy as np
import tensorflow as tf
tf.random.set_seed(1)

X = []
y = []
for i in range(8):
    for j in range(8):
        X.append([i,j])
        y.append([i+j])
X=np.array(X, dtype=float)
y=np.array(y, dtype=float)
print(X)
print(y)
X = (X - 3.5) / 3.5 # 0 center
y = (y - 3.5) / 3.5
print(X)
print(y)

import tensorflow.keras as keras
model = keras.Sequential([
    keras.layers.Dense(16, input_shape=(X.shape[1],)),
    keras.layers.Dense(16),
    keras.layers.Dense(1)
  ])

model.compile(loss="mse")
model.summary()

history = model.fit(X, y, epochs=10, batch_size=1)

res = model.predict(X)
print(res)
for i in range(8):
    for j in range(8):
        predict = int(round(res[i * 8 + j][0] * 3.5 + 3.5))
        print(f"{i}+{j}={predict} {i+j==predict}")


