import numpy as np
import random
import scipy.optimize as opt
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)
noise = 5
def f(x):
    delta = (np.random.rand(x.shape[0]) - 0.5) * noise
    # f(x) = 2.5x.sin(0.7x)+2
    return  2.5 * x * np.sin(0.7 * x) + 2 + delta

dataset_x = np.linspace(0, 10, 1000)
dataset_y = f(dataset_x)
plt.plot(dataset_x, dataset_x * 2.5 * np.sin(dataset_x * 0.7) + 2 , color="green")
plt.scatter(dataset_x[::10], dataset_y[::10])
# plt.show()

x = dataset_x.reshape(-1, 1)
scaler = pp.RobustScaler()
scaler.fit(x)
x = scaler.transform(x)
y = np.c_[dataset_x, dataset_y] # Add a column

scalery = pp.MinMaxScaler()
scalery.fit(y)
y = scalery.transform(y)

model = tf.keras.Sequential([
      tf.keras.layers.Dense(200, activation='relu', input_shape=(x.shape[1],)),
      tf.keras.layers.Dense(200, activation='relu'),
      tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(loss='mse')
model.fit(x, y, validation_split=0.2, epochs=200)

score = model.evaluate(x, y)
print(score)
pred = model.predict(x)
pred = scalery.inverse_transform(pred)
plt.plot(pred[:,0], pred[:,1], color="red")
plt.show()
