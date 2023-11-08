import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.model_selection as ms
import sklearn.preprocessing as pp

tf.random.set_seed(0)
dataframe = pd.read_csv("data/auto/mpg.csv")
dataframe = dataframe.dropna()
dataframe['Origin'] = dataframe['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataframe = pd.get_dummies(dataframe, columns=['Origin'], prefix='', prefix_sep='') # identique à to_categorical
print(dataframe)
y = dataframe.MPG
x = dataframe.drop("MPG", axis=1)
scaler = pp.RobustScaler()
scaler.fit(x)
x = scaler.transform(x)

model = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(1)
  ])

model.compile(loss='mse')

history = model.fit(x, y, validation_split=0.2, epochs=100)

score = model.evaluate(x, y)
print(score)

pred = model.predict(x).reshape(-1)

a = plt.axes(aspect='equal')
plt.scatter(y, pred)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims, color="red")
plt.show()






