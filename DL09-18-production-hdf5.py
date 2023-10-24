import sklearn.preprocessing

import tensorflow as tf
import pandas


tf.random.set_seed(1)

dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)

scaler = sklearn.preprocessing.RobustScaler()
scaler.fit(x)
X = scaler.transform(x)

model = tf.keras.models.load_model("data/h5/cancer-mlp.h5")
predicted = model.predict(x)
print(predicted[0][0], y.values[0])

