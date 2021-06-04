from sklearn.datasets import load_breast_cancer
import tensorflow.keras as keras
import tensorflow as tf
import pandas
import sklearn.preprocessing as pp

tf.random.set_seed(1)

dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)
print(x.shape)
X_train,X_test,y_train,y_test = ms.train_test_split(X,y)


import tensorflow as tf
import tensorflow.keras as keras
model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X_train.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=200, batch_size=10)
eval = model.evaluate(X_test, y_test)
print(eval)

